// Package kservefake is an in-process fake of Triton's KServe v2 HTTP endpoint, for tests
// only: nothing in the binary imports it. It serves model "m" (input x INT32[-1,8],
// output p FP32[-1,1], max_batch_size 16), checks that every infer request is a well-formed
// binary-extension request, and answers in the binary extension.
package kservefake

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// Response modes.
const (
	OK          = iota
	BadShape    // the output has the wrong shape
	Unavailable // 503 with a JSON error body
	NoBinary    // the output comes back as JSON data, not binary
	ShortBody   // the body is shorter than binary_data_size says
)

// Server is the fake.
type Server struct {
	Addr string

	// Delay holds every infer response for this many nanoseconds.
	Delay atomic.Int64
	// Mode is one of the response modes above.
	Mode atomic.Int32
	// CloseAfter, when > 0, adds Connection: close to every n-th response (counted over
	// all connections), which makes the server end that connection.
	CloseAfter atomic.Int64

	Received atomic.Int64 // infer requests that were well-formed
	Bad      atomic.Int64 // infer requests that were not
	Metadata atomic.Int64 // metadata and config requests

	mu       sync.Mutex
	lastErr  string
	lastBody []byte
	srv      *http.Server
}

// New starts the fake on a free port and stops it when the test ends.
func New(tb testing.TB) *Server {
	tb.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		tb.Fatal(err)
	}
	s := &Server{Addr: ln.Addr().String()}
	s.srv = &http.Server{Handler: s, ReadHeaderTimeout: 10 * time.Second}
	go func() {
		if err := s.srv.Serve(ln); !errors.Is(err, http.ErrServerClosed) {
			tb.Error(err)
		}
	}()
	tb.Cleanup(func() {
		if err := s.srv.Close(); err != nil {
			tb.Error(err)
		}
	})
	return s
}

// LastError is the reason the last malformed infer request was refused.
func (s *Server) LastError() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.lastErr
}

// LastBody is the last well-formed infer request's body (header and tensors).
func (s *Server) LastBody() []byte {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.lastBody
}

func (s *Server) fail(w http.ResponseWriter, code int, msg string) {
	s.mu.Lock()
	s.lastErr = msg
	s.mu.Unlock()
	reply(w, code, "", []byte(`{"error":`+strconv.Quote(msg)+`}`))
}

func reply(w http.ResponseWriter, code int, extra string, body []byte) {
	if extra != "" {
		w.Header().Set("Inference-Header-Content-Length", extra)
	}
	w.Header().Set("Content-Length", strconv.Itoa(len(body)))
	w.WriteHeader(code)
	_, _ = w.Write(body) // the client may be gone; nothing to report it to
}

// ServeHTTP implements http.Handler.
func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	path := strings.TrimPrefix(r.URL.Path, "/v2/models/m")
	if path == r.URL.Path {
		s.fail(w, http.StatusNotFound, "unknown model "+r.URL.Path)
		return
	}
	if strings.HasPrefix(path, "/versions/") {
		path = strings.TrimPrefix(path, "/versions/")
		_, path, _ = strings.Cut(path, "/")
		path = "/" + path
		if path == "/" {
			path = ""
		}
	}
	switch {
	case r.Method == http.MethodGet && path == "":
		s.Metadata.Add(1)
		reply(w, 200, "", []byte(`{"name":"m","versions":["1"],"platform":"fake","inputs":[{"name":"x","datatype":"INT32","shape":[-1,8]}],"outputs":[{"name":"p","datatype":"FP32","shape":[-1,1]}]}`))
	case r.Method == http.MethodGet && path == "/config":
		s.Metadata.Add(1)
		reply(w, 200, "", []byte(`{"name":"m","max_batch_size":16}`))
	case r.Method == http.MethodPost && path == "/infer":
		s.infer(w, r)
	default:
		s.fail(w, http.StatusNotFound, "no such endpoint "+r.Method+" "+r.URL.Path)
	}
}

func (s *Server) infer(w http.ResponseWriter, r *http.Request) {
	if ct := r.Header.Get("Content-Type"); ct != "application/octet-stream" {
		s.Bad.Add(1)
		s.fail(w, 400, "content type "+ct)
		return
	}
	hl, err := strconv.Atoi(r.Header.Get("Inference-Header-Content-Length"))
	if err != nil {
		s.Bad.Add(1)
		s.fail(w, 400, "no Inference-Header-Content-Length")
		return
	}
	body, err := io.ReadAll(r.Body)
	if err != nil || hl > len(body) {
		s.Bad.Add(1)
		s.fail(w, 400, "short body")
		return
	}
	var h struct {
		Inputs []struct {
			Name       string
			Shape      []int64
			Datatype   string
			Parameters struct {
				Size int `json:"binary_data_size"`
			}
		}
		Parameters map[string]any
	}
	if err := json.Unmarshal(body[:hl], &h); err != nil || len(h.Inputs) == 0 || h.Parameters["binary_data_output"] != true {
		s.Bad.Add(1)
		s.fail(w, 400, fmt.Sprintf("bad header: %v %v", err, h.Parameters))
		return
	}
	total := 0
	for _, in := range h.Inputs {
		total += in.Parameters.Size
	}
	if hl+total != len(body) || h.Inputs[0].Name != "x" || len(h.Inputs[0].Shape) != 2 {
		s.Bad.Add(1)
		s.fail(w, 400, "tensor sizes do not add up")
		return
	}
	s.Received.Add(1)
	s.mu.Lock()
	s.lastBody = body
	s.mu.Unlock()
	batch := h.Inputs[0].Shape[0]
	if d := s.Delay.Load(); d > 0 {
		time.Sleep(time.Duration(d))
	}
	if n := s.CloseAfter.Load(); n > 0 {
		// Closing is decided per connection by net/http's own accounting: the simple way
		// to get a Connection: close answer is to ask for it on every n-th request.
		if s.Received.Load()%n == 0 {
			w.Header().Set("Connection", "close")
		}
	}
	size := 4 * batch
	hdr := fmt.Sprintf(`{"model_name":"m","model_version":"1","outputs":[{"name":"p","datatype":"FP32","shape":[%d,1],"parameters":{"binary_data_size":%d}}]}`, batch, size)
	bin := make([]byte, size)
	switch s.Mode.Load() {
	case BadShape:
		hdr = strings.Replace(hdr, fmt.Sprintf("[%d,1]", batch), fmt.Sprintf("[%d,2]", batch), 1)
	case Unavailable:
		reply(w, 503, "", []byte(`{"error":"Request timeout expired"}`))
		return
	case NoBinary:
		hdr = fmt.Sprintf(`{"model_name":"m","outputs":[{"name":"p","datatype":"FP32","shape":[%d,1],"data":[0]}]}`, batch)
		bin = nil
	case ShortBody:
		bin = bin[:len(bin)/2]
	}
	reply(w, 200, strconv.Itoa(len(hdr)), append([]byte(hdr), bin...))
}
