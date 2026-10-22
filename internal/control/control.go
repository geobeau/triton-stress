// Package control is the control API of one triton-stress process: GET /ready says whether
// its startup has finished, and POST /start releases the load. An operator that runs
// several processes against one server polls /ready on each, then posts /start to all of
// them so that they offer load at the same time.
package control

import (
	"context"
	"errors"
	"net"
	"net/http"
	"sync"
	"sync/atomic"
	"time"
)

// Server is a running control endpoint.
type Server struct {
	srv   *http.Server
	ln    net.Listener
	ready atomic.Bool
	start chan struct{}
	once  sync.Once
}

// Serve listens on addr and serves the control API in the background. /ready and /start
// answer 503 until SetReady is called.
func Serve(addr string) (*Server, error) {
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return nil, err
	}
	s := &Server{ln: ln, start: make(chan struct{})}
	s.srv = &http.Server{Handler: s.Handler(), ReadHeaderTimeout: 10 * time.Second}
	go func() {
		if err := s.srv.Serve(ln); err != nil && !errors.Is(err, http.ErrServerClosed) {
			// the listener is gone; nothing else to do for a control endpoint
			_ = err
		}
	}()
	return s, nil
}

// Handler serves GET /ready and POST /start.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/ready", s.handleReady)
	mux.HandleFunc("/start", s.handleStart)
	return mux
}

func (s *Server) handleReady(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.Header().Set("Allow", http.MethodGet)
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	if !s.ready.Load() {
		http.Error(w, "starting", http.StatusServiceUnavailable)
		return
	}
	_, _ = w.Write([]byte("ready\n")) // the client may be gone; nothing to report it to
}

func (s *Server) handleStart(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", http.MethodPost)
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	if !s.ready.Load() {
		http.Error(w, "starting", http.StatusServiceUnavailable)
		return
	}
	s.once.Do(func() { close(s.start) })
	_, _ = w.Write([]byte("started\n")) // the client may be gone; nothing to report it to
}

// SetReady marks the startup as finished: /ready answers 200 from now on, and /start is
// accepted.
func (s *Server) SetReady() { s.ready.Store(true) }

// Wait blocks until /start has been posted, and returns ctx's error if ctx ends first.
func (s *Server) Wait(ctx context.Context) error {
	select {
	case <-s.start:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// Addr is the address being served.
func (s *Server) Addr() string { return s.ln.Addr().String() }

// Close stops the server.
func (s *Server) Close() error {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	return s.srv.Shutdown(ctx)
}
