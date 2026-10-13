package h2client

import (
	"bufio"
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"fmt"
	"io"
	"math/big"
	"net"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/criteo/triton-stress/internal/clock"
	"github.com/criteo/triton-stress/internal/grpcclient"
	pb "github.com/criteo/triton-stress/pkg/tritonpb"
	"golang.org/x/net/http2"
	"golang.org/x/net/http2/hpack"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/keepalive"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
)

// The server under test is grpc-go's own, so the client is checked against an
// independent HTTP/2 implementation. (This is a test fixture, not part of the binary.)

type fake struct {
	pb.UnimplementedGRPCInferenceServiceServer
	received    atomic.Int64
	sawDeadline atomic.Bool
	lastLen     atomic.Int64
	delay       atomic.Int64 // ns
	blockMu     sync.Mutex
	block       chan struct{}
	started     chan struct{} // receives a value when a request starts, if non-nil
	cancelled   atomic.Int64  // requests whose context ended while blocked
	err         atomic.Pointer[error]
	respLen     atomic.Int64
}

func (f *fake) setBlock(ch chan struct{}) {
	f.blockMu.Lock()
	f.block = ch
	f.blockMu.Unlock()
}

func (f *fake) ModelInfer(ctx context.Context, req *pb.ModelInferRequest) (*pb.ModelInferResponse, error) {
	f.received.Add(1)
	if _, ok := ctx.Deadline(); ok {
		f.sawDeadline.Store(true)
	}
	n := 0
	for _, b := range req.RawInputContents {
		n += len(b)
	}
	f.lastLen.Store(int64(n))
	if f.started != nil {
		select {
		case f.started <- struct{}{}:
		default:
		}
	}
	f.blockMu.Lock()
	block := f.block
	f.blockMu.Unlock()
	if block != nil {
		select {
		case <-block:
		case <-ctx.Done():
			f.cancelled.Add(1)
			return nil, ctx.Err()
		}
	}
	if d := f.delay.Load(); d > 0 {
		time.Sleep(time.Duration(d))
	}
	if e := f.err.Load(); e != nil {
		return nil, *e
	}
	rl := int(f.respLen.Load())
	if rl == 0 {
		rl = 4
	}
	return &pb.ModelInferResponse{
		ModelName:         req.ModelName,
		Outputs:           []*pb.ModelInferResponse_InferOutputTensor{{Name: "p", Datatype: "FP32", Shape: []int64{1, int64(rl / 4)}}},
		RawOutputContents: [][]byte{make([]byte, rl)},
	}, nil
}

type server struct {
	f    *fake
	addr string
	mu   sync.Mutex
	srv  *grpc.Server
	opts []grpc.ServerOption
}

func newServer(t *testing.T, f *fake, opts ...grpc.ServerOption) *server {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	s := &server{f: f, addr: l.Addr().String(), opts: opts}
	s.serve(l)
	t.Cleanup(s.stop)
	return s
}

func (s *server) serve(l net.Listener) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.srv = grpc.NewServer(s.opts...)
	pb.RegisterGRPCInferenceServiceServer(s.srv, s.f)
	go func() { _ = s.srv.Serve(l) }()
}

func (s *server) stop() {
	s.mu.Lock()
	srv := s.srv
	s.srv = nil
	s.mu.Unlock()
	if srv != nil {
		srv.Stop()
	}
}

// restart serves again on the same address.
func (s *server) restart(t *testing.T) {
	t.Helper()
	var l net.Listener
	var err error
	for i := 0; i < 50; i++ {
		if l, err = net.Listen("tcp", s.addr); err == nil {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if err != nil {
		t.Fatal(err)
	}
	s.serve(l)
}

func newConn(t *testing.T, addr string, mod func(*Options)) *Conn {
	t.Helper()
	o := Options{Target: addr, BackoffMin: 5 * time.Millisecond, BackoffMax: 50 * time.Millisecond}
	if mod != nil {
		mod(&o)
	}
	c, err := New(o)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = c.Close() })
	return c
}

func waitReady(t *testing.T, c *Conn) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := c.WaitReady(ctx); err != nil {
		t.Fatal(err)
	}
}

func waitNotReady(t *testing.T, c *Conn) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for c.Ready() {
		if time.Now().After(deadline) {
			t.Fatal("still Ready")
		}
		time.Sleep(time.Millisecond)
	}
}

// body is a ModelInferRequest with n bytes of raw input.
func body(t testing.TB, n int) []byte {
	t.Helper()
	b, err := proto.Marshal(&pb.ModelInferRequest{
		ModelName:        "m",
		Inputs:           []*pb.ModelInferRequest_InferInputTensor{{Name: "x", Datatype: "INT8", Shape: []int64{1, int64(n)}}},
		RawInputContents: [][]byte{make([]byte, n)},
	})
	if err != nil {
		t.Fatal(err)
	}
	return b
}

// start submits a request and returns the channel its completion arrives on.
func start(c *Conn, b []byte, timeout time.Duration, keep bool) (*Request, chan *Request) {
	ch := make(chan *Request, 1)
	r := &Request{Body: b, Keep: keep, Timeout: timeout, Done: func(r *Request) { ch <- r }}
	c.Submit(r)
	return r, ch
}

func wait(t *testing.T, ch chan *Request) *Request {
	t.Helper()
	select {
	case r := <-ch:
		return r
	case <-time.After(10 * time.Second):
		t.Fatal("request did not complete")
		return nil
	}
}

func do(t *testing.T, c *Conn, b []byte, timeout time.Duration) *Request {
	t.Helper()
	_, ch := start(c, b, timeout, false)
	return wait(t, ch)
}

func TestOKRoundTripAndTimestamps(t *testing.T) {
	f := &fake{}
	f.respLen.Store(16)
	s := newServer(t, f)
	c := newConn(t, s.addr, nil)
	waitReady(t, c)
	before := clockNow()
	_, ch := start(c, body(t, 1172), time.Second, true)
	r := wait(t, ch)
	if r.Status != grpcclient.StatusOK {
		t.Fatalf("status %s", r.Status.Label())
	}
	if before > r.TReady || r.TReady > r.TSend || r.TSend > r.TDone {
		t.Fatalf("timestamps out of order: %d %d %d %d", before, r.TReady, r.TSend, r.TDone)
	}
	resp := &pb.ModelInferResponse{}
	if err := proto.Unmarshal(r.Data, resp); err != nil {
		t.Fatalf("response data does not decode: %v", err)
	}
	if r.RespBytes != len(r.Data) || len(resp.RawOutputContents[0]) != 16 {
		t.Fatalf("size %d, data %d", r.RespBytes, len(r.Data))
	}
	if f.lastLen.Load() != 1172 {
		t.Fatalf("server saw %d input bytes", f.lastLen.Load())
	}
	if f.sawDeadline.Load() {
		t.Fatal("a deadline reached the server")
	}
	// Without Keep the data is not copied, but the size is known.
	_, ch = start(c, body(t, 8), time.Second, false)
	if r := wait(t, ch); r.Status != grpcclient.StatusOK || r.Data != nil || r.RespBytes == 0 {
		t.Fatalf("%+v", r)
	}
}

func TestStatusesFromTheServer(t *testing.T) {
	f := &fake{}
	s := newServer(t, f)
	c := newConn(t, s.addr, nil)
	waitReady(t, c)
	for code, label := range map[codes.Code]string{
		codes.NotFound: "NOT_FOUND", codes.Unavailable: "UNAVAILABLE", codes.ResourceExhausted: "RESOURCE_EXHAUSTED",
		codes.Internal: "INTERNAL", codes.InvalidArgument: "INVALID_ARGUMENT", codes.Unauthenticated: "UNAUTHENTICATED",
	} {
		e := error(status.Error(code, "server says no"))
		f.err.Store(&e)
		if r := do(t, c, body(t, 8), time.Second); r.Status.Label() != label {
			t.Fatalf("%v reported as %s", code, r.Status.Label())
		}
	}
	f.err.Store(nil)
	if r := do(t, c, body(t, 8), time.Second); r.Status != grpcclient.StatusOK {
		t.Fatal(r.Status.Label())
	}
}

func TestTimeoutCancelsTheStream(t *testing.T) {
	f := &fake{block: make(chan struct{})}
	s := newServer(t, f)
	c := newConn(t, s.addr, nil)
	waitReady(t, c)
	start0 := time.Now()
	r := do(t, c, body(t, 8), 100*time.Millisecond)
	if r.Status != grpcclient.StatusTimeout {
		t.Fatalf("status %s", r.Status.Label())
	}
	if d := time.Since(start0); d < 100*time.Millisecond || d > time.Second {
		t.Fatalf("timed out after %v", d)
	}
	// The stream was reset: the server's handler saw its context end.
	deadline := time.Now().Add(2 * time.Second)
	for f.cancelled.Load() == 0 && time.Now().Before(deadline) {
		time.Sleep(5 * time.Millisecond)
	}
	if f.cancelled.Load() != 1 {
		t.Fatal("the server was not told to cancel")
	}
	// The connection is fine afterwards.
	close(f.block)
	f.setBlock(nil)
	if r := do(t, c, body(t, 8), time.Second); r.Status != grpcclient.StatusOK {
		t.Fatal(r.Status.Label())
	}
}

func TestTimeoutsAreIndependentAndInOrder(t *testing.T) {
	f := &fake{}
	f.delay.Store(int64(300 * time.Millisecond))
	s := newServer(t, f)
	c := newConn(t, s.addr, nil)
	waitReady(t, c)
	// A request with a long timeout in front of ones with short timeouts: only the
	// short ones expire, and the long one still completes.
	_, long := start(c, body(t, 8), 5*time.Second, false)
	var shorts []chan *Request
	for i := 0; i < 5; i++ {
		_, ch := start(c, body(t, 8), 50*time.Millisecond, false)
		shorts = append(shorts, ch)
	}
	for _, ch := range shorts {
		if r := wait(t, ch); r.Status != grpcclient.StatusTimeout {
			t.Fatalf("short request: %s", r.Status.Label())
		}
	}
	if r := wait(t, long); r.Status != grpcclient.StatusOK {
		t.Fatalf("long request: %s", r.Status.Label())
	}
}

func TestNotReadyCompletesAsConnectAtOnce(t *testing.T) {
	c, err := New(Options{
		Target: "x:1", BackoffMin: 5 * time.Millisecond, BackoffMax: 20 * time.Millisecond,
		Dialer: func(context.Context, string) (net.Conn, error) { return nil, fmt.Errorf("connection refused") },
	})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c.Close() }()
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	if c.WaitReady(ctx) == nil {
		t.Fatal("WaitReady succeeded")
	}
	t0 := time.Now()
	r, ch := start(c, []byte("x"), time.Second, false)
	if got := wait(t, ch); got != r || r.Status != grpcclient.StatusConnect || r.TReady != 0 || r.TSend != 0 || r.TDone == 0 {
		t.Fatalf("%+v", r)
	}
	if time.Since(t0) > 50*time.Millisecond {
		t.Fatalf("connect took %v", time.Since(t0))
	}
}

func TestReconnectAndEvents(t *testing.T) {
	f := &fake{block: make(chan struct{}), started: make(chan struct{}, 1)}
	s := newServer(t, f)
	var mu sync.Mutex
	var events []string
	c := newConn(t, s.addr, func(o *Options) {
		o.OnState = func(from, to connectivity.State) {
			mu.Lock()
			events = append(events, from.String()+">"+to.String())
			mu.Unlock()
		}
	})
	waitReady(t, c)
	// A request in flight when the connection dies completes as transport.
	_, ch := start(c, body(t, 8), 10*time.Second, false)
	<-f.started
	s.stop()
	if r := wait(t, ch); r.Status != grpcclient.StatusTransport {
		t.Fatalf("mid-flight failure reported as %s", r.Status.Label())
	}
	waitNotReady(t, c)
	// While it is down, requests fail at once as connect.
	t0 := time.Now()
	if r := do(t, c, body(t, 8), 10*time.Second); r.Status != grpcclient.StatusConnect || time.Since(t0) > 50*time.Millisecond {
		t.Fatalf("%s after %v", r.Status.Label(), time.Since(t0))
	}
	f.setBlock(nil)
	s.restart(t)
	waitReady(t, c)
	if r := do(t, c, body(t, 8), time.Second); r.Status != grpcclient.StatusOK {
		t.Fatal(r.Status.Label())
	}
	mu.Lock()
	defer mu.Unlock()
	want := []string{"IDLE>CONNECTING", "CONNECTING>READY", "READY>IDLE", "IDLE>CONNECTING"}
	if len(events) < len(want) {
		t.Fatalf("events %v", events)
	}
	for i, w := range want {
		if events[i] != w {
			t.Fatalf("events %v, want prefix %v", events, want)
		}
	}
}

func TestFirstReconnectIsImmediate(t *testing.T) {
	f := &fake{}
	s := newServer(t, f)
	var mu sync.Mutex
	var ready []time.Time
	c := newConn(t, s.addr, func(o *Options) {
		o.BackoffMin, o.BackoffMax = 2*time.Second, 2*time.Second
		o.OnState = func(_, to connectivity.State) {
			if to == connectivity.Ready {
				mu.Lock()
				ready = append(ready, time.Now())
				mu.Unlock()
			}
		}
	})
	waitReady(t, c)
	// Kill only the connection (the listener stays up): the first retry must not wait for
	// the 2 s backoff.
	killed := time.Now()
	_ = c.cur.Load().nc.Close()
	deadline := time.Now().Add(time.Second)
	for {
		mu.Lock()
		n := len(ready)
		mu.Unlock()
		if n >= 2 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("no reconnect within 1 s with a 2 s backoff")
		}
		time.Sleep(time.Millisecond)
	}
	mu.Lock()
	defer mu.Unlock()
	if d := ready[1].Sub(killed); d > 500*time.Millisecond {
		t.Fatalf("reconnected after %v", d)
	}
}

func TestBackoffGrowsAfterFailedAttempts(t *testing.T) {
	var mu sync.Mutex
	var at []time.Time
	c, err := New(Options{
		Target: "x:1", BackoffMin: 20 * time.Millisecond, BackoffMax: 80 * time.Millisecond,
		Dialer: func(context.Context, string) (net.Conn, error) {
			mu.Lock()
			at = append(at, time.Now())
			mu.Unlock()
			return nil, fmt.Errorf("connection refused")
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c.Close() }()
	time.Sleep(600 * time.Millisecond)
	mu.Lock()
	defer mu.Unlock()
	if len(at) < 5 {
		t.Fatalf("%d attempts", len(at))
	}
	// gaps: ~20, 40, 80, 80 ms (jitter +-20%), capped at the maximum
	for i := 1; i < 5; i++ {
		gap := at[i].Sub(at[i-1])
		want := min(20*time.Millisecond<<(i-1), 80*time.Millisecond)
		if gap < want*7/10 || gap > want*14/10+30*time.Millisecond {
			t.Fatalf("attempt gap %d is %v, want about %v", i, gap, want)
		}
	}
}

func TestLargeBodyFlowControl(t *testing.T) {
	f := &fake{}
	s := newServer(t, f)
	c := newConn(t, s.addr, nil)
	waitReady(t, c)
	// Far above the 64 KiB default window and the 16 KiB frame size.
	for _, n := range []int{20 << 10, 70 << 10, 3 << 20} {
		if r := do(t, c, body(t, n), 10*time.Second); r.Status != grpcclient.StatusOK {
			t.Fatalf("%d bytes: %s", n, r.Status.Label())
		}
		if got := f.lastLen.Load(); got != int64(n) {
			t.Fatalf("server saw %d of %d bytes", got, n)
		}
	}
	// Many large requests at once share the connection window.
	var chs []chan *Request
	for i := 0; i < 40; i++ {
		_, ch := start(c, body(t, 200<<10), 20*time.Second, false)
		chs = append(chs, ch)
	}
	for _, ch := range chs {
		if r := wait(t, ch); r.Status != grpcclient.StatusOK {
			t.Fatal(r.Status.Label())
		}
	}
}

func TestLargeResponse(t *testing.T) {
	f := &fake{}
	f.respLen.Store(3 << 20)
	s := newServer(t, f)
	c := newConn(t, s.addr, nil)
	waitReady(t, c)
	for i := 0; i < 3; i++ { // enough to need connection- and stream-level window updates
		_, ch := start(c, body(t, 8), 10*time.Second, true)
		r := wait(t, ch)
		if r.Status != grpcclient.StatusOK {
			t.Fatal(r.Status.Label())
		}
		resp := &pb.ModelInferResponse{}
		if err := proto.Unmarshal(r.Data, resp); err != nil || len(resp.RawOutputContents[0]) != 3<<20 {
			t.Fatalf("bad response (%v)", err)
		}
	}
}

func TestMaxConcurrentStreamsQueuesRequests(t *testing.T) {
	f := &fake{}
	f.delay.Store(int64(20 * time.Millisecond))
	s := newServer(t, f, grpc.MaxConcurrentStreams(2))
	c := newConn(t, s.addr, nil)
	waitReady(t, c)
	time.Sleep(50 * time.Millisecond) // let the server's SETTINGS arrive
	const n = 20
	var chs []chan *Request
	t0 := time.Now()
	for i := 0; i < n; i++ {
		_, ch := start(c, body(t, 8), 10*time.Second, false)
		chs = append(chs, ch)
	}
	var maxReady time.Duration
	for _, ch := range chs {
		r := wait(t, ch)
		if r.Status != grpcclient.StatusOK {
			t.Fatal(r.Status.Label())
		}
		if d := time.Duration(r.TReady) - time.Duration(clockOf(t0)); d > maxReady {
			maxReady = d
		}
	}
	// 20 requests, 2 at a time, 20 ms each: the last one waits ~180 ms for a stream. That
	// wait is between submit and t_ready (backpressure), not in the service time.
	if maxReady < 100*time.Millisecond {
		t.Fatalf("the last request found capacity after only %v", maxReady)
	}
	if time.Since(t0) < 150*time.Millisecond {
		t.Fatalf("finished in %v: the limit was not respected", time.Since(t0))
	}
}

func TestBackpressureShowsBetweenSubmitAndReady(t *testing.T) {
	f := &fake{}
	f.delay.Store(int64(50 * time.Millisecond))
	s := newServer(t, f, grpc.MaxConcurrentStreams(1))
	c := newConn(t, s.addr, nil)
	waitReady(t, c)
	time.Sleep(50 * time.Millisecond)
	a, cha := start(c, body(t, 8), 10*time.Second, false)
	b, chb := start(c, body(t, 8), 10*time.Second, false)
	wait(t, cha)
	wait(t, chb)
	if gap := time.Duration(b.TReady - a.TDone); gap < -5*time.Millisecond || gap > 20*time.Millisecond {
		t.Fatalf("second request became ready %v after the first finished", gap)
	}
	if time.Duration(b.TReady-a.TReady) < 40*time.Millisecond {
		t.Fatalf("second request was ready only %v after the first", time.Duration(b.TReady-a.TReady))
	}
}

func TestGracefulShutdownGoaway(t *testing.T) {
	f := &fake{block: make(chan struct{}), started: make(chan struct{}, 1)}
	s := newServer(t, f)
	c := newConn(t, s.addr, nil)
	waitReady(t, c)
	_, ch := start(c, body(t, 8), 10*time.Second, false)
	<-f.started
	s.mu.Lock()
	srv := s.srv
	s.mu.Unlock()
	go srv.GracefulStop() // sends GOAWAY, waits for the request in flight
	waitNotReady(t, c)
	// The request already in flight is not cut: it completes once the server answers.
	select {
	case <-ch:
		t.Fatal("completed before the server answered")
	case <-time.After(100 * time.Millisecond):
	}
	// A new request does not wait for a reconnect.
	if r := do(t, c, body(t, 8), time.Second); r.Status != grpcclient.StatusConnect {
		t.Fatalf("after GOAWAY: %s", r.Status.Label())
	}
	close(f.block)
	if r := wait(t, ch); r.Status != grpcclient.StatusOK {
		t.Fatalf("request in flight at GOAWAY: %s", r.Status.Label())
	}
}

func TestCloseCompletesInFlightAsTransport(t *testing.T) {
	f := &fake{block: make(chan struct{}), started: make(chan struct{}, 1)}
	s := newServer(t, f)
	c, err := New(Options{Target: s.addr})
	if err != nil {
		t.Fatal(err)
	}
	waitReady(t, c)
	_, ch := start(c, body(t, 8), 10*time.Second, false)
	<-f.started
	if err := c.Close(); err != nil {
		t.Fatal(err)
	}
	if r := wait(t, ch); r.Status != grpcclient.StatusTransport {
		t.Fatalf("%s", r.Status.Label())
	}
	if c.Ready() {
		t.Fatal("Ready after Close")
	}
	if r := do(t, c, body(t, 8), time.Second); r.Status != grpcclient.StatusConnect {
		t.Fatalf("%s", r.Status.Label())
	}
	close(f.block)
}

func TestManyRequestsOutOfOrder(t *testing.T) {
	f := &fake{started: make(chan struct{}, 1)}
	s := newServer(t, f)
	c := newConn(t, s.addr, nil)
	waitReady(t, c)
	// One request is held open until the end (not for a fixed time, which a busy machine
	// can outlast) while thousands of others come and go: the stream table has to grow
	// past its first size.
	hold := make(chan struct{})
	f.setBlock(hold)
	_, slow := start(c, body(t, 8), 10*time.Second, false)
	<-f.started
	f.setBlock(nil)
	var done atomic.Int64
	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 400; i++ {
				ch := make(chan *Request, 1)
				r := &Request{Body: body(t, 8), Timeout: 10 * time.Second, Done: func(r *Request) { ch <- r }}
				c.Submit(r)
				if got := <-ch; got.Status == grpcclient.StatusOK {
					done.Add(1)
				}
			}
		}()
	}
	wg.Wait()
	if done.Load() != 8*400 {
		t.Fatalf("%d of %d completed OK", done.Load(), 8*400)
	}
	tr := c.cur.Load()
	tr.mu.Lock()
	grown := len(tr.streams.slots) > 1024
	tr.mu.Unlock()
	if !grown {
		t.Fatal("the stream table never had to grow")
	}
	close(hold)
	if r := wait(t, slow); r.Status != grpcclient.StatusOK {
		t.Fatal(r.Status.Label())
	}
}

func TestStreamIDsRunningOutMoveToANewConnection(t *testing.T) {
	f := &fake{}
	s := newServer(t, f)
	c := newConn(t, s.addr, nil)
	waitReady(t, c)
	old := c.cur.Load()
	old.mu.Lock()
	old.nextID = maxWindow - idSlack - 4 // odd, two streams before the reserve
	old.mu.Unlock()
	// A few more requests are fine; then the transport stops taking new ones and the
	// Conn switches to a fresh connection, without failing anything.
	for i := 0; i < 10; i++ {
		if r := do(t, c, body(t, 8), 5*time.Second); r.Status != grpcclient.StatusOK && r.Status != grpcclient.StatusConnect {
			t.Fatalf("request %d: %s", i, r.Status.Label())
		}
	}
	deadline := time.Now().Add(5 * time.Second)
	for cur := c.cur.Load(); cur == nil || cur == old; cur = c.cur.Load() {
		if time.Now().After(deadline) {
			t.Fatal("still on the old connection")
		}
		time.Sleep(time.Millisecond)
	}
	if r := do(t, c, body(t, 8), 5*time.Second); r.Status != grpcclient.StatusOK {
		t.Fatal(r.Status.Label())
	}
}

func TestServerPingsAreAnswered(t *testing.T) {
	f := &fake{}
	s := newServer(t, f, grpc.KeepaliveParams(keepalive.ServerParameters{Time: 50 * time.Millisecond, Timeout: 100 * time.Millisecond}))
	c := newConn(t, s.addr, nil)
	waitReady(t, c)
	// The server pings every 50 ms and closes the connection if a ping is not answered
	// within 100 ms; it must stay up.
	time.Sleep(500 * time.Millisecond)
	if !c.Ready() {
		t.Fatal("connection lost: a server PING went unanswered")
	}
	if r := do(t, c, body(t, 8), time.Second); r.Status != grpcclient.StatusOK {
		t.Fatal(r.Status.Label())
	}
}

// frameLog is a minimal HTTP/2 server that records the frames the client sends.
func TestClientNeverSendsPing(t *testing.T) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = l.Close() }()
	var mu sync.Mutex
	types := map[http2.FrameType]int{}
	var gotPreface atomic.Bool
	go func() {
		nc, err := l.Accept()
		if err != nil {
			return
		}
		defer func() { _ = nc.Close() }()
		br := bufio.NewReader(nc)
		pre := make([]byte, len(clientPreface))
		if _, err := io.ReadFull(br, pre); err != nil || string(pre) != clientPreface {
			return
		}
		gotPreface.Store(true)
		fr := http2.NewFramer(nc, br)
		if err := fr.WriteSettings(); err != nil {
			return
		}
		for {
			f, err := fr.ReadFrame()
			if err != nil {
				return
			}
			mu.Lock()
			types[f.Header().Type]++
			mu.Unlock()
			if sf, ok := f.(*http2.SettingsFrame); ok && !sf.IsAck() {
				if err := fr.WriteSettingsAck(); err != nil {
					return
				}
			}
		}
	}()
	c := newConn(t, l.Addr().String(), nil)
	waitReady(t, c)
	time.Sleep(300 * time.Millisecond) // idle: a keepalive would show up now
	if err := c.Close(); err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	defer mu.Unlock()
	if !gotPreface.Load() {
		t.Fatal("no preface")
	}
	if types[http2.FramePing] != 0 {
		t.Fatalf("the client sent %d PING frames", types[http2.FramePing])
	}
	if types[http2.FrameSettings] == 0 || types[http2.FrameWindowUpdate] == 0 {
		t.Fatalf("expected SETTINGS and a connection WINDOW_UPDATE, got %v", types)
	}
}

func TestInfersOverTLS(t *testing.T) {
	cert := selfSigned(t)
	f := &fake{}
	s := newServer(t, f, grpc.Creds(credentials.NewServerTLSFromCert(&cert)))
	c := newConn(t, s.addr, func(o *Options) { o.TLS = grpcclient.TLSInsecure })
	waitReady(t, c)
	for i := 0; i < 20; i++ {
		if r := do(t, c, body(t, 1172), time.Second); r.Status != grpcclient.StatusOK {
			t.Fatal(r.Status.Label())
		}
	}
	// System roots do not trust a self-signed certificate; plaintext cannot talk to a TLS server.
	for _, mode := range []grpcclient.TLSMode{grpcclient.TLSSystem, grpcclient.TLSOff} {
		c2, err := New(Options{Target: s.addr, TLS: mode, BackoffMin: 5 * time.Millisecond})
		if err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 300*time.Millisecond)
		if c2.WaitReady(ctx) == nil {
			t.Fatalf("mode %d got ready against a TLS server it cannot verify or speak to", mode)
		}
		cancel()
		if err := c2.Close(); err != nil {
			t.Fatal(err)
		}
	}
}

func selfSigned(t *testing.T) tls.Certificate {
	t.Helper()
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	tmpl := &x509.Certificate{
		SerialNumber: big.NewInt(1), Subject: pkix.Name{CommonName: "localhost"},
		NotBefore: time.Now().Add(-time.Hour), NotAfter: time.Now().Add(time.Hour),
		DNSNames: []string{"localhost"}, IPAddresses: []net.IP{net.ParseIP("127.0.0.1")},
		KeyUsage: x509.KeyUsageDigitalSignature, ExtKeyUsage: []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
	}
	der, err := x509.CreateCertificate(rand.Reader, tmpl, tmpl, &key.PublicKey, key)
	if err != nil {
		t.Fatal(err)
	}
	return tls.Certificate{Certificate: [][]byte{der}, PrivateKey: key}
}

func TestOptionsAndTargets(t *testing.T) {
	if _, err := New(Options{Target: "x:1", BackoffMin: time.Second, BackoffMax: time.Millisecond}); err == nil {
		t.Fatal("backoff max below min accepted")
	}
	if _, err := New(Options{Target: "x:1", TLS: 9}); err == nil {
		t.Fatal("bad TLS mode accepted")
	}
	for in, want := range map[string]string{
		"127.0.0.1:8001":            "127.0.0.1:8001",
		"dns:///triton.svc:8001":    "triton.svc:8001",
		"passthrough:///10.0.0.1:1": "10.0.0.1:1",
		"triton.svc":                "triton.svc:443",
	} {
		c, err := New(Options{Target: in, Dialer: func(context.Context, string) (net.Conn, error) { return nil, io.EOF }})
		if err != nil {
			t.Fatal(in, err)
		}
		if c.addr != want {
			t.Fatalf("%q resolved to %q, want %q", in, c.addr, want)
		}
		if err := c.Close(); err != nil {
			t.Fatal(err)
		}
	}
}

func TestRequestHeadersDecode(t *testing.T) {
	// The pre-encoded header block must decode, with a standard HPACK decoder, to the
	// pseudo-headers and gRPC headers a server expects.
	for _, tlsOn := range []bool{false, true} {
		blk := requestHeaders("triton:8001", tlsOn)
		got := map[string]string{}
		d := hpackDecoder(func(k, v string) { got[k] = v })
		if err := d(blk); err != nil {
			t.Fatal(err)
		}
		scheme := "http"
		if tlsOn {
			scheme = "https"
		}
		want := map[string]string{
			":method": "POST", ":scheme": scheme, ":path": grpcclient.InferMethod, ":authority": "triton:8001",
			"content-type": "application/grpc+proto", "te": "trailers", "user-agent": "triton-stress",
		}
		for k, v := range want {
			if got[k] != v {
				t.Fatalf("header %s = %q, want %q (all: %v)", k, got[k], v, got)
			}
		}
	}
}

func TestStatusMappings(t *testing.T) {
	for code, want := range map[int]grpcclient.Status{
		400: grpcclient.StatusInternal, 401: grpcclient.StatusUnauthenticated, 403: grpcclient.StatusPermissionDenied,
		404: grpcclient.StatusUnimplemented, 429: grpcclient.StatusUnavailable, 502: grpcclient.StatusUnavailable,
		503: grpcclient.StatusUnavailable, 504: grpcclient.StatusUnavailable, 500: grpcclient.StatusUnknown,
		200: grpcclient.StatusInternal, 0: grpcclient.StatusUnknown,
	} {
		if got := statusFromHTTP(code); got != want {
			t.Errorf("HTTP %d -> %s, want %s", code, got.Label(), want.Label())
		}
	}
	if statusFromCode(5) != grpcclient.Status(codes.NotFound) || statusFromCode(99) != grpcclient.StatusUnknown {
		t.Fatal("statusFromCode")
	}
	if rstStatus(http2.ErrCodeRefusedStream) != grpcclient.StatusTransport ||
		rstStatus(http2.ErrCodeCancel) != grpcclient.StatusCanceled ||
		rstStatus(http2.ErrCodeEnhanceYourCalm) != grpcclient.StatusResourceExhausted ||
		rstStatus(http2.ErrCodeInternal) != grpcclient.StatusInternal {
		t.Fatal("rstStatus")
	}
}

func TestStreamTableGrowsAndFinds(t *testing.T) {
	s := newStreamTable()
	var all []*Request
	for i := 0; i < 5000; i++ { // ids 1, 3, 5, ...: more than the initial 1024 slots
		r := &Request{id: uint32(2*i + 1)}
		s.put(r)
		all = append(all, r)
	}
	for _, r := range all {
		if s.get(r.id) != r {
			t.Fatalf("lost stream %d", r.id)
		}
	}
	if s.get(2) != nil || s.get(20001) != nil {
		t.Fatal("found a stream that is not there")
	}
	for _, r := range all {
		s.del(r)
	}
	for _, r := range all {
		if s.get(r.id) != nil {
			t.Fatal("still there after delete")
		}
	}
}

func clockNow() int64 { return clock.Now() }

// clockOf is the clock value of the wall-clock instant t0 (t0 must be recent).
func clockOf(t0 time.Time) int64 { return clock.Now() - int64(time.Since(t0)) }

func hpackDecoder(emit func(k, v string)) func([]byte) error {
	d := hpack.NewDecoder(4096, func(f hpack.HeaderField) { emit(f.Name, f.Value) })
	return func(b []byte) error {
		if _, err := d.Write(b); err != nil {
			return err
		}
		return d.Close()
	}
}
