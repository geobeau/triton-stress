package h1client

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"net"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/criteo/triton-stress/internal/clock"
	"github.com/criteo/triton-stress/internal/grpcclient"
	"google.golang.org/grpc/connectivity"
)

// server is a raw TCP fake for the tests (a test fixture, not part of the binary). Its
// handler runs once per accepted connection.
type server struct {
	ln      net.Listener
	mu      sync.Mutex
	conns   []net.Conn
	accepts atomic.Int64
}

func newServer(t *testing.T, handle func(nc net.Conn, br *bufio.Reader, seq int)) *server {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	s := &server{ln: ln}
	t.Cleanup(s.close)
	go func() {
		for {
			nc, err := ln.Accept()
			if err != nil {
				return
			}
			seq := int(s.accepts.Add(1)) - 1
			s.mu.Lock()
			s.conns = append(s.conns, nc)
			s.mu.Unlock()
			go func() {
				defer func() { _ = nc.Close() }()
				handle(nc, bufio.NewReader(nc), seq)
			}()
		}
	}()
	return s
}

func (s *server) addr() string { return s.ln.Addr().String() }

func (s *server) close() {
	_ = s.ln.Close()
	s.mu.Lock()
	for _, c := range s.conns {
		_ = c.Close()
	}
	s.mu.Unlock()
}

// readReq reads one request and returns its body.
func readReq(br *bufio.Reader) (string, error) {
	n := 0
	for {
		line, err := br.ReadString('\n')
		if err != nil {
			return "", err
		}
		line = strings.TrimRight(line, "\r\n")
		if line == "" {
			break
		}
		if k, v, ok := strings.Cut(line, ": "); ok && strings.EqualFold(k, "content-length") {
			n, _ = strconv.Atoi(v)
		}
	}
	b := make([]byte, n)
	_, err := io.ReadFull(br, b)
	return string(b), err
}

func reply(nc net.Conn, code int, extra, body string) error {
	_, err := fmt.Fprintf(nc, "HTTP/1.1 %d X\r\n%sContent-Length: %d\r\n\r\n%s", code, extra, len(body), body)
	return err
}

// echo answers every request with 200 and the request's body.
func echo(nc net.Conn, br *bufio.Reader, _ int) {
	for {
		b, err := readReq(br)
		if err != nil {
			return
		}
		if reply(nc, 200, "", b) != nil {
			return
		}
	}
}

func post(id string) []byte {
	return []byte(fmt.Sprintf("POST /v2/models/m/infer HTTP/1.1\r\nHost: t\r\nContent-Length: %d\r\n\r\n%s", len(id), id))
}

type event struct {
	idx      int
	from, to connectivity.State
}

type events struct {
	mu sync.Mutex
	ev []event
}

func (e *events) add(idx int, from, to connectivity.State) {
	e.mu.Lock()
	e.ev = append(e.ev, event{idx, from, to})
	e.mu.Unlock()
}

func (e *events) states(idx int) []connectivity.State {
	e.mu.Lock()
	defer e.mu.Unlock()
	var s []connectivity.State
	for _, v := range e.ev {
		if v.idx == idx {
			s = append(s, v.to)
		}
	}
	return s
}

func newGroup(t *testing.T, o Options, idx ...int) (*Group, *events) {
	t.Helper()
	ev := &events{}
	o.OnState = ev.add
	if o.BackoffMin == 0 {
		o.BackoffMin = 5 * time.Millisecond
	}
	if o.BackoffMax == 0 {
		o.BackoffMax = 40 * time.Millisecond
	}
	g, err := NewGroup(o, idx)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = g.Close() })
	return g, ev
}

func waitFor(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", what)
		}
		time.Sleep(time.Millisecond)
	}
}

func ready(t *testing.T, g *Group) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := g.WaitReady(ctx); err != nil {
		t.Fatal(err)
	}
}

// call submits a request and returns its (heap-allocated) result channel.
func call(g *Group, body string, timeout time.Duration, keep bool) <-chan *Request {
	ch := make(chan *Request, 1)
	r := &Request{Body: post(body), Timeout: timeout, Keep: keep, Done: func(r *Request) { ch <- r }}
	g.Submit(r)
	return ch
}

func get(t *testing.T, ch <-chan *Request) *Request {
	t.Helper()
	select {
	case r := <-ch:
		return r
	case <-time.After(5 * time.Second):
		t.Fatal("request did not complete")
		return nil
	}
}

func TestRequestAndKeep(t *testing.T) {
	s := newServer(t, echo)
	g, _ := newGroup(t, Options{Target: s.addr()}, 0, 1)
	ready(t, g)
	r := get(t, call(g, "hello", time.Second, true))
	if r.Status != grpcclient.StatusHTTP200 || !r.Status.IsOK() || string(r.Data) != "hello" || r.RespBytes != 5 || r.HdrLen != -1 {
		t.Fatalf("%+v", r)
	}
	if r.TReady <= 0 || r.TReady > r.TSend || r.TSend > r.TDone || r.Conn < 0 || r.Conn > 1 {
		t.Fatalf("timestamps/conn: %d %d %d conn %d", r.TReady, r.TSend, r.TDone, r.Conn)
	}
	r = get(t, call(g, "x", time.Second, false))
	if r.Status != grpcclient.StatusHTTP200 || len(r.Data) != 0 || r.RespBytes != 1 {
		t.Fatalf("%+v", r)
	}
	if lbl := r.Status.Label(); lbl != "http_200" {
		t.Fatal(lbl)
	}
}

func TestStatusMapping(t *testing.T) {
	s := newServer(t, func(nc net.Conn, br *bufio.Reader, _ int) {
		for {
			b, err := readReq(br)
			if err != nil {
				return
			}
			code, _ := strconv.Atoi(b)
			if reply(nc, code, "", `{"error":"e"}`) != nil {
				return
			}
		}
	})
	g, _ := newGroup(t, Options{Target: s.addr()}, 0)
	ready(t, g)
	for _, c := range []struct {
		code  int
		label string
		ok    bool
	}{{200, "http_200", true}, {400, "http_400", false}, {404, "http_404", false}, {429, "http_429", false},
		{500, "http_500", false}, {503, "http_503", false}, {201, "http_201", false}} {
		r := get(t, call(g, strconv.Itoa(c.code), time.Second, false))
		if r.Status.Label() != c.label || r.Status.IsOK() != c.ok {
			t.Errorf("%d: %s ok=%v", c.code, r.Status.Label(), r.Status.IsOK())
		}
	}
	if grpcclient.StatusHTTP(99) != grpcclient.StatusInvalidResponse || grpcclient.StatusHTTP(600) != grpcclient.StatusInvalidResponse {
		t.Error("out-of-range code")
	}
}

// One request in flight per connection; the rest wait in a FIFO and are served in order.
func TestFIFOAndOneInFlightPerConnection(t *testing.T) {
	var inflight, maxInflight atomic.Int64
	gate := make(chan struct{})
	s := newServer(t, func(nc net.Conn, br *bufio.Reader, _ int) {
		for {
			b, err := readReq(br)
			if err != nil {
				return
			}
			n := inflight.Add(1)
			for {
				m := maxInflight.Load()
				if n <= m || maxInflight.CompareAndSwap(m, n) {
					break
				}
			}
			<-gate
			inflight.Add(-1)
			if reply(nc, 200, "", b) != nil {
				return
			}
		}
	})
	g, _ := newGroup(t, Options{Target: s.addr()}, 0, 1)
	ready(t, g)
	var chs []<-chan *Request
	for i := 0; i < 8; i++ {
		chs = append(chs, call(g, fmt.Sprintf("r%d", i), 5*time.Second, false))
		time.Sleep(2 * time.Millisecond)
	}
	waitFor(t, "two requests at the server", func() bool { return inflight.Load() == 2 })
	time.Sleep(20 * time.Millisecond)
	if inflight.Load() != 2 {
		t.Fatalf("in flight %d, want 2", inflight.Load())
	}
	released := clock.Now()
	for i := 0; i < 8; i++ { // let one through at a time
		gate <- struct{}{}
	}
	var res []*Request
	for _, ch := range chs {
		res = append(res, get(t, ch))
	}
	if maxInflight.Load() != 2 {
		t.Fatalf("max in flight at the server %d, want 2", maxInflight.Load())
	}
	// Connections are handed out in submit order (the order the server sees two
	// connections' requests in is a race, the order they were taken in is not).
	for i := 1; i < len(res); i++ {
		if res[i].TReady < res[i-1].TReady {
			t.Fatalf("r%d took a connection before r%d", i, i-1)
		}
	}
	// The first two took a connection at once; the others waited for a free one, and that
	// wait is between their emit and their TReady.
	for i, r := range res {
		if r.Status != grpcclient.StatusHTTP200 {
			t.Fatalf("r%d: %s", i, r.Status.Label())
		}
		if i < 2 && (r.TReady >= released || r.TSend-r.TReady > int64(5*time.Millisecond)) {
			t.Errorf("r%d waited for a connection (ready %v before the release)", i, time.Duration(released-r.TReady))
		}
		if i >= 2 && r.TReady < released {
			t.Errorf("r%d took a connection while both were busy", i)
		}
		if r.TSend < r.TReady {
			t.Errorf("r%d: TSend before TReady", i)
		}
	}
}

func TestNoHealthyConnectionFailsAtOnceAsConnect(t *testing.T) {
	var dials atomic.Int64
	dial := func(ctx context.Context, _ string) (net.Conn, error) {
		dials.Add(1)
		return nil, fmt.Errorf("refused")
	}
	g, ev := newGroup(t, Options{Target: "x:1", Dialer: dial}, 0, 1)
	if g.Up() || g.Healthy() != 0 {
		t.Fatal("up")
	}
	start := time.Now()
	r := get(t, call(g, "a", time.Second, false))
	if r.Status != grpcclient.StatusConnect || r.TReady != 0 || r.TSend != 0 || r.TDone == 0 || r.Conn != -1 {
		t.Fatalf("%+v", r)
	}
	if time.Since(start) > 100*time.Millisecond {
		t.Fatal("connect failure was not immediate")
	}
	waitFor(t, "retries", func() bool { return dials.Load() > 4 })
	st := ev.states(0)
	if st[0] != connectivity.Connecting || st[1] != connectivity.TransientFailure {
		t.Fatalf("states %v", st)
	}
	// Nothing was queued.
	g.mu.Lock()
	q := g.qHead
	g.mu.Unlock()
	if q != nil {
		t.Fatal("request queued")
	}
}

func TestLastHealthyConnectionDiesWhileRequestsAreQueued(t *testing.T) {
	var accepted atomic.Int64
	hold := make(chan struct{})
	s := newServer(t, func(nc net.Conn, br *bufio.Reader, _ int) {
		accepted.Add(1)
		_, _ = readReq(br)
		<-hold // never answers: closes when told to
	})
	var allowDial atomic.Bool
	allowDial.Store(true)
	dial := func(ctx context.Context, addr string) (net.Conn, error) {
		if !allowDial.Load() {
			return nil, fmt.Errorf("refused")
		}
		var d net.Dialer
		return d.DialContext(ctx, "tcp", addr)
	}
	g, _ := newGroup(t, Options{Target: s.addr(), Dialer: dial}, 0)
	ready(t, g)
	first := call(g, "in-flight", 10*time.Second, false)
	waitFor(t, "first at the server", func() bool { return accepted.Load() == 1 })
	q1 := call(g, "q1", 10*time.Second, false)
	q2 := call(g, "q2", 10*time.Second, false)
	select {
	case <-q1:
		t.Fatal("queued request completed early")
	case <-time.After(20 * time.Millisecond):
	}
	allowDial.Store(false)
	close(hold) // the handler returns and closes the connection
	r := get(t, first)
	if r.Status != grpcclient.StatusTransport {
		t.Fatalf("in flight: %s", r.Status.Label())
	}
	for _, ch := range []<-chan *Request{q1, q2} {
		r := get(t, ch)
		if r.Status != grpcclient.StatusConnect || r.TSend != 0 {
			t.Fatalf("queued: %s", r.Status.Label())
		}
	}
	if g.Up() {
		t.Fatal("group still up")
	}
}

func TestTimeoutClosesTheConnectionAndReconnects(t *testing.T) {
	var closed atomic.Int64
	s := newServer(t, func(nc net.Conn, br *bufio.Reader, seq int) {
		if seq > 0 {
			echo(nc, br, seq)
			return
		}
		_, _ = readReq(br)
		_, err := br.ReadByte() // wait for the client to close
		if err != nil {
			closed.Add(1)
		}
	})
	g, ev := newGroup(t, Options{Target: s.addr()}, 0)
	ready(t, g)
	r := get(t, call(g, "slow", 60*time.Millisecond, false))
	if r.Status != grpcclient.StatusTimeout || r.TSend == 0 {
		t.Fatalf("%s", r.Status.Label())
	}
	waitFor(t, "server sees the close", func() bool { return closed.Load() == 1 })
	waitFor(t, "reconnect", func() bool { return g.Healthy() == 1 })
	if r := get(t, call(g, "again", time.Second, false)); r.Status != grpcclient.StatusHTTP200 {
		t.Fatalf("after reconnect: %s", r.Status.Label())
	}
	want := []connectivity.State{connectivity.Connecting, connectivity.Ready, connectivity.Idle, connectivity.Connecting, connectivity.Ready}
	st := ev.states(0)
	if fmt.Sprint(st[:len(want)]) != fmt.Sprint(want) {
		t.Fatalf("states %v, want %v", st, want)
	}
}

// A request that waits too long for a connection times out in the queue, without ever being sent.
func TestQueuedRequestTimesOutUnsent(t *testing.T) {
	var got atomic.Int64
	s := newServer(t, func(nc net.Conn, br *bufio.Reader, _ int) {
		for {
			if _, err := readReq(br); err != nil {
				return
			}
			got.Add(1)
			time.Sleep(time.Hour)
		}
	})
	g, _ := newGroup(t, Options{Target: s.addr()}, 0)
	ready(t, g)
	a := call(g, "a", 400*time.Millisecond, false)
	waitFor(t, "a at the server", func() bool { return got.Load() == 1 })
	b := call(g, "b", 80*time.Millisecond, false)
	rb := get(t, b)
	if rb.Status != grpcclient.StatusTimeout || rb.TReady != 0 || rb.TSend != 0 || rb.Conn != -1 {
		t.Fatalf("%+v", rb)
	}
	if ra := get(t, a); ra.Status != grpcclient.StatusTimeout || ra.TSend == 0 {
		t.Fatalf("a: %s", ra.Status.Label())
	}
	if got.Load() != 1 {
		t.Fatal("the queued request reached the server")
	}
}

func TestConnectionCloseHeaderEndsTheConnection(t *testing.T) {
	s := newServer(t, func(nc net.Conn, br *bufio.Reader, seq int) {
		b, err := readReq(br)
		if err != nil {
			return
		}
		if seq == 0 {
			_ = reply(nc, 200, "Connection: close\r\n", b)
			return
		}
		if err := reply(nc, 200, "", b); err != nil {
			return
		}
		echo(nc, br, seq)
	})
	g, ev := newGroup(t, Options{Target: s.addr()}, 0)
	ready(t, g)
	r := get(t, call(g, "one", time.Second, true))
	if r.Status != grpcclient.StatusHTTP200 || string(r.Data) != "one" {
		t.Fatalf("a complete response is delivered as it is: %s", r.Status.Label())
	}
	waitFor(t, "second connection", func() bool { return s.accepts.Load() == 2 && g.Healthy() == 1 })
	if r := get(t, call(g, "two", time.Second, false)); r.Status != grpcclient.StatusHTTP200 {
		t.Fatal(r.Status.Label())
	}
	st := ev.states(0)
	if fmt.Sprint(st[:5]) != fmt.Sprint([]connectivity.State{connectivity.Connecting, connectivity.Ready, connectivity.Idle, connectivity.Connecting, connectivity.Ready}) {
		t.Fatalf("states %v", st)
	}
}

func TestServerClosingAnIdleConnectionIsNoticed(t *testing.T) {
	s := newServer(t, func(nc net.Conn, br *bufio.Reader, seq int) {
		if seq == 0 {
			return // close at once
		}
		echo(nc, br, seq)
	})
	g, _ := newGroup(t, Options{Target: s.addr()}, 0)
	waitFor(t, "second connection", func() bool { return s.accepts.Load() >= 2 && g.Healthy() == 1 })
	// The first request does not run into the dead connection.
	if r := get(t, call(g, "x", time.Second, false)); r.Status != grpcclient.StatusHTTP200 {
		t.Fatal(r.Status.Label())
	}
}

func TestTransportErrors(t *testing.T) {
	for name, handler := range map[string]func(nc net.Conn, br *bufio.Reader){
		"closed before the response": func(nc net.Conn, br *bufio.Reader) { _, _ = readReq(br) },
		"cut in the body": func(nc net.Conn, br *bufio.Reader) {
			_, _ = readReq(br)
			_, _ = io.WriteString(nc, "HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\nabc")
		},
		"cut in the headers": func(nc net.Conn, br *bufio.Reader) {
			_, _ = readReq(br)
			_, _ = io.WriteString(nc, "HTTP/1.1 200 OK\r\nContent-Le")
		},
	} {
		s := newServer(t, func(nc net.Conn, br *bufio.Reader, seq int) {
			if seq == 0 {
				handler(nc, br)
				return
			}
			echo(nc, br, seq)
		})
		g, _ := newGroup(t, Options{Target: s.addr()}, 0)
		ready(t, g)
		if r := get(t, call(g, "x", time.Second, true)); r.Status != grpcclient.StatusTransport {
			t.Errorf("%s: %s", name, r.Status.Label())
		}
		waitFor(t, name+": reconnect", func() bool { return g.Healthy() == 1 })
		if r := get(t, call(g, "y", time.Second, true)); r.Status != grpcclient.StatusHTTP200 {
			t.Errorf("%s after reconnect: %s", name, r.Status.Label())
		}
		if err := g.Close(); err != nil {
			t.Fatal(err)
		}
		s.close()
	}
}

func TestInvalidResponse(t *testing.T) {
	s := newServer(t, func(nc net.Conn, br *bufio.Reader, seq int) {
		if seq == 0 {
			_, _ = readReq(br)
			_, _ = io.WriteString(nc, "this is not http\r\n\r\n")
			time.Sleep(time.Second)
			return
		}
		echo(nc, br, seq)
	})
	g, _ := newGroup(t, Options{Target: s.addr()}, 0)
	ready(t, g)
	if r := get(t, call(g, "x", time.Second, false)); r.Status != grpcclient.StatusInvalidResponse || r.TSend == 0 {
		t.Fatalf("%s", r.Status.Label())
	}
	waitFor(t, "reconnect", func() bool { return s.accepts.Load() == 2 && g.Healthy() == 1 })
}

func TestReconnectBackoff(t *testing.T) {
	var mu sync.Mutex
	var at []time.Time
	var fail atomic.Int64
	fail.Store(100) // fail every attempt until told otherwise
	s := newServer(t, echo)
	dial := func(ctx context.Context, addr string) (net.Conn, error) {
		mu.Lock()
		at = append(at, time.Now())
		mu.Unlock()
		if fail.Add(-1) >= 0 {
			return nil, fmt.Errorf("refused")
		}
		var d net.Dialer
		return d.DialContext(ctx, "tcp", addr)
	}
	g, _ := newGroup(t, Options{Target: s.addr(), Dialer: dial, BackoffMin: 20 * time.Millisecond, BackoffMax: 80 * time.Millisecond}, 0)
	waitFor(t, "6 attempts", func() bool { mu.Lock(); defer mu.Unlock(); return len(at) >= 6 })
	mu.Lock()
	gaps := make([]time.Duration, 0, 5)
	for i := 1; i < 6; i++ {
		gaps = append(gaps, at[i].Sub(at[i-1]))
	}
	mu.Unlock()
	// 20, 40, 80, 80, 80 ms with 20% jitter. A wait can only be longer than asked on a busy
	// machine, so the upper bound leaves room for scheduling delays.
	exp := []time.Duration{20, 40, 80, 80, 80}
	for i, g := range gaps {
		e := exp[i] * time.Millisecond
		if g < e*7/10 || g > e*2+250*time.Millisecond {
			t.Errorf("gap %d = %v, want about %v (%v)", i, g, e, gaps)
		}
	}
	fail.Store(0)
	waitFor(t, "connected", func() bool { return g.Healthy() == 1 })
	fail.Store(0)
	waitFor(t, "reconnected", func() bool { return g.Healthy() == 1 })
}

// After a success the backoff is reset: the first retry after a lost connection is at
// once, not after the backoff the earlier failures had built up. The backoff here is long
// next to any scheduling delay, so a slow machine cannot make a delayed retry look instant.
func TestReconnectAfterLossIsImmediate(t *testing.T) {
	var mu sync.Mutex
	var at []time.Time
	var fail atomic.Int64
	fail.Store(2) // the first two attempts are refused
	s := newServer(t, echo)
	dial := func(ctx context.Context, addr string) (net.Conn, error) {
		mu.Lock()
		at = append(at, time.Now())
		mu.Unlock()
		if fail.Add(-1) >= 0 {
			return nil, fmt.Errorf("refused")
		}
		var d net.Dialer
		return d.DialContext(ctx, "tcp", addr)
	}
	g, _ := newGroup(t, Options{Target: s.addr(), Dialer: dial, BackoffMin: time.Second, BackoffMax: time.Second}, 0)
	waitFor(t, "connected", func() bool { return g.Healthy() == 1 })
	mu.Lock()
	n := len(at)
	mu.Unlock()
	killed := time.Now()
	s.mu.Lock()
	for _, c := range s.conns {
		_ = c.Close()
	}
	s.mu.Unlock()
	waitFor(t, "a retry", func() bool { mu.Lock(); defer mu.Unlock(); return len(at) > n })
	mu.Lock()
	if gap := at[n].Sub(killed); gap > 500*time.Millisecond {
		t.Errorf("first retry after a loss came %v later, want it instant (backoff is 1s)", gap)
	}
	mu.Unlock()
	waitFor(t, "reconnected", func() bool { return g.Healthy() == 1 })
}

func TestAddress(t *testing.T) {
	for _, c := range []struct {
		in   string
		tls  bool
		want string
		err  bool
	}{
		{"h:8000", false, "h:8000", false},
		{"h", false, "h:80", false},
		{"h", true, "h:443", false},
		{"127.0.0.1:8000", true, "127.0.0.1:8000", false},
		{"dns:///h:1", false, "", true},
		{"http://h", false, "", true},
	} {
		got, err := Address(c.in, c.tls)
		if (err != nil) != c.err || got != c.want {
			t.Errorf("Address(%q,%v) = %q, %v", c.in, c.tls, got, err)
		}
	}
}

func TestCloseCompletesRequests(t *testing.T) {
	s := newServer(t, func(nc net.Conn, br *bufio.Reader, _ int) { _, _ = readReq(br); time.Sleep(time.Hour) })
	g, _ := newGroup(t, Options{Target: s.addr()}, 0)
	ready(t, g)
	a := call(g, "a", time.Minute, false)
	b := call(g, "b", time.Minute, false)
	time.Sleep(20 * time.Millisecond)
	if err := g.Close(); err != nil {
		t.Fatal(err)
	}
	if r := get(t, a); r.Status != grpcclient.StatusTransport {
		t.Fatal(r.Status.Label())
	}
	if r := get(t, b); r.Status != grpcclient.StatusConnect {
		t.Fatal(r.Status.Label())
	}
	// After Close, Submit fails as connect.
	if r := get(t, call(g, "c", time.Second, false)); r.Status != grpcclient.StatusConnect {
		t.Fatal(r.Status.Label())
	}
}

// Many requests across several connections, all completing exactly once.
func TestManyRequests(t *testing.T) {
	s := newServer(t, echo)
	g, _ := newGroup(t, Options{Target: s.addr()}, 0, 1, 2, 3)
	ready(t, g)
	const n = 5000
	var wg sync.WaitGroup
	var bad atomic.Int64
	var done atomic.Int64
	wg.Add(n)
	for i := 0; i < n; i++ {
		id := strconv.Itoa(i)
		r := &Request{Body: post(id), Timeout: 5 * time.Second, Keep: true}
		r.Done = func(r *Request) {
			if r.Status != grpcclient.StatusHTTP200 || string(r.Data) != id {
				bad.Add(1)
			}
			done.Add(1)
			wg.Done()
		}
		g.Submit(r)
	}
	wg.Wait()
	if bad.Load() != 0 || done.Load() != n {
		t.Fatalf("bad %d done %d", bad.Load(), done.Load())
	}
}
