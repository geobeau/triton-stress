// Package h1client is a minimal HTTP/1.1 client for one purpose: sending pre-built
// requests over a fixed set of persistent connections at a high rate and timing them.
//
// It is not net/http: the load path must never dial inside a request, and the plan's
// rules about connection failure (nothing waits for a reconnect) need the connections to
// be owned here. A Group is the connections of one worker: one request in flight per
// connection, due requests waiting in a FIFO for a free one, reconnection in the
// background.
//
// # Rules
//
//   - A request goes to a free connection at once (written by the submitting goroutine)
//     or waits in the Group's FIFO; a connection that finishes a response takes the head
//     of the FIFO. TReady is the moment a free connection took the request and TSend just
//     before its bytes are written, so the wait is backpressure and nothing else.
//   - With no healthy connection at Submit, the request completes at once as
//     StatusConnect. When the last healthy connection dies, the queued requests complete
//     as StatusConnect at that moment. Nothing waits for a reconnect.
//   - Any read or write error ends the connection: the request in flight completes as
//     StatusTransport. A response with Connection: close (or HTTP/1.0 without
//     keep-alive, or a body that runs to the end of the connection) completes normally
//     and then ends the connection. A malformed response is StatusInvalidResponse and
//     ends it. A timeout (from Submit, a wait for a connection included) is
//     StatusTimeout and closes the connection, since its state is unknown.
//   - A lost connection is re-dialled at once; failed attempts back off from BackoffMin
//     to BackoffMax, doubling, and the backoff resets after a success.
//   - A connection that is idle is read from, so a server closing it is noticed at once.
//
// Statuses of responses are StatusHTTP(code); StatusHTTP200 is success.
//
// One goroutine per connection reads the responses; nothing is allocated per request
// and no goroutine is created for one.
package h1client

import (
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"math/rand/v2"
	"net"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/criteo/triton-stress/internal/clock"
	"github.com/criteo/triton-stress/internal/grpcclient"
	"google.golang.org/grpc/connectivity"
)

// connectTimeout bounds one connection attempt, the TLS handshake included.
const connectTimeout = 5 * time.Second

// readBufSize is each connection's read buffer.
const readBufSize = 4096

// keepAliveIdle is the shortest idle time before a connection's first TCP keepalive
// probe; each connection draws its own from [keepAliveIdle, 2*keepAliveIdle].
const keepAliveIdle = 15 * time.Second  // net.Dialer default value

// Options configures a Group.
type Options struct {
	// Target is host:port; without a port it is 80 (443 with TLS).
	Target string
	TLS    grpcclient.TLSMode
	// BackoffMin and BackoffMax bound the reconnect backoff (defaults 10 ms and 1 s).
	BackoffMin, BackoffMax time.Duration
	// OnState is called, from the connection's own goroutine, on every state change of
	// connection idx (the index given to NewGroup).
	OnState func(idx int, from, to connectivity.State)
	// Dialer replaces the network dialer (tests).
	Dialer func(ctx context.Context, addr string) (net.Conn, error)
}

// Address is the host:port a target resolves to: the port defaults to 80, or 443 with
// TLS. It is also the Host header value.
func Address(target string, tlsOn bool) (string, error) {
	if strings.Contains(target, "://") {
		return "", fmt.Errorf("target %q: only host:port is supported for http", target)
	}
	if _, _, err := net.SplitHostPort(target); err == nil {
		return target, nil
	}
	port := "80"
	if tlsOn {
		port = "443"
	}
	addr := net.JoinHostPort(target, port)
	if _, _, err := net.SplitHostPort(addr); err != nil {
		return "", fmt.Errorf("target %q: %w", target, err)
	}
	return addr, nil
}

// Group is a set of connections sharing one FIFO.
type Group struct {
	o      Options
	addr   string
	tlsCfg *tls.Config

	mu      sync.Mutex
	conns   []*conn
	idle    []*conn // free healthy connections (a stack: the last one freed is reused first)
	healthy int
	up      atomic.Int32 // healthy, readable without the lock
	closed  bool
	qHead   *Request // requests waiting for a connection, in submit order
	qTail   *Request
	dHead   *Request // every unfinished request, in submit order, so in deadline order
	dTail   *Request
	changed chan struct{} // closed and replaced on every state change

	reapWake chan struct{}
	cancel   context.CancelFunc
	wg       sync.WaitGroup
}

// NewGroup creates one connection per entry of idx (the indices reported to OnState and
// in Request.Conn) and starts connecting them in the background.
func NewGroup(o Options, idx []int) (*Group, error) {
	if o.BackoffMin <= 0 {
		o.BackoffMin = 10 * time.Millisecond
	}
	if o.BackoffMax <= 0 {
		o.BackoffMax = time.Second
	}
	if o.BackoffMax < o.BackoffMin {
		return nil, fmt.Errorf("reconnect backoff max %v is below min %v", o.BackoffMax, o.BackoffMin)
	}
	addr, err := Address(o.Target, o.TLS != grpcclient.TLSOff)
	if err != nil {
		return nil, err
	}
	host, _, _ := net.SplitHostPort(addr)
	g := &Group{o: o, addr: addr, changed: make(chan struct{}), reapWake: make(chan struct{}, 1)}
	switch o.TLS {
	case grpcclient.TLSOff:
	case grpcclient.TLSSystem:
		g.tlsCfg = &tls.Config{MinVersion: tls.VersionTLS12, ServerName: host, NextProtos: []string{"http/1.1"}}
	case grpcclient.TLSInsecure:
		g.tlsCfg = &tls.Config{MinVersion: tls.VersionTLS12, InsecureSkipVerify: true, NextProtos: []string{"http/1.1"}}
	default:
		return nil, fmt.Errorf("bad TLS mode %d", o.TLS)
	}
	ctx, cancel := context.WithCancel(context.Background())
	g.cancel = cancel
	for _, i := range idx {
		c := &conn{g: g, idx: i, idleAt: -1}
		c.state.Store(int32(connectivity.Idle))
		g.conns = append(g.conns, c)
	}
	g.wg.Add(len(g.conns) + 1)
	for _, c := range g.conns {
		go func() { defer g.wg.Done(); c.run(ctx) }()
	}
	go func() { defer g.wg.Done(); g.reapLoop(ctx) }()
	return g, nil
}

// Len is the number of connections.
func (g *Group) Len() int { return len(g.conns) }

// Healthy is the number of connections that are up.
func (g *Group) Healthy() int {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.healthy
}

// Up reports whether any connection is healthy (lock-free). A dispatcher uses it to fail
// a request as connect without involving the Group; Submit decides again under its lock.
func (g *Group) Up() bool { return g.up.Load() > 0 }

// WaitReady blocks until every connection is up or ctx ends.
func (g *Group) WaitReady(ctx context.Context) error {
	for {
		g.mu.Lock()
		ch, n := g.changed, g.healthy
		g.mu.Unlock()
		if n == len(g.conns) {
			return nil
		}
		select {
		case <-ch:
		case <-ctx.Done():
			return fmt.Errorf("%d of %d connections up: %w", n, len(g.conns), ctx.Err())
		}
	}
}

// notifyLocked wakes the WaitReady callers.
func (g *Group) notifyLocked() {
	close(g.changed)
	g.changed = make(chan struct{})
}

// Close ends the connections; requests in flight complete as transport errors and those
// waiting as connect.
func (g *Group) Close() error {
	g.mu.Lock()
	g.closed = true
	var ncs []net.Conn
	for _, c := range g.conns {
		if c.nc != nil {
			ncs = append(ncs, c.nc)
		}
	}
	g.mu.Unlock()
	g.cancel()
	for _, nc := range ncs {
		_ = nc.Close()
	}
	g.wg.Wait()
	return nil
}

// Submit sends r. It never blocks: it is written on a free connection, queued for one,
// or, with no healthy connection, completed at once as StatusConnect.
func (g *Group) Submit(r *Request) {
	r.reset()
	now := clock.Now()
	r.deadline = now + int64(r.Timeout)
	g.mu.Lock()
	if g.healthy == 0 || g.closed {
		g.mu.Unlock()
		r.Status, r.TDone = grpcclient.StatusConnect, clock.Now()
		r.Done(r)
		return
	}
	g.linkDL(r)
	if n := len(g.idle); n > 0 {
		c := g.idle[n-1]
		g.idle = g.idle[:n-1]
		c.idleAt = -1
		nc := c.assign(r, now)
		g.mu.Unlock()
		c.send(nc, r)
		return
	}
	g.pushQ(r)
	g.mu.Unlock()
}

func (g *Group) linkDL(r *Request) {
	r.inDL = true
	r.dprev, r.dnx = g.dTail, nil
	if g.dTail != nil {
		g.dTail.dnx = r
	} else {
		g.dHead = r
		select { // the reaper may be sleeping with nothing to wait for
		case g.reapWake <- struct{}{}:
		default:
		}
	}
	g.dTail = r
}

func (g *Group) unlinkDL(r *Request) {
	if !r.inDL {
		return
	}
	r.inDL = false
	if r.dprev != nil {
		r.dprev.dnx = r.dnx
	} else {
		g.dHead = r.dnx
	}
	if r.dnx != nil {
		r.dnx.dprev = r.dprev
	} else {
		g.dTail = r.dprev
	}
	r.dprev, r.dnx = nil, nil
}

func (g *Group) pushQ(r *Request) {
	r.inQ = true
	r.qprev, r.qnx = g.qTail, nil
	if g.qTail != nil {
		g.qTail.qnx = r
	} else {
		g.qHead = r
	}
	g.qTail = r
}

func (g *Group) unlinkQ(r *Request) {
	if !r.inQ {
		return
	}
	r.inQ = false
	if r.qprev != nil {
		r.qprev.qnx = r.qnx
	} else {
		g.qHead = r.qnx
	}
	if r.qnx != nil {
		r.qnx.qprev = r.qprev
	} else {
		g.qTail = r.qprev
	}
	r.qprev, r.qnx = nil, nil
}

// failQueuedLocked takes every waiting request off the queue, as connect failures.
func (g *Group) failQueuedLocked(now int64, out []*Request) []*Request {
	for r := g.qHead; r != nil; r = g.qHead {
		g.unlinkQ(r)
		g.unlinkDL(r)
		r.Status, r.TDone = grpcclient.StatusConnect, now
		out = append(out, r)
	}
	return out
}

func deliver(rs []*Request) {
	for _, r := range rs {
		r.Done(r)
	}
}

// reapLoop times requests out. They are in submit order, so only the head can be the
// first to expire.
func (g *Group) reapLoop(ctx context.Context) {
	timer := time.NewTimer(time.Hour)
	defer timer.Stop()
	var done []*Request
	var ncs []net.Conn
	for {
		g.mu.Lock()
		head := g.dHead
		var wait time.Duration
		if head != nil {
			wait = time.Duration(head.deadline - clock.Now())
		}
		if head != nil && wait <= 0 {
			now := clock.Now()
			for h := g.dHead; h != nil && h.deadline <= now; h = g.dHead {
				g.unlinkDL(h)
				if h.inQ { // still waiting for a connection
					g.unlinkQ(h)
					h.Status, h.TDone = grpcclient.StatusTimeout, now
					done = append(done, h)
				} else if c := h.c; c != nil && c.cur.Load() == h { // in flight: its connection's state is unknown
					h.timedOut = true
					ncs = append(ncs, c.nc)
				}
			}
			g.mu.Unlock()
			for _, nc := range ncs {
				_ = nc.Close() // the connection's goroutine completes the request as timeout
			}
			ncs = ncs[:0]
			deliver(done)
			done = done[:0]
			continue
		}
		g.mu.Unlock()
		if head == nil {
			select {
			case <-g.reapWake:
			case <-ctx.Done():
				return
			}
			continue
		}
		if !timer.Stop() {
			select {
			case <-timer.C:
			default:
			}
		}
		timer.Reset(wait)
		select {
		case <-timer.C:
		case <-g.reapWake:
		case <-ctx.Done():
			return
		}
	}
}

// conn is one persistent connection.
type conn struct {
	g   *Group
	idx int

	state atomic.Int32
	cur   atomic.Pointer[Request] // the request in flight; written under g.mu

	// Under g.mu.
	nc     net.Conn
	up     bool
	idleAt int // position in g.idle, -1 when not there
}

func (c *conn) set(to connectivity.State) {
	from := connectivity.State(c.state.Swap(int32(to)))
	if from == to {
		return
	}
	g := c.g
	g.mu.Lock()
	g.notifyLocked()
	g.mu.Unlock()
	if g.o.OnState != nil {
		g.o.OnState(c.idx, from, to)
	}
}

// assign makes r the request in flight on c (g.mu held) and stamps TReady and TSend; the
// caller writes r.Body to the returned connection once it has released the lock.
func (c *conn) assign(r *Request, ready int64) net.Conn {
	r.c, r.Conn, r.inQ = c, c.idx, false
	r.TReady = ready
	r.TSend = clock.Now()
	c.cur.Store(r)
	return c.nc
}

// send writes the request. A failure closes the connection; its goroutine, which owns the
// completion of the request, then finds the error on its next read.
func (c *conn) send(nc net.Conn, r *Request) {
	if _, err := nc.Write(r.Body); err != nil {
		_ = nc.Close()
	}
}

func (c *conn) dial(ctx context.Context) (net.Conn, error) {
	g := c.g
	ctx, cancel := context.WithTimeout(ctx, connectTimeout)
	defer cancel()
	var nc net.Conn
	var err error
	if g.o.Dialer != nil {
		nc, err = g.o.Dialer(ctx, g.addr)
	} else {
		// All the connections are dialled at once and most stay idle (the idle stack reuses
		// the last connection freed), so with Go's fixed 15 s default they all probed in the
		// same second, every 15 s. This can have a signiifcant impact if a large number
		// of concurrent connections are open, even if not used.
		// A randomly shifted per-connection idle time spreads the probes out.
		d := net.Dialer{KeepAliveConfig: net.KeepAliveConfig{
			Enable: true,
			Idle:   keepAliveIdle + rand.N(keepAliveIdle),
		}}
		nc, err = d.DialContext(ctx, "tcp", g.addr)
	}
	if err != nil {
		return nil, fmt.Errorf("error while dialing: %w", err)
	}
	if g.tlsCfg != nil {
		tc := tls.Client(nc, g.tlsCfg.Clone())
		if err := tc.HandshakeContext(ctx); err != nil {
			_ = nc.Close()
			return nil, fmt.Errorf("tls handshake: %w", err)
		}
		nc = tc
	}
	return nc, nil
}

// run keeps the connection up. The first retry after a lost connection is immediate;
// after failed attempts the delay starts at BackoffMin and doubles to BackoffMax.
func (c *conn) run(ctx context.Context) {
	g := c.g
	defer c.set(connectivity.Shutdown)
	delay := time.Duration(0)
	for ctx.Err() == nil {
		c.set(connectivity.Connecting)
		nc, err := c.dial(ctx)
		if err != nil {
			if ctx.Err() != nil {
				return
			}
			c.set(connectivity.TransientFailure)
			if delay == 0 {
				delay = g.o.BackoffMin
			}
			d := time.Duration(float64(delay) * (1 + 0.2*(2*rand.Float64()-1)))
			select {
			case <-time.After(d):
			case <-ctx.Done():
				return
			}
			delay = min(delay*2, g.o.BackoffMax)
			continue
		}
		delay = 0
		c.serve(nc)
		c.set(connectivity.Idle)
	}
}

// serve runs one established connection until it ends.
func (c *conn) serve(nc net.Conn) {
	g := c.g
	br := newReader(nc)
	var next *Request
	g.mu.Lock()
	if g.closed {
		g.mu.Unlock()
		_ = nc.Close()
		return
	}
	c.nc, c.up = nc, true
	g.healthy++
	g.up.Store(int32(g.healthy))
	if next = g.qHead; next != nil { // requests were waiting for a connection
		g.unlinkQ(next)
		c.assign(next, clock.Now())
	} else {
		c.idleAt = len(g.idle)
		g.idle = append(g.idle, c)
	}
	g.mu.Unlock()
	c.set(connectivity.Ready)
	if next != nil {
		c.send(nc, next)
	}
	for {
		_, err := br.Peek(1) // a request's response, or the server closing an idle connection
		r := c.cur.Load()
		if err != nil {
			c.lose(nc, nil, err)
			return
		}
		if r == nil {
			c.lose(nc, nil, bad("data on an idle connection"))
			return
		}
		var data *[]byte
		if r.Keep {
			data = &r.Data
		}
		resp, n, err := readResponse(br, data)
		r.TDone = clock.Now()
		if err != nil {
			c.lose(nc, nil, err)
			return
		}
		r.RespBytes, r.HdrLen = n, resp.hdrLen
		r.Status = grpcclient.StatusHTTP(resp.code)
		if resp.close {
			c.lose(nc, r, errServerClose)
			return
		}
		next = c.finish(r)
		if next != nil {
			c.send(nc, next)
		}
		r.Done(r)
	}
}

var errServerClose = errors.New("server closed the connection")

// finish takes r off the books and gives c its next request, if one is waiting.
func (c *conn) finish(r *Request) (next *Request) {
	g := c.g
	g.mu.Lock()
	g.unlinkDL(r)
	c.cur.Store(nil)
	if next = g.qHead; next != nil {
		g.unlinkQ(next)
		c.assign(next, clock.Now())
	} else {
		c.idleAt = len(g.idle)
		g.idle = append(g.idle, c)
	}
	g.mu.Unlock()
	return next
}

// lose ends the connection after err. The request in flight (if any) completes: with its
// own status when the server closed the connection after answering (err is
// errServerClose, r the answered request), otherwise as timeout, invalid_response or
// transport. When this was the last healthy connection, the waiting requests fail as
// connect.
func (c *conn) lose(nc net.Conn, answered *Request, err error) {
	g := c.g
	_ = nc.Close()
	now := clock.Now()
	var done []*Request
	g.mu.Lock()
	c.up = false
	g.healthy--
	g.up.Store(int32(g.healthy))
	if c.idleAt >= 0 {
		last := len(g.idle) - 1
		g.idle[c.idleAt] = g.idle[last]
		g.idle[c.idleAt].idleAt = c.idleAt
		g.idle = g.idle[:last]
		c.idleAt = -1
	}
	r := c.cur.Swap(nil) // authoritative: a Submit may have taken this connection after our read failed
	if r != nil {
		g.unlinkDL(r)
		if r != answered {
			r.TDone = now
			var bad *badResponse
			switch {
			case r.timedOut:
				r.Status = grpcclient.StatusTimeout
			case errors.As(err, &bad):
				r.Status = grpcclient.StatusInvalidResponse
			default:
				r.Status = grpcclient.StatusTransport
			}
			r.RespBytes = 0
		}
		done = append(done, r)
	}
	if g.healthy == 0 {
		done = g.failQueuedLocked(now, done)
	}
	g.notifyLocked()
	g.mu.Unlock()
	deliver(done)
}
