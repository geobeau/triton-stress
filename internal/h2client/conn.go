// Package h2client is a minimal HTTP/2 gRPC client for one purpose: sending
// pre-encoded unary requests at a high rate and timing them.
//
// It exists because grpc-go spends tens of microseconds of CPU and about ninety heap
// allocations (5-6 KB) per call, and needs a goroutine for every request in flight. Here a
// request is a struct handed to Submit; the connection's reader goroutine completes it
// and calls its Done function. Nothing is allocated per request and no goroutine is
// created for it.
//
// What it implements of HTTP/2 and gRPC is what a unary client needs: the preface and
// SETTINGS, flow control in both directions, GOAWAY, RST_STREAM, PING replies (it never
// sends a PING), the five-byte message prefix, and grpc-status in the trailers.
// Requests are never retried, compressed, or given a deadline on the wire.
package h2client

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

// InferPath is the gRPC method path of ModelInfer.
const InferPath = grpcclient.InferMethod

// Options configures a Conn.
type Options struct {
	// Target is host:port, optionally prefixed by dns:/// or passthrough:///.
	Target string
	TLS    grpcclient.TLSMode
	// BackoffMin and BackoffMax bound the reconnect backoff (defaults 10 ms and 1 s).
	BackoffMin, BackoffMax time.Duration
	// OnState is called, from the connection's own goroutine, on every state change.
	OnState func(from, to connectivity.State)
	// Dialer replaces the network dialer (tests).
	Dialer func(ctx context.Context, addr string) (net.Conn, error)
}

// connectTimeout bounds one connection attempt, the TLS and HTTP/2 handshakes included.
const connectTimeout = 5 * time.Second

// Conn is one HTTP/2 connection to the server, kept connected in the background. A
// Conn that is not Ready completes requests at once as StatusConnect: nothing waits for
// a reconnect.
type Conn struct {
	o        Options
	addr     string
	hdrBlock []byte
	tlsCfg   *tls.Config

	state atomic.Int32
	cur   atomic.Pointer[transport]

	mu       sync.Mutex
	changed  chan struct{} // closed and replaced on every state change
	draining []*transport  // transports that got a GOAWAY and still have requests
	cancel   context.CancelFunc
	done     chan struct{}
}

// New creates a Conn and starts connecting in the background.
func New(o Options) (*Conn, error) {
	if o.BackoffMin <= 0 {
		o.BackoffMin = 10 * time.Millisecond
	}
	if o.BackoffMax <= 0 {
		o.BackoffMax = time.Second
	}
	if o.BackoffMax < o.BackoffMin {
		return nil, fmt.Errorf("reconnect backoff max %v is below min %v", o.BackoffMax, o.BackoffMin)
	}
	addr := strings.TrimPrefix(strings.TrimPrefix(o.Target, "dns:///"), "passthrough:///")
	if strings.Contains(addr, "://") || strings.HasPrefix(addr, "unix:") {
		return nil, fmt.Errorf("target %q: only host:port, dns:///host:port and passthrough:///host:port are supported", o.Target)
	}
	host, _, err := net.SplitHostPort(addr)
	if err != nil { // no port: the HTTPS port, as grpc-go's DNS resolver does
		if host, _, err = net.SplitHostPort(net.JoinHostPort(addr, "443")); err != nil {
			return nil, fmt.Errorf("target %q: %w", o.Target, err)
		}
		addr = net.JoinHostPort(addr, "443")
	}
	c := &Conn{o: o, addr: addr, changed: make(chan struct{}), done: make(chan struct{})}
	switch o.TLS {
	case grpcclient.TLSOff:
	case grpcclient.TLSSystem:
		c.tlsCfg = &tls.Config{MinVersion: tls.VersionTLS12, ServerName: host, NextProtos: []string{"h2"}}
	case grpcclient.TLSInsecure:
		c.tlsCfg = &tls.Config{MinVersion: tls.VersionTLS12, InsecureSkipVerify: true, NextProtos: []string{"h2"}}
	default:
		return nil, fmt.Errorf("bad TLS mode %d", o.TLS)
	}
	c.hdrBlock = requestHeaders(addr, o.TLS != grpcclient.TLSOff)
	c.state.Store(int32(connectivity.Idle))
	ctx, cancel := context.WithCancel(context.Background())
	c.cancel = cancel
	go c.run(ctx)
	return c, nil
}

// requestHeaders is the HPACK block of a ModelInfer request. It uses only static-table
// references and literals without indexing, so it does not depend on (or change) the
// dynamic table and can be sent unchanged on every stream.
func requestHeaders(authority string, tlsOn bool) []byte {
	lit := func(b []byte, nameIdx int, v string) []byte {
		// literal header field without indexing, name from the static table
		if nameIdx < 15 {
			b = append(b, byte(nameIdx))
		} else {
			b = append(b, 0x0f, byte(nameIdx-15))
		}
		b = append(b, byte(len(v)))
		return append(b, v...)
	}
	var b []byte
	b = append(b, 0x83) // :method POST
	if tlsOn {
		b = append(b, 0x87) // :scheme https
	} else {
		b = append(b, 0x86) // :scheme http
	}
	b = lit(b, 4, InferPath)                 // :path
	b = lit(b, 1, authority)                 // :authority
	b = lit(b, 31, "application/grpc+proto") // content-type
	b = append(b, 0x00, 2, 't', 'e', 8)      // te: trailers
	b = append(b, "trailers"...)
	b = lit(b, 58, "triton-stress") // user-agent
	return b
}

func (c *Conn) set(to connectivity.State) {
	from := connectivity.State(c.state.Swap(int32(to)))
	if from == to {
		return
	}
	c.mu.Lock()
	close(c.changed)
	c.changed = make(chan struct{})
	c.mu.Unlock()
	if c.o.OnState != nil {
		c.o.OnState(from, to)
	}
}

// State is the connectivity state (lock-free).
func (c *Conn) State() connectivity.State { return connectivity.State(c.state.Load()) }

// Ready reports whether the Conn can take a request now (lock-free).
func (c *Conn) Ready() bool { return c.State() == connectivity.Ready }

// WaitReady blocks until the Conn is Ready or ctx ends.
func (c *Conn) WaitReady(ctx context.Context) error {
	for {
		c.mu.Lock()
		ch := c.changed
		c.mu.Unlock()
		if c.Ready() {
			return nil
		}
		select {
		case <-ch:
		case <-ctx.Done():
			return fmt.Errorf("connection not ready (state %s): %w", c.State(), ctx.Err())
		}
	}
}

// Submit sends r. It never blocks: when the Conn is not Ready, or the connection
// refuses the request (it is shutting down), r completes at once as StatusConnect.
func (c *Conn) Submit(r *Request) {
	r.reset()
	if t := c.cur.Load(); t != nil && t.submit(r, clock.Now()) {
		return
	}
	r.Status = grpcclient.StatusConnect
	r.TDone = clock.Now()
	r.Done(r)
}

// Close shuts the connection down; requests in flight complete as transport errors.
func (c *Conn) Close() error {
	c.cancel()
	<-c.done
	return nil
}

// run keeps the connection up. The first retry after a lost connection is immediate;
// after failed attempts the delay starts at BackoffMin and doubles to BackoffMax.
func (c *Conn) run(ctx context.Context) {
	defer close(c.done)
	defer func() {
		c.set(connectivity.Shutdown)
		if t := c.cur.Swap(nil); t != nil {
			t.close()
		}
		c.mu.Lock()
		old := c.draining
		c.draining = nil
		c.mu.Unlock()
		for _, t := range old {
			t.close()
		}
	}()
	delay := time.Duration(0)
	for ctx.Err() == nil {
		c.set(connectivity.Connecting)
		t, err := c.connect(ctx)
		if err != nil {
			if ctx.Err() != nil {
				return
			}
			c.set(connectivity.TransientFailure)
			if delay == 0 {
				delay = c.o.BackoffMin
			}
			d := time.Duration(float64(delay) * (1 + 0.2*(2*rand.Float64()-1)))
			select {
			case <-time.After(d):
			case <-ctx.Done():
				return
			}
			delay = min(delay*2, c.o.BackoffMax)
			continue
		}
		delay = 0
		c.cur.Store(t)
		c.set(connectivity.Ready)
		select {
		case <-t.lost:
		case <-ctx.Done():
			return
		}
		c.cur.Store(nil)
		c.set(connectivity.Idle)
		t.mu.Lock()
		drain := !t.closed
		t.mu.Unlock()
		if drain { // GOAWAY: the transport finishes its requests, then closes itself
			c.mu.Lock()
			c.draining = append(c.draining, t)
			c.mu.Unlock()
			go func() {
				<-t.done
				c.mu.Lock()
				for i, d := range c.draining {
					if d == t {
						c.draining = append(c.draining[:i], c.draining[i+1:]...)
						break
					}
				}
				c.mu.Unlock()
			}()
		}
	}
}

// connect dials and completes the HTTP/2 handshake (the server's SETTINGS).
func (c *Conn) connect(ctx context.Context) (*transport, error) {
	ctx, cancel := context.WithTimeout(ctx, connectTimeout)
	defer cancel()
	var nc net.Conn
	var err error
	if c.o.Dialer != nil {
		nc, err = c.o.Dialer(ctx, c.addr)
	} else {
		var d net.Dialer
		nc, err = d.DialContext(ctx, "tcp", c.addr)
	}
	if err != nil {
		return nil, fmt.Errorf("error while dialing: %w", err)
	}
	if c.tlsCfg != nil {
		tc := tls.Client(nc, c.tlsCfg.Clone())
		if err := tc.HandshakeContext(ctx); err != nil {
			_ = nc.Close()
			return nil, fmt.Errorf("tls handshake: %w", err)
		}
		if tc.ConnectionState().NegotiatedProtocol != "h2" {
			_ = nc.Close()
			return nil, errors.New("server did not negotiate h2")
		}
		nc = tc
	}
	t := newTransport(nc, c.hdrBlock)
	t.start()
	select {
	case <-t.ready:
		return t, nil
	case <-t.done:
		return nil, fmt.Errorf("error while dialing: %v", t.err)
	case <-ctx.Done():
		t.fail(ctx.Err(), grpcclient.StatusConnect)
		return nil, fmt.Errorf("http2 handshake: %w", ctx.Err())
	}
}
