package grpcclient

import (
	"context"
	"crypto/tls"
	"fmt"
	"net"
	"sync/atomic"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/backoff"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"
)

// TLSMode selects the transport security.
type TLSMode uint8

// TLS modes.
const (
	TLSOff      TLSMode = iota // plaintext HTTP/2
	TLSSystem                  // TLS, server verified against the system roots
	TLSInsecure                // TLS, no verification
)

// HTTP/2 flow-control windows. Setting any window size makes grpc-go use fixed
// windows, which turns its BDP estimator (and the PINGs it sends) off.
const (
	streamWindow = 1 << 20
	connWindow   = 16 << 20
)

// Options configures a Conn.
type Options struct {
	// Target is the grpc-go target, usually host:port.
	Target string
	TLS    TLSMode
	// BackoffMin and BackoffMax bound the reconnect backoff (defaults 10 ms and 1 s).
	BackoffMin, BackoffMax time.Duration
	// OnState, if set, is called from the watcher goroutine on each observed
	// connectivity change (changes faster than the watcher can observe are coalesced).
	OnState func(from, to connectivity.State)
	// Dialer replaces the network dialer (tests).
	Dialer func(ctx context.Context, addr string) (net.Conn, error)
}

// Conn is one gRPC connection to Triton.
type Conn struct {
	cc        *grpc.ClientConn
	state     atomic.Int32
	cancel    context.CancelFunc
	done      chan struct{}
	inferOpts []grpc.CallOption
}

// New creates a Conn and starts connecting in the background; it does not wait.
// Use WaitReady to block until the connection is up.
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
	var creds credentials.TransportCredentials
	switch o.TLS {
	case TLSOff:
		creds = insecure.NewCredentials()
	case TLSSystem:
		creds = credentials.NewTLS(&tls.Config{MinVersion: tls.VersionTLS12})
	case TLSInsecure:
		creds = credentials.NewTLS(&tls.Config{MinVersion: tls.VersionTLS12, InsecureSkipVerify: true})
	default:
		return nil, fmt.Errorf("bad TLS mode %d", o.TLS)
	}
	opts := []grpc.DialOption{
		grpc.WithTransportCredentials(creds),
		grpc.WithInitialWindowSize(streamWindow),
		grpc.WithInitialConnWindowSize(connWindow),
		grpc.WithConnectParams(grpc.ConnectParams{
			Backoff:           backoff.Config{BaseDelay: o.BackoffMin, Multiplier: 2, Jitter: 0.2, MaxDelay: o.BackoffMax},
			MinConnectTimeout: 5 * time.Second,
		}),
		grpc.WithNoProxy(),
		grpc.WithDisableServiceConfig(),
	}
	if o.Dialer != nil {
		opts = append(opts, grpc.WithContextDialer(o.Dialer))
	}
	cc, err := grpc.NewClient(o.Target, opts...)
	if err != nil {
		return nil, err
	}
	ctx, cancel := context.WithCancel(context.Background())
	c := &Conn{
		cc:        cc,
		cancel:    cancel,
		done:      make(chan struct{}),
		inferOpts: []grpc.CallOption{grpc.ForceCodecV2(RawCodec{})},
	}
	c.state.Store(int32(connectivity.Idle))
	go c.watch(ctx, o.OnState)
	return c, nil
}

// watch mirrors the connectivity state into an atomic and reconnects at once
// whenever grpc-go parks the channel in IDLE (it does so after a lost connection).
func (c *Conn) watch(ctx context.Context, on func(from, to connectivity.State)) {
	defer close(c.done)
	s := connectivity.Idle
	for {
		if s == connectivity.Idle {
			c.cc.Connect()
		}
		if !c.cc.WaitForStateChange(ctx, s) {
			return
		}
		n := c.cc.GetState()
		c.state.Store(int32(n))
		if on != nil && n != s {
			on(s, n)
		}
		if n == connectivity.Shutdown {
			return
		}
		s = n
	}
}

// State is the last observed connectivity state (lock-free).
func (c *Conn) State() connectivity.State { return connectivity.State(c.state.Load()) }

// Ready reports whether the connection can take a request now (lock-free).
func (c *Conn) Ready() bool { return c.State() == connectivity.Ready }

// WaitReady blocks until the Conn is Ready or ctx ends.
func (c *Conn) WaitReady(ctx context.Context) error {
	for !c.Ready() {
		s := c.cc.GetState()
		if s == connectivity.Ready {
			return nil
		}
		if !c.cc.WaitForStateChange(ctx, s) {
			return fmt.Errorf("connection not ready (state %s): %w", c.cc.GetState(), ctx.Err())
		}
	}
	return nil
}

// ClientConn exposes the underlying grpc-go connection (for the metadata calls and tests).
func (c *Conn) ClientConn() *grpc.ClientConn { return c.cc }

// Close shuts the connection down.
func (c *Conn) Close() error {
	c.cancel()
	err := c.cc.Close()
	<-c.done
	return err
}
