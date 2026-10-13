package h2client

import (
	"bufio"
	"context"
	"io"
	"net"
	"sync"
	"sync/atomic"
	"syscall"
	"testing"
	"time"

	"github.com/criteo/triton-stress/internal/grpcclient"
	"golang.org/x/net/http2"
)

// responder is a minimal HTTP/2 server for benchmarks (a test fixture, not part of the
// binary): it answers every request at once with a fixed small gRPC response and does
// no protobuf or HPACK work, so a benchmark measures the client.
func responder(tb testing.TB) (addr string) {
	tb.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		tb.Fatal(err)
	}
	var conns sync.WaitGroup
	tb.Cleanup(func() { _ = l.Close() })
	go func() {
		for {
			nc, err := l.Accept()
			if err != nil {
				return
			}
			conns.Add(1)
			go func() { defer conns.Done(); respond(nc) }()
		}
	}()
	return l.Addr().String()
}

func respond(nc net.Conn) {
	defer func() { _ = nc.Close() }()
	br := bufio.NewReaderSize(nc, 64<<10)
	if _, err := io.ReadFull(br, make([]byte, len(clientPreface))); err != nil {
		return
	}
	fr := http2.NewFramer(nil, br)
	fr.SetReuseFrames()
	var mu sync.Mutex
	var wbuf []byte
	wake := make(chan struct{}, 1)
	push := func(b []byte) {
		mu.Lock()
		wbuf = append(wbuf, b...)
		mu.Unlock()
		select {
		case wake <- struct{}{}:
		default:
		}
	}
	go func() {
		var spare []byte
		for range wake {
			for {
				mu.Lock()
				if len(wbuf) == 0 {
					mu.Unlock()
					break
				}
				b := wbuf
				wbuf = spare[:0]
				mu.Unlock()
				if _, err := nc.Write(b); err != nil {
					_ = nc.Close()
					return
				}
				spare = b
			}
		}
	}()
	hdr := func(b []byte, n int, typ, flags byte, id uint32) []byte {
		return append(b, byte(n>>16), byte(n>>8), byte(n), typ, flags, byte(id>>24)&0x7f, byte(id>>16), byte(id>>8), byte(id))
	}
	push(append(hdr(nil, 0, 0x4, 0, 0), append(hdr(nil, 4, 0x8, 0, 0), 0, 0xff, 0, 0)...)) // SETTINGS, WINDOW_UPDATE
	respHdr := append([]byte{0x88, 0x0f, 0x10, 16}, "application/grpc"...)
	trailer := append([]byte{0x00, 11}, "grpc-status"...)
	trailer = append(trailer, 1, '0')
	var consumed int64
	for {
		f, err := fr.ReadFrame()
		if err != nil {
			return
		}
		switch f := f.(type) {
		case *http2.SettingsFrame:
			if !f.IsAck() {
				push(hdr(nil, 0, 0x4, 0x1, 0))
			}
		case *http2.DataFrame:
			consumed += int64(f.Length)
			var out []byte
			if consumed >= 64<<10 {
				out = append(hdr(out, 4, 0x8, 0, 0), byte(consumed>>24), byte(consumed>>16), byte(consumed>>8), byte(consumed))
				consumed = 0
			}
			if f.StreamEnded() {
				id := f.StreamID
				out = append(hdr(out, len(respHdr), 0x1, 0x4, id), respHdr...)
				out = append(hdr(out, 5+8, 0x0, 0, id), 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0)
				out = append(hdr(out, len(trailer), 0x1, 0x5, id), trailer...)
			}
			if len(out) > 0 {
				push(out)
			}
		}
	}
}

func cpuTime() time.Duration {
	var ru syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &ru); err != nil {
		panic(err)
	}
	return time.Duration(ru.Utime.Nano() + ru.Stime.Nano())
}

const benchConns = 4

// BenchmarkSubmit is the hot path of the load generator: submit a request and take its
// completion, from many goroutines over a few connections to a trivial local responder.
// cpu-ns/op is the CPU of the whole process, the responder included.
func BenchmarkSubmit(b *testing.B) {
	addr := responder(b)
	conns := make([]*Conn, benchConns)
	for i := range conns {
		c, err := New(Options{Target: addr})
		if err != nil {
			b.Fatal(err)
		}
		defer func() { _ = c.Close() }()
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		if err := c.WaitReady(ctx); err != nil {
			b.Fatal(err)
		}
		cancel()
		conns[i] = c
	}
	payload := make([]byte, 1300)
	var next atomic.Int64
	b.SetParallelism(16)
	b.ReportAllocs()
	b.ResetTimer()
	cpu0 := cpuTime()
	b.RunParallel(func(pb *testing.PB) {
		c := conns[int(next.Add(1))%benchConns]
		ch := make(chan *Request, 1)
		r := &Request{Body: payload, Timeout: 10 * time.Second, Done: func(r *Request) { ch <- r }}
		for pb.Next() {
			c.Submit(r)
			if got := <-ch; got.Status != grpcclient.StatusOK {
				b.Errorf("status %s", got.Status.Label())
				return
			}
		}
	})
	b.ReportMetric(float64(cpuTime()-cpu0)/float64(b.N), "cpu-ns/op")
}

// BenchmarkGRPCGoInfer is the same load through grpc-go's Invoke (the client the load
// generator used before h2client), for comparison.
func BenchmarkGRPCGoInfer(b *testing.B) {
	addr := responder(b)
	conns := make([]*grpcclient.Conn, benchConns)
	for i := range conns {
		c, err := grpcclient.New(grpcclient.Options{Target: addr})
		if err != nil {
			b.Fatal(err)
		}
		defer func() { _ = c.Close() }()
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		if err := c.WaitReady(ctx); err != nil {
			b.Fatal(err)
		}
		cancel()
		conns[i] = c
	}
	payload := make([]byte, 1300)
	var next atomic.Int64
	b.SetParallelism(16)
	b.ReportAllocs()
	b.ResetTimer()
	cpu0 := cpuTime()
	b.RunParallel(func(pb *testing.PB) {
		c := conns[int(next.Add(1))%benchConns]
		for pb.Next() {
			if r := c.Infer(context.Background(), payload, 10*time.Second, nil); r.Status != grpcclient.StatusOK {
				b.Errorf("status %s", r.Status.Label())
				return
			}
		}
	})
	b.ReportMetric(float64(cpuTime()-cpu0)/float64(b.N), "cpu-ns/op")
}
