// Command triton-stress is an open-loop load generator for NVIDIA Triton Inference Server.
//
// It connects, reads the model's metadata and config, builds the payload pool, then
// sends requests on a rate schedule (warmup, ramp) whether or not earlier responses have
// returned, until the ramp reaches its maximum, --duration ends, a guard trips or a signal
// arrives. It writes intervals.csv, events.jsonl (and requests.csv with
// --request-log-every) under --out-dir, one JSON line per tick on stdout, and Prometheus
// metrics on --metrics-addr. With --control-addr it waits for POST /start before sending.
// Human-readable messages go to stderr.
package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/criteo/triton-stress/internal/control"
	"github.com/criteo/triton-stress/internal/grpcclient"
	"github.com/criteo/triton-stress/internal/h1client"
	"github.com/criteo/triton-stress/internal/h2client"
	"github.com/criteo/triton-stress/internal/load"
	"github.com/criteo/triton-stress/internal/metrics"
	"github.com/criteo/triton-stress/internal/output"
	"github.com/criteo/triton-stress/internal/payload"
	"github.com/criteo/triton-stress/pkg/tensorgen"
	"google.golang.org/grpc/connectivity"
)

func main() {
	ctx, cancel := context.WithCancelCause(context.Background())
	sigs := make(chan os.Signal, 1)
	signal.Notify(sigs, os.Interrupt, syscall.SIGTERM)
	go func() {
		s := <-sigs
		signal.Stop(sigs) // a second signal gets the default action and kills the process
		if s == syscall.SIGTERM {
			cancel(load.ErrTerminated)
		} else {
			cancel(load.ErrInterrupted)
		}
	}()
	os.Exit(run(ctx, os.Args[1:], os.Stdout, os.Stderr))
}

// syncWriter serialises writes: connection watchers log from their own goroutines.
type syncWriter struct {
	mu sync.Mutex
	w  io.Writer
}

func (s *syncWriter) Write(b []byte) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.w.Write(b)
}

// printf writes to the console. A failed console write has nowhere to be reported.
func printf(w io.Writer, format string, a ...any) { _, _ = fmt.Fprintf(w, format, a...) }

// run returns the exit code: 0 when the run completed without a guard tripping, 2 when a
// guard tripped, 1 for a fatal error (bad configuration, no connection, metadata failure,
// output files that could not be written).
func run(ctx context.Context, args []string, stdout, stderr io.Writer) (code int) {
	stderr = &syncWriter{w: stderr}
	cfg, err := parseConfig(args, stderr)
	if err != nil {
		printf(stderr, "triton-stress: error: %v\n", err)
		return 1
	}
	dir := cfg.OutDir
	if dir == "" {
		dir = filepath.Join("runs", output.RunDirName(time.Now()))
	}
	out, err := output.Open(dir, cfg.RequestLogEvery > 0, stdout)
	if err != nil {
		printf(stderr, "triton-stress: fatal: %v\n", err)
		return 1
	}
	defer func() {
		if err := out.Close(); err != nil {
			printf(stderr, "triton-stress: error: closing the output files: %v\n", err)
			code = 1
		}
	}()
	printf(stderr, "triton-stress: writing to %s\n", dir)

	reason, err := execute(ctx, cfg, out, stderr)
	if err != nil {
		fatalReason := "fatal:" + err.Error()
		code = 1
		if ctx.Err() != nil { // a signal during startup is a normal end
			fatalReason, code = load.ReasonInterrupted, 0
			if errors.Is(context.Cause(ctx), load.ErrTerminated) {
				fatalReason = load.ReasonTerminated
			}
		}
		if code != 0 {
			printf(stderr, "triton-stress: fatal: %v\n", err)
		}
		if err := out.WriteStop(output.Event{TS: time.Now(), Kind: "stop", Fields: []output.KV{
			output.Str("reason", fatalReason), output.Rate("offered_rps", 0), output.Int("inflight", 0)}}); err != nil {
			printf(stderr, "triton-stress: error: writing the stop event: %v\n", err)
			code = 1
		}
		return code
	}
	if strings.HasPrefix(reason, "guard:") {
		printf(stderr, "triton-stress: guard tripped: %s\n", reason)
		return 2
	}
	return 0
}

// clients are the load connections and what startup learnt through them.
type clients struct {
	conns  []*h2client.Conn  // gRPC
	groups []*h1client.Group // HTTP, one per worker
	model  *tensorgen.Model
	tpl    *tensorgen.Template
	pool   *payload.Pool
}

func (c *clients) close() {
	for _, x := range c.conns {
		if x != nil {
			_ = x.Close()
		}
	}
	for _, g := range c.groups {
		if g != nil {
			_ = g.Close()
		}
	}
}

func (c *clients) total() int {
	n := len(c.conns)
	for _, g := range c.groups {
		n += g.Len()
	}
	return n
}

// execute does the startup (connections, metadata, payload pool) and runs the load, after
// POST /start when --control-addr is set. Its error is fatal; a run that ends normally
// returns its stop reason and a nil error, after writing its stop event.
func execute(ctx context.Context, cfg *config, out *output.Files, stderr io.Writer) (string, error) {
	var ctl *control.Server
	if cfg.ControlAddr != "" {
		var err error
		if ctl, err = control.Serve(cfg.ControlAddr); err != nil {
			return "", fmt.Errorf("control endpoint: %w", err)
		}
		defer func() { _ = ctl.Close() }()
	}
	var cl clients
	defer cl.close()
	var err error
	if cfg.Protocol == "http" {
		err = startHTTP(ctx, cfg, out, stderr, &cl)
	} else {
		err = startGRPC(ctx, cfg, out, stderr, &cl)
	}
	if err != nil {
		return "", err
	}
	printDescription(stderr, cfg, cl.model, cl.tpl, cl.pool)

	deps := load.Deps{Conns: cl.conns, Groups: cl.groups, Pool: cl.pool, Out: out, Log: stderr}
	if cfg.Validate {
		deps.Validator = tensorgen.NewValidator(cl.model.Outputs, cfg.BatchSize, cl.model.Batched)
	}
	runner, err := load.New(load.Config{
		Schedule: cfg.Schedule, Arrival: cfg.Arrival, Seed: cfg.Seed, Workers: cfg.Workers,
		Timeout: cfg.Timeout, Interval: cfg.Interval, Guards: cfg.Guards, LogEvery: cfg.RequestLogEvery,
	}, deps)
	if err != nil {
		return "", err
	}
	if cfg.MetricsAddr != "" {
		m := metrics.New(cfg.Protocol, cfg.Model, runner.Stats())
		srv, err := metrics.Serve(cfg.MetricsAddr, m)
		if err != nil {
			return "", fmt.Errorf("metrics endpoint: %w", err)
		}
		defer func() { _ = srv.Close() }()
		runner.SetMetrics(m)
		printf(stderr, "triton-stress: metrics on http://%s/metrics\n", srv.Addr())
	}
	printf(stderr, "triton-stress: %d workers, %d connections\n", runner.Workers(), cl.total())
	if ctl != nil {
		ctl.SetReady()
		printf(stderr, "triton-stress: waiting for POST /start on http://%s\n", ctl.Addr())
		if err := ctl.Wait(ctx); err != nil {
			return "", err
		}
	}
	res, err := runner.Run(ctx)
	return res.Reason, err
}

// startGRPC connects, reads the model over a regular grpc-go connection (used only here)
// and builds the payload pool.
func startGRPC(ctx context.Context, cfg *config, out *output.Files, stderr io.Writer, cl *clients) error {
	cl.conns = make([]*h2client.Conn, cfg.Connections)
	for i := range cl.conns {
		ev := connEvent(out, stderr)
		c, err := h2client.New(h2client.Options{
			Target:     cfg.Target,
			TLS:        cfg.TLS,
			BackoffMin: cfg.BackoffMin,
			BackoffMax: cfg.BackoffMax,
			OnState:    func(from, to connectivity.State) { ev(i, from, to) },
		})
		if err != nil {
			return err
		}
		cl.conns[i] = c
	}

	// Wait up to --request-timeout for the connections; at least one must come up.
	wctx, cancel := context.WithTimeout(ctx, cfg.Timeout)
	var wg sync.WaitGroup
	for _, c := range cl.conns {
		wg.Add(1)
		go func() { defer wg.Done(); _ = c.WaitReady(wctx) }() // readiness is counted below
	}
	wg.Wait()
	cancel()
	ready := 0
	for _, c := range cl.conns {
		if c.Ready() {
			ready++
		}
	}
	if err := checkReady(ctx, cfg, ready, len(cl.conns), stderr); err != nil {
		return err
	}

	mctx, mcancel := context.WithTimeout(ctx, cfg.Timeout)
	defer mcancel()
	meta, err := grpcclient.New(grpcclient.Options{Target: cfg.Target, TLS: cfg.TLS})
	if err != nil {
		return err
	}
	defer func() { _ = meta.Close() }()
	if err := meta.WaitReady(mctx); err != nil {
		return err
	}
	if cl.model, err = meta.FetchModel(mctx, cfg.Model, cfg.ModelVersion); err != nil {
		return err
	}
	if cl.tpl, err = cl.model.NewTemplate(cfg.Inputs, cfg.BatchSize); err != nil {
		return err
	}
	cl.pool, err = payload.Build(tensorgen.NewGenerator(cfg.Seed, tensorgen.WithDist(cfg.Gen)), cl.tpl, cfg.PoolSize)
	return err
}

// startHTTP creates one connection Group per worker (worker k of n owns connections
// k, k+n, ...), reads the model over a plain HTTP request and builds the pool of complete
// HTTP requests.
func startHTTP(ctx context.Context, cfg *config, out *output.Files, stderr io.Writer, cl *clients) error {
	addr, err := h1client.Address(cfg.Target, cfg.TLS != grpcclient.TLSOff)
	if err != nil {
		return err
	}
	n := min(max(cfg.Workers, 1), cfg.Connections)
	ev := connEvent(out, stderr)
	for k := 0; k < n; k++ {
		var idx []int
		for i := k; i < cfg.Connections; i += n {
			idx = append(idx, i)
		}
		g, err := h1client.NewGroup(h1client.Options{
			Target: addr, TLS: cfg.TLS, BackoffMin: cfg.BackoffMin, BackoffMax: cfg.BackoffMax, OnState: ev,
		}, idx)
		if err != nil {
			return err
		}
		cl.groups = append(cl.groups, g)
	}
	wctx, cancel := context.WithTimeout(ctx, cfg.Timeout)
	var wg sync.WaitGroup
	for _, g := range cl.groups {
		wg.Add(1)
		go func() { defer wg.Done(); _ = g.WaitReady(wctx) }() // readiness is counted below
	}
	wg.Wait()
	cancel()
	ready := 0
	for _, g := range cl.groups {
		ready += g.Healthy()
	}
	if err := checkReady(ctx, cfg, ready, cfg.Connections, stderr); err != nil {
		return err
	}

	mctx, mcancel := context.WithTimeout(ctx, cfg.Timeout)
	defer mcancel()
	if cl.model, err = h1client.FetchModel(mctx, addr, cfg.TLS, cfg.Model, cfg.ModelVersion); err != nil {
		return err
	}
	if cl.tpl, err = cl.model.NewTemplate(cfg.Inputs, cfg.BatchSize); err != nil {
		return err
	}
	gen := tensorgen.NewGenerator(cfg.Seed, tensorgen.WithDist(cfg.Gen))
	cl.pool, err = payload.BuildFunc(func() ([]byte, error) { return gen.EncodeHTTP(cl.tpl, addr) }, cfg.PoolSize)
	return err
}

// checkReady is fatal when no connection is up after the startup wait.
func checkReady(ctx context.Context, cfg *config, ready, total int, stderr io.Writer) error {
	if ready == 0 {
		if err := ctx.Err(); err != nil {
			return err
		}
		return fmt.Errorf("no connection to %s within %v", cfg.Target, cfg.Timeout)
	}
	if ready < total {
		printf(stderr, "triton-stress: warning: %d of %d connections are ready; the others keep reconnecting\n", ready, total)
	}
	return nil
}

// connEvent records a connection state change in events.jsonl and on stderr.
func connEvent(out *output.Files, stderr io.Writer) func(idx int, from, to connectivity.State) {
	return func(idx int, from, to connectivity.State) {
		err := out.WriteEvent(output.Event{TS: time.Now(), Kind: "conn", Fields: []output.KV{
			output.Int("conn", int64(idx)), output.Str("from", from.String()), output.Str("to", to.String())}})
		if err != nil {
			printf(stderr, "triton-stress: connection %d: %s -> %s (writing events.jsonl: %v)\n", idx, from, to, err)
			return
		}
		printf(stderr, "triton-stress: connection %d: %s -> %s\n", idx, from, to)
	}
}

func printDescription(w io.Writer, cfg *config, m *tensorgen.Model, tpl *tensorgen.Template, pool *payload.Pool) {
	version := cfg.ModelVersion
	if version == "" {
		version = "latest"
	}
	batching := fmt.Sprintf("max_batch_size %d", m.MaxBatchSize)
	if !m.Batched {
		batching = "does not batch"
	}
	printf(w, "model %s (version %s), %s, batch size %d\n", m.Name, version, batching, cfg.BatchSize)
	printf(w, "inputs:\n")
	for _, s := range tpl.Inputs {
		printf(w, "  %s %s %v\n", s.Name, s.DType, s.Shape(int64(tpl.BatchSize), tpl.Batched))
	}
	printf(w, "outputs (per batch element):\n")
	for _, s := range m.Outputs {
		printf(w, "  %s %s %v\n", s.Name, s.DType, s.Dims)
	}
	lo, hi := pool.SizeRange()
	size := fmt.Sprintf("%d bytes", lo)
	if lo != hi {
		size = fmt.Sprintf("%d-%d bytes", lo, hi)
	}
	printf(w, "request: %s; pool: %d requests, %.1f MiB total\n", size, pool.Len(), float64(pool.TotalBytes())/(1<<20))
	printf(w, "gen: %s, seed %d; validate: %s; request timeout %v\n", cfg.Gen, cfg.Seed, onOff(cfg.Validate), cfg.Timeout)
	if m.Sequence {
		printf(w, "sequence batching: each request carries sequence_id and sequence_start\n")
	}
	printf(w, "protocol %s, target %s (%s, %d connections)\n", cfg.Protocol, cfg.Target, strings.ToLower(tlsName(cfg)), cfg.Connections)
}

func onOff(b bool) string {
	if b {
		return "on"
	}
	return "off"
}

func tlsName(cfg *config) string {
	switch cfg.TLS {
	case grpcclient.TLSSystem:
		return "TLS"
	case grpcclient.TLSInsecure:
		return "TLS, unverified"
	}
	return "plaintext"
}
