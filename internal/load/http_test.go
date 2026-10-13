package load

import (
	"context"
	"encoding/json"
	"strconv"
	"testing"
	"time"

	"github.com/criteo/triton-stress/internal/guard"
	"github.com/criteo/triton-stress/internal/h1client"
	"github.com/criteo/triton-stress/internal/h2client"
	"github.com/criteo/triton-stress/internal/kservefake"
	"github.com/criteo/triton-stress/internal/payload"
	"github.com/criteo/triton-stress/pkg/tensorgen"
)

// httpRig is a fake KServe server with workers' groups of connections and a pool.
type httpRig struct {
	f      *kservefake.Server
	rig    *rig
	groups []*h1client.Group
}

// newHTTPRig starts the fake and one Group per worker with conns connections each.
func newHTTPRig(t *testing.T, workers, conns int, target string) *httpRig {
	t.Helper()
	h := &httpRig{f: kservefake.New(t)}
	if target == "" {
		target = h.f.Addr
	}
	idx := 0
	for k := 0; k < workers; k++ {
		var ids []int
		for i := 0; i < conns; i++ {
			ids = append(ids, idx)
			idx++
		}
		g, err := h1client.NewGroup(h1client.Options{Target: target, BackoffMin: 5 * time.Millisecond, BackoffMax: 50 * time.Millisecond}, ids)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = g.Close() })
		h.groups = append(h.groups, g)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if target == h.f.Addr {
		for _, g := range h.groups {
			if err := g.WaitReady(ctx); err != nil {
				t.Fatal(err)
			}
		}
	}
	m, err := h1client.FetchModel(ctx, h.f.Addr, 0, "m", "")
	if err != nil {
		t.Fatal(err)
	}
	tpl, err := m.NewTemplate(nil, 2)
	if err != nil {
		t.Fatal(err)
	}
	gen := tensorgen.NewGenerator(1)
	pool, err := payload.BuildFunc(func() ([]byte, error) { return gen.EncodeHTTP(tpl, h.f.Addr) }, 16)
	if err != nil {
		t.Fatal(err)
	}
	h.rig = &rig{pool: pool, val: tensorgen.NewValidator(m.Outputs, 2, m.Batched)}
	return h
}

func (h *httpRig) mod(d *Deps) { d.Conns, d.Groups = nil, h.groups }

func httpCfg() Config {
	cfg := baseCfg()
	cfg.Workers = 2
	cfg.Schedule = Schedule{Start: 1000, Max: 1e6, Duration: time.Second}
	return cfg
}

func TestHTTPRunValidatesEveryResponse(t *testing.T) {
	h := newHTTPRig(t, 2, 4, "")
	ro := execute(t, h.rig, httpCfg(), nil, h.mod)
	if ro.res.Reason != ReasonDuration {
		t.Fatal(ro.res.Reason)
	}
	if h.f.Bad.Load() != 0 {
		t.Fatalf("the server refused %d requests: %s", h.f.Bad.Load(), h.f.LastError())
	}
	got := h.f.Received.Load()
	if got < 950 || got > 1050 {
		t.Fatalf("%d requests at 1000/s for 1 s", got)
	}
	// every completion is http_200 (counted OK), validated
	tot := 0.0
	for _, l := range ro.stdout[:len(ro.stdout)-1] {
		for _, st := range statusCounts(t, l) {
			tot += st.n
			if st.label != "http_200" {
				t.Fatalf("status %s", st.label)
			}
		}
	}
	if int64(tot) != got {
		t.Fatalf("http_200 completions %v, server got %d", tot, got)
	}
	// The per-tick rates are not summed against the counts: the last tick is a partial one
	// of its own length. The exact check is the per-status counts above.
	if ro.sum("ok_rps") <= 0 || ro.sum("error_rps") != 0 {
		t.Fatalf("ok_rps %v, error_rps %v", ro.sum("ok_rps"), ro.sum("error_rps"))
	}
	if ro.ticks[2]["lat_sent_p99_us"] == "" || ro.ticks[2]["send_delay_p99_us"] == "" {
		t.Fatal("latency columns are empty")
	}
}

func TestHTTPInvalidResponses(t *testing.T) {
	for name, c := range map[string]struct {
		mode  int32
		label string
	}{
		"wrong shape":     {kservefake.BadShape, "invalid_response"},
		"json not binary": {kservefake.NoBinary, "invalid_response"},
		"short body":      {kservefake.ShortBody, "invalid_response"},
		"503":             {kservefake.Unavailable, "http_503"},
	} {
		t.Run(name, func(t *testing.T) {
			h := newHTTPRig(t, 1, 4, "")
			h.f.Mode.Store(c.mode)
			cfg := httpCfg()
			cfg.Workers = 1
			cfg.Schedule.Duration = 600 * time.Millisecond
			ro := execute(t, h.rig, cfg, nil, h.mod)
			seen := map[string]float64{}
			for _, l := range ro.stdout[:len(ro.stdout)-1] {
				for _, st := range statusCounts(t, l) {
					seen[st.label] += st.n
				}
			}
			if len(seen) != 1 || seen[c.label] < 400 {
				t.Fatalf("statuses %v, want only %s", seen, c.label)
			}
			if ro.sum("ok_rps") != 0 || ro.sum("error_rps") == 0 {
				t.Fatal("these are errors")
			}
		})
	}
}

func TestHTTPErrorRateGuardCountsHTTPErrors(t *testing.T) {
	h := newHTTPRig(t, 1, 4, "")
	h.f.Mode.Store(kservefake.Unavailable)
	cfg := httpCfg()
	cfg.Workers = 1
	cfg.Schedule.Duration = 0
	rate := 0.01
	cfg.Guards = guard.Config{Consecutive: 2, ErrorRate: &rate}
	ro := execute(t, h.rig, cfg, nil, h.mod)
	if ro.res.Reason != "guard:error_rate" {
		t.Fatalf("reason %q", ro.res.Reason)
	}
}

func TestHTTPNoHealthyConnectionIsConnect(t *testing.T) {
	// A target nobody listens on: the groups never come up.
	h := newHTTPRig(t, 2, 3, "127.0.0.1:1")
	cfg := httpCfg()
	cfg.Schedule.Duration = 500 * time.Millisecond
	cfg.LogEvery = 1
	ro := execute(t, h.rig, cfg, nil, h.mod)
	seen := map[string]float64{}
	var sent float64
	for _, l := range ro.stdout[:len(ro.stdout)-1] {
		for _, st := range statusCounts(t, l) {
			seen[st.label] += st.n
		}
		sent += jsonNum(t, l, "sent_rps") * 0.2
	}
	if len(seen) != 1 || seen["connect"] < 400 {
		t.Fatalf("%v", seen)
	}
	if sent > 5 {
		t.Fatalf("requests that were never sent count as sent: %v", sent)
	}
	if ro.ticks[1]["send_delay_p99_us"] != "" {
		t.Fatal("a request that was never sent has no send delay")
	}
}

// With one connection and a slow server the wait for a connection is backpressure: it
// shows in the send delay and not in sched_lag.
func TestHTTPBackpressureIsInSendDelayNotSchedLag(t *testing.T) {
	h := newHTTPRig(t, 1, 1, "")
	h.f.Delay.Store(int64(20 * time.Millisecond)) // one connection: at most 50 req/s
	cfg := httpCfg()
	cfg.Workers = 1
	cfg.Schedule = Schedule{Start: 100, Max: 1e6, Duration: time.Second}
	cfg.Guards = guard.Config{Consecutive: 3}
	ro := execute(t, h.rig, cfg, nil, h.mod)
	last := ro.ticks[3]
	delay := mustFloat(t, last["send_delay_p99_us"])
	lag := mustFloat(t, last["sched_lag_p99_us"])
	if delay < 100_000 {
		t.Fatalf("send delay p99 %v us, want the queue wait (>100 ms)", delay)
	}
	// The wait is over 100 ms: anything far below that is scheduling noise (a busy machine
	// adds milliseconds), not the wait being counted.
	if lag > 50_000 {
		t.Fatalf("sched_lag p99 %v us: the wait must not be counted as scheduling delay", lag)
	}
}

func TestHTTPConnectionCloseKeepsTheRunGoing(t *testing.T) {
	h := newHTTPRig(t, 1, 4, "")
	h.f.CloseAfter.Store(25)
	cfg := httpCfg()
	cfg.Workers = 1
	cfg.Schedule.Duration = time.Second
	cfg.Schedule.Start = 400
	ro := execute(t, h.rig, cfg, nil, h.mod)
	seen := map[string]float64{}
	for _, l := range ro.stdout[:len(ro.stdout)-1] {
		for _, st := range statusCounts(t, l) {
			seen[st.label] += st.n
		}
	}
	// A response that came with Connection: close is delivered as it is, and the
	// reconnect keeps the other requests going.
	if seen["http_200"] < 380 {
		t.Fatalf("%v", seen)
	}
	var bad float64
	for k, v := range seen {
		if k != "http_200" {
			bad += v
		}
	}
	if bad > 10 {
		t.Fatalf("%v", seen)
	}
}

func TestNewRejectsMixedOrMissingTransports(t *testing.T) {
	h := newHTTPRig(t, 1, 1, "")
	cfg := httpCfg()
	if _, err := New(cfg, Deps{Pool: h.rig.pool, Out: nil}); err == nil {
		t.Fatal("no connections accepted")
	}
	if _, err := New(cfg, Deps{Pool: h.rig.pool, Out: nil, Groups: h.groups, Conns: []*h2client.Conn{nil}}); err == nil {
		t.Fatal("gRPC connections and HTTP groups accepted together")
	}
}

type statusCount struct {
	label string
	n     float64
}

// statusCounts reads the status counts of one stdout tick line.
func statusCounts(t *testing.T, line string) []statusCount {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal([]byte(line), &m); err != nil {
		t.Fatal(err)
	}
	var out []statusCount
	for st, n := range m["status"].(map[string]any) {
		out = append(out, statusCount{st, n.(float64)})
	}
	return out
}

func jsonNum(t *testing.T, line, key string) float64 {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal([]byte(line), &m); err != nil {
		t.Fatal(err)
	}
	v, _ := m[key].(float64)
	return v
}

func mustFloat(t *testing.T, s string) float64 {
	t.Helper()
	v, err := strconv.ParseFloat(s, 64)
	if err != nil {
		t.Fatalf("%q: %v", s, err)
	}
	return v
}
