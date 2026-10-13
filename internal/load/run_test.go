package load

import (
	"bytes"
	"context"
	"encoding/csv"
	"encoding/json"
	"google.golang.org/grpc"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/criteo/triton-stress/internal/arrival"
	"github.com/criteo/triton-stress/internal/clock"
	"github.com/criteo/triton-stress/internal/guard"
	"github.com/criteo/triton-stress/internal/metrics"
	"github.com/criteo/triton-stress/internal/output"
	"github.com/criteo/triton-stress/internal/plan"
	"github.com/criteo/triton-stress/internal/units"
)

type runOut struct {
	dir    string
	res    Result
	stdout []string
	events []map[string]any
	ticks  []map[string]string
	took   time.Duration
}

// execute runs the load against the rig and loads everything it wrote.
func execute(t *testing.T, rg *rig, cfg Config, ctx context.Context, mod func(*Deps)) runOut {
	t.Helper()
	dir := filepath.Join(t.TempDir(), "run")
	var stdout bytes.Buffer
	out, err := output.Open(dir, cfg.LogEvery > 0, &stdout)
	if err != nil {
		t.Fatal(err)
	}
	d := Deps{Conns: rg.conns, Pool: rg.pool, Validator: rg.val, Out: out}
	if mod != nil {
		mod(&d)
	}
	r, err := New(cfg, d)
	if err != nil {
		t.Fatal(err)
	}
	if ctx == nil {
		ctx = context.Background()
	}
	start := time.Now()
	res, err := r.Run(ctx)
	if err != nil {
		t.Fatal(err)
	}
	ro := runOut{dir: dir, res: res, took: time.Since(start)}
	if err := out.Close(); err != nil {
		t.Fatal(err)
	}
	ro.stdout = strings.Split(strings.TrimSpace(stdout.String()), "\n")
	b, _ := os.ReadFile(filepath.Join(dir, "events.jsonl"))
	for _, l := range strings.Split(strings.TrimSpace(string(b)), "\n") {
		var m map[string]any
		if err := json.Unmarshal([]byte(l), &m); err != nil {
			t.Fatalf("bad event line %q: %v", l, err)
		}
		ro.events = append(ro.events, m)
	}
	f, err := os.Open(filepath.Join(dir, "intervals.csv"))
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = f.Close() }()
	recs, err := csv.NewReader(f).ReadAll()
	if err != nil {
		t.Fatal(err)
	}
	for _, rec := range recs[1:] {
		line := map[string]string{}
		for i, c := range recs[0] {
			line[c] = rec[i]
		}
		ro.ticks = append(ro.ticks, line)
	}
	return ro
}

func (ro runOut) sum(col string) float64 {
	s := 0.0
	for _, r := range ro.ticks {
		v, _ := strconv.ParseFloat(r[col], 64)
		s += v
	}
	return s
}

func (ro runOut) eventsOf(kind string) []map[string]any {
	var out []map[string]any
	for _, e := range ro.events {
		if e["event"] == kind {
			out = append(out, e)
		}
	}
	return out
}

func (ro runOut) stop(t *testing.T) map[string]any {
	t.Helper()
	s := ro.eventsOf("stop")
	if len(s) != 1 || ro.events[len(ro.events)-1]["event"] != "stop" {
		t.Fatalf("want exactly one stop event, last in the file: %v", ro.events)
	}
	return s[0]
}

func baseCfg() Config {
	return Config{
		Schedule: Schedule{Warmup: 0, Start: 500, Slope: 0, Max: 1e6, Duration: time.Second},
		Arrival:  arrival.Constant, Seed: 1, Workers: 3,
		Timeout: 2 * time.Second, Interval: 200 * time.Millisecond,
		Guards: guard.Config{Consecutive: 3},
	}
}

func TestRampToMaxRate(t *testing.T) {
	f := &fakeTriton{}
	rg := newRig(t, 6, true, f)
	cfg := baseCfg()
	cfg.Schedule = Schedule{Warmup: 400 * time.Millisecond, Start: 200, Slope: 400, Max: 1000}
	cfg.Workers = 4
	ro := execute(t, rg, cfg, nil, nil)

	if ro.res.Reason != ReasonMaxRate {
		t.Fatalf("reason %q", ro.res.Reason)
	}
	if ro.took < 2300*time.Millisecond || ro.took > 3500*time.Millisecond {
		t.Fatalf("a 2.4 s schedule took %v", ro.took)
	}
	stop := ro.stop(t)
	if stop["reason"] != "max_rate_reached" || math.Abs(stop["offered_rps"].(float64)-1000) > 1 {
		t.Fatalf("%v", stop)
	}
	if _, ok := stop["inflight"].(float64); !ok {
		t.Fatalf("%v", stop)
	}
	// phase events in order
	var phases []string
	for _, e := range ro.eventsOf("phase") {
		phases = append(phases, e["phase"].(string))
	}
	if strings.Join(phases, ",") != "warmup,ramp,drain,done" {
		t.Fatalf("phases %v", phases)
	}
	// the number of requests matches the plan: 0.4 s at 200 + 2 s from 200 to 1000
	want := 0.4*200 + 2*(200+1000)/2.0
	ok := ro.sum("ok_rps")
	_ = ok
	if got := float64(f.received.Load()); math.Abs(got-want)/want > 0.03 {
		t.Fatalf("server received %v requests, plan has %v", got, want)
	}
	// ticks: phases and the offered rate follow the plan
	if len(ro.ticks) < 6 { // 12 on an idle machine; a starved one emits late ticks
		t.Fatalf("%d ticks", len(ro.ticks))
	}
	if ro.ticks[0]["phase"] != "warmup" || ro.ticks[1]["phase"] != "warmup" || ro.ticks[2]["phase"] != "ramp" {
		t.Fatalf("phases %v %v %v", ro.ticks[0]["phase"], ro.ticks[1]["phase"], ro.ticks[2]["phase"])
	}
	if o, _ := strconv.ParseFloat(ro.ticks[0]["offered_rps"], 64); math.Abs(o-200) > 10 {
		t.Fatalf("warmup offered %v", o)
	}
	last := ro.ticks[len(ro.ticks)-3]
	if o, _ := strconv.ParseFloat(last["offered_rps"], 64); o < 800 || o > 1050 {
		t.Fatalf("offered near the end %v", o)
	}
	// every request completed OK (statuses) and the totals tie out
	var tot float64
	for _, l := range ro.stdout[:len(ro.stdout)-1] {
		var m map[string]any
		if err := json.Unmarshal([]byte(l), &m); err != nil {
			t.Fatal(err)
		}
		for st, n := range m["status"].(map[string]any) {
			if st != "OK" {
				t.Fatalf("unexpected status %s", st)
			}
			tot += n.(float64)
		}
	}
	if int64(tot) != f.received.Load() {
		t.Fatalf("OK completions %v vs server %d", tot, f.received.Load())
	}
	// stdout: one JSON line per tick, then the stop event repeated
	if len(ro.stdout) != len(ro.ticks)+1 || !strings.Contains(ro.stdout[len(ro.stdout)-1], `"event":"stop"`) {
		t.Fatalf("%d stdout lines for %d ticks", len(ro.stdout), len(ro.ticks))
	}
	last0 := ro.stdout[len(ro.stdout)-1]
	b, _ := os.ReadFile(filepath.Join(ro.dir, "events.jsonl"))
	if !strings.HasSuffix(strings.TrimSpace(string(b)), last0) {
		t.Fatal("last stdout line must repeat the stop event")
	}
	if _, err := os.Stat(filepath.Join(ro.dir, "requests.csv")); err == nil {
		t.Fatal("requests.csv without --request-log-every")
	}
	// timestamps
	tsRe := regexp.MustCompile(`^\d{10}\.\d{3}$`)
	prev := 0.0
	for _, r := range ro.ticks {
		if !tsRe.MatchString(r["ts"]) {
			t.Fatalf("ts %q", r["ts"])
		}
		v, _ := strconv.ParseFloat(r["ts"], 64)
		if v < prev || math.Abs(v-float64(time.Now().Unix())) > 60 {
			t.Fatalf("ts %v", v)
		}
		prev = v
	}
	if !strings.HasSuffix(ro.ticks[0]["iso"], "Z") || ro.ticks[0]["cpu_user_pct"] == "" && len(ro.ticks) > 2 && ro.ticks[2]["cpu_user_pct"] == "" {
		t.Fatalf("iso %q, cpu %q", ro.ticks[0]["iso"], ro.ticks[2]["cpu_user_pct"])
	}
	if ro.ticks[3]["lat_sched_p99_us"] == "" || ro.ticks[3]["sched_lag_p99_us"] == "" {
		t.Fatal("percentile columns are empty")
	}
}

func TestFlatPoissonUntilDuration(t *testing.T) {
	f := &fakeTriton{}
	rg := newRig(t, 4, true, f)
	cfg := baseCfg()
	cfg.Arrival = arrival.Poisson
	cfg.Schedule = Schedule{Start: 2000, Max: 1e6, Duration: time.Second}
	ro := execute(t, rg, cfg, nil, nil)
	if ro.res.Reason != ReasonDuration {
		t.Fatal(ro.res.Reason)
	}
	if got := float64(f.received.Load()); math.Abs(got-2000)/2000 > 0.1 {
		t.Fatalf("%v requests at 2000/s for 1 s", got)
	}
	if ro.stop(t)["offered_rps"].(float64) != 2000 {
		t.Fatal(ro.stop(t))
	}
}

func TestLateArrivalsKeepTheirScheduledTime(t *testing.T) {
	f := &fakeTriton{}
	rg := newRig(t, 1, true, f)
	cfg := baseCfg()
	cfg.Workers = 1
	cfg.LogEvery = 1
	out, err := output.Open(filepath.Join(t.TempDir(), "r"), true, &bytes.Buffer{})
	if err != nil {
		t.Fatal(err)
	}
	r, err := New(cfg, Deps{Conns: rg.conns, Pool: rg.pool, Out: out})
	if err != nil {
		t.Fatal(err)
	}
	// a plan that started 1 s ago at 1000 req/s: the first ~1000 arrivals are already late
	t0 := float64(clock.Now())/1e9 - 1
	var p atomic.Pointer[plan.Plan]
	pl := &plan.Plan{}
	// The 1100th arrival is at t0+1.1 exactly; the plan ends half an interval later, so
	// float rounding cannot decide whether it is the last one.
	pl.PushLinear(t0, 1000, 0, t0+1.0+0.1+0.0005, 1e9)
	p.Store(pl)
	var stopped atomic.Bool
	w := r.workers[0]
	tm := newTimer(t)
	done := make(chan struct{})
	go func() { w.run(&p, &stopped, t0, tm); close(done) }()
	deadline := time.Now().Add(5 * time.Second)
	for f.received.Load() < 1100 && time.Now().Before(deadline) {
		time.Sleep(5 * time.Millisecond)
	}
	stopped.Store(true)
	<-done
	time.Sleep(100 * time.Millisecond)
	if got := f.received.Load(); got != 1100 {
		t.Fatalf("%d requests, the plan has 1100: nothing may be skipped", got)
	}
	log := drainLog(r)
	sort.Slice(log, func(i, j int) bool { return log[i].TSched < log[j].TSched })
	if len(log) != 1100 {
		t.Fatalf("%d logged", len(log))
	}
	first := log[0]
	if want := int64(math.Round((t0 + 0.001) * 1e9)); first.TSched < want-1000 || first.TSched > want+1000 {
		t.Fatalf("first t_sched %d, want %d", first.TSched, want)
	}
	if lag := first.TEmit - first.TSched; lag < 900_000_000 {
		t.Fatalf("a request due 1 s ago was emitted only %d ns late", lag)
	}
	for i := 1; i < len(log); i++ {
		if d := log[i].TSched - log[i-1].TSched; math.Abs(float64(d)-1e6) > 10 {
			t.Fatalf("t_sched spacing %d at %d", d, i)
		}
	}
}

func TestLatencyGuardTripsAfterConsecutiveTicks(t *testing.T) {
	f := &fakeTriton{}
	f.delay.Store(int64(30 * time.Millisecond))
	rg := newRig(t, 4, true, f)
	cfg := baseCfg()
	cfg.Schedule = Schedule{Start: 500, Max: 1e6}
	cfg.Guards = guard.Config{Consecutive: 2, Latency: &units.PctThreshold{Pct: 99, Max: 5 * time.Millisecond}}
	mm := new(*metrics.Metrics)
	ro := execute(t, rg, cfg, nil, func(d *Deps) {
		*mm = metrics.New("grpc", "m", nil)
		d.Metrics = *mm
	})
	if ro.res.Reason != "guard:latency" {
		t.Fatalf("reason %q", ro.res.Reason)
	}
	br := ro.eventsOf("breach")
	if len(br) != 2 {
		t.Fatalf("%d breach events: %v", len(br), br)
	}
	b := br[0]
	if b["guard"] != "latency" || b["threshold"].(float64) != 0.005 || b["observed"].(float64) < 0.029 {
		t.Fatalf("%v", b)
	}
	if b["ts"] == nil || b["iso"] == nil {
		t.Fatal("events carry ts and iso")
	}
	if ro.stop(t)["reason"] != "guard:latency" || ro.took > 2500*time.Millisecond {
		t.Fatalf("%v took %v", ro.stop(t), ro.took)
	}
	// breach events share the tick's ts
	if br[0]["ts"] != parseStdout(t, ro.stdout[0])["ts"] {
		t.Fatalf("breach ts %v vs tick ts %v", br[0]["ts"], parseStdout(t, ro.stdout[0])["ts"])
	}
}

func parseStdout(t *testing.T, line string) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal([]byte(line), &m); err != nil {
		t.Fatal(err)
	}
	return m
}

func TestWarmupIsReportOnlyEndToEnd(t *testing.T) {
	f := &fakeTriton{}
	f.delay.Store(int64(30 * time.Millisecond))
	rg := newRig(t, 4, true, f)
	cfg := baseCfg()
	cfg.Schedule = Schedule{Warmup: 1200 * time.Millisecond, Start: 500, Slope: 0, Max: 1e6, Duration: 1400 * time.Millisecond}
	cfg.Guards = guard.Config{Consecutive: 2, Latency: &units.PctThreshold{Pct: 99, Max: 5 * time.Millisecond}}
	ro := execute(t, rg, cfg, nil, nil)
	if ro.res.Reason != ReasonDuration {
		t.Fatalf("a guard tripped during warmup: %q", ro.res.Reason)
	}
	if n := len(ro.eventsOf("breach")); n < 3 {
		t.Fatalf("warmup breaches must still be logged, got %d", n)
	}
}

func TestMaxInflightTripsImmediately(t *testing.T) {
	f := &fakeTriton{block: make(chan struct{})}
	rg := newRig(t, 4, true, f)
	t.Cleanup(func() { close(f.block) })
	cfg := baseCfg()
	cfg.Timeout = 500 * time.Millisecond
	cfg.Interval = time.Second // the trip must not wait for a tick
	cfg.Schedule = Schedule{Start: 2000, Max: 1e6}
	limit := uint64(50)
	cfg.Guards = guard.Config{Consecutive: 3, MaxInflight: &limit}
	ro := execute(t, rg, cfg, nil, nil)
	if ro.res.Reason != "guard:max_inflight" {
		t.Fatalf("reason %q", ro.res.Reason)
	}
	if ro.took > 1500*time.Millisecond {
		t.Fatalf("took %v: the trip should be immediate and the drain bounded by the 500 ms timeout", ro.took)
	}
	if ro.res.Inflight < 51 || ro.res.Inflight > 120 {
		t.Fatalf("in flight at the stop: %d", ro.res.Inflight)
	}
	stop := ro.stop(t)
	if stop["inflight"].(float64) != float64(ro.res.Inflight) {
		t.Fatalf("%v", stop)
	}
	br := ro.eventsOf("breach")
	if len(br) != 1 || br[0]["guard"] != "max_inflight" || br[0]["threshold"].(float64) != 50 {
		t.Fatalf("%v", br)
	}
	// the stuck requests were counted as timeouts in the drain
	if ro.sum("error_rps") <= 0 {
		t.Fatal("the drained, timed-out requests must show as errors")
	}
}

func TestMaxInflightInWarmupIsReportOnly(t *testing.T) {
	f := &fakeTriton{block: make(chan struct{})}
	rg := newRig(t, 4, true, f)
	t.Cleanup(func() { close(f.block) })
	cfg := baseCfg()
	cfg.Timeout = 300 * time.Millisecond
	cfg.Schedule = Schedule{Warmup: 700 * time.Millisecond, Start: 2000, Max: 1e6, Duration: 700 * time.Millisecond}
	limit := uint64(50)
	cfg.Guards = guard.Config{Consecutive: 3, MaxInflight: &limit}
	ro := execute(t, rg, cfg, nil, nil)
	if ro.res.Reason != ReasonDuration {
		t.Fatalf("reason %q", ro.res.Reason)
	}
	warm := 0
	for _, r := range ro.ticks {
		if r["phase"] == "warmup" {
			warm++
		}
	}
	if n := len(ro.eventsOf("breach")); n < 1 || n > warm {
		t.Fatalf("report-only in-flight breaches are logged at most once per tick: %d events, %d warmup ticks", n, warm)
	}
}

func TestSignalsStopGracefully(t *testing.T) {
	for cause, want := range map[error]string{ErrInterrupted: "interrupted", ErrTerminated: "terminated", context.Canceled: "interrupted"} {
		f := &fakeTriton{}
		rg := newRig(t, 2, true, f)
		cfg := baseCfg()
		cfg.Schedule = Schedule{Start: 500, Max: 1e6}
		ctx, cancel := context.WithCancelCause(context.Background())
		time.AfterFunc(450*time.Millisecond, func() { cancel(cause) })
		ro := execute(t, rg, cfg, ctx, nil)
		if ro.res.Reason != want || ro.stop(t)["reason"] != want {
			t.Fatalf("%v: reason %q", cause, ro.res.Reason)
		}
		// Graceful: the stop comes at 450 ms and the drain is bounded by the 2 s timeout.
		if ro.took > 4*time.Second || ro.took < 400*time.Millisecond {
			t.Fatalf("took %v", ro.took)
		}
		if f.received.Load() < 100 {
			t.Fatalf("only %d requests", f.received.Load())
		}
	}
}

func TestNoReadyConnectionCompletesAsConnect(t *testing.T) {
	rg := newRig(t, 2, false, &fakeTriton{})
	cfg := baseCfg()
	cfg.Schedule = Schedule{Start: 1000, Max: 1e6}
	cfg.Interval = 100 * time.Millisecond
	rate := 0.01
	cfg.Guards = guard.Config{Consecutive: 2, ErrorRate: &rate}
	ro := execute(t, rg, cfg, nil, nil)
	if ro.res.Reason != "guard:error_rate" {
		t.Fatalf("reason %q", ro.res.Reason)
	}
	m := parseStdout(t, ro.stdout[0])
	if st := m["status"].(map[string]any); st["connect"] == nil || len(st) != 1 {
		t.Fatalf("%v", m["status"])
	}
	// never sent: no send-based percentiles, but sched_lag is there
	if m["lat_sent_p99_us"] != nil || m["send_delay_p99_us"] != nil || m["sched_lag_p99_us"] == nil {
		t.Fatalf("%v", m)
	}
	if m["sent_rps"].(float64) != 0 || m["inflight"].(float64) != 0 {
		t.Fatalf("%v", m)
	}
}

func TestRequestLogAndConnectionOwnership(t *testing.T) {
	f := &fakeTriton{}
	rg := newRig(t, 7, true, f)
	cfg := baseCfg()
	cfg.Workers = 3
	cfg.LogEvery = 5
	cfg.Schedule = Schedule{Start: 1500, Max: 1e6, Duration: time.Second}
	before := time.Now().UnixNano()
	ro := execute(t, rg, cfg, nil, nil)
	after := time.Now().UnixNano()
	fh, err := os.Open(filepath.Join(ro.dir, "requests.csv"))
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = fh.Close() }()
	recs, err := csv.NewReader(fh).ReadAll()
	if err != nil {
		t.Fatal(err)
	}
	if strings.Join(recs[0], ",") != strings.TrimSpace(output.RequestsHeader) {
		t.Fatalf("%v", recs[0])
	}
	// one in 5 per worker: about a fifth of the requests
	got, want := float64(len(recs)-1), float64(f.received.Load())/5
	if math.Abs(got-want) > 3+0.02*want {
		t.Fatalf("%v logged for %v expected", got, want)
	}
	for _, r := range recs[1:] {
		var ts [5]int64
		for i := range ts {
			ts[i], _ = strconv.ParseInt(r[i], 10, 64)
			if ts[i] < before-int64(time.Second) || ts[i] > after+int64(time.Second) {
				t.Fatalf("timestamp %d is not UTC epoch ns: %v", ts[i], r)
			}
		}
		if ts[0] > ts[1] || ts[1] > ts[2] || ts[2] > ts[3] || ts[3] > ts[4] {
			t.Fatalf("timestamps out of order %v", r)
		}
		worker, _ := strconv.Atoi(r[6])
		conn, _ := strconv.Atoi(r[7])
		if conn%3 != worker || r[5] != "OK" {
			t.Fatalf("worker %d used connection %d (%s)", worker, conn, r[5])
		}
	}
}

func TestWorkersCappedAtConnections(t *testing.T) {
	rg := newRig(t, 2, true, &fakeTriton{})
	cfg := baseCfg()
	cfg.Workers = 8
	d := Deps{Conns: rg.conns, Pool: rg.pool, Out: mustOpen(t)}
	r, err := New(cfg, d)
	if err != nil {
		t.Fatal(err)
	}
	if r.Workers() != 2 || len(r.Stats()) != 2 {
		t.Fatal(r.Workers())
	}
	for k, w := range r.workers {
		if len(w.conns) != 1 || w.connIdx[0] != k {
			t.Fatalf("worker %d: %v", k, w.connIdx)
		}
	}
}

func TestNewRejectsBadConfig(t *testing.T) {
	rg := newRig(t, 1, true, &fakeTriton{})
	out := mustOpen(t)
	good := baseCfg()
	bad := []func(*Config){
		func(c *Config) { c.Schedule.Start = 0 },
		func(c *Config) { c.Interval = 0 },
		func(c *Config) { c.Timeout = 0 },
	}
	for i, m := range bad {
		c := good
		m(&c)
		if _, err := New(c, Deps{Conns: rg.conns, Pool: rg.pool, Out: out}); err == nil {
			t.Errorf("case %d accepted", i)
		}
	}
	if _, err := New(good, Deps{Pool: rg.pool, Out: out}); err == nil {
		t.Error("no connections accepted")
	}
}

func mustOpen(t *testing.T) *output.Files {
	t.Helper()
	out, err := output.Open(filepath.Join(t.TempDir(), "r"), false, &bytes.Buffer{})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = out.Close() })
	return out
}

// maxCol is the largest value of a column over the ticks.
func (ro runOut) maxCol(col string) float64 {
	m := 0.0
	for _, r := range ro.ticks {
		if v, err := strconv.ParseFloat(r[col], 64); err == nil && v > m {
			m = v
		}
	}
	return m
}

// A server that takes one request at a time makes later requests wait for a stream:
// that wait is backpressure (between t_emit and t_send), so send_delay grows while
// sched_lag, the dispatcher's own lateness, stays small.
func TestBackpressureIsInSendDelayNotSchedLag(t *testing.T) {
	f := &fakeTriton{}
	f.delay.Store(int64(20 * time.Millisecond))
	rg := newRigOpts(t, 1, true, f, grpc.MaxConcurrentStreams(1))
	time.Sleep(50 * time.Millisecond) // the server's SETTINGS reach the client
	cfg := baseCfg()
	cfg.Workers = 1
	cfg.Timeout = 10 * time.Second
	cfg.Schedule = Schedule{Start: 100, Max: 1e6, Duration: time.Second} // twice what the server can do
	ro := execute(t, rg, cfg, nil, nil)
	sd, lag := ro.maxCol("send_delay_p99_us"), ro.maxCol("sched_lag_p99_us")
	if sd < 300_000 {
		t.Fatalf("send_delay p99 %.0f us: the wait for a stream is not in the send delay", sd)
	}
	if lag > 20_000 {
		t.Fatalf("sched_lag p99 %.0f us: the dispatcher itself was not late, backpressure must not land there", lag)
	}
}
