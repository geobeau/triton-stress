package load

import (
	"context"
	"errors"
	"fmt"
	"io"
	"sync"
	"sync/atomic"
	"time"

	"github.com/criteo/triton-stress/internal/arrival"
	"github.com/criteo/triton-stress/internal/clock"
	"github.com/criteo/triton-stress/internal/grpcclient"
	"github.com/criteo/triton-stress/internal/guard"
	"github.com/criteo/triton-stress/internal/h1client"
	"github.com/criteo/triton-stress/internal/h2client"
	"github.com/criteo/triton-stress/internal/health"
	"github.com/criteo/triton-stress/internal/metrics"
	"github.com/criteo/triton-stress/internal/output"
	"github.com/criteo/triton-stress/internal/payload"
	"github.com/criteo/triton-stress/internal/plan"
	"github.com/criteo/triton-stress/internal/stats"
	"github.com/criteo/triton-stress/internal/timer"
	"github.com/criteo/triton-stress/pkg/tensorgen"
)

// Causes to cancel the Run context with, so that the stop reason says which signal
// stopped the run. A context cancelled for any other reason counts as interrupted.
var (
	ErrInterrupted = errors.New("interrupted")
	ErrTerminated  = errors.New("terminated")
)

// checkEvery is how often the controller looks at the clock, the in-flight count and
// the end of the schedule.
const checkEvery = 10 * time.Millisecond

// Config is the load configuration.
type Config struct {
	Schedule Schedule
	Arrival  arrival.Mode
	Seed     uint64
	// Workers is the number of dispatchers (at most the number of connections). With HTTP
	// it is the number of Groups.
	Workers int
	// Timeout is the client-side request timeout; it also bounds the drain.
	Timeout  time.Duration
	Interval time.Duration
	Guards   guard.Config
	// LogEvery samples one request in that many into requests.csv (0: none).
	LogEvery uint64
}

// Deps are the run's collaborators.
type Deps struct {
	// Conns are the gRPC connections, spread over the workers round-robin. For HTTP,
	// Groups is given instead: one per worker (each owns its connections and its FIFO).
	Conns     []*h2client.Conn
	Groups    []*h1client.Group
	Pool      *payload.Pool
	Validator *tensorgen.Validator // nil: responses are not decoded
	Out       *output.Files
	Metrics   *metrics.Metrics // optional
	Log       io.Writer        // human-readable progress (stderr); optional
	Health    *health.Sampler  // default: the running process
}

// Result is how the run ended.
type Result struct {
	// Reason is guard:<name>, max_rate_reached, duration, interrupted or terminated.
	Reason string
	// OfferedRPS and Inflight are the offered rate and the in-flight count at the stop.
	OfferedRPS float64
	Inflight   int64
}

// Runner runs the load.
type Runner struct {
	cfg     Config
	deps    Deps
	workers []*worker
	stats   []*stats.Worker
}

// New validates the configuration and creates the workers and their statistics, so that
// metrics can be attached before Run.
func New(cfg Config, d Deps) (*Runner, error) {
	if err := cfg.Schedule.Validate(); err != nil {
		return nil, err
	}
	switch {
	case len(d.Conns) == 0 && len(d.Groups) == 0:
		return nil, errors.New("no connections")
	case len(d.Conns) > 0 && len(d.Groups) > 0:
		return nil, errors.New("gRPC connections and HTTP groups are exclusive")
	case d.Pool == nil || d.Out == nil:
		return nil, errors.New("pool and output are required")
	case cfg.Interval <= 0:
		return nil, errors.New("--interval must be positive")
	case cfg.Timeout <= 0:
		return nil, errors.New("--request-timeout must be positive")
	}
	if d.Health == nil {
		d.Health = health.New()
	}
	n := min(max(cfg.Workers, 1), len(d.Conns))
	if len(d.Groups) > 0 {
		n = len(d.Groups)
	}
	cfg.Workers = n
	r := &Runner{cfg: cfg, deps: d}
	for k := 0; k < n && len(d.Groups) > 0; k++ {
		w := &worker{id: k, n: n, r: r, st: stats.NewWorker(k, cfg.LogEvery), pool: k * d.Pool.Len() / n, group: d.Groups[k]}
		r.workers = append(r.workers, w)
		r.stats = append(r.stats, w.st)
	}
	for k := 0; k < n && len(d.Groups) == 0; k++ {
		w := &worker{id: k, n: n, r: r, st: stats.NewWorker(k, cfg.LogEvery), pool: k * d.Pool.Len() / n}
		for i := k; i < len(d.Conns); i += n {
			w.conns = append(w.conns, d.Conns[i])
			w.connIdx = append(w.connIdx, i)
		}
		r.workers = append(r.workers, w)
		r.stats = append(r.stats, w.st)
	}
	return r, nil
}

// Stats are the workers' statistics (for the metrics).
func (r *Runner) Stats() []*stats.Worker { return r.stats }

// SetMetrics attaches the Prometheus metrics (before Run).
func (r *Runner) SetMetrics(m *metrics.Metrics) { r.deps.Metrics = m }

// Workers is the number of dispatchers in use.
func (r *Runner) Workers() int { return len(r.workers) }

func (r *Runner) logf(format string, a ...any) {
	if r.deps.Log != nil {
		_, _ = fmt.Fprintf(r.deps.Log, format+"\n", a...) // a failed log write has nowhere to go
	}
}

func (r *Runner) inflight() int64 {
	var n int64
	for _, w := range r.stats {
		n += w.Inflight()
	}
	return max(n, 0)
}

// controller is the state of one Run.
type controller struct {
	r      *Runner
	guards *guard.Guards
	t0     int64 // clock.Now at the start
	plan   atomic.Pointer[plan.Plan]
	stop   atomic.Bool

	phase     Phase
	tickPhase Phase // the phase in which the current tick began
	lastTick  int64
	nextTick  int64
	tickSent  uint64
	tickBytes uint64
	merged    *stats.Tick
	sampler   *health.Sampler
	reqLog    []stats.ReqLog

	stopped  bool
	stopAt   int64
	stopWall time.Time
	result   Result

	reportedInflight int // tick index of the last report-only in-flight breach event
	tickIndex        int
	warned           bool
}

// Run runs the load until it ends by itself (max rate reached, duration), a guard trips,
// or ctx is cancelled (cause ErrInterrupted, ErrTerminated, or anything else, which
// counts as interrupted). It returns after the drain. Output errors are reported on the
// log and do not stop the run.
func (r *Runner) Run(ctx context.Context) (Result, error) {
	c := &controller{
		r: r, guards: guard.New(r.cfg.Guards), merged: stats.NewTick(), sampler: r.deps.Health,
	}
	timers := make([]*timer.Timer, len(r.workers))
	for i := range timers {
		tm, err := timer.New()
		if err != nil {
			for _, t := range timers[:i] {
				_ = t.Close()
			}
			return Result{}, err
		}
		timers[i] = tm
	}

	c.t0 = clock.Now()
	t0s := float64(c.t0) / 1e9
	c.plan.Store(r.cfg.Schedule.Plan(t0s))
	c.lastTick, c.nextTick = c.t0, c.t0+r.cfg.Interval.Nanoseconds()
	c.sampler.Sample(time.Now()) // baseline for the first tick
	if r.cfg.Schedule.Warmup > 0 {
		c.setPhase(Warmup, c.t0)
	} else {
		c.setPhase(Ramp, c.t0)
	}
	c.tickPhase = c.phase

	var wg sync.WaitGroup
	for i, w := range r.workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			w.run(&c.plan, &c.stop, t0s, timers[i])
		}()
	}

	endElapsed, endReason, hasEnd := r.cfg.Schedule.End()
	tk := time.NewTicker(checkEvery)
	defer tk.Stop()
loop:
	for {
		select {
		case <-ctx.Done():
			if !c.stopped {
				reason := ReasonInterrupted
				if errors.Is(context.Cause(ctx), ErrTerminated) {
					reason = ReasonTerminated
				}
				c.beginStop(reason, clock.Now())
			}
			ctx = context.Background() // from here on, only the drain deadline ends the loop
		case <-tk.C:
		}
		now := clock.Now()
		switch {
		case !c.stopped:
			if c.phase == Warmup && now-c.t0 >= r.cfg.Schedule.Warmup.Nanoseconds() {
				c.setPhase(Ramp, now)
			}
			if hasEnd && now-c.t0 >= endElapsed.Nanoseconds() {
				c.beginStop(endReason, now)
				break
			}
			if b := c.guards.CheckInflight(r.inflight()); b != nil {
				c.inflightBreach(b, now)
			}
		default: // draining
			if r.inflight() == 0 || now-c.stopAt >= r.cfg.Timeout.Nanoseconds() {
				break loop
			}
		}
		if now >= c.nextTick {
			c.doTick(now, false)
			c.nextTick += r.cfg.Interval.Nanoseconds()
			for c.nextTick <= now { // a stalled controller skips ticks rather than bursting
				c.nextTick += r.cfg.Interval.Nanoseconds()
			}
		}
	}
	// Nothing more to wait for: the drain is over (or never was needed).
	end := clock.Now()
	c.stop.Store(true)
	wg.Wait()
	for _, t := range timers {
		_ = t.Close()
	}
	if end-c.lastTick > 0 {
		c.doTick(end, true)
	}
	c.setPhase(Done, end)
	c.writeStop()
	return c.result, nil
}

func (c *controller) wall(mono int64) time.Time {
	return output.Anchor{Mono: clock.Now(), Wall: time.Now()}.At(mono)
}

func (c *controller) event(kind string, at time.Time, fields ...output.KV) {
	if err := c.r.deps.Out.WriteEvent(output.Event{TS: at, Kind: kind, Fields: fields}); err != nil {
		c.warnOnce("writing events.jsonl", err)
	}
}

func (c *controller) warnOnce(what string, err error) {
	if !c.warned {
		c.warned = true
		c.r.logf("rts: warning: %s: %v (further output errors are not reported)", what, err)
	}
}

func (c *controller) setPhase(p Phase, now int64) {
	c.phase = p
	at := c.wall(now)
	c.event("phase", at, output.Str("phase", p.String()))
	if c.r.deps.Metrics != nil {
		c.r.deps.Metrics.SetPhase(p.String())
	}
	c.r.logf("rts: phase %s", p)
}

// inflightBreach handles an in-flight check above the limit: it trips at once, except in
// warmup, where it is only reported (at most once per tick).
func (c *controller) inflightBreach(b *guard.Breach, now int64) {
	if c.phase == Warmup {
		if c.reportedInflight == c.tickIndex+1 {
			return
		}
		c.reportedInflight = c.tickIndex + 1
		c.reportBreach(b, c.wall(now))
		return
	}
	c.reportBreach(b, c.wall(now))
	c.trip(b, now)
}

func (c *controller) reportBreach(b *guard.Breach, at time.Time) {
	c.event("breach", at, output.Str("guard", b.Guard), output.Float("observed", b.Observed), output.Float("threshold", b.Threshold))
	if c.r.deps.Metrics != nil {
		c.r.deps.Metrics.Breach(b.Guard)
	}
	c.r.logf("rts: guard %s breached: %g (threshold %g)", b.Guard, b.Observed, b.Threshold)
}

func (c *controller) trip(b *guard.Breach, now int64) {
	if c.r.deps.Metrics != nil {
		c.r.deps.Metrics.Trip(b.Guard)
	}
	c.beginStop("guard:"+b.Guard, now)
}

// beginStop ends the sending: the plan is truncated (so the offered rate of later ticks is
// 0), the dispatchers stop at once, and the drain starts.
func (c *controller) beginStop(reason string, now int64) {
	if c.stopped {
		return
	}
	c.stopped = true
	c.stopAt = now
	c.stopWall = c.wall(now)
	c.result = Result{
		Reason:     reason,
		OfferedRPS: c.r.cfg.Schedule.RateAt(time.Duration(now - c.t0)),
		Inflight:   c.r.inflight(),
	}
	next := c.plan.Load().Clone()
	next.Truncate(float64(now) / 1e9)
	next.SwitchT = float64(now) / 1e9
	c.plan.Store(next)
	c.stop.Store(true)
	c.setPhase(Drain, now)
	c.r.logf("rts: stopping: %s (offered %.1f req/s, %d in flight); draining for up to %v",
		reason, c.result.OfferedRPS, c.result.Inflight, c.r.cfg.Timeout)
}

func (c *controller) writeStop() {
	e := output.Event{TS: c.stopWall, Kind: "stop", Fields: []output.KV{
		output.Str("reason", c.result.Reason),
		output.Rate("offered_rps", c.result.OfferedRPS),
		output.Int("inflight", c.result.Inflight),
	}}
	if err := c.r.deps.Out.WriteStop(e); err != nil {
		c.warnOnce("writing the stop event", err)
	}
	c.r.logf("rts: done: %s", c.result.Reason)
}

// doTick closes the tick ending at now: it drains the workers, computes the record,
// writes it and (unless final or draining) evaluates the guards.
func (c *controller) doTick(now int64, final bool) {
	r := c.r
	a, b := c.lastTick, now
	secs := float64(b-a) / 1e9
	anchor := output.Anchor{Mono: now, Wall: time.Now()}
	at := anchor.At(now)

	c.merged.Reset()
	c.reqLog = c.reqLog[:0]
	var sent, bytes uint64
	for _, w := range r.stats {
		w.Drain(c.merged, &c.reqLog)
		s, by := w.Totals()
		sent += s
		bytes += by
	}
	sentTick := sent - c.tickSent
	c.tickSent, c.tickBytes = sent, bytes
	if err := r.deps.Out.WriteRequests(c.reqLog, anchor); err != nil {
		c.warnOnce("writing requests.csv", err)
	}

	offered := c.plan.Load().Integral(float64(a)/1e9, float64(b)/1e9) / secs
	completed, ok := c.merged.Completed(), c.merged.OK()
	iv := &output.Interval{
		TS: at, Phase: c.tickPhase.String(),
		OfferedRPS: offered, SentRPS: float64(sentTick) / secs,
		OKRPS: float64(ok) / secs, ErrorRPS: float64(completed-ok) / secs,
		Inflight: r.inflight(),
		Status:   map[string]uint64{},
	}
	for s, n := range c.merged.ByStatus {
		if n > 0 {
			iv.Status[grpcclient.Status(s).Label()] = n
		}
	}
	for h := 0; h < stats.NumHists; h++ {
		hist := c.merged.Hist[h]
		if hist.TotalCount() == 0 {
			continue
		}
		sum := output.HistSummary{Valid: true}
		for i, p := range output.Percentiles {
			sum.Ns[i] = hist.ValueAtPercentile(p)
		}
		sum.Ns[4] = hist.Max()
		iv.Hist[h] = sum
	}
	iv.Health, iv.HasHealth = c.sampler.Sample(time.Now())
	if secs < r.cfg.Interval.Seconds()/2 {
		iv.HasHealth = false // a short final tick: the CPU counters tick in 10 ms steps, too coarse to be meaningful
	}

	if r.deps.Metrics != nil {
		r.deps.Metrics.SetOffered(offered)
	}
	if err := r.deps.Out.WriteInterval(iv); err != nil {
		c.warnOnce("writing the interval record", err)
	}

	c.lastTick = now
	label := c.tickPhase
	c.tickPhase = c.phase
	c.tickIndex++
	if final || c.stopped {
		return
	}
	breaches, trip := c.guards.EvaluateTick(guard.Tick{
		OfferedRPS: offered, OKRPS: float64(ok) / secs,
		Completed: completed, Errors: completed - ok,
		Latency: c.merged.Hist[stats.LatSched], SendDelay: c.merged.Hist[stats.SendDelay],
	}, label == Warmup)
	for i := range breaches {
		c.reportBreach(&breaches[i], at)
	}
	if trip != nil {
		c.trip(trip, now)
	}
}
