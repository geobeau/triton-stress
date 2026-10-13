// Package guard is the stop-guard logic: per-tick breach checks debounced over
// consecutive ticks, and the immediate in-flight check.
package guard

import (
	"github.com/HdrHistogram/hdrhistogram-go"
	"github.com/criteo/triton-stress/internal/units"
)

// Guard names, as used in events and metrics and in the stop reason (guard:<name>).
const (
	Achieved    = "achieved"
	ErrorRate   = "error_rate"
	Latency     = "latency"
	SendDelay   = "send_delay"
	MaxInflight = "max_inflight"
)

// Names lists every guard.
var Names = []string{Achieved, ErrorRate, Latency, SendDelay, MaxInflight}

// Config holds the thresholds; a nil threshold turns that guard off.
type Config struct {
	// Achieved breaches when ok/s < Achieved * offered/s.
	Achieved *float64
	// ErrorRate breaches when errors/completed > ErrorRate.
	ErrorRate *float64
	// Latency breaches when the percentile of latency from scheduled exceeds the threshold.
	Latency *units.PctThreshold
	// SendDelay breaches when the percentile of the send delay exceeds the threshold.
	SendDelay *units.PctThreshold
	// MaxInflight breaches when in-flight requests exceed it.
	MaxInflight *uint64
	// Consecutive is the number of consecutive breaching ticks that trips a guard.
	Consecutive int
}

// Breach describes one breach. Observed and Threshold share a unit: a ratio for
// achieved (ok/offered, with the threshold as configured), a fraction for error_rate,
// seconds for latency and send_delay, a count for max_inflight.
type Breach struct {
	Guard     string
	Observed  float64
	Threshold float64
}

// Tick is what one interval measured.
type Tick struct {
	OfferedRPS, OKRPS float64
	Completed, Errors uint64
	Latency           *hdrhistogram.Histogram // latency from scheduled, ns
	SendDelay         *hdrhistogram.Histogram // send delay, ns
}

// Guards evaluates the guards and keeps the debounce state.
type Guards struct {
	cfg    Config
	streak map[string]int
}

// New creates the evaluator. Consecutive below 1 is treated as 1.
func New(cfg Config) *Guards {
	cfg.Consecutive = max(cfg.Consecutive, 1)
	return &Guards{cfg: cfg, streak: map[string]int{}}
}

// Config returns the configuration.
func (g *Guards) Config() Config { return g.cfg }

// EvaluateTick checks the tick-based guards. It returns every breach (to be logged)
// and, when a guard has now breached Consecutive ticks in succession, that guard's breach
// as trip. With reportOnly (warmup) breaches are returned but never counted, and the
// debounce streaks are cleared, so a trip needs Consecutive breaching ticks after
// warmup.
func (g *Guards) EvaluateTick(t Tick, reportOnly bool) (breaches []Breach, trip *Breach) {
	check := func(name string, b *Breach) {
		if b == nil || reportOnly {
			g.streak[name] = 0
		} else {
			g.streak[name]++
		}
		if b == nil {
			return
		}
		breaches = append(breaches, *b)
		if !reportOnly && g.streak[name] >= g.cfg.Consecutive && trip == nil {
			c := *b
			trip = &c
		}
	}
	c := g.cfg
	if c.Achieved != nil {
		var b *Breach
		if t.OfferedRPS > 0 && t.OKRPS < *c.Achieved*t.OfferedRPS {
			b = &Breach{Achieved, t.OKRPS / t.OfferedRPS, *c.Achieved}
		}
		check(Achieved, b)
	}
	if c.ErrorRate != nil {
		var b *Breach
		if t.Completed > 0 {
			if r := float64(t.Errors) / float64(t.Completed); r > *c.ErrorRate {
				b = &Breach{ErrorRate, r, *c.ErrorRate}
			}
		}
		check(ErrorRate, b)
	}
	if c.Latency != nil {
		check(Latency, pctBreach(Latency, t.Latency, c.Latency))
	}
	if c.SendDelay != nil {
		check(SendDelay, pctBreach(SendDelay, t.SendDelay, c.SendDelay))
	}
	return breaches, trip
}

func pctBreach(name string, h *hdrhistogram.Histogram, th *units.PctThreshold) *Breach {
	if h == nil || h.TotalCount() == 0 {
		return nil
	}
	v := h.ValueAtPercentile(th.Pct)
	if v > th.Max.Nanoseconds() {
		return &Breach{name, float64(v) / 1e9, th.Max.Seconds()}
	}
	return nil
}

// CheckInflight is the 10 ms check of the in-flight guard. It returns the breach when
// inflight is above the limit; the caller trips immediately unless in warmup.
func (g *Guards) CheckInflight(inflight int64) *Breach {
	if g.cfg.MaxInflight == nil || inflight <= 0 || uint64(inflight) <= *g.cfg.MaxInflight {
		return nil
	}
	return &Breach{MaxInflight, float64(inflight), float64(*g.cfg.MaxInflight)}
}
