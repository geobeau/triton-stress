// Package load runs the open-loop load: the rate schedule and phases, the per-worker
// dispatchers, and the controller that ticks, applies the guards and stops the run.
package load

import (
	"errors"
	"fmt"
	"math"
	"time"

	"github.com/criteo/triton-stress/internal/plan"
)

// Phase is the run's phase.
type Phase uint8

// Phases: warmup -> ramp -> drain -> done.
const (
	Warmup Phase = iota
	Ramp
	Drain
	Done
)

// String is the phase name used in output and metrics.
func (p Phase) String() string { return [...]string{"warmup", "ramp", "drain", "done"}[p] }

// Stop reasons that do not carry a guard name or a message.
const (
	ReasonMaxRate     = "max_rate_reached"
	ReasonDuration    = "duration"
	ReasonInterrupted = "interrupted"
	ReasonTerminated  = "terminated"
)

// Schedule is the rate schedule: Warmup holds Start; then the rate rises by Slope per
// second. Reaching Max ends the run. With Slope 0 the run is flat until Duration (zero
// means none) or an external stop. Duration counts from the start of the warmup and
// caps a ramping run too.
type Schedule struct {
	Warmup   time.Duration
	Start    float64 // req/s
	Slope    float64 // req/s per second
	Max      float64 // req/s
	Duration time.Duration
}

// Validate checks the schedule.
func (s Schedule) Validate() error {
	switch {
	case !(s.Start > 0):
		return errors.New("--start-rate must be above 0")
	case !(s.Max > 0):
		return errors.New("--max-rate must be above 0")
	case s.Start > s.Max:
		return fmt.Errorf("--start-rate %g is above --max-rate %g", s.Start, s.Max)
	case s.Slope < 0 || math.IsNaN(s.Slope) || math.IsInf(s.Slope, 0):
		return errors.New("--slope must be finite and not negative")
	case s.Warmup < 0:
		return errors.New("--warmup must not be negative")
	case s.Duration < 0:
		return errors.New("--duration must not be negative")
	}
	return nil
}

// rampEnd is the elapsed time at which the ramp reaches Max (ok=false when it never does).
func (s Schedule) rampEnd() (time.Duration, bool) {
	if s.Slope <= 0 {
		return 0, false
	}
	secs := (s.Max - s.Start) / s.Slope
	if secs > float64(math.MaxInt64)/1e9/2 {
		return 0, false
	}
	return s.Warmup + time.Duration(math.Round(secs*1e9)), true
}

// End is when the schedule ends by itself, as time since the start, and why. ok is false
// for a schedule with no end (flat, no duration).
func (s Schedule) End() (elapsed time.Duration, reason string, ok bool) {
	ramp, hasRamp := s.rampEnd()
	switch {
	case hasRamp && (s.Duration == 0 || ramp <= s.Duration):
		return ramp, ReasonMaxRate, true
	case s.Duration > 0:
		return s.Duration, ReasonDuration, true
	}
	return 0, "", false
}

// PhaseAt is the phase at the given time since the start, while the run is sending.
func (s Schedule) PhaseAt(elapsed time.Duration) Phase {
	if elapsed < s.Warmup {
		return Warmup
	}
	return Ramp
}

// RateAt is the scheduled rate at the given time since the start, held at the end of the
// schedule.
func (s Schedule) RateAt(elapsed time.Duration) float64 {
	if end, _, ok := s.End(); ok && elapsed > end {
		elapsed = end
	}
	if elapsed <= s.Warmup {
		return s.Start
	}
	return math.Min(s.Max, s.Start+s.Slope*(elapsed-s.Warmup).Seconds())
}

// Plan builds the rate plan with the start at t0 seconds on the clock.
func (s Schedule) Plan(t0 float64) *plan.Plan {
	end := math.Inf(1)
	if e, _, ok := s.End(); ok {
		end = t0 + e.Seconds()
	}
	p := &plan.Plan{SwitchT: t0}
	w := t0 + s.Warmup.Seconds()
	p.PushLinear(t0, s.Start, 0, math.Min(w, end), s.Max)
	p.PushLinear(w, s.Start, s.Slope, end, s.Max)
	return p
}
