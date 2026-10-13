// Package plan is the piecewise-linear request-rate plan and its exact integration.
//
// Times are seconds since the clock base. Each segment has r(t) = R0 + S*(t-T0) on
// [T0, T1) and is non-negative over its whole span; caps and zero crossings are split
// into separate segments by PushLinear. Outside every segment the rate is 0.
//
// A Plan is not safe for concurrent modification: build it, then share it read-only
// and publish changes as a new Plan.
package plan

import (
	"math"
	"sort"
)

// Seg is one linear piece of the plan.
type Seg struct {
	T0, T1 float64
	R0, S  float64
}

func (s Seg) rate(t float64) float64 { return math.Max(s.R0+s.S*(t-s.T0), 0) }

// integral of the rate over [a, b], both clipped to the segment.
func (s Seg) integral(a, b float64) float64 {
	a = math.Max(a, s.T0)
	b = math.Min(b, s.T1)
	if b <= a {
		return 0
	}
	ra := s.rate(a)
	l := b - a
	return math.Max(ra*l+0.5*s.S*l*l, 0)
}

// Plan is a list of segments ordered by time.
type Plan struct {
	Segs []Seg
	// SwitchT is the time from which this plan may differ from its predecessor
	// (see arrival.Gen.Replan).
	SwitchT float64
}

// Clone copies the plan.
func (p *Plan) Clone() *Plan {
	return &Plan{Segs: append([]Seg(nil), p.Segs...), SwitchT: p.SwitchT}
}

// firstEndingAfter is the index of the first segment whose end is after t.
func (p *Plan) firstEndingAfter(t float64) int {
	return sort.Search(len(p.Segs), func(i int) bool { return p.Segs[i].T1 > t })
}

// RateAt is the rate at time t.
func (p *Plan) RateAt(t float64) float64 {
	i := p.firstEndingAfter(t)
	if i < len(p.Segs) && p.Segs[i].T0 <= t {
		return p.Segs[i].rate(t)
	}
	return 0
}

// Integral is the number of requests the plan offers over [a, b].
func (p *Plan) Integral(a, b float64) float64 {
	if b <= a {
		return 0
	}
	sum := 0.0
	for _, s := range p.Segs[p.firstEndingAfter(a):] {
		if s.T0 >= b {
			break
		}
		sum += s.integral(a, b)
	}
	return sum
}

// Solve is the smallest t' >= t with the integral of the rate over [t, t'] equal to e,
// or ok=false if the plan ends first.
func (p *Plan) Solve(t, e float64) (tp float64, ok bool) {
	e = math.Max(e, 0)
	for _, seg := range p.Segs[p.firstEndingAfter(t):] {
		a := math.Max(t, seg.T0)
		if a >= seg.T1 {
			continue
		}
		if e <= 0 {
			return a, true
		}
		r := seg.rate(a)
		l := seg.T1 - a
		// S = 0 is special-cased: on an unbounded flat segment 0*Inf^2 would be NaN.
		var capacity float64
		if seg.S == 0 {
			capacity = r * l
		} else {
			capacity = r*l + 0.5*seg.S*l*l
		}
		if capacity >= e {
			// d solves r*d + S*d^2/2 = e; this form is stable for S = 0 and S < 0.
			disc := math.Max(r*r+2*seg.S*e, 0)
			den := r + math.Sqrt(disc)
			d := l
			if den > 0 {
				d = 2 * e / den
			}
			return a + math.Min(d, l), true
		}
		e -= capacity
		t = seg.T1
	}
	return 0, false
}

// PushLinear appends r(t) = r0 + slope*(t-t0) on [t0, tEnd), clamped to [0, rmax].
func (p *Plan) PushLinear(t0, r0, slope, tEnd, rmax float64) {
	r0 = math.Min(math.Max(r0, 0), rmax)
	if tEnd <= t0 {
		return
	}
	push := func(s Seg) {
		if s.T1 > s.T0 && (s.R0 > 0 || s.S > 0) {
			p.Segs = append(p.Segs, s)
		}
	}
	switch {
	case slope > 0:
		tCap := t0 + (rmax-r0)/slope
		if tCap < tEnd {
			push(Seg{t0, tCap, r0, slope})
			push(Seg{tCap, tEnd, rmax, 0})
		} else {
			push(Seg{t0, tEnd, r0, slope})
		}
	case slope < 0:
		tZero := t0 + r0/-slope
		push(Seg{t0, math.Min(tZero, tEnd), r0, slope})
	default:
		push(Seg{t0, tEnd, r0, 0})
	}
}

// Truncate drops everything at or after t: the rate is 0 from t on.
func (p *Plan) Truncate(t float64) {
	kept := p.Segs[:0]
	for _, s := range p.Segs {
		if s.T0 < t {
			kept = append(kept, s)
		}
	}
	p.Segs = kept
	if n := len(p.Segs); n > 0 && p.Segs[n-1].T1 > t {
		p.Segs[n-1].T1 = t
	}
}

// PruneBefore forgets segments that ended before t.
func (p *Plan) PruneBefore(t float64) {
	kept := p.Segs[:0]
	for _, s := range p.Segs {
		if s.T1 > t {
			kept = append(kept, s)
		}
	}
	p.Segs = kept
}

// EndTime is when the rate becomes 0 for good (+Inf for an unbounded plan).
func (p *Plan) EndTime() float64 {
	if len(p.Segs) == 0 {
		return 0
	}
	return p.Segs[len(p.Segs)-1].T1
}
