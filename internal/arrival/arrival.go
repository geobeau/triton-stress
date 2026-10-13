// Package arrival is the per-worker arrival process over a shared rate plan.
//
// Worker k of N consumes every N-th unit of the plan's integral (time-rescaling
// theorem): constant mode steps exactly N (first step k+1, so N workers interleave
// evenly), Poisson mode steps N*Exp(1). The superposition over workers is the
// requested process at the full rate.
package arrival

import (
	"math"
	"math/bits"

	"github.com/criteo/triton-stress/internal/plan"
)

// Mode is the spacing between requests.
type Mode uint8

// Arrival modes.
const (
	Constant Mode = iota
	Poisson
)

// String is the flag value.
func (m Mode) String() string {
	if m == Poisson {
		return "poisson"
	}
	return "constant"
}

// Rng is xoshiro256++ seeded through splitmix64.
type Rng [4]uint64

// NewRng seeds a generator.
func NewRng(seed uint64) *Rng {
	x := seed
	next := func() uint64 {
		x += 0x9E3779B97F4A7C15
		z := x
		z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
		z = (z ^ (z >> 27)) * 0x94D049BB133111EB
		return z ^ (z >> 31)
	}
	return &Rng{next(), next(), next(), next()}
}

// Uint64 is the next 64 random bits.
func (r *Rng) Uint64() uint64 {
	s := r
	res := bits.RotateLeft64(s[0]+s[3], 23) + s[0]
	t := s[1] << 17
	s[2] ^= s[0]
	s[3] ^= s[1]
	s[1] ^= s[2]
	s[0] ^= s[3]
	s[2] ^= t
	s[3] = bits.RotateLeft64(s[3], 45)
	return res
}

// Float64 is uniform in [0, 1).
func (r *Rng) Float64() float64 { return float64(r.Uint64()>>11) * (1.0 / (1 << 53)) }

// Exp1 is Exp(1).
func (r *Rng) Exp1() float64 { return -math.Log(1 - r.Float64()) }

// Gen is one worker's arrival generator.
type Gen struct {
	mode Mode
	n    float64
	rng  *Rng
	// tPrev is the time from which the current step's integral is measured.
	tPrev float64
	// step is the integral mass separating tPrev from next.
	step    float64
	next    float64
	hasNext bool
}

// New creates worker k of n (0 <= k < n), starting at tStart.
func New(mode Mode, k, n int, seed uint64, p *plan.Plan, tStart float64) *Gen {
	rng := NewRng((seed + uint64(k)) * 0x2545F4914F6CDD1D)
	nf := float64(n)
	var step float64
	if mode == Constant {
		step = float64(k + 1)
	} else {
		step = nf * rng.Exp1()
	}
	g := &Gen{mode: mode, n: nf, rng: rng, tPrev: tStart, step: step}
	g.next, g.hasNext = p.Solve(tStart, step)
	return g
}

// Peek is the pending arrival time, or ok=false when the plan has no more.
func (g *Gen) Peek() (t float64, ok bool) { return g.next, g.hasNext }

func (g *Gen) draw() float64 {
	if g.mode == Constant {
		return g.n
	}
	return g.n * g.rng.Exp1()
}

// Advance consumes the pending arrival and schedules the next one.
func (g *Gen) Advance(p *plan.Plan) {
	if !g.hasNext {
		return
	}
	t := g.next
	g.tPrev = t
	g.step = g.draw()
	g.next, g.hasNext = p.Solve(t, g.step)
}

// Replan switches to next, which equals old before next.SwitchT.
func (g *Gen) Replan(old, next *plan.Plan) {
	ts := math.Max(next.SwitchT, g.tPrev)
	if g.hasNext && g.next <= ts {
		return // the pending arrival is before the change and still valid
	}
	rem := math.Max(g.step-old.Integral(g.tPrev, ts), 0)
	g.tPrev = ts
	g.step = rem
	g.next, g.hasNext = next.Solve(ts, rem)
}
