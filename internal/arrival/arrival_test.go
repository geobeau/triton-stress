package arrival

import (
	"math"
	"slices"
	"testing"

	"github.com/criteo/triton-stress/internal/plan"
)

func flat(rate, until float64) *plan.Plan {
	var p plan.Plan
	p.PushLinear(0, rate, 0, until, math.Inf(1))
	return &p
}

func collect(g *Gen, p *plan.Plan) []float64 {
	var v []float64
	for {
		t, ok := g.Peek()
		if !ok {
			return v
		}
		v = append(v, t)
		g.Advance(p)
	}
}

func TestRngReferenceAndRange(t *testing.T) {
	a, b := NewRng(1), NewRng(1)
	for i := 0; i < 100; i++ {
		if a.Uint64() != b.Uint64() {
			t.Fatal("same seed must give the same stream")
		}
	}
	if NewRng(1).Uint64() == NewRng(2).Uint64() {
		t.Fatal("different seeds should differ")
	}
	r := NewRng(5)
	sum := 0.0
	const n = 200000
	for i := 0; i < n; i++ {
		f := r.Float64()
		if f < 0 || f >= 1 {
			t.Fatal(f)
		}
		sum += f
	}
	if m := sum / n; math.Abs(m-0.5) > 0.005 {
		t.Fatalf("uniform mean %v", m)
	}
}

func TestConstantWorkersInterleaveEvenly(t *testing.T) {
	// The plan ends just past the 1000th arrival, which sits exactly on a segment boundary.
	p := flat(1000, 1+1e-6)
	const n = 4
	var all []float64
	for k := 0; k < n; k++ {
		all = append(all, collect(New(Constant, k, n, 1, p, 0), p)...)
	}
	slices.Sort(all)
	if len(all) != 1000 {
		t.Fatalf("%d arrivals", len(all))
	}
	for i := 1; i < len(all); i++ {
		if gap := all[i] - all[i-1]; math.Abs(gap-1e-3) > 1e-9 {
			t.Fatalf("gap %v", gap)
		}
	}
}

func TestPoissonRateAndDispersion(t *testing.T) {
	const rate, dur, n = 50000.0, 20.0, 8
	p := flat(rate, dur)
	var all []float64
	for k := 0; k < n; k++ {
		all = append(all, collect(New(Poisson, k, n, 42, p, 0), p)...)
	}
	slices.Sort(all)
	expected := rate * dur
	got := float64(len(all))
	if math.Abs(got-expected)/expected > 0.005 {
		t.Fatalf("count %v vs %v", got, expected)
	}
	// Merged gaps ~ Exp(rate): mean 1/rate, coefficient of variation 1.
	gaps := make([]float64, len(all)-1)
	mean := 0.0
	for i := range gaps {
		gaps[i] = all[i+1] - all[i]
		mean += gaps[i]
	}
	mean /= float64(len(gaps))
	varr := 0.0
	for _, g := range gaps {
		varr += (g - mean) * (g - mean)
	}
	varr /= float64(len(gaps))
	cv := math.Sqrt(varr) / mean
	if math.Abs(mean*rate-1) > 0.01 {
		t.Errorf("mean gap %v", mean)
	}
	if math.Abs(cv-1) > 0.02 {
		t.Errorf("cv %v", cv)
	}
	// Per-second counts of a Poisson process: variance ~ mean.
	counts := make([]float64, int(dur))
	for _, x := range all {
		counts[min(int(x), len(counts)-1)]++
	}
	m := 0.0
	for _, c := range counts {
		m += c
	}
	m /= float64(len(counts))
	v := 0.0
	for _, c := range counts {
		v += (c - m) * (c - m)
	}
	v /= float64(len(counts) - 1)
	if v/m <= 0.4 || v/m >= 2 {
		t.Errorf("dispersion %v", v/m)
	}
}

func TestRampCountMatchesIntegral(t *testing.T) {
	var p plan.Plan
	p.PushLinear(0, 100, 2000, 10, 1e9)
	expected := p.Integral(0, 10)
	const n = 3
	got := 0
	for k := 0; k < n; k++ {
		got += len(collect(New(Constant, k, n, 7, &p, 0), &p))
	}
	if math.Abs(float64(got)-expected) > n {
		t.Fatalf("%d vs %v", got, expected)
	}
}

func TestReplanKeepsContinuity(t *testing.T) {
	old := flat(1000, math.Inf(1))
	next := old.Clone()
	next.Truncate(0.5)
	next.PushLinear(0.5, 2000, 0, 1, math.Inf(1))
	next.SwitchT = 0.5
	g := New(Constant, 0, 1, 1, old, 0)
	var v []float64
	for {
		t, ok := g.Peek()
		if !ok || t > 0.45 {
			break
		}
		v = append(v, t)
		g.Advance(old)
	}
	g.Replan(old, next)
	for {
		t, ok := g.Peek()
		if !ok {
			break
		}
		v = append(v, t)
		g.Advance(next)
	}
	expected := old.Integral(0, 0.5) + next.Integral(0.5, 1)
	if math.Abs(float64(len(v))-expected) > 1 {
		t.Fatalf("%d vs %v", len(v), expected)
	}
}

func TestStopByTruncation(t *testing.T) {
	old := flat(1000, math.Inf(1))
	g := New(Poisson, 0, 2, 3, old, 0)
	for i := 0; i < 10; i++ {
		g.Advance(old)
	}
	next := old.Clone()
	next.Truncate(0.2)
	next.SwitchT = 0.2
	g.Replan(old, next)
	for {
		x, ok := g.Peek()
		if !ok {
			break
		}
		if x > 0.2+1e-12 {
			t.Fatalf("arrival %v after truncation", x)
		}
		g.Advance(next)
	}
}

func TestUnboundedFlatArrivalIsFinite(t *testing.T) {
	p := flat(1000, math.Inf(1))
	g := New(Poisson, 3, 8, 1, p, 5)
	for i := 0; i < 1000; i++ {
		x, ok := g.Peek()
		if !ok || math.IsNaN(x) || math.IsInf(x, 0) || x < 5 {
			t.Fatal(x, ok)
		}
		g.Advance(p)
	}
}
