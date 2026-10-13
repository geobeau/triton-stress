package plan

import (
	"math"
	"testing"
)

func close(a, b float64) bool {
	return math.Abs(a-b) <= 1e-9*math.Max(math.Max(math.Abs(a), math.Abs(b)), 1)
}

func TestFlatAndRampIntegrals(t *testing.T) {
	var p Plan
	inf := math.Inf(1)
	p.PushLinear(0, 100, 0, 10, inf)
	p.PushLinear(10, 100, 50, inf, 300)
	// ramp reaches 300 at t=14, then flat
	if len(p.Segs) != 3 {
		t.Fatalf("segments: %+v", p.Segs)
	}
	for _, c := range []struct{ t, want float64 }{{5, 100}, {12, 200}, {20, 300}} {
		if got := p.RateAt(c.t); !close(got, c.want) {
			t.Errorf("rate(%v) = %v, want %v", c.t, got, c.want)
		}
	}
	if got := p.Integral(0, 10); !close(got, 1000) {
		t.Error(got)
	}
	if got := p.Integral(10, 14); !close(got, 800) { // avg 200 * 4 s
		t.Error(got)
	}
	if got := p.Integral(0, 15); !close(got, 1000+800+300) {
		t.Error(got)
	}
}

func TestSolveInvertsIntegral(t *testing.T) {
	var p Plan
	inf := math.Inf(1)
	p.PushLinear(1, 10, 0, 3, inf)
	p.PushLinear(3, 10, 25, 9, 1e9)
	p.PushLinear(9, 160, -40, inf, 1e9)
	for _, c := range [][2]float64{{0, 5}, {1.5, 3}, {2.5, 30}, {4, 100}, {8.5, 250}} {
		t2, ok := p.Solve(c[0], c[1])
		if !ok {
			t.Fatalf("no solution for %v", c)
		}
		if got := p.Integral(c[0], t2); !close(got, c[1]) {
			t.Errorf("t=%v e=%v t2=%v integral=%v", c[0], c[1], t2, got)
		}
	}
	// Total mass is finite: the ramp-down hits 0 at t=13.
	total := p.Integral(0, 100)
	if _, ok := p.Solve(0, total+1); ok {
		t.Error("solution beyond the plan's mass")
	}
	if !close(p.EndTime(), 13) {
		t.Error(p.EndTime())
	}
}

// Regression: an unbounded flat segment made the capacity 0*Inf^2 = NaN.
func TestSolveUnboundedFlatSegment(t *testing.T) {
	var p Plan
	p.PushLinear(0, 1000, 0, math.Inf(1), math.Inf(1))
	for _, e := range []float64{1, 1000, 1e9} {
		t2, ok := p.Solve(0, e)
		if !ok || math.IsNaN(t2) || !close(t2, e/1000) {
			t.Fatalf("Solve(0, %v) = %v, %v", e, t2, ok)
		}
	}
	// the same with the plan's start not at zero and a clamped ramp reaching an unbounded plateau
	var q Plan
	q.PushLinear(0, 0, 100, math.Inf(1), 500)
	if t2, ok := q.Solve(0, 1e6); !ok || math.IsNaN(t2) || math.IsInf(t2, 0) {
		t.Fatal(t2, ok)
	}
}

func TestTruncateEndsPlan(t *testing.T) {
	var p Plan
	p.PushLinear(0, 100, 10, math.Inf(1), 1000)
	p.Truncate(5)
	if p.RateAt(6) != 0 {
		t.Error("rate after truncate")
	}
	if !close(p.EndTime(), 5) {
		t.Error(p.EndTime())
	}
	if _, ok := p.Solve(4.99, 10); ok {
		t.Error("solved past the truncation")
	}
}

func TestPushLinearEdgeCases(t *testing.T) {
	var p Plan
	p.PushLinear(5, 10, 1, 5, 100) // empty span
	p.PushLinear(0, 0, 0, 10, 100) // zero rate, no slope: nothing
	if len(p.Segs) != 0 {
		t.Fatalf("%+v", p.Segs)
	}
	p.PushLinear(0, 200, 0, 1, 100) // clamped to rmax
	if p.RateAt(0.5) != 100 {
		t.Error(p.RateAt(0.5))
	}
}

func TestPruneBefore(t *testing.T) {
	var p Plan
	p.PushLinear(0, 10, 0, 1, 100)
	p.PushLinear(1, 10, 0, 2, 100)
	p.PruneBefore(1.5)
	if len(p.Segs) != 1 || p.Segs[0].T0 != 1 {
		t.Fatalf("%+v", p.Segs)
	}
}
