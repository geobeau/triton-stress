package load

import (
	"math"
	"testing"
	"time"
)

func TestScheduleValidate(t *testing.T) {
	ok := Schedule{Warmup: time.Second, Start: 1000, Slope: 1000, Max: 1e6}
	if err := ok.Validate(); err != nil {
		t.Fatal(err)
	}
	bad := []Schedule{
		{Start: 0, Max: 10},
		{Start: 10, Max: 0},
		{Start: 20, Max: 10},
		{Start: 1, Max: 10, Slope: -1},
		{Start: 1, Max: 10, Warmup: -1},
		{Start: 1, Max: 10, Duration: -1},
		{Start: 1, Max: 10, Slope: math.NaN()},
	}
	for i, s := range bad {
		if s.Validate() == nil {
			t.Errorf("case %d should fail: %+v", i, s)
		}
	}
}

func TestPhasesAndMaxRateEnd(t *testing.T) {
	s := Schedule{Warmup: 10 * time.Second, Start: 1000, Slope: 1000, Max: 5000}
	end, reason, ok := s.End()
	if !ok || reason != ReasonMaxRate || end != 14*time.Second {
		t.Fatal(end, reason, ok)
	}
	if s.PhaseAt(9999*time.Millisecond) != Warmup || s.PhaseAt(10*time.Second) != Ramp {
		t.Fatal("phase boundary")
	}
	if s.RateAt(5*time.Second) != 1000 || s.RateAt(12*time.Second) != 3000 || s.RateAt(14*time.Second) != 5000 ||
		s.RateAt(100*time.Second) != 5000 {
		t.Fatal(s.RateAt(5*time.Second), s.RateAt(12*time.Second), s.RateAt(14*time.Second))
	}
	// the plan integrates to the same thing: 10 s at 1k, 4 s from 1k to 5k (avg 3k)
	p := s.Plan(100)
	if got := p.Integral(100, 200); math.Abs(got-(10000+12000)) > 1e-6 {
		t.Fatalf("plan mass %v", got)
	}
	if p.EndTime() != 114 || p.RateAt(113.999) < 4998.9 || p.RateAt(114) != 0 {
		t.Fatal(p.EndTime(), p.RateAt(113.999), p.RateAt(114))
	}
}

func TestDurationCapsARamp(t *testing.T) {
	s := Schedule{Warmup: time.Second, Start: 100, Slope: 100, Max: 1e6, Duration: 5 * time.Second}
	end, reason, ok := s.End()
	if !ok || reason != ReasonDuration || end != 5*time.Second {
		t.Fatal(end, reason, ok)
	}
	if got := s.Plan(0).Integral(0, 100); math.Abs(got-(100+4*100+0.5*100*16)) > 1e-6 {
		t.Fatalf("mass %v", got)
	}
	// a ramp that reaches max exactly at the duration counts as max_rate_reached
	s = Schedule{Warmup: 0, Start: 100, Slope: 100, Max: 500, Duration: 4 * time.Second}
	if _, reason, _ := s.End(); reason != ReasonMaxRate {
		t.Fatal(reason)
	}
}

func TestFlatSchedule(t *testing.T) {
	s := Schedule{Warmup: 2 * time.Second, Start: 500, Slope: 0, Max: 1e6}
	if _, _, ok := s.End(); ok {
		t.Fatal("flat without duration has no end")
	}
	p := s.Plan(0)
	if !math.IsInf(p.EndTime(), 1) || p.RateAt(1000) != 500 {
		t.Fatal("unbounded flat plan")
	}
	// the unbounded segment must solve (regression: NaN capacity)
	if x, ok := p.Solve(0, 1e5); !ok || math.Abs(x-200) > 1e-9 {
		t.Fatal(x, ok)
	}
	s.Duration = 10 * time.Second
	if end, reason, ok := s.End(); !ok || reason != ReasonDuration || end != 10*time.Second {
		t.Fatal(end, reason, ok)
	}
	if got := s.Plan(0).Integral(0, 100); math.Abs(got-5000) > 1e-6 {
		t.Fatal(got)
	}
}

func TestStartEqualsMax(t *testing.T) {
	s := Schedule{Warmup: time.Second, Start: 1000, Slope: 10, Max: 1000}
	end, reason, ok := s.End()
	if !ok || reason != ReasonMaxRate || end != time.Second {
		t.Fatal(end, reason, ok)
	}
	if got := s.Plan(0).Integral(0, 10); math.Abs(got-1000) > 1e-6 {
		t.Fatal(got)
	}
}

func TestZeroWarmup(t *testing.T) {
	s := Schedule{Start: 100, Slope: 100, Max: 300}
	p := s.Plan(0)
	if got := p.Integral(0, 10); math.Abs(got-(0.5*(100+300)*2)) > 1e-6 {
		t.Fatal(got)
	}
	if s.PhaseAt(0) != Ramp {
		t.Fatal("no warmup: straight to ramp")
	}
}
