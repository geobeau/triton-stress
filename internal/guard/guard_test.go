package guard

import (
	"testing"
	"time"

	"github.com/HdrHistogram/hdrhistogram-go"
	"github.com/criteo/triton-stress/internal/units"
)

func hist(vals ...int64) *hdrhistogram.Histogram {
	h := hdrhistogram.New(1, 3_600_000_000_000, 3)
	for _, v := range vals {
		if err := h.RecordValue(v); err != nil {
			panic(err)
		}
	}
	return h
}

func f(v float64) *float64 { return &v }
func u(v uint64) *uint64   { return &v }

func ok() Tick {
	return Tick{OfferedRPS: 1000, OKRPS: 1000, Completed: 1000, Latency: hist(1e6), SendDelay: hist(1e3)}
}

func defaults() Config {
	return Config{
		Achieved: f(0.95), ErrorRate: f(0.01),
		Latency:     &units.PctThreshold{Pct: 99, Max: 100 * time.Millisecond},
		SendDelay:   &units.PctThreshold{Pct: 99, Max: time.Millisecond},
		MaxInflight: u(100000), Consecutive: 3,
	}
}

func TestHealthyTickBreachesNothing(t *testing.T) {
	g := New(defaults())
	if b, trip := g.EvaluateTick(ok(), false); len(b) != 0 || trip != nil {
		t.Fatal(b, trip)
	}
}

func TestEachGuardBreach(t *testing.T) {
	cases := map[string]func(*Tick){
		Achieved:  func(x *Tick) { x.OKRPS = 900 },
		ErrorRate: func(x *Tick) { x.Errors = 20 },
		Latency:   func(x *Tick) { x.Latency = hist(200e6) },
		SendDelay: func(x *Tick) { x.SendDelay = hist(2e6) },
	}
	for name, mutate := range cases {
		g := New(defaults())
		x := ok()
		mutate(&x)
		b, trip := g.EvaluateTick(x, false)
		if len(b) != 1 || b[0].Guard != name || trip != nil {
			t.Fatalf("%s: %+v trip %v", name, b, trip)
		}
	}
	// observed and threshold values, in the documented units
	g := New(defaults())
	x := ok()
	x.Latency = hist(200e6)
	b, _ := g.EvaluateTick(x, false)
	if b[0].Threshold != 0.1 || b[0].Observed < 0.199 || b[0].Observed > 0.201 {
		t.Fatalf("%+v", b[0])
	}
	g = New(defaults())
	x = ok()
	x.OKRPS = 500
	b, _ = g.EvaluateTick(x, false)
	if b[0].Observed != 0.5 || b[0].Threshold != 0.95 {
		t.Fatalf("%+v", b[0])
	}
}

func TestDebounceNeedsConsecutiveTicks(t *testing.T) {
	g := New(defaults())
	bad := ok()
	bad.Latency = hist(500e6)
	for i := 1; i <= 2; i++ {
		if _, trip := g.EvaluateTick(bad, false); trip != nil {
			t.Fatalf("tripped after %d ticks", i)
		}
	}
	// a clean tick resets the streak
	if _, trip := g.EvaluateTick(ok(), false); trip != nil {
		t.Fatal("clean tick tripped")
	}
	for i := 1; i <= 2; i++ {
		if _, trip := g.EvaluateTick(bad, false); trip != nil {
			t.Fatalf("tripped after %d ticks following a reset", i)
		}
	}
	_, trip := g.EvaluateTick(bad, false)
	if trip == nil || trip.Guard != Latency {
		t.Fatalf("trip %+v", trip)
	}
}

func TestWarmupIsReportOnly(t *testing.T) {
	g := New(defaults())
	bad := ok()
	bad.OKRPS = 10
	for i := 0; i < 10; i++ {
		b, trip := g.EvaluateTick(bad, true)
		if len(b) != 1 || trip != nil {
			t.Fatalf("warmup tick %d: %v %v", i, b, trip)
		}
	}
	// The warmup breaches did not count: after warmup it takes Consecutive new ticks.
	for i := 1; i <= 2; i++ {
		if _, trip := g.EvaluateTick(bad, false); trip != nil {
			t.Fatalf("tripped on post-warmup tick %d", i)
		}
	}
	if _, trip := g.EvaluateTick(bad, false); trip == nil || trip.Guard != Achieved {
		t.Fatal("no trip after 3 post-warmup ticks")
	}
}

func TestOffGuardsAndEmptyTicks(t *testing.T) {
	g := New(Config{Consecutive: 1})
	bad := Tick{OfferedRPS: 1000, OKRPS: 0, Completed: 100, Errors: 100, Latency: hist(9e9), SendDelay: hist(9e9)}
	if b, trip := g.EvaluateTick(bad, false); len(b) != 0 || trip != nil {
		t.Fatal("guards that are off must not breach")
	}
	if g.CheckInflight(1<<40) != nil {
		t.Fatal("max inflight off")
	}
	g = New(defaults())
	// nothing completed, nothing offered: no division by zero, no breach
	if b, _ := g.EvaluateTick(Tick{}, false); len(b) != 0 {
		t.Fatal(b)
	}
	if b, _ := g.EvaluateTick(Tick{Latency: hist(), SendDelay: hist()}, false); len(b) != 0 {
		t.Fatal(b)
	}
}

func TestConsecutiveOne(t *testing.T) {
	c := defaults()
	c.Consecutive = 1
	_, trip := New(c).EvaluateTick(Tick{OfferedRPS: 1000, OKRPS: 0}, false)
	if trip == nil || trip.Guard != Achieved {
		t.Fatal(trip)
	}
	c.Consecutive = 0 // treated as 1
	if _, trip := New(c).EvaluateTick(Tick{OfferedRPS: 1000, OKRPS: 0}, false); trip == nil {
		t.Fatal("0 means 1")
	}
}

func TestMaxInflight(t *testing.T) {
	g := New(defaults())
	if g.CheckInflight(100000) != nil {
		t.Fatal("at the limit is fine")
	}
	b := g.CheckInflight(100001)
	if b == nil || b.Guard != MaxInflight || b.Observed != 100001 || b.Threshold != 100000 {
		t.Fatalf("%+v", b)
	}
}
