package units

import (
	"testing"
	"time"
)

func TestSI(t *testing.T) {
	ok := map[string]float64{"1500": 1500, "2.5k": 2500, "1M": 1e6, "3G": 3e9, " 7K ": 7000}
	for in, want := range ok {
		if got, err := ParseSI(in); err != nil || got != want {
			t.Errorf("ParseSI(%q) = %v, %v; want %v", in, got, err, want)
		}
	}
	for _, in := range []string{"1m", "-1", "", "k", "abc", "inf", "NaN"} {
		if _, err := ParseSI(in); err == nil {
			t.Errorf("ParseSI(%q) should fail", in)
		}
	}
	if v, err := ParseRate("10k/s"); err != nil || v != 1e4 {
		t.Error(v, err)
	}
	if v, err := ParseSlope("2k"); err != nil || v != 2e3 {
		t.Error(v, err)
	}
	if v, err := ParseCount("100k"); err != nil || v != 100_000 {
		t.Error(v, err)
	}
	if _, err := ParseCount("1.5"); err == nil {
		t.Error("1.5 is not a count")
	}
}

func TestDurations(t *testing.T) {
	ok := map[string]time.Duration{
		"100ms": 100 * time.Millisecond, "30": 30 * time.Second, "50us": 50 * time.Microsecond,
		"2m": 2 * time.Minute, "1.5s": 1500 * time.Millisecond, "250ns": 250, "1h": time.Hour,
	}
	for in, want := range ok {
		if got, err := ParseDuration(in); err != nil || got != want {
			t.Errorf("ParseDuration(%q) = %v, %v; want %v", in, got, err, want)
		}
	}
	for _, in := range []string{"3 parsecs", "", "ms", "-1s", "1e400s"} {
		if _, err := ParseDuration(in); err == nil {
			t.Errorf("ParseDuration(%q) should fail", in)
		}
	}
}

func TestPct(t *testing.T) {
	p, err := ParsePctThreshold("p99.9:1ms")
	if err != nil || p == nil || p.Pct != 99.9 || p.Max != time.Millisecond {
		t.Fatal(p, err)
	}
	if p, err := ParsePctThreshold("OFF"); p != nil || err != nil {
		t.Fatal(p, err)
	}
	for _, in := range []string{"99:1ms", "p99", "p101:1ms", "pxx:1ms", "p99:zz"} {
		if _, err := ParsePctThreshold(in); err == nil {
			t.Errorf("%q should fail", in)
		}
	}
}

func TestFractionAndCountOrOff(t *testing.T) {
	if f, err := ParseFractionOrOff("0.95"); err != nil || *f != 0.95 {
		t.Fatal(f, err)
	}
	if f, err := ParseFractionOrOff("off"); err != nil || f != nil {
		t.Fatal(f, err)
	}
	for _, in := range []string{"1.5", "-0.1", "x"} {
		if _, err := ParseFractionOrOff(in); err == nil {
			t.Errorf("%q should fail", in)
		}
	}
	if c, err := ParseCountOrOff("100k"); err != nil || *c != 100000 {
		t.Fatal(c, err)
	}
	if c, err := ParseCountOrOff("off"); err != nil || c != nil {
		t.Fatal(c, err)
	}
}
