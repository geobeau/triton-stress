// Package units parses the human-friendly quantities of the command line:
// 10k, 1k/s, 100ms, p99:100ms, off.
package units

import (
	"fmt"
	"math"
	"strconv"
	"strings"
	"time"
)

// ParseSI parses 1500, 2.5k, 1M, 3G. A lowercase m is rejected (milli or mega).
func ParseSI(s string) (float64, error) {
	s = strings.TrimSpace(s)
	num, mult := s, 1.0
	if s != "" {
		switch s[len(s)-1] {
		case 'k', 'K':
			num, mult = s[:len(s)-1], 1e3
		case 'M':
			num, mult = s[:len(s)-1], 1e6
		case 'G':
			num, mult = s[:len(s)-1], 1e9
		case 'm':
			return 0, fmt.Errorf("ambiguous suffix 'm' in %q: use k, M or G", s)
		}
	}
	v, err := strconv.ParseFloat(strings.TrimSpace(num), 64)
	if err != nil {
		return 0, fmt.Errorf("not a number: %q", s)
	}
	if math.IsInf(v, 0) || math.IsNaN(v) || v < 0 {
		return 0, fmt.Errorf("expected a non-negative finite number, got %q", s)
	}
	return v * mult, nil
}

// ParseRate parses a request rate in req/s: 10k or 10k/s.
func ParseRate(s string) (float64, error) {
	return ParseSI(strings.TrimSuffix(strings.TrimSpace(s), "/s"))
}

// ParseSlope parses a rate slope in req/s per second: 2k/s or 2k.
func ParseSlope(s string) (float64, error) { return ParseRate(s) }

// ParseCount parses a non-negative integer, with the SI suffixes.
func ParseCount(s string) (uint64, error) {
	v, err := ParseSI(s)
	if err != nil {
		return 0, err
	}
	if v != math.Trunc(v) {
		return 0, fmt.Errorf("expected an integer, got %q", s)
	}
	return uint64(v), nil
}

// ParseDuration parses 250ns, 50us, 100ms, 30s, 5m, 2h. A bare number is seconds.
func ParseDuration(s string) (time.Duration, error) {
	s = strings.TrimSpace(s)
	split := strings.IndexFunc(s, func(r rune) bool { return (r < '0' || r > '9') && r != '.' })
	if split < 0 {
		split = len(s)
	}
	num, unit := s[:split], strings.TrimSpace(s[split:])
	v, err := strconv.ParseFloat(num, 64)
	if err != nil {
		return 0, fmt.Errorf("bad duration %q", s)
	}
	var secs float64
	switch unit {
	case "ns":
		secs = v * 1e-9
	case "us", "µs":
		secs = v * 1e-6
	case "ms":
		secs = v * 1e-3
	case "", "s":
		secs = v
	case "m", "min":
		secs = v * 60
	case "h":
		secs = v * 3600
	default:
		return 0, fmt.Errorf("unknown duration unit %q in %q", unit, s)
	}
	if math.IsInf(secs, 0) || math.IsNaN(secs) || secs < 0 || secs > float64(math.MaxInt64)/1e9 {
		return 0, fmt.Errorf("bad duration %q", s)
	}
	return time.Duration(math.Round(secs * 1e9)), nil
}

func isOff(s string) bool { return strings.EqualFold(strings.TrimSpace(s), "off") }

// PctThreshold is a percentile and a duration: p99:100ms.
type PctThreshold struct {
	Pct float64
	Max time.Duration
}

// ParsePctThreshold parses p99:100ms. "off" gives nil.
func ParsePctThreshold(s string) (*PctThreshold, error) {
	if isOff(s) {
		return nil, nil
	}
	p, d, ok := strings.Cut(strings.TrimSpace(s), ":")
	if !ok {
		return nil, fmt.Errorf("expected pNN:<duration>, got %q", s)
	}
	p, ok = strings.CutPrefix(p, "p")
	if !ok {
		return nil, fmt.Errorf("percentile must look like p99, got %q", p)
	}
	pct, err := strconv.ParseFloat(p, 64)
	if err != nil {
		return nil, fmt.Errorf("bad percentile %q", p)
	}
	if !(pct >= 0 && pct <= 100) {
		return nil, fmt.Errorf("percentile out of range: %v", pct)
	}
	dur, err := ParseDuration(d)
	if err != nil {
		return nil, err
	}
	return &PctThreshold{Pct: pct, Max: dur}, nil
}

// ParseFractionOrOff parses a fraction in [0, 1], or "off" (nil).
func ParseFractionOrOff(s string) (*float64, error) {
	if isOff(s) {
		return nil, nil
	}
	v, err := strconv.ParseFloat(strings.TrimSpace(s), 64)
	if err != nil {
		return nil, fmt.Errorf("bad fraction %q", s)
	}
	if !(v >= 0 && v <= 1) {
		return nil, fmt.Errorf("fraction out of range [0,1]: %v", v)
	}
	return &v, nil
}

// ParseCountOrOff parses a count, or "off" (nil).
func ParseCountOrOff(s string) (*uint64, error) {
	if isOff(s) {
		return nil, nil
	}
	v, err := ParseCount(s)
	if err != nil {
		return nil, err
	}
	return &v, nil
}
