package stats

import (
	"math"
	"sort"
)

// The Prometheus view of a latency histogram: classic buckets (1-2-5 from 10 us to
// 20 s) and native buckets at schema 3 (growth factor 2^(1/8), about 9%). Both are
// counted per worker on the hot path and summed at scrape time.

// NativeSchema is the native-histogram resolution: bucket boundaries at 2^(k/8).
const NativeSchema = 3

// Native bucket keys are clamped to [nativeMinKey, nativeMaxKey]; keys below the range
// (at or under 2^-32 s) count as the zero bucket, keys above it land in the last bucket
// (2^16 s is far beyond any timeout).
const (
	nativeMinKey = -255
	nativeMaxKey = 127
	nativeKeys   = nativeMaxKey - nativeMinKey + 1
)

// ClassicBounds are the classic bucket upper bounds in seconds.
var ClassicBounds = func() []float64 {
	var b []float64
	for _, us := range []int64{10, 20, 50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000,
		200_000, 500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000, 20_000_000} {
		b = append(b, float64(us)/1e6)
	}
	return b
}()

// fracBounds[i] = 2^((i-8)/8), i = 0..8: the upper bounds of the 8 native buckets
// inside one octave, for a mantissa in [0.5, 1).
var fracBounds = func() [9]float64 {
	var b [9]float64
	for i := range b {
		b[i] = math.Pow(2, float64(i-8)/8)
	}
	return b
}()

// nativeKey is the schema-3 bucket key k of v > 0: 2^((k-1)/8) < v <= 2^(k/8).
func nativeKey(v float64) int {
	frac, exp := math.Frexp(v) // v = frac * 2^exp, frac in [0.5, 1)
	i := 0
	for i < 8 && frac > fracBounds[i] {
		i++
	}
	return exp*8 + i - 8
}

// PromHist accumulates one histogram for Prometheus. Not safe for concurrent use:
// the owner serialises access.
type PromHist struct {
	Count   uint64
	Sum     float64 // seconds
	Classic [20]uint64
	// Above is the count over the last classic bound (the +Inf bucket minus the others).
	Above  uint64
	Zero   uint64
	Native [nativeKeys]uint64
}

// Observe adds one value in seconds.
func (h *PromHist) Observe(sec float64) {
	h.Count++
	h.Sum += sec
	if i := sort.SearchFloat64s(ClassicBounds, sec); i < len(ClassicBounds) {
		h.Classic[i]++
	} else {
		h.Above++
	}
	if sec <= 0 {
		h.Zero++
		return
	}
	k := nativeKey(sec)
	switch {
	case k < nativeMinKey:
		h.Zero++
	case k > nativeMaxKey:
		h.Native[nativeKeys-1]++
	default:
		h.Native[k-nativeMinKey]++
	}
}

// Add merges o into h.
func (h *PromHist) Add(o *PromHist) {
	h.Count += o.Count
	h.Sum += o.Sum
	h.Above += o.Above
	h.Zero += o.Zero
	for i := range h.Classic {
		h.Classic[i] += o.Classic[i]
	}
	for i := range h.Native {
		h.Native[i] += o.Native[i]
	}
}

// ClassicCumulative is the cumulative count at each of ClassicBounds.
func (h *PromHist) ClassicCumulative() []uint64 {
	out := make([]uint64, len(h.Classic))
	var c uint64
	for i, n := range h.Classic {
		c += n
		out[i] = c
	}
	return out
}

// Span is a run of consecutive native buckets: Offset from the end of the previous
// span (from key 0 for the first), Length buckets.
type Span struct {
	Offset int32
	Length uint32
}

// NativePositive returns the positive native buckets as spans and delta-encoded counts.
func (h *PromHist) NativePositive() (spans []Span, deltas []int64) {
	var prevKey = 0 // key just past the last bucket of the previous span
	var prevCount int64
	inSpan := false
	for i, n := range h.Native {
		if n == 0 {
			inSpan = false
			continue
		}
		key := i + nativeMinKey
		if !inSpan {
			spans = append(spans, Span{Offset: int32(key - prevKey)})
			inSpan = true
		}
		spans[len(spans)-1].Length++
		deltas = append(deltas, int64(n)-prevCount)
		prevCount = int64(n)
		prevKey = key + 1
	}
	return spans, deltas
}
