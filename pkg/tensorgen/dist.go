package tensorgen

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand/v2"
	"strconv"
	"strings"
)

// DistKind selects how tensor values are drawn.
type DistKind uint8

// Distribution kinds.
const (
	DistZeros DistKind = iota
	DistConst
	DistUniform
	DistRandomBytes
	// DistProbe is blackbox-prober's own generator (PR #64); see ProbeDist.
	DistProbe
)

// Dist is a value distribution applied to every input. Build one with ParseDist.
type Dist struct {
	Kind DistKind
	// A is the constant (DistConst) or the lower bound (DistUniform).
	A float64
	// B is the upper bound (DistUniform): inclusive for integer types, exclusive for floats.
	B float64
}

// DefaultDist is uniform:0..1000.
var DefaultDist = Dist{Kind: DistUniform, A: 0, B: 1000}

// ProbeDist reproduces the values blackbox-prober's Triton probe draws, whatever the
// datatype, so the probe can adopt this package without changing what it sends:
//
//   - integer types: a random value in [0, 255] in the low byte, the other bytes zero
//     (for INT8 that byte is reinterpreted, so -128..127);
//   - FP32 and FP64: uniform in [0, 1);
//   - BOOL: 0 or 1 (the probe draws a whole random byte, which Triton may reject);
//   - FP16 and BF16: random bits (the probe rejects BF16);
//   - BYTES: decimal strings of integers in [0, 1000000).
//
// It is only available through this variable; ParseDist (RTS's --gen) does not accept it.
var ProbeDist = Dist{Kind: DistProbe}

// ParseDist parses "zeros", "const:<v>", "uniform:<lo>..<hi>" or "random-bytes".
// uniform needs finite lo <= hi here; Check rejects lo == hi for floats, where
// the range is half-open, and accepts it for integers, where it is inclusive.
func ParseDist(s string) (Dist, error) {
	s = strings.TrimSpace(s)
	switch {
	case s == "zeros":
		return Dist{Kind: DistZeros}, nil
	case s == "random-bytes":
		return Dist{Kind: DistRandomBytes}, nil
	case strings.HasPrefix(s, "const:"):
		v, err := strconv.ParseFloat(s[len("const:"):], 64)
		if err != nil {
			return Dist{}, fmt.Errorf("bad const value in %q", s)
		}
		if math.IsInf(v, 0) || math.IsNaN(v) {
			return Dist{}, fmt.Errorf("const value must be finite in %q", s)
		}
		return Dist{Kind: DistConst, A: v}, nil
	case strings.HasPrefix(s, "uniform:"):
		lo, hi, ok := strings.Cut(s[len("uniform:"):], "..")
		if !ok {
			return Dist{}, fmt.Errorf("expected uniform:<lo>..<hi>, got %q", s)
		}
		l, err := strconv.ParseFloat(lo, 64)
		if err != nil {
			return Dist{}, fmt.Errorf("bad lower bound in %q", s)
		}
		h, err := strconv.ParseFloat(hi, 64)
		if err != nil {
			return Dist{}, fmt.Errorf("bad upper bound in %q", s)
		}
		if math.IsInf(l, 0) || math.IsNaN(l) || math.IsInf(h, 0) || math.IsNaN(h) || l > h {
			return Dist{}, fmt.Errorf("uniform needs finite lo <= hi, got %q", s)
		}
		return Dist{Kind: DistUniform, A: l, B: h}, nil
	}
	return Dist{}, fmt.Errorf("unknown --gen %q: expected zeros | const:<v> | uniform:<lo>..<hi> | random-bytes", s)
}

// String is the ParseDist form.
func (d Dist) String() string {
	f := func(v float64) string { return strconv.FormatFloat(v, 'g', -1, 64) }
	switch d.Kind {
	case DistConst:
		return "const:" + f(d.A)
	case DistUniform:
		return "uniform:" + f(d.A) + ".." + f(d.B)
	case DistRandomBytes:
		return "random-bytes"
	case DistProbe:
		return "probe"
	}
	return "zeros"
}

// Check reports whether d can fill a tensor of dtype: FP16/BF16 take only
// zeros and random-bytes; values must lie inside the datatype's range;
// integer types (and BOOL, and the integers behind BYTES) need integer
// bounds; floats need lo < hi.
func (d Dist) Check(dtype DType, input string) error {
	if d.Kind == DistZeros || d.Kind == DistRandomBytes || d.Kind == DistProbe {
		if dtype < Bool || dtype > Bytes {
			return fmt.Errorf("input %s: invalid datatype", input)
		}
		return nil
	}
	if dtype == Fp16 || dtype == Bf16 {
		return fmt.Errorf("input %s: %s supports only --gen zeros or random-bytes", input, dtype)
	}
	lo, hi := d.A, d.A
	if d.Kind == DistUniform {
		lo, hi = d.A, d.B
	}
	isFloat := dtype.IsFloat()
	if d.Kind == DistUniform && isFloat && !(lo < hi) {
		return fmt.Errorf("input %s: uniform over %s needs lo < hi, got %v..%v", input, dtype, lo, hi)
	}
	if !isFloat && (lo != math.Trunc(lo) || hi != math.Trunc(hi)) {
		return fmt.Errorf("input %s: --gen values must be integers for %s", input, dtype)
	}
	min, maxIncl, maxExcl := dtypeRange(dtype)
	if lo < min || (maxExcl != 0 && hi >= maxExcl) || (maxExcl == 0 && hi > maxIncl) {
		return fmt.Errorf("input %s: --gen values [%v, %v] do not fit %s", input, lo, hi, dtype)
	}
	return nil
}

// dtypeRange: min, and the maximum either inclusive (exact in float64) or, for
// 64-bit integers whose maximum is not representable, as an exclusive bound.
func dtypeRange(d DType) (min, maxIncl, maxExcl float64) {
	switch d {
	case Bool:
		return 0, 1, 0
	case Uint8:
		return 0, math.MaxUint8, 0
	case Uint16:
		return 0, math.MaxUint16, 0
	case Uint32:
		return 0, math.MaxUint32, 0
	case Uint64:
		return 0, 0, 1 << 64
	case Int8:
		return math.MinInt8, math.MaxInt8, 0
	case Int16:
		return math.MinInt16, math.MaxInt16, 0
	case Int32:
		return math.MinInt32, math.MaxInt32, 0
	case Int64, Bytes: // BYTES: the integers behind the strings are int64
		return math.MinInt64, 0, 1 << 63
	case Fp32:
		return -math.MaxFloat32, math.MaxFloat32, 0
	}
	return -math.MaxFloat64, math.MaxFloat64, 0
}

// fill appends nothing and returns n little-endian elements of dtype (not BYTES).
// The distribution must have passed Check.
func fill(d Dist, dtype DType, n int, rng *rand.Rand) []byte {
	size, _ := dtype.Size()
	out := make([]byte, n*size)
	switch d.Kind {
	case DistProbe:
		fillProbe(out, dtype, n, rng)
	case DistZeros:
	case DistRandomBytes:
		for i := 0; i < len(out); i += 8 {
			var b [8]byte
			binary.LittleEndian.PutUint64(b[:], rng.Uint64())
			copy(out[i:], b[:])
		}
		if dtype == Bool {
			for i := range out {
				out[i] &= 1
			}
		}
	default:
		next := sampler(d, dtype, rng)
		for i := 0; i < n; i++ {
			next(out[i*size : (i+1)*size])
		}
	}
	return out
}

// fillProbe fills out (n elements of dtype, not BYTES) as ProbeDist describes.
func fillProbe(out []byte, dtype DType, n int, rng *rand.Rand) {
	size, _ := dtype.Size()
	switch dtype {
	case Fp32:
		for i := 0; i < n; i++ {
			binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(rng.Float32()))
		}
	case Fp64:
		for i := 0; i < n; i++ {
			binary.LittleEndian.PutUint64(out[i*8:], math.Float64bits(rng.Float64()))
		}
	case Bool:
		for i := range out {
			out[i] = byte(rng.Uint32() & 1)
		}
	case Fp16, Bf16:
		for i := 0; i < len(out); i += 8 {
			var b [8]byte
			binary.LittleEndian.PutUint64(b[:], rng.Uint64())
			copy(out[i:], b[:])
		}
	default: // integers
		for i := 0; i < n; i++ {
			out[i*size] = byte(rng.Uint32())
		}
	}
}

// sampler returns a function writing one value of dtype into dst.
func sampler(d Dist, dtype DType, rng *rand.Rand) func(dst []byte) {
	if dtype.IsFloat() {
		lo, hi := d.A, d.A
		uniform := d.Kind == DistUniform
		if uniform {
			hi = d.B
		}
		draw := func() float64 {
			if !uniform {
				return lo
			}
			return lo + rng.Float64()*(hi-lo)
		}
		if dtype == Fp32 {
			return func(dst []byte) {
				v := float32(draw())
				if uniform && v >= float32(hi) { // float32 rounding can reach the open bound
					v = math.Nextafter32(float32(hi), float32(lo))
				}
				binary.LittleEndian.PutUint32(dst, math.Float32bits(v))
			}
		}
		return func(dst []byte) { binary.LittleEndian.PutUint64(dst, math.Float64bits(draw())) }
	}
	next := intDraw(d, dtype, rng)
	size, _ := dtype.Size()
	return func(dst []byte) {
		v := next()
		switch size {
		case 1:
			dst[0] = byte(v)
		case 2:
			binary.LittleEndian.PutUint16(dst, uint16(v))
		case 4:
			binary.LittleEndian.PutUint32(dst, uint32(v))
		default:
			binary.LittleEndian.PutUint64(dst, v)
		}
	}
}

// intDraw returns the two's-complement bits (in a uint64) of one integer value.
func intDraw(d Dist, dtype DType, rng *rand.Rand) func() uint64 {
	signed := dtype.isSigned() || dtype == Bytes
	lo, hi := d.A, d.A
	if d.Kind == DistUniform {
		hi = d.B
	}
	var loBits, span uint64 // value = loBits + draw in [0, span], wrapping
	if signed {
		loBits = uint64(int64(lo))
		span = uint64(int64(hi)) - loBits
	} else {
		loBits = uint64(lo)
		span = uint64(hi) - loBits
	}
	if d.Kind != DistUniform || span == 0 {
		return func() uint64 { return loBits }
	}
	if span == math.MaxUint64 {
		return func() uint64 { return rng.Uint64() }
	}
	return func() uint64 { return loBits + rng.Uint64N(span+1) }
}

// fillBytes returns the BYTES tensor wire form of n elements: each the decimal
// string of an integer from d, as a 4-byte little-endian length plus the text.
func fillBytes(d Dist, n int, rng *rand.Rand) []byte {
	out := make([]byte, 0, n*8)
	var draw func() string
	switch d.Kind {
	case DistZeros:
		draw = func() string { return "0" }
	case DistRandomBytes:
		draw = func() string { return strconv.FormatUint(uint64(rng.Uint32()), 10) }
	case DistProbe:
		draw = func() string { return strconv.FormatUint(rng.Uint64N(1000000), 10) }
	default:
		next := intDraw(d, Bytes, rng)
		draw = func() string { return strconv.FormatInt(int64(next()), 10) }
	}
	for i := 0; i < n; i++ {
		s := draw()
		out = binary.LittleEndian.AppendUint32(out, uint32(len(s)))
		out = append(out, s...)
	}
	return out
}
