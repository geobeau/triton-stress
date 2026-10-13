package tensorgen

import (
	"fmt"
	"strconv"
	"strings"
)

// DType is a Triton / KServe v2 tensor datatype.
type DType uint8

// Datatypes. The zero value is invalid.
const (
	Bool DType = iota + 1
	Uint8
	Uint16
	Uint32
	Uint64
	Int8
	Int16
	Int32
	Int64
	Fp16
	Fp32
	Fp64
	Bf16
	Bytes
)

var dtypeNames = [...]string{
	Bool: "BOOL", Uint8: "UINT8", Uint16: "UINT16", Uint32: "UINT32", Uint64: "UINT64",
	Int8: "INT8", Int16: "INT16", Int32: "INT32", Int64: "INT64",
	Fp16: "FP16", Fp32: "FP32", Fp64: "FP64", Bf16: "BF16", Bytes: "BYTES",
}

// ParseDType parses a Triton datatype name such as "INT32" (upper case only).
func ParseDType(s string) (DType, error) {
	for d := Bool; d <= Bytes; d++ {
		if dtypeNames[d] == s {
			return d, nil
		}
	}
	return 0, fmt.Errorf("unknown datatype %q", s)
}

// String is the Triton datatype name.
func (d DType) String() string {
	if d < Bool || d > Bytes {
		return fmt.Sprintf("DType(%d)", uint8(d))
	}
	return dtypeNames[d]
}

// Size is the element size in bytes; ok is false for BYTES (variable length).
func (d DType) Size() (size int, ok bool) {
	switch d {
	case Bool, Uint8, Int8:
		return 1, true
	case Uint16, Int16, Fp16, Bf16:
		return 2, true
	case Uint32, Int32, Fp32:
		return 4, true
	case Uint64, Int64, Fp64:
		return 8, true
	}
	return 0, false
}

// IsFloat reports FP16, FP32, FP64 and BF16.
func (d DType) IsFloat() bool { return d == Fp16 || d == Fp32 || d == Fp64 || d == Bf16 }

func (d DType) isSigned() bool { return d == Int8 || d == Int16 || d == Int32 || d == Int64 }

// Spec is a tensor as the model sees it per batch element: the batch dimension
// is not part of Dims. A dim of -1 is variable (for an output it matches
// anything; for an input it must be fixed by an override).
type Spec struct {
	Name  string
	DType DType
	Dims  []int64
}

// NumElements is the element count per batch element; ok is false when a dim is variable.
func (s Spec) NumElements() (n int64, ok bool) {
	n = 1
	for _, d := range s.Dims {
		if d < 0 {
			return 0, false
		}
		if d != 0 && n > (1<<62)/d {
			return 0, false
		}
		n *= d
	}
	return n, true
}

// Shape is [batchSize, Dims...] for a batching model, or Dims as they are.
func (s Spec) Shape(batchSize int64, batched bool) []int64 {
	if !batched {
		return append([]int64(nil), s.Dims...)
	}
	return append([]int64{batchSize}, s.Dims...)
}

// String renders "name:DTYPE:d1,d2".
func (s Spec) String() string {
	dims := make([]string, len(s.Dims))
	for i, d := range s.Dims {
		dims[i] = strconv.FormatInt(d, 10)
	}
	return s.Name + ":" + s.DType.String() + ":" + strings.Join(dims, ",")
}

// ParseSpec parses "name:DTYPE:d1,d2" (dims comma-separated, each >= 1, batch
// dimension excluded), the format of RTS's --input flag.
func ParseSpec(s string) (Spec, error) {
	parts := strings.SplitN(s, ":", 3)
	if len(parts) != 3 || parts[0] == "" {
		return Spec{}, fmt.Errorf("expected name:DTYPE:dims, got %q", s)
	}
	dt, err := ParseDType(parts[1])
	if err != nil {
		return Spec{}, fmt.Errorf("%q: %w", s, err)
	}
	var dims []int64
	for _, d := range strings.Split(parts[2], ",") {
		v, err := strconv.ParseInt(strings.TrimSpace(d), 10, 64)
		if err != nil {
			return Spec{}, fmt.Errorf("bad dim %q in %q", d, s)
		}
		if v < 1 {
			return Spec{}, fmt.Errorf("dims must be >= 1, got %d in %q", v, s)
		}
		dims = append(dims, v)
	}
	return Spec{Name: parts[0], DType: dt, Dims: dims}, nil
}
