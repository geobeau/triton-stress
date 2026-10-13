package tensorgen

import (
	"encoding/binary"
	"fmt"
)

// MismatchKind says why a response failed validation.
type MismatchKind uint8

// Mismatch kinds.
const (
	MismatchDecode MismatchKind = iota + 1
	MismatchMissingOutput
	MismatchDuplicateOutput
	MismatchDatatype
	MismatchShape
	MismatchRawCount
	MismatchRawSize
	MismatchRawBytes
)

// Mismatch is the error returned by Validator.
type Mismatch struct {
	Kind   MismatchKind
	Output string
	Detail string
}

func (m *Mismatch) Error() string {
	if m.Output == "" {
		return "invalid response: " + m.Detail
	}
	return fmt.Sprintf("invalid response: output %s: %s", m.Output, m.Detail)
}

// Validator checks inference responses against a model's outputs. Outputs the
// model's metadata does not list are ignored. It is safe for concurrent use.
type Validator struct {
	outputs   []Spec
	batchSize int
	batched   bool
}

// NewValidator expects outputs (per batch element) at the given client-side
// batch size. For a non-batching model (batched false) outputs carry no batch
// dimension. A Dims entry of -1 matches any size.
func NewValidator(outputs []Spec, batchSize int, batched bool) *Validator {
	return &Validator{outputs: outputs, batchSize: batchSize, batched: batched}
}

// ValidateBytes decodes the protobuf wire bytes of a ModelInferResponse (with
// protowire: only the outputs and raw_output_contents are read) and validates it.
func (v *Validator) ValidateBytes(b []byte) error {
	outs, raws, err := decodeResponse(b)
	if err != nil {
		return &Mismatch{Kind: MismatchDecode, Detail: err.Error()}
	}
	return v.Validate(outs, raws)
}

// outView is one response output as the checks see it, whichever protocol it came from.
type outView struct {
	name, dtype string
	shape       []int64
	raw         []byte
}

// Validate checks output names, datatypes, shapes and raw content sizes of a gRPC
// response given as its outputs (ModelInferResponse.outputs) and raw_output_contents,
// one raw content per output.
func (v *Validator) Validate(outputs []TensorMeta, raws [][]byte) error {
	if len(raws) != len(outputs) {
		return &Mismatch{Kind: MismatchRawCount, Detail: fmt.Sprintf("%d outputs but %d raw contents", len(outputs), len(raws))}
	}
	views := make([]outView, len(outputs))
	for i, t := range outputs {
		views[i] = outView{name: t.Name, dtype: t.Datatype, shape: t.Shape, raw: raws[i]}
	}
	return v.check(views)
}

// check is the protocol-independent part: duplicates, then for every expected output its
// presence, datatype, shape and content size.
func (v *Validator) check(outs []outView) error {
	for i, t := range outs {
		for _, p := range outs[:i] {
			if p.name == t.name {
				return &Mismatch{Kind: MismatchDuplicateOutput, Output: t.name, Detail: "listed twice"}
			}
		}
	}
	for _, want := range v.outputs {
		idx := -1
		for i, t := range outs {
			if t.name == want.Name {
				idx = i
				break
			}
		}
		if idx < 0 {
			return &Mismatch{Kind: MismatchMissingOutput, Output: want.Name, Detail: "missing"}
		}
		got := outs[idx]
		if got.dtype != want.DType.String() {
			return &Mismatch{Kind: MismatchDatatype, Output: want.Name,
				Detail: fmt.Sprintf("datatype %s, want %s", got.dtype, want.DType)}
		}
		shape := got.shape
		lead := 0
		if v.batched {
			lead = 1
		}
		ok := len(shape) == len(want.Dims)+lead
		if ok && v.batched && shape[0] != int64(v.batchSize) {
			ok = false
		}
		for i := 0; ok && i < len(want.Dims); i++ {
			if want.Dims[i] != -1 && shape[lead+i] != want.Dims[i] {
				ok = false
			}
		}
		if !ok {
			return &Mismatch{Kind: MismatchShape, Output: want.Name,
				Detail: fmt.Sprintf("shape %v, want %v", shape, want.Shape(int64(v.batchSize), v.batched))}
		}
		numel := int64(1)
		for _, d := range shape {
			if d < 0 || (d != 0 && numel > (1<<40)/d) {
				return &Mismatch{Kind: MismatchShape, Output: want.Name, Detail: fmt.Sprintf("shape %v is not usable", shape)}
			}
			numel *= d
		}
		raw := got.raw
		if size, fixed := want.DType.Size(); fixed {
			if int64(len(raw)) != numel*int64(size) {
				return &Mismatch{Kind: MismatchRawSize, Output: want.Name,
					Detail: fmt.Sprintf("%d raw bytes, want %d", len(raw), numel*int64(size))}
			}
		} else if !validBytesTensor(raw, numel) {
			return &Mismatch{Kind: MismatchRawBytes, Output: want.Name, Detail: "malformed BYTES content"}
		}
	}
	return nil
}

// validBytesTensor: n elements, each a 4-byte little-endian length plus that many bytes.
func validBytesTensor(raw []byte, n int64) bool {
	for i := int64(0); i < n; i++ {
		if len(raw) < 4 {
			return false
		}
		l := uint64(binary.LittleEndian.Uint32(raw))
		if uint64(len(raw)-4) < l {
			return false
		}
		raw = raw[4+l:]
	}
	return len(raw) == 0
}
