package tensorgen

import (
	"fmt"
	"sort"

	"google.golang.org/protobuf/encoding/protowire"
)

// Field numbers of ModelInferRequest and its nested messages (proto/grpc_service.proto,
// package inference). Pre-encoding with protowire keeps this package free of generated
// protobuf types, which cannot be linked twice into one binary.
const (
	reqModelName    = 1
	reqModelVersion = 2
	reqParameters   = 4
	reqInputs       = 5
	reqOutputs      = 6
	reqRawInputs    = 7
	tensorName      = 1 // InferInputTensor and InferRequestedOutputTensor
	inputDatatype   = 2
	inputShape      = 3
	mapKey          = 1
	mapValue        = 2
	paramBool       = 1 // InferParameter oneof
	paramInt64      = 2
	paramString     = 3
	respOutputs     = 5 // ModelInferResponse
	respRawOutputs  = 6
	outName         = 1 // ModelInferResponse.InferOutputTensor
	outDatatype     = 2
	outShape        = 3
)

// MarshalWire returns the protobuf wire encoding of r as a ModelInferRequest. The
// bytes are those of proto.MarshalOptions{Deterministic: true} on the generated type:
// fields in number order, parameters sorted by name, shapes packed, zero-valued
// strings omitted.
func (r *Request) MarshalWire() ([]byte, error) {
	return r.AppendWire(make([]byte, 0, r.wireSizeHint()))
}

func (r *Request) wireSizeHint() int {
	n := 64 + len(r.Model) + len(r.Version)
	for _, in := range r.Inputs {
		n += 32 + len(in.Name) + len(in.Datatype) + 10*len(in.Shape) + len(in.Raw)
	}
	return n
}

// AppendWire appends the encoding MarshalWire returns to dst.
func (r *Request) AppendWire(dst []byte) ([]byte, error) {
	dst = appendString(dst, reqModelName, r.Model)
	dst = appendString(dst, reqModelVersion, r.Version)
	if len(r.Parameters) > 0 {
		keys := make([]string, 0, len(r.Parameters))
		for k := range r.Parameters {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			var val []byte
			switch v := r.Parameters[k].(type) {
			case int64:
				val = protowire.AppendVarint(protowire.AppendTag(nil, paramInt64, protowire.VarintType), uint64(v))
			case bool:
				b := uint64(0)
				if v {
					b = 1
				}
				val = protowire.AppendVarint(protowire.AppendTag(nil, paramBool, protowire.VarintType), b)
			case string:
				val = protowire.AppendString(protowire.AppendTag(nil, paramString, protowire.BytesType), v)
			default:
				return nil, fmt.Errorf("parameter %s: unsupported type %T (want int64, bool or string)", k, v)
			}
			// A map entry always carries its key and value, even when they are empty.
			entry := protowire.AppendString(protowire.AppendTag(nil, mapKey, protowire.BytesType), k)
			entry = protowire.AppendBytes(protowire.AppendTag(entry, mapValue, protowire.BytesType), val)
			dst = protowire.AppendBytes(protowire.AppendTag(dst, reqParameters, protowire.BytesType), entry)
		}
	}
	for _, in := range r.Inputs {
		var shape []byte
		for _, d := range in.Shape {
			shape = protowire.AppendVarint(shape, uint64(d))
		}
		n := stringSize(tensorName, in.Name) + stringSize(inputDatatype, in.Datatype)
		if len(shape) > 0 {
			n += protowire.SizeTag(inputShape) + protowire.SizeBytes(len(shape))
		}
		dst = protowire.AppendTag(dst, reqInputs, protowire.BytesType)
		dst = protowire.AppendVarint(dst, uint64(n))
		dst = appendString(dst, tensorName, in.Name)
		dst = appendString(dst, inputDatatype, in.Datatype)
		if len(shape) > 0 {
			dst = protowire.AppendBytes(protowire.AppendTag(dst, inputShape, protowire.BytesType), shape)
		}
	}
	for _, name := range r.Outputs {
		dst = protowire.AppendTag(dst, reqOutputs, protowire.BytesType)
		dst = protowire.AppendVarint(dst, uint64(stringSize(tensorName, name)))
		dst = appendString(dst, tensorName, name)
	}
	for _, in := range r.Inputs {
		dst = protowire.AppendBytes(protowire.AppendTag(dst, reqRawInputs, protowire.BytesType), in.Raw)
	}
	return dst, nil
}

// appendString appends a proto3 string field, which is omitted when empty.
func appendString(dst []byte, num protowire.Number, s string) []byte {
	if s == "" {
		return dst
	}
	return protowire.AppendString(protowire.AppendTag(dst, num, protowire.BytesType), s)
}

func stringSize(num protowire.Number, s string) int {
	if s == "" {
		return 0
	}
	return protowire.SizeTag(num) + protowire.SizeBytes(len(s))
}

// decodeResponse reads the outputs and raw_output_contents of a ModelInferResponse.
// The raw contents alias b.
func decodeResponse(b []byte) (outs []TensorMeta, raws [][]byte, err error) {
	for len(b) > 0 {
		num, typ, n := protowire.ConsumeTag(b)
		if n < 0 {
			return nil, nil, protowire.ParseError(n)
		}
		b = b[n:]
		switch {
		case num == respOutputs && typ == protowire.BytesType:
			msg, n := protowire.ConsumeBytes(b)
			if n < 0 {
				return nil, nil, protowire.ParseError(n)
			}
			b = b[n:]
			o, err := decodeOutput(msg)
			if err != nil {
				return nil, nil, err
			}
			outs = append(outs, o)
		case num == respRawOutputs && typ == protowire.BytesType:
			raw, n := protowire.ConsumeBytes(b)
			if n < 0 {
				return nil, nil, protowire.ParseError(n)
			}
			b = b[n:]
			raws = append(raws, raw)
		default:
			n := protowire.ConsumeFieldValue(num, typ, b)
			if n < 0 {
				return nil, nil, protowire.ParseError(n)
			}
			b = b[n:]
		}
	}
	return outs, raws, nil
}

func decodeOutput(b []byte) (TensorMeta, error) {
	var o TensorMeta
	for len(b) > 0 {
		num, typ, n := protowire.ConsumeTag(b)
		if n < 0 {
			return o, protowire.ParseError(n)
		}
		b = b[n:]
		switch {
		case (num == outName || num == outDatatype) && typ == protowire.BytesType:
			s, n := protowire.ConsumeString(b)
			if n < 0 {
				return o, protowire.ParseError(n)
			}
			b = b[n:]
			if num == outName {
				o.Name = s
			} else {
				o.Datatype = s
			}
		case num == outShape && typ == protowire.BytesType: // packed
			p, n := protowire.ConsumeBytes(b)
			if n < 0 {
				return o, protowire.ParseError(n)
			}
			b = b[n:]
			for len(p) > 0 {
				v, n := protowire.ConsumeVarint(p)
				if n < 0 {
					return o, protowire.ParseError(n)
				}
				p = p[n:]
				o.Shape = append(o.Shape, int64(v))
			}
		case num == outShape && typ == protowire.VarintType: // unpacked
			v, n := protowire.ConsumeVarint(b)
			if n < 0 {
				return o, protowire.ParseError(n)
			}
			b = b[n:]
			o.Shape = append(o.Shape, int64(v))
		default:
			n := protowire.ConsumeFieldValue(num, typ, b)
			if n < 0 {
				return o, protowire.ParseError(n)
			}
			b = b[n:]
		}
	}
	return o, nil
}
