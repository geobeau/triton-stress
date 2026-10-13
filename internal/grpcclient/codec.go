package grpcclient

import (
	"fmt"

	"google.golang.org/grpc/encoding"
	"google.golang.org/grpc/mem"
)

// RawRequest is a pre-encoded ModelInferRequest (the bare protobuf message,
// without gRPC framing). The bytes must not be modified while a call uses them.
type RawRequest struct{ Body []byte }

// RawResponse receives the undecoded response message. Size is always set.
// Data is set (a copy, safe to keep) only when Keep is true on entry.
type RawResponse struct {
	Keep bool
	Size int
	Data []byte
}

// RawCodec is a grpc-go CodecV2 that passes pre-encoded bytes through. Use it
// with grpc.ForceCodecV2 on a single call. Its name is "proto", so the
// content-type on the wire is the usual application/grpc+proto.
type RawCodec struct{}

var _ encoding.CodecV2 = RawCodec{}

// Name implements encoding.CodecV2.
func (RawCodec) Name() string { return "proto" }

// Marshal implements encoding.CodecV2. It wraps the body without copying it.
func (RawCodec) Marshal(v any) (mem.BufferSlice, error) {
	r, ok := v.(*RawRequest)
	if !ok {
		return nil, fmt.Errorf("RawCodec: cannot marshal %T, want *RawRequest", v)
	}
	return mem.BufferSlice{mem.SliceBuffer(r.Body)}, nil
}

// Unmarshal implements encoding.CodecV2.
func (RawCodec) Unmarshal(data mem.BufferSlice, v any) error {
	r, ok := v.(*RawResponse)
	if !ok {
		return fmt.Errorf("RawCodec: cannot unmarshal into %T, want *RawResponse", v)
	}
	r.Size = data.Len()
	if r.Keep {
		r.Data = data.Materialize()
	}
	return nil
}
