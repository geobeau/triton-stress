// Package tritonpbadapter connects pkg/tensorgen to the generated protobuf types in
// pkg/tritonpb, for programs (like triton-stress) that use them.
//
// pkg/tensorgen itself works on plain Go values and never imports generated code: a
// binary that links two generated copies of Triton's protobuf package (one in
// pkg/tritonpb, one in another module such as a prober's own client) panics at init
// with a registration conflict. This package is the only bridge to pkg/tritonpb.
package tritonpbadapter

import (
	"fmt"

	"github.com/criteo/triton-stress/pkg/tensorgen"
	pb "github.com/criteo/triton-stress/pkg/tritonpb"
)

// Metadata converts a ModelMetadataResponse.
func Metadata(m *pb.ModelMetadataResponse) tensorgen.Metadata {
	conv := func(ts []*pb.ModelMetadataResponse_TensorMetadata) []tensorgen.TensorMeta {
		out := make([]tensorgen.TensorMeta, len(ts))
		for i, t := range ts {
			out[i] = tensorgen.TensorMeta{Name: t.GetName(), Datatype: t.GetDatatype(), Shape: t.GetShape()}
		}
		return out
	}
	return tensorgen.Metadata{Inputs: conv(m.GetInputs()), Outputs: conv(m.GetOutputs())}
}

// Config converts a ModelConfig; nil stays nil.
func Config(c *pb.ModelConfig) *tensorgen.Config {
	if c == nil {
		return nil
	}
	return &tensorgen.Config{MaxBatchSize: c.GetMaxBatchSize(), Sequence: c.GetSequenceBatching() != nil}
}

// NewModel is tensorgen.NewModel for the generated metadata and config (cfg may be nil).
func NewModel(meta *pb.ModelMetadataResponse, cfg *pb.ModelConfig, name, version string) (*tensorgen.Model, error) {
	if meta == nil {
		return nil, fmt.Errorf("nil model metadata")
	}
	return tensorgen.NewModel(Metadata(meta), Config(cfg), name, version)
}

// Probeable is tensorgen.Probeable for the generated metadata.
func Probeable(meta *pb.ModelMetadataResponse) (ok bool, reason string) {
	return tensorgen.Probeable(Metadata(meta))
}

// BuildInferRequest is Generator.BuildInferRequest returning a generated
// *pb.ModelInferRequest, the shape of blackbox-prober's method of the same name.
func BuildInferRequest(g *tensorgen.Generator, meta *pb.ModelMetadataResponse, cfg *pb.ModelConfig, modelName, modelVersion string, batchSize int64) (*pb.ModelInferRequest, error) {
	if meta == nil {
		return nil, fmt.Errorf("nil model metadata")
	}
	r, err := g.BuildInferRequest(Metadata(meta), Config(cfg), modelName, modelVersion, batchSize)
	if err != nil {
		return nil, err
	}
	return Request(r)
}

// Request converts a generated request to the protobuf type (raw contents in
// raw_input_contents, one per input).
func Request(r *tensorgen.Request) (*pb.ModelInferRequest, error) {
	req := &pb.ModelInferRequest{
		ModelName:        r.Model,
		ModelVersion:     r.Version,
		Inputs:           make([]*pb.ModelInferRequest_InferInputTensor, len(r.Inputs)),
		RawInputContents: make([][]byte, len(r.Inputs)),
	}
	for i, in := range r.Inputs {
		req.Inputs[i] = &pb.ModelInferRequest_InferInputTensor{Name: in.Name, Datatype: in.Datatype, Shape: in.Shape}
		req.RawInputContents[i] = in.Raw
	}
	for _, name := range r.Outputs {
		req.Outputs = append(req.Outputs, &pb.ModelInferRequest_InferRequestedOutputTensor{Name: name})
	}
	if len(r.Parameters) > 0 {
		req.Parameters = make(map[string]*pb.InferParameter, len(r.Parameters))
		for k, v := range r.Parameters {
			switch c := v.(type) {
			case int64:
				req.Parameters[k] = &pb.InferParameter{ParameterChoice: &pb.InferParameter_Int64Param{Int64Param: c}}
			case bool:
				req.Parameters[k] = &pb.InferParameter{ParameterChoice: &pb.InferParameter_BoolParam{BoolParam: c}}
			case string:
				req.Parameters[k] = &pb.InferParameter{ParameterChoice: &pb.InferParameter_StringParam{StringParam: c}}
			default:
				return nil, fmt.Errorf("parameter %s: unsupported type %T", k, v)
			}
		}
	}
	return req, nil
}

// Validator is a tensorgen.Validator that also checks generated responses.
type Validator struct{ *tensorgen.Validator }

// NewValidator is tensorgen.NewValidator wrapped.
func NewValidator(outputs []tensorgen.Spec, batchSize int, batched bool) *Validator {
	return &Validator{tensorgen.NewValidator(outputs, batchSize, batched)}
}

// Validate checks output names, datatypes, shapes and raw content sizes of a response.
func (v *Validator) Validate(resp *pb.ModelInferResponse) error {
	outs := make([]tensorgen.TensorMeta, len(resp.GetOutputs()))
	for i, o := range resp.GetOutputs() {
		outs[i] = tensorgen.TensorMeta{Name: o.GetName(), Datatype: o.GetDatatype(), Shape: o.GetShape()}
	}
	return v.Validator.Validate(outs, resp.GetRawOutputContents())
}
