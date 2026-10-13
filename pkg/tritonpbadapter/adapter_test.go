package tritonpbadapter

import (
	"bytes"
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/criteo/triton-stress/pkg/tensorgen"
	pb "github.com/criteo/triton-stress/pkg/tritonpb"
	"google.golang.org/protobuf/proto"
)

var det = proto.MarshalOptions{Deterministic: true}

// checkSame asserts that the protowire encoding equals proto.Marshal of the generated
// equivalent, byte for byte, and that tritonpb decodes it back to that equivalent.
func checkSame(t *testing.T, name string, r *tensorgen.Request) {
	t.Helper()
	got, err := r.MarshalWire()
	if err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	p, err := Request(r)
	if err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	want, err := det.Marshal(p)
	if err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("%s: protowire and proto.Marshal differ\n got % x\nwant % x", name, got, want)
	}
	var back pb.ModelInferRequest
	if err := proto.Unmarshal(got, &back); err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	if !proto.Equal(&back, p) {
		t.Fatalf("%s: decoded request differs:\n%v\n%v", name, &back, p)
	}
}

func TestWireMatchesProtoMarshalEdgeCases(t *testing.T) {
	in := func(name, dt string, raw []byte, shape ...int64) tensorgen.Tensor {
		return tensorgen.Tensor{Name: name, Datatype: dt, Shape: shape, Raw: raw}
	}
	cases := map[string]*tensorgen.Request{
		"empty":         {},
		"model only":    {Model: "m"},
		"version":       {Model: "m", Version: "12"},
		"one input":     {Model: "m", Inputs: []tensorgen.Tensor{in("x", "INT32", []byte{1, 0, 0, 0}, 1)}},
		"empty names":   {Inputs: []tensorgen.Tensor{in("", "", nil)}},
		"empty raw":     {Model: "m", Inputs: []tensorgen.Tensor{in("x", "FP32", []byte{}, 0, 4)}},
		"scalar shape":  {Model: "m", Inputs: []tensorgen.Tensor{in("x", "FP32", []byte{0, 0, 0, 0})}},
		"negative dims": {Model: "m", Inputs: []tensorgen.Tensor{in("x", "FP32", nil, -1, -9223372036854775808)}},
		"big dims":      {Model: "m", Inputs: []tensorgen.Tensor{in("x", "FP32", nil, 1<<62, 127, 128, 16383, 16384)}},
		"unicode":       {Model: "mødèl-模型", Version: "é", Inputs: []tensorgen.Tensor{in("入力", "BYTES", []byte("hello"), 1)}},
		"big raw":       {Model: "m", Inputs: []tensorgen.Tensor{in("x", "UINT8", make([]byte, 70000), 70000)}},
		"outputs":       {Model: "m", Outputs: []string{"a", "", "long-output-name"}},
		"params": {Model: "m", Parameters: map[string]any{
			"i0": int64(0), "ineg": int64(-1), "imax": int64(1<<63 - 1), "imin": int64(-1 << 63),
			"bt": true, "bf": false, "s": "", "s2": "text", "": "empty key", "z": int64(7),
		}},
		"many inputs": {Model: "m", Inputs: []tensorgen.Tensor{
			in("a", "INT8", []byte{1}, 1), in("b", "BOOL", []byte{0}, 1), in("c", "FP16", []byte{0, 0}, 1)}},
	}
	for name, r := range cases {
		checkSame(t, name, r)
	}
}

// Whole generated requests: every datatype and distribution, batching and not, with
// sequence parameters and requested outputs.
func TestWireMatchesProtoMarshalGenerated(t *testing.T) {
	var all []tensorgen.Spec
	for _, dt := range []tensorgen.DType{tensorgen.Bool, tensorgen.Uint8, tensorgen.Uint16, tensorgen.Uint32, tensorgen.Uint64,
		tensorgen.Int8, tensorgen.Int16, tensorgen.Int32, tensorgen.Int64, tensorgen.Fp16, tensorgen.Fp32, tensorgen.Fp64,
		tensorgen.Bf16, tensorgen.Bytes} {
		all = append(all, tensorgen.Spec{Name: "in_" + dt.String(), DType: dt, Dims: []int64{3, 2}})
	}
	dists := []string{"zeros", "random-bytes", "uniform:0..1", "const:1"}
	for _, ds := range dists {
		d, err := tensorgen.ParseDist(ds)
		if err != nil {
			t.Fatal(err)
		}
		for _, batched := range []bool{true, false} {
			for _, seq := range []bool{true, false} {
				specs := all
				if ds == "uniform:0..1" || ds == "const:1" { // FP16 and BF16 take only zeros and random-bytes
					specs = nil
					for _, s := range all {
						if s.DType != tensorgen.Fp16 && s.DType != tensorgen.Bf16 {
							specs = append(specs, s)
						}
					}
				}
				bs := 1
				if batched {
					bs = 5
				}
				tpl := &tensorgen.Template{Model: "mdl", Version: "2", Inputs: specs, BatchSize: bs, Batched: batched, Sequence: seq,
					Outputs: []string{"o1", "o2"}}
				for _, opts := range [][]tensorgen.Option{{tensorgen.WithDist(d)}, {tensorgen.WithDist(d), tensorgen.WithRequestedOutputs()}} {
					g := tensorgen.NewGenerator(9, opts...)
					for i := 0; i < 3; i++ {
						r, err := g.Build(tpl)
						if err != nil {
							t.Fatal(err)
						}
						checkSame(t, fmt.Sprintf("%s batched=%v seq=%v #%d", ds, batched, seq, i), r)
					}
				}
			}
		}
	}
	// The probe's distribution too.
	g := tensorgen.NewGenerator(1, tensorgen.WithDist(tensorgen.ProbeDist), tensorgen.WithSequenceIDStart(1<<60))
	r, err := g.Build(&tensorgen.Template{Model: "m", Inputs: all, BatchSize: 2, Batched: true, Sequence: true})
	if err != nil {
		t.Fatal(err)
	}
	checkSame(t, "probe dist", r)
	// Encode is Build + MarshalWire, whatever the seed.
	tpl := &tensorgen.Template{Model: "m", Inputs: all[:3], BatchSize: 1}
	zeros := tensorgen.WithDist(tensorgen.Dist{Kind: tensorgen.DistZeros})
	a, err := tensorgen.NewGenerator(4, zeros).Encode(tpl)
	if err != nil {
		t.Fatal(err)
	}
	b, err := tensorgen.NewGenerator(4, zeros).Build(tpl)
	if err != nil {
		t.Fatal(err)
	}
	p, _ := Request(b)
	if w, _ := det.Marshal(p); !bytes.Equal(a, w) {
		t.Fatal("Encode differs from proto.Marshal")
	}
}

// Random requests: random tensors, shapes, outputs and parameters.
func TestWireMatchesProtoMarshalRandom(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	str := func(max int) string {
		b := make([]byte, rng.IntN(max+1))
		for i := range b {
			b[i] = byte('a' + rng.IntN(26))
		}
		return string(b)
	}
	for i := 0; i < 300; i++ {
		r := &tensorgen.Request{Model: str(6), Version: str(2)}
		for j := rng.IntN(5); j > 0; j-- {
			raw := make([]byte, rng.IntN(300))
			for k := range raw {
				raw[k] = byte(rng.Uint32())
			}
			shape := make([]int64, rng.IntN(5))
			for k := range shape {
				shape[k] = int64(rng.Uint64())
			}
			r.Inputs = append(r.Inputs, tensorgen.Tensor{Name: str(8), Datatype: str(6), Shape: shape, Raw: raw})
		}
		for j := rng.IntN(3); j > 0; j-- {
			r.Outputs = append(r.Outputs, str(5))
		}
		if rng.IntN(2) == 0 {
			r.Parameters = map[string]any{}
			for j := rng.IntN(5); j > 0; j-- {
				switch rng.IntN(3) {
				case 0:
					r.Parameters[str(10)] = int64(rng.Uint64())
				case 1:
					r.Parameters[str(10)] = rng.IntN(2) == 0
				default:
					r.Parameters[str(10)] = str(20)
				}
			}
		}
		checkSame(t, fmt.Sprintf("random #%d", i), r)
	}
}

func TestRequestRejectsUnsupportedParameter(t *testing.T) {
	r := &tensorgen.Request{Parameters: map[string]any{"x": 1.5}}
	if _, err := Request(r); err == nil {
		t.Error("float accepted")
	}
}

func tm(name, dt string, shape ...int64) *pb.ModelMetadataResponse_TensorMetadata {
	return &pb.ModelMetadataResponse_TensorMetadata{Name: name, Datatype: dt, Shape: shape}
}

func meta(in ...*pb.ModelMetadataResponse_TensorMetadata) *pb.ModelMetadataResponse {
	return &pb.ModelMetadataResponse{Name: "m", Inputs: in}
}

func TestNewModelAndBuildInferRequest(t *testing.T) {
	md := meta(tm("x", "FP32", -1, 4))
	md.Outputs = []*pb.ModelMetadataResponse_TensorMetadata{tm("p", "FP32", -1, 1)}
	m, err := NewModel(md, &pb.ModelConfig{MaxBatchSize: 8}, "m", "1")
	if err != nil || !m.Batched || m.MaxBatchSize != 8 || m.Sequence || m.Inputs[0].Dims[0] != 4 || m.Outputs[0].Dims[0] != 1 {
		t.Fatalf("%+v %v", m, err)
	}
	if _, err := NewModel(nil, nil, "m", ""); err == nil {
		t.Error("nil metadata accepted")
	}
	g := tensorgen.NewGenerator(5)
	// Without a config the leading -1 marks a batching model.
	req, err := BuildInferRequest(g, md, nil, "mdl", "3", 2)
	if err != nil {
		t.Fatal(err)
	}
	if req.ModelName != "mdl" || req.ModelVersion != "3" || req.Inputs[0].Shape[0] != 2 || len(req.RawInputContents[0]) != 2*4*4 || len(req.Outputs) != 0 {
		t.Fatalf("%+v", req)
	}
	if _, err := BuildInferRequest(g, nil, nil, "m", "", 1); err == nil {
		t.Error("nil metadata accepted")
	}
	if _, err := BuildInferRequest(g, meta(tm("x", "INT32", -1, -1)), nil, "m", "", 1); err == nil {
		t.Error("dynamic dim accepted")
	}
}

func TestSequenceParams(t *testing.T) {
	md := meta(tm("x", "INT32", -1, 2))
	cfg := &pb.ModelConfig{
		MaxBatchSize:     4,
		SchedulingChoice: &pb.ModelConfig_SequenceBatching{SequenceBatching: &pb.ModelSequenceBatching{}},
	}
	g := tensorgen.NewGenerator(1)
	for want := int64(1); want <= 3; want++ {
		req, err := BuildInferRequest(g, md, cfg, "m", "", 1)
		if err != nil {
			t.Fatal(err)
		}
		if got := req.Parameters["sequence_id"].GetInt64Param(); got != want {
			t.Fatalf("sequence_id %d, want %d", got, want)
		}
		if !req.Parameters["sequence_start"].GetBoolParam() || req.Parameters["sequence_end"] != nil {
			t.Fatal("sequence_start must be true and sequence_end unset")
		}
	}
	req, _ := BuildInferRequest(g, md, &pb.ModelConfig{MaxBatchSize: 4}, "m", "", 1)
	if len(req.Parameters) != 0 {
		t.Fatal("non-sequence model got parameters")
	}
}

func TestProbeable(t *testing.T) {
	for _, c := range []struct {
		md *pb.ModelMetadataResponse
		ok bool
	}{
		{meta(tm("x", "FP32", -1, 4)), true},
		{meta(tm("s", "BYTES", -1)), false},
		{meta(tm("s", "BYTES", -1, 16)), true},
		{meta(tm("x", "FP32", -1, -1)), false},
	} {
		if ok, _ := Probeable(c.md); ok != c.ok {
			t.Errorf("%v: ok=%v", c.md, ok)
		}
	}
}

func out(name, dt string, shape ...int64) *pb.ModelInferResponse_InferOutputTensor {
	return &pb.ModelInferResponse_InferOutputTensor{Name: name, Datatype: dt, Shape: shape}
}

func resp(outs []*pb.ModelInferResponse_InferOutputTensor, raw ...[]byte) *pb.ModelInferResponse {
	return &pb.ModelInferResponse{ModelName: "m", Id: "7", Outputs: outs, RawOutputContents: raw}
}

// Validate on the generated type and ValidateBytes on its proto.Marshal wire form must
// agree, for good and bad responses alike.
func TestValidatorAgreesOnWireAndGenerated(t *testing.T) {
	v := NewValidator([]tensorgen.Spec{
		{Name: "p", DType: tensorgen.Fp32, Dims: []int64{1}},
		{Name: "s", DType: tensorgen.Bytes, Dims: []int64{2}},
	}, 4, true)
	zero16 := make([]byte, 16)
	good := out("p", "FP32", 4, 1)
	sOut := out("s", "BYTES", 4, 2)
	var s8 []byte
	for i := 0; i < 8; i++ {
		s8 = append(s8, 1, 0, 0, 0, 'z')
	}
	extra := &pb.ModelInferResponse_InferOutputTensor{Name: "x", Datatype: "INT8", Shape: []int64{1},
		Parameters: map[string]*pb.InferParameter{"k": {ParameterChoice: &pb.InferParameter_BoolParam{BoolParam: true}}}}
	cases := map[string]*pb.ModelInferResponse{
		"ok":             resp([]*pb.ModelInferResponse_InferOutputTensor{good, sOut}, zero16, s8),
		"extra output":   resp([]*pb.ModelInferResponse_InferOutputTensor{extra, good, sOut}, []byte{0}, zero16, s8),
		"missing":        resp([]*pb.ModelInferResponse_InferOutputTensor{good}, zero16),
		"duplicate":      resp([]*pb.ModelInferResponse_InferOutputTensor{good, good, sOut}, zero16, zero16, s8),
		"datatype":       resp([]*pb.ModelInferResponse_InferOutputTensor{out("p", "INT32", 4, 1), sOut}, zero16, s8),
		"shape":          resp([]*pb.ModelInferResponse_InferOutputTensor{out("p", "FP32", 3, 1), sOut}, zero16, s8),
		"scalar shape":   resp([]*pb.ModelInferResponse_InferOutputTensor{out("p", "FP32"), sOut}, zero16, s8),
		"raw size":       resp([]*pb.ModelInferResponse_InferOutputTensor{good, sOut}, make([]byte, 12), s8),
		"raw count":      resp([]*pb.ModelInferResponse_InferOutputTensor{good, sOut}, zero16),
		"bytes content":  resp([]*pb.ModelInferResponse_InferOutputTensor{good, sOut}, zero16, s8[:len(s8)-1]),
		"empty response": {},
	}
	for name, r := range cases {
		wire, err := proto.Marshal(r)
		if err != nil {
			t.Fatal(err)
		}
		e1, e2 := v.Validate(r), v.ValidateBytes(wire)
		if fmt.Sprint(e1) != fmt.Sprint(e2) {
			t.Errorf("%s: Validate says %v, ValidateBytes says %v", name, e1, e2)
		}
		if (name == "ok" || name == "extra output") != (e1 == nil) {
			t.Errorf("%s: verdict %v", name, e1)
		}
		// The same with proto's own deterministic encoding and a different field order.
		wire2, _ := det.Marshal(r)
		if e3 := v.ValidateBytes(wire2); fmt.Sprint(e3) != fmt.Sprint(e1) {
			t.Errorf("%s: deterministic wire says %v", name, e3)
		}
	}
	// Truncations of a valid response never pass as valid with different content, and never panic.
	wire, _ := proto.Marshal(cases["ok"])
	for n := 0; n < len(wire); n++ {
		if err := v.ValidateBytes(wire[:n]); err == nil {
			t.Fatalf("truncated to %d bytes validated", n)
		}
	}
}
