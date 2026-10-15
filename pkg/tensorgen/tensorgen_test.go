package tensorgen

import (
	"bytes"
	"encoding/binary"
	"math"
	"strconv"
	"testing"

	"google.golang.org/protobuf/encoding/protowire"
)

func spec(name string, dt DType, dims ...int64) Spec { return Spec{Name: name, DType: dt, Dims: dims} }

func mustDist(t *testing.T, s string) Dist {
	t.Helper()
	d, err := ParseDist(s)
	if err != nil {
		t.Fatal(err)
	}
	return d
}

func template(inputs []Spec, batch int, batched bool) *Template {
	return &Template{Model: "m", Inputs: inputs, BatchSize: batch, Batched: batched}
}

func TestParseDist(t *testing.T) {
	good := map[string]Dist{
		"zeros":             {Kind: DistZeros},
		"random-bytes":      {Kind: DistRandomBytes},
		"const:3":           {Kind: DistConst, A: 3},
		"const:-1.5":        {Kind: DistConst, A: -1.5},
		"uniform:0..1000":   {Kind: DistUniform, A: 0, B: 1000},
		"uniform:-1.5..2.5": {Kind: DistUniform, A: -1.5, B: 2.5},
		"uniform:5..5":      {Kind: DistUniform, A: 5, B: 5},
	}
	for s, want := range good {
		got, err := ParseDist(s)
		if err != nil || got != want {
			t.Errorf("%q: got %v, %v want %v", s, got, err, want)
		}
		if got.String() != s {
			t.Errorf("%q round trips to %q", s, got.String())
		}
	}
	for _, s := range []string{"uniform:6..5", "uniform:5", "const:x", "gaussian", "const:inf", "const:NaN", "uniform:a..2", ""} {
		if _, err := ParseDist(s); err == nil {
			t.Errorf("%q should be rejected", s)
		}
	}
}

func TestDistCheck(t *testing.T) {
	type c struct {
		dist string
		dt   DType
		ok   bool
	}
	cases := []c{
		{"zeros", Fp16, true}, {"random-bytes", Bf16, true},
		{"const:1", Fp16, false}, {"uniform:0..1", Bf16, false},
		{"zeros", Bytes, true}, {"uniform:0..1000", Int32, true},
		{"uniform:0..1000", Uint8, false},
		// Inclusive upper bound for integers: 255 fits UINT8, 256 does not.
		{"uniform:0..255", Uint8, true}, {"uniform:0..256", Uint8, false},
		{"uniform:-128..127", Int8, true}, {"uniform:-129..0", Int8, false},
		// lo == hi: fine for integers, not for floats.
		{"uniform:5..5", Int32, true}, {"uniform:5..5", Fp32, false},
		{"uniform:0..1.5", Int32, false}, {"uniform:0..1.5", Fp32, true},
		{"uniform:0..1", Bool, true}, {"uniform:0..2", Bool, false},
		{"const:1.5", Int32, false}, {"const:1.5", Fp32, true}, {"const:2", Bool, false},
		{"const:4294967295", Uint32, true}, {"const:4294967296", Uint32, false},
		{"const:9223372036854775808", Int64, false}, {"const:-9223372036854775808", Int64, true},
		{"const:18446744073709551616", Uint64, false}, {"const:1e39", Fp32, false}, {"const:1e39", Fp64, true},
		{"uniform:0..1000", Bytes, true}, {"const:1.5", Bytes, false},
	}
	for _, c := range cases {
		err := mustDist(t, c.dist).Check(c.dt, "a")
		if (err == nil) != c.ok {
			t.Errorf("%s on %s: err=%v, want ok=%v", c.dist, c.dt, err, c.ok)
		}
	}
}

func TestParseSpec(t *testing.T) {
	s, err := ParseSpec("img:FP32:3,224,224")
	if err != nil || s.Name != "img" || s.DType != Fp32 || len(s.Dims) != 3 || s.Dims[2] != 224 {
		t.Fatalf("got %+v, %v", s, err)
	}
	if n, ok := s.NumElements(); !ok || n != 3*224*224 {
		t.Fatalf("numel %d %v", n, ok)
	}
	if got := s.Shape(4, true); len(got) != 4 || got[0] != 4 {
		t.Fatalf("shape %v", got)
	}
	if got := s.Shape(1, false); len(got) != 3 {
		t.Fatalf("unbatched shape %v", got)
	}
	if s.String() != "img:FP32:3,224,224" {
		t.Fatal(s.String())
	}
	if b, err := ParseSpec("s:BYTES:2"); err != nil || b.DType != Bytes {
		t.Fatalf("BYTES input: %v", err)
	}
	if c, err := ParseSpec("knn/feed/timestamps_sec:0:INT64:100"); err != nil || c.Name != "knn/feed/timestamps_sec:0" || c.DType != Int64 || len(c.Dims) != 1 || c.Dims[0] != 100 {
		t.Fatalf("name with a colon: %+v, %v", c, err)
	}
	for _, bad := range []string{"x:INT32", ":INT32:1", "x:INT32:0", "x:INT32:-1", "x:INT32:a", "x:INT32:", "x:int32:1", "x:INT7:1"} {
		if _, err := ParseSpec(bad); err == nil {
			t.Errorf("%q should be rejected", bad)
		}
	}
}

func TestDTypeRoundTrip(t *testing.T) {
	for _, n := range []string{"BOOL", "UINT8", "UINT16", "UINT32", "UINT64", "INT8", "INT16", "INT32", "INT64", "FP16", "FP32", "FP64", "BF16", "BYTES"} {
		d, err := ParseDType(n)
		if err != nil || d.String() != n {
			t.Errorf("%s: %v %v", n, d, err)
		}
	}
	if _, err := ParseDType("int32"); err == nil {
		t.Error("lower case accepted")
	}
	if s, ok := Int32.Size(); !ok || s != 4 {
		t.Error("INT32 size")
	}
	if s, ok := Bf16.Size(); !ok || s != 2 {
		t.Error("BF16 size")
	}
	if _, ok := Bytes.Size(); ok {
		t.Error("BYTES has no fixed size")
	}
}

func TestRequestShapesAndSizesPerDType(t *testing.T) {
	sizes := map[DType]int{Bool: 1, Uint8: 1, Int8: 1, Uint16: 2, Int16: 2, Fp16: 2, Bf16: 2, Uint32: 4, Int32: 4, Fp32: 4, Uint64: 8, Int64: 8, Fp64: 8}
	for dt, size := range sizes {
		g := NewGenerator(1, WithDist(mustDist(t, "zeros")))
		req, err := g.Build(template([]Spec{spec("a", dt, 3, 5)}, 4, true))
		if err != nil {
			t.Fatal(dt, err)
		}
		in := req.Inputs[0]
		if in.Datatype != dt.String() || len(in.Shape) != 3 || in.Shape[0] != 4 || in.Shape[1] != 3 || in.Shape[2] != 5 {
			t.Errorf("%s: %+v", dt, in)
		}
		if got, want := len(req.Inputs[0].Raw), 4*3*5*size; got != want {
			t.Errorf("%s: %d bytes, want %d", dt, got, want)
		}
	}
}

func TestRequestRoundTrip(t *testing.T) {
	g := NewGenerator(1)
	tpl := template([]Spec{spec("x", Int32, 208), spec("m", Int32, 85)}, 4, true)
	var reqs []*Request
	for i := 0; i < 8; i++ {
		r, err := g.Build(tpl)
		if err != nil {
			t.Fatal(err)
		}
		reqs = append(reqs, r)
	}
	req := reqs[3]
	if req.Model != "m" || len(req.Inputs) != 2 || req.Inputs[0].Name != "x" || req.Inputs[0].Datatype != "INT32" {
		t.Fatalf("%+v", req)
	}
	if req.Inputs[0].Shape[0] != 4 || req.Inputs[0].Shape[1] != 208 || req.Inputs[1].Shape[1] != 85 {
		t.Fatalf("shapes %v %v", req.Inputs[0].Shape, req.Inputs[1].Shape)
	}
	if len(req.Inputs[0].Raw) != 4*208*4 || len(req.Inputs[1].Raw) != 4*85*4 {
		t.Fatal("raw sizes")
	}
	for i := 0; i < len(req.Inputs[0].Raw); i += 4 {
		if v := int32(binary.LittleEndian.Uint32(req.Inputs[0].Raw[i:])); v < 0 || v > 1000 {
			t.Fatalf("value %d outside 0..1000", v)
		}
	}
	if bytes.Equal(reqs[0].Inputs[0].Raw, reqs[1].Inputs[0].Raw) {
		t.Error("requests of one generator should differ")
	}
	if len(req.Outputs) != 0 {
		t.Error("no outputs should be selected")
	}
}

func TestRequestedOutputs(t *testing.T) {
	m, err := NewModel(Metadata{Inputs: []TensorMeta{tm("x", "FP32", -1, 4)}, Outputs: []TensorMeta{tm("p", "FP32", -1, 1), tm("q", "INT32", -1)}},
		&Config{MaxBatchSize: 4}, "m", "")
	if err != nil {
		t.Fatal(err)
	}
	tpl, err := m.NewTemplate(nil, 1)
	if err != nil {
		t.Fatal(err)
	}
	req, _ := NewGenerator(1).Build(tpl)
	if len(req.Outputs) != 0 {
		t.Errorf("outputs selected without the option: %v", req.Outputs)
	}
	req, _ = NewGenerator(1, WithRequestedOutputs()).Build(tpl)
	if len(req.Outputs) != 2 || req.Outputs[0] != "p" || req.Outputs[1] != "q" {
		t.Errorf("outputs %v", req.Outputs)
	}
}

func TestDeterminismPerSeed(t *testing.T) {
	tpl := template([]Spec{spec("x", Int32, 16), spec("s", Bytes, 3)}, 2, true)
	gen := func(seed uint64, n int) [][]byte {
		g := NewGenerator(seed, WithDist(mustDist(t, "uniform:0..9")))
		var out [][]byte
		for i := 0; i < n; i++ {
			b, err := g.Encode(tpl)
			if err != nil {
				t.Fatal(err)
			}
			out = append(out, b)
		}
		return out
	}
	a, b, c := gen(1, 3), gen(1, 3), gen(2, 3)
	for i := range a {
		if !bytes.Equal(a[i], b[i]) {
			t.Errorf("same seed differs at %d", i)
		}
		if bytes.Equal(a[i], c[i]) {
			t.Errorf("different seeds equal at %d", i)
		}
	}
}

func TestGensAndDTypes(t *testing.T) {
	inputs := []Spec{spec("f", Fp32, 2), spec("b", Bool, 4), spec("h", Fp16, 3)}
	g := NewGenerator(1, WithDist(mustDist(t, "zeros")))
	req, err := g.Build(template(inputs, 1, true))
	if err != nil {
		t.Fatal(err)
	}
	for _, in := range req.Inputs {
		raw := in.Raw
		for _, b := range raw {
			if b != 0 {
				t.Fatal("zeros produced a non-zero byte")
			}
		}
	}
	g = NewGenerator(1, WithDist(mustDist(t, "const:1")))
	req, err = g.Build(template(inputs[:2], 1, true))
	if err != nil {
		t.Fatal(err)
	}
	one := binary.LittleEndian.AppendUint32(binary.LittleEndian.AppendUint32(nil, math.Float32bits(1)), math.Float32bits(1))
	if !bytes.Equal(req.Inputs[0].Raw, one) || !bytes.Equal(req.Inputs[1].Raw, []byte{1, 1, 1, 1}) {
		t.Fatalf("const:1 gave %v", req.Inputs)
	}
	g = NewGenerator(1, WithDist(mustDist(t, "random-bytes")))
	req, err = g.Build(template(inputs, 1, true))
	if err != nil {
		t.Fatal(err)
	}
	if len(req.Inputs[2].Raw) != 6 {
		t.Fatal("FP16 size")
	}
	for _, b := range req.Inputs[1].Raw {
		if b > 1 {
			t.Fatalf("BOOL random-bytes not masked: %d", b)
		}
	}
	// FP16 with const is refused.
	g = NewGenerator(1, WithDist(mustDist(t, "const:1")))
	if _, err := g.Build(template(inputs, 1, true)); err == nil {
		t.Fatal("FP16 + const accepted")
	}
}

func TestUniformBounds(t *testing.T) {
	g := NewGenerator(7, WithDist(mustDist(t, "uniform:-2..2")))
	req, err := g.Build(template([]Spec{spec("a", Int32, 50000)}, 1, true))
	if err != nil {
		t.Fatal(err)
	}
	var hist [5]int
	raw := req.Inputs[0].Raw
	for i := 0; i < len(raw); i += 4 {
		v := int32(binary.LittleEndian.Uint32(raw[i:]))
		if v < -2 || v > 2 {
			t.Fatalf("%d outside -2..2", v)
		}
		hist[v+2]++
	}
	for i, n := range hist {
		if n < 9000 || n > 11000 {
			t.Errorf("bucket %d has %d, histogram %v", i, n, hist)
		}
	}
	// lo == hi for integers always gives that value.
	g = NewGenerator(7, WithDist(mustDist(t, "uniform:5..5")))
	req, _ = g.Build(template([]Spec{spec("a", Int16, 100)}, 1, true))
	for i := 0; i < 200; i += 2 {
		if binary.LittleEndian.Uint16(req.Inputs[0].Raw[i:]) != 5 {
			t.Fatal("lo == hi must give lo")
		}
	}
	// BOOL: both values appear.
	g = NewGenerator(7, WithDist(mustDist(t, "uniform:0..1")))
	req, _ = g.Build(template([]Spec{spec("a", Bool, 1000)}, 1, true))
	if !bytes.Contains(req.Inputs[0].Raw, []byte{0}) || !bytes.Contains(req.Inputs[0].Raw, []byte{1}) {
		t.Fatal("BOOL uniform:0..1 should hit both values")
	}
	// Full-range 64-bit does not hang or panic; signed negatives appear.
	g = NewGenerator(7, WithDist(mustDist(t, "uniform:-9223372036854775808..9223372036854774784")))
	if _, err := g.Build(template([]Spec{spec("a", Int64, 100)}, 1, true)); err != nil {
		t.Fatal(err)
	}
	// Floats: half-open.
	for _, dt := range []DType{Fp32, Fp64} {
		g = NewGenerator(7, WithDist(mustDist(t, "uniform:0..1")))
		req, _ = g.Build(template([]Spec{spec("a", dt, 50000)}, 1, true))
		raw := req.Inputs[0].Raw
		step := 4
		if dt == Fp64 {
			step = 8
		}
		for i := 0; i < len(raw); i += step {
			var v float64
			if dt == Fp32 {
				v = float64(math.Float32frombits(binary.LittleEndian.Uint32(raw[i:])))
			} else {
				v = math.Float64frombits(binary.LittleEndian.Uint64(raw[i:]))
			}
			if v < 0 || v >= 1 {
				t.Fatalf("%s value %v outside [0,1)", dt, v)
			}
		}
	}
}

func TestBytesEncoding(t *testing.T) {
	g := NewGenerator(3, WithDist(mustDist(t, "uniform:10..99")))
	req, err := g.Build(template([]Spec{spec("s", Bytes, 3)}, 2, true))
	if err != nil {
		t.Fatal(err)
	}
	if req.Inputs[0].Datatype != "BYTES" || req.Inputs[0].Shape[0] != 2 || req.Inputs[0].Shape[1] != 3 {
		t.Fatalf("%+v", req.Inputs[0])
	}
	raw := req.Inputs[0].Raw
	for i := 0; i < 6; i++ {
		l := int(binary.LittleEndian.Uint32(raw))
		s := string(raw[4 : 4+l])
		v, err := strconv.Atoi(s)
		if err != nil || v < 10 || v > 99 {
			t.Fatalf("element %d = %q", i, s)
		}
		raw = raw[4+l:]
	}
	if len(raw) != 0 {
		t.Fatal("trailing bytes")
	}
	g = NewGenerator(3, WithDist(mustDist(t, "zeros")))
	req, _ = g.Build(template([]Spec{spec("s", Bytes, 1)}, 1, true))
	if !bytes.Equal(req.Inputs[0].Raw, []byte{1, 0, 0, 0, '0'}) {
		t.Fatalf("zeros: %v", req.Inputs[0].Raw)
	}
	g = NewGenerator(3, WithDist(mustDist(t, "random-bytes")))
	if _, err := g.Build(template([]Spec{spec("s", Bytes, 5)}, 1, true)); err != nil {
		t.Fatal(err)
	}
}

func TestUnbatchedRequestShape(t *testing.T) {
	g := NewGenerator(1, WithDist(mustDist(t, "zeros")))
	req, err := g.Build(template([]Spec{spec("x", Int32, 5)}, 1, false))
	if err != nil {
		t.Fatal(err)
	}
	if len(req.Inputs[0].Shape) != 1 || req.Inputs[0].Shape[0] != 5 || len(req.Inputs[0].Raw) != 20 {
		t.Fatalf("%+v", req)
	}
}

func meta(inputs ...TensorMeta) Metadata { return Metadata{Inputs: inputs} }

func tm(name, dt string, shape ...int64) TensorMeta {
	return TensorMeta{Name: name, Datatype: dt, Shape: shape}
}

func TestModelOverridesAndBatchLimit(t *testing.T) {
	md := meta(tm("x", "INT32", -1, 208), tm("v", "INT32", -1, -1))
	md.Outputs = []TensorMeta{tm("p", "FP32", -1, 1)}
	m, err := NewModel(md, &Config{MaxBatchSize: 8}, "m", "")
	if err != nil {
		t.Fatal(err)
	}
	if !m.Batched || len(m.Inputs[0].Dims) != 1 || m.Inputs[0].Dims[0] != 208 || m.Outputs[0].Dims[0] != 1 {
		t.Fatalf("%+v", m)
	}
	// `v` has a variable dim: unresolved is an error, an override fixes it.
	if _, err := m.NewTemplate(nil, 1); err == nil {
		t.Error("variable dim accepted")
	}
	tpl, err := m.NewTemplate([]Spec{spec("v", Int32, 5), spec("x", Int32, 10)}, 8)
	if err != nil {
		t.Fatal(err)
	}
	if tpl.Inputs[0].Dims[0] != 10 || tpl.Inputs[1].Dims[0] != 5 {
		t.Fatalf("%+v", tpl.Inputs)
	}
	if _, err := m.NewTemplate([]Spec{spec("v", Int32, 5)}, 9); err == nil {
		t.Error("batch size above max_batch_size accepted")
	}
	if _, err := m.NewTemplate([]Spec{spec("v", Int32, 5)}, 0); err == nil {
		t.Error("batch size 0 accepted")
	}
	// An unknown name is appended as an extra input.
	tpl, err = m.NewTemplate([]Spec{spec("extra", Int32, 2), spec("v", Int32, 5)}, 1)
	if err != nil || len(tpl.Inputs) != 3 || tpl.Inputs[2].Name != "extra" {
		t.Fatalf("%+v %v", tpl, err)
	}
}

func TestNonBatchingModel(t *testing.T) {
	md := meta(tm("v", "INT32", 7))
	m, err := NewModel(md, &Config{MaxBatchSize: 0}, "m", "")
	if err != nil {
		t.Fatal(err)
	}
	if m.Batched || m.Inputs[0].Dims[0] != 7 {
		t.Fatalf("%+v", m)
	}
	tpl, err := m.NewTemplate(nil, 1)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := m.NewTemplate(nil, 2); err == nil {
		t.Error("batch size 2 accepted for a non-batching model")
	}
	req, err := NewGenerator(1).Build(tpl)
	if err != nil || len(req.Inputs[0].Shape) != 1 || req.Inputs[0].Shape[0] != 7 {
		t.Fatalf("%+v %v", req, err)
	}
}

func TestBuildInferRequestConvenience(t *testing.T) {
	md := meta(tm("x", "FP32", -1, 4))
	g := NewGenerator(5)
	// Without a config the leading -1 marks a batching model.
	req, err := g.BuildInferRequest(md, nil, "mdl", "3", 2)
	if err != nil {
		t.Fatal(err)
	}
	if req.Model != "mdl" || req.Version != "3" || req.Inputs[0].Shape[0] != 2 || len(req.Inputs[0].Raw) != 2*4*4 {
		t.Fatalf("%+v", req)
	}
	if _, err := g.BuildInferRequest(meta(tm("x", "INT32", -1, -1)), nil, "m", "", 1); err == nil {
		t.Error("dynamic dim accepted")
	}
}

func TestSequenceParams(t *testing.T) {
	md := meta(tm("x", "INT32", -1, 2))
	cfg := &Config{MaxBatchSize: 4, Sequence: true}
	g := NewGenerator(1)
	var ids []int64
	for i := 0; i < 3; i++ {
		req, err := g.BuildInferRequest(md, cfg, "m", "", 1)
		if err != nil {
			t.Fatal(err)
		}
		ids = append(ids, req.Parameters["sequence_id"].(int64))
		if req.Parameters["sequence_start"] != true {
			t.Fatal("sequence_start must be true")
		}
		if _, ok := req.Parameters["sequence_end"]; ok {
			t.Fatal("sequence_end is not set")
		}
	}
	if ids[0] != 1 || ids[1] != 2 || ids[2] != 3 {
		t.Fatalf("ids %v", ids)
	}
	g = NewGenerator(1, WithSequenceIDStart(1000))
	req, _ := g.BuildInferRequest(md, cfg, "m", "", 1)
	if req.Parameters["sequence_id"] != int64(1000) {
		t.Fatal("WithSequenceIDStart ignored")
	}
	// A plain model has no parameters.
	req, _ = g.BuildInferRequest(md, &Config{MaxBatchSize: 4}, "m", "", 1)
	if len(req.Parameters) != 0 {
		t.Fatal("non-sequence model got parameters")
	}
}

func TestProbeable(t *testing.T) {
	cases := []struct {
		name string
		md   Metadata
		ok   bool
	}{
		{"plain", meta(tm("x", "FP32", -1, 4)), true},
		{"bytes one dim", meta(tm("s", "BYTES", -1)), false},
		{"bytes two dims", meta(tm("s", "BYTES", -1, 16)), true},
		{"dynamic inner dim", meta(tm("x", "FP32", -1, -1)), false},
		{"dynamic first dim only", meta(tm("x", "FP32", -1, 3), tm("y", "INT32", -1)), true},
		{"no inputs", meta(), true},
	}
	for _, c := range cases {
		ok, reason := Probeable(c.md)
		if ok != c.ok || (!ok && reason == "") {
			t.Errorf("%s: ok=%v reason=%q", c.name, ok, reason)
		}
	}
}

func out(name, dt string, shape ...int64) TensorMeta { return tm(name, dt, shape...) }

func outs(o ...TensorMeta) []TensorMeta { return o }

func raws(r ...[]byte) [][]byte { return r }

func wantKind(t *testing.T, err error, k MismatchKind) {
	t.Helper()
	m, ok := err.(*Mismatch)
	if !ok || m.Kind != k {
		t.Errorf("got %v, want kind %d", err, k)
	}
}

// respWire encodes a ModelInferResponse the way Triton does.
func respWire(os []TensorMeta, raw ...[]byte) []byte {
	var b []byte
	b = protowire.AppendString(protowire.AppendTag(b, 1, protowire.BytesType), "model")
	for _, o := range os {
		var m []byte
		m = protowire.AppendString(protowire.AppendTag(m, 1, protowire.BytesType), o.Name)
		m = protowire.AppendString(protowire.AppendTag(m, 2, protowire.BytesType), o.Datatype)
		var sh []byte
		for _, d := range o.Shape {
			sh = protowire.AppendVarint(sh, uint64(d))
		}
		m = protowire.AppendBytes(protowire.AppendTag(m, 3, protowire.BytesType), sh)
		b = protowire.AppendBytes(protowire.AppendTag(b, 5, protowire.BytesType), m)
	}
	for _, r := range raw {
		b = protowire.AppendBytes(protowire.AppendTag(b, 6, protowire.BytesType), r)
	}
	return b
}

func TestValidator(t *testing.T) {
	v := NewValidator([]Spec{spec("p", Fp32, 1)}, 4, true)
	zero16 := make([]byte, 16)
	if err := v.Validate(outs(out("p", "FP32", 4, 1)), raws(zero16)); err != nil {
		t.Fatal(err)
	}
	// Extra outputs the metadata does not list are fine.
	if err := v.Validate(outs(out("p", "FP32", 4, 1), out("q", "INT32", 4)), raws(zero16, zero16)); err != nil {
		t.Fatal(err)
	}
	wire := respWire(outs(out("p", "FP32", 4, 1)), zero16)
	if err := v.ValidateBytes(wire); err != nil {
		t.Fatal(err)
	}
	wantKind(t, v.ValidateBytes([]byte{0xff, 0xff, 0xff}), MismatchDecode)
	wantKind(t, v.ValidateBytes(wire[:len(wire)-1]), MismatchDecode)
	wantKind(t, v.Validate(outs(out("other", "FP32", 4, 1)), raws(zero16)), MismatchMissingOutput)
	wantKind(t, v.Validate(outs(out("p", "FP32", 4, 1), out("p", "FP32", 4, 1)), raws(zero16, zero16)), MismatchDuplicateOutput)
	wantKind(t, v.Validate(outs(out("p", "INT32", 4, 1)), raws(zero16)), MismatchDatatype)
	for _, shape := range [][]int64{{3, 1}, {4, 2}, {4}, {4, 1, 1}} {
		wantKind(t, v.Validate(outs(out("p", "FP32", shape...)), raws(zero16)), MismatchShape)
	}
	wantKind(t, v.Validate(outs(out("p", "FP32", 4, 1)), raws(make([]byte, 12))), MismatchRawSize)
	wantKind(t, v.Validate(outs(out("p", "FP32", 4, 1)), nil), MismatchRawCount)
	wantKind(t, v.ValidateBytes(respWire(outs(out("p", "FP32", 4, 1)))), MismatchRawCount)
}

// The decoder reads what proto.Marshal writes whatever the encoding variant: unpacked
// shapes, unknown fields, fields in any order, the last of a repeated scalar.
func TestValidateBytesToleratesWireVariants(t *testing.T) {
	v := NewValidator([]Spec{spec("p", Fp32, 1)}, 4, true)
	// Output with an unpacked shape, an unknown varint field and a parameters map.
	var o []byte
	o = protowire.AppendVarint(protowire.AppendTag(o, 3, protowire.VarintType), 4)
	o = protowire.AppendVarint(protowire.AppendTag(o, 3, protowire.VarintType), 1)
	o = protowire.AppendString(protowire.AppendTag(o, 2, protowire.BytesType), "FP32")
	o = protowire.AppendVarint(protowire.AppendTag(o, 99, protowire.VarintType), 7)
	o = protowire.AppendString(protowire.AppendTag(o, 1, protowire.BytesType), "p")
	o = protowire.AppendString(protowire.AppendTag(o, 4, protowire.BytesType), "ignored")
	var b []byte
	b = protowire.AppendBytes(protowire.AppendTag(b, 6, protowire.BytesType), make([]byte, 16)) // raw before outputs
	b = protowire.AppendBytes(protowire.AppendTag(b, 5, protowire.BytesType), o)
	b = protowire.AppendFixed32(protowire.AppendTag(b, 42, protowire.Fixed32Type), 1)
	if err := v.ValidateBytes(b); err != nil {
		t.Fatal(err)
	}
	if err := v.ValidateBytes(nil); err == nil {
		t.Error("an empty response lacks the output")
	}
}

func TestValidatorUnbatched(t *testing.T) {
	v := NewValidator([]Spec{spec("p", Fp32, 3)}, 1, false)
	if err := v.Validate(outs(out("p", "FP32", 3)), raws(make([]byte, 12))); err != nil {
		t.Fatal(err)
	}
	wantKind(t, v.Validate(outs(out("p", "FP32", 1, 3)), raws(make([]byte, 12))), MismatchShape)
}

func TestValidatorWildcardAndBytes(t *testing.T) {
	v := NewValidator([]Spec{spec("a", Int32, -1), spec("s", Bytes)}, 2, true)
	var s []byte
	for _, e := range []string{"hi", "there"} {
		s = binary.LittleEndian.AppendUint32(s, uint32(len(e)))
		s = append(s, e...)
	}
	os := outs(out("a", "INT32", 2, 7), out("s", "BYTES", 2))
	if err := v.Validate(os, raws(make([]byte, 56), s)); err != nil {
		t.Fatal(err)
	}
	wantKind(t, v.Validate(os, raws(make([]byte, 56), s[:len(s)-1])), MismatchRawBytes)
}
