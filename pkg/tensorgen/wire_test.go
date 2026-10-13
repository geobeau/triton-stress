package tensorgen

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"
)

func TestMarshalWireGolden(t *testing.T) {
	r := &Request{
		Model: "m",
		Inputs: []Tensor{{Name: "x", Datatype: "INT32", Shape: []int64{2, 1},
			Raw: []byte{1, 0, 0, 0, 2, 0, 0, 0}}},
		Outputs:    []string{"p"},
		Parameters: map[string]any{"sequence_id": int64(5), "a": true, "s": "v"},
	}
	want := []byte{
		0x0a, 0x01, 'm',
		0x22, 0x07, 0x0a, 0x01, 'a', 0x12, 0x02, 0x08, 0x01, // parameters sorted by name
		0x22, 0x08, 0x0a, 0x01, 's', 0x12, 0x03, 0x1a, 0x01, 'v',
		0x22, 0x11, 0x0a, 0x0b, 's', 'e', 'q', 'u', 'e', 'n', 'c', 'e', '_', 'i', 'd', 0x12, 0x02, 0x10, 0x05,
		0x2a, 0x0e, 0x0a, 0x01, 'x', 0x12, 0x05, 'I', 'N', 'T', '3', '2', 0x1a, 0x02, 0x02, 0x01,
		0x32, 0x03, 0x0a, 0x01, 'p',
		0x3a, 0x08, 1, 0, 0, 0, 2, 0, 0, 0,
	}
	got, err := r.MarshalWire()
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("got  % x\nwant % x", got, want)
	}
	app, err := r.AppendWire([]byte("pre"))
	if err != nil || !bytes.Equal(app, append([]byte("pre"), want...)) {
		t.Fatalf("AppendWire: % x %v", app, err)
	}
}

func TestMarshalWireRejectsParameterType(t *testing.T) {
	r := &Request{Model: "m", Parameters: map[string]any{"x": 1.5}}
	if _, err := r.MarshalWire(); err == nil {
		t.Error("float parameter accepted")
	}
	if _, _, err := HTTPBody(r); err == nil {
		t.Error("float parameter accepted over HTTP")
	}
}

func TestHTTPRequestedOutputs(t *testing.T) {
	r := &Request{Model: "m", Inputs: []Tensor{{Name: "x", Datatype: "INT32", Shape: []int64{1}, Raw: []byte{0, 0, 0, 0}}}, Outputs: []string{"p", "q"}}
	body, n, err := HTTPBody(r)
	if err != nil {
		t.Fatal(err)
	}
	want := `{"inputs":[{"name":"x","shape":[1],"datatype":"INT32","parameters":{"binary_data_size":4}}],` +
		`"outputs":[{"name":"p"},{"name":"q"}],"parameters":{"binary_data_output":true}}`
	if string(body[:n]) != want {
		t.Fatalf("%s", body[:n])
	}
}

func TestProbeDist(t *testing.T) {
	if err := ProbeDist.Check(Bf16, "x"); err != nil {
		t.Fatal(err)
	}
	if _, err := ParseDist("probe"); err == nil {
		t.Error("ParseDist must not accept probe: --gen is RTS's contract")
	}
	if ProbeDist.String() != "probe" {
		t.Error(ProbeDist.String())
	}
	g := NewGenerator(3, WithDist(ProbeDist))
	tpl := template([]Spec{
		spec("i32", Int32, 500), spec("i8", Int8, 500), spec("u64", Uint64, 500), spec("i16", Int16, 500),
		spec("f32", Fp32, 500), spec("f64", Fp64, 500), spec("b", Bool, 500), spec("h", Fp16, 500), spec("bf", Bf16, 500),
		spec("s", Bytes, 500),
	}, 1, true)
	req, err := g.Build(tpl)
	if err != nil {
		t.Fatal(err)
	}
	raw := func(name string) []byte {
		for _, in := range req.Inputs {
			if in.Name == name {
				return in.Raw
			}
		}
		t.Fatal(name)
		return nil
	}
	// Integers: low byte only, any value of that byte.
	seen := map[byte]bool{}
	for _, name := range []string{"i32", "u64", "i16"} {
		b := raw(name)
		size := len(b) / 500
		for i := 0; i < 500; i++ {
			for _, hi := range b[i*size+1 : (i+1)*size] {
				if hi != 0 {
					t.Fatalf("%s: upper byte %d set", name, hi)
				}
			}
			seen[b[i*size]] = true
		}
	}
	if len(seen) < 150 {
		t.Errorf("only %d distinct low bytes", len(seen))
	}
	neg := false
	for _, b := range raw("i8") {
		neg = neg || b >= 128
	}
	if !neg {
		t.Error("INT8 should use the whole byte")
	}
	for i := 0; i < 500; i++ {
		if v := math.Float32frombits(binary.LittleEndian.Uint32(raw("f32")[i*4:])); v < 0 || v >= 1 {
			t.Fatalf("fp32 %v", v)
		}
		if v := math.Float64frombits(binary.LittleEndian.Uint64(raw("f64")[i*8:])); v < 0 || v >= 1 {
			t.Fatalf("fp64 %v", v)
		}
		if raw("b")[i] > 1 {
			t.Fatal("bool")
		}
	}
	if len(raw("h")) != 1000 || len(raw("bf")) != 1000 {
		t.Fatal("fp16 sizes")
	}
	if !validBytesTensor(raw("s"), 500) {
		t.Fatal("BYTES encoding")
	}
	b := raw("s")
	for i := 0; i < 500; i++ {
		l := int(binary.LittleEndian.Uint32(b))
		var v int
		for _, c := range b[4 : 4+l] {
			if c < '0' || c > '9' {
				t.Fatalf("BYTES element %q", b[4:4+l])
			}
			v = v*10 + int(c-'0')
		}
		if v >= 1000000 || (l > 1 && b[4] == '0') {
			t.Fatalf("BYTES element %d", v)
		}
		b = b[4+l:]
	}
}
