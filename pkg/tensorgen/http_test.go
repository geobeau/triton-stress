package tensorgen

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"testing"
)

func TestInferPath(t *testing.T) {
	for _, c := range []struct{ model, ver, want string }{
		{"m", "", "/v2/models/m/infer"},
		{"m", "3", "/v2/models/m/versions/3/infer"},
		{"a b/c", "", "/v2/models/a%20b%2Fc/infer"},
	} {
		if got := InferPath(c.model, c.ver); got != c.want {
			t.Errorf("InferPath(%q,%q) = %q, want %q", c.model, c.ver, got, c.want)
		}
	}
	if got := ModelPath("m", "2"); got != "/v2/models/m/versions/2" {
		t.Error(got)
	}
}

// splitRequest parses a complete request into its header lines and body.
func splitRequest(t *testing.T, b []byte) (lines []string, body []byte) {
	t.Helper()
	i := bytes.Index(b, []byte("\r\n\r\n"))
	if i < 0 {
		t.Fatal("no end of headers")
	}
	return strings.Split(string(b[:i]), "\r\n"), b[i+4:]
}

func TestEncodeHTTPByteExact(t *testing.T) {
	tpl := &Template{Model: "model_x", Version: "7", Batched: true, BatchSize: 2,
		Inputs: []Spec{spec("a", Int32, 3), spec("b", Uint8, 2)}}
	g := NewGenerator(5, WithDist(mustDist(t, "const:9")))
	got, err := g.EncodeHTTP(tpl, "triton:8000")
	if err != nil {
		t.Fatal(err)
	}
	js := `{"inputs":[` +
		`{"name":"a","shape":[2,3],"datatype":"INT32","parameters":{"binary_data_size":24}},` +
		`{"name":"b","shape":[2,2],"datatype":"UINT8","parameters":{"binary_data_size":4}}],` +
		`"parameters":{"binary_data_output":true}}`
	var want []byte
	want = append(want, "POST /v2/models/model_x/versions/7/infer HTTP/1.1\r\n"+
		"Host: triton:8000\r\n"+
		"Content-Type: application/octet-stream\r\n"+
		"Inference-Header-Content-Length: "+strconv.Itoa(len(js))+"\r\n"+
		"Content-Length: "+strconv.Itoa(len(js)+28)+"\r\n\r\n"+js...)
	for i := 0; i < 6; i++ {
		want = append(want, 9, 0, 0, 0)
	}
	want = append(want, 9, 9, 9, 9)
	if !bytes.Equal(got, want) {
		t.Fatalf("request differs:\n got %q\nwant %q", got, want)
	}
}

func TestEncodeHTTPDecodesBack(t *testing.T) {
	tpl := &Template{Model: "m", Batched: true, BatchSize: 4, Inputs: []Spec{spec("x", Int32, 208), spec("mc", Int32, 85)}}
	g := NewGenerator(1)
	req, err := g.EncodeHTTP(tpl, "h:1")
	if err != nil {
		t.Fatal(err)
	}
	lines, body := splitRequest(t, req)
	if lines[0] != "POST /v2/models/m/infer HTTP/1.1" {
		t.Fatal(lines[0])
	}
	hdr := map[string]string{}
	for _, l := range lines[1:] {
		k, v, _ := strings.Cut(l, ": ")
		hdr[k] = v
	}
	n, _ := strconv.Atoi(hdr["Inference-Header-Content-Length"])
	if cl, _ := strconv.Atoi(hdr["Content-Length"]); cl != len(body) || hdr["Host"] != "h:1" ||
		hdr["Content-Type"] != "application/octet-stream" {
		t.Fatalf("headers %v, body %d", hdr, len(body))
	}
	var h struct {
		Inputs []struct {
			Name       string
			Shape      []int64
			Datatype   string
			Parameters map[string]int
		}
		Parameters map[string]any
	}
	if err := json.Unmarshal(body[:n], &h); err != nil {
		t.Fatal(err)
	}
	if h.Parameters["binary_data_output"] != true || len(h.Parameters) != 1 || len(h.Inputs) != 2 {
		t.Fatalf("%+v", h)
	}
	if h.Inputs[0].Name != "x" || h.Inputs[0].Datatype != "INT32" || fmt.Sprint(h.Inputs[0].Shape) != "[4 208]" ||
		h.Inputs[0].Parameters["binary_data_size"] != 4*208*4 || h.Inputs[1].Parameters["binary_data_size"] != 4*85*4 {
		t.Fatalf("%+v", h.Inputs)
	}
	if len(body)-n != 4*208*4+4*85*4 {
		t.Fatalf("tensor bytes %d", len(body)-n)
	}
}

// The same generator, seed and distribution put the same tensors on the wire for both
// protocols.
func TestHTTPAndGRPCCarryIdenticalTensors(t *testing.T) {
	tpl := &Template{Model: "m", Batched: true, BatchSize: 3, Inputs: []Spec{spec("x", Int32, 5), spec("f", Fp32, 2), spec("s", Bytes, 2)}}
	g1, g2 := NewGenerator(42), NewGenerator(42)
	for i := 0; i < 4; i++ {
		pbReq, err := g1.Build(tpl)
		if err != nil {
			t.Fatal(err)
		}
		httpReq, err := g2.EncodeHTTP(tpl, "h:1")
		if err != nil {
			t.Fatal(err)
		}
		_, body := splitRequest(t, httpReq)
		var raw []byte
		for _, in := range pbReq.Inputs {
			raw = append(raw, in.Raw...)
		}
		if !bytes.HasSuffix(body, raw) {
			t.Fatalf("request %d: tensors differ", i)
		}
	}
}

func TestHTTPBytesInputUsesLengthPrefix(t *testing.T) {
	tpl := &Template{Model: "m", Batched: false, BatchSize: 1, Inputs: []Spec{spec("s", Bytes, 2)}}
	g := NewGenerator(1, WithDist(mustDist(t, "const:12")))
	req, _ := g.EncodeHTTP(tpl, "h")
	_, body := splitRequest(t, req)
	want := []byte{2, 0, 0, 0, '1', '2', 2, 0, 0, 0, '1', '2'}
	if !bytes.HasSuffix(body, want) {
		t.Fatalf("%q", body)
	}
	if !strings.Contains(string(body), `"binary_data_size":12`) {
		t.Fatalf("%s", body)
	}
}

func TestHTTPSequenceParameters(t *testing.T) {
	tpl := &Template{Model: "m", Batched: true, BatchSize: 1, Sequence: true, Inputs: []Spec{spec("x", Int32, 1)}}
	g := NewGenerator(1)
	if _, err := g.EncodeHTTP(tpl, "h"); err != nil { // takes sequence_id 1
		t.Fatal(err)
	}
	req, _ := g.EncodeHTTP(tpl, "h")
	_, body := splitRequest(t, req)
	if !strings.Contains(string(body), `"parameters":{"binary_data_output":true,"sequence_id":2,"sequence_start":true}`) {
		t.Fatalf("%s", body)
	}
}

// httpResp builds a response body: JSON header plus binary outputs.
func httpResp(js string, bin ...[]byte) (hdrLen int, body []byte) {
	body = []byte(js)
	for _, b := range bin {
		body = append(body, b...)
	}
	return len(js), body
}

func TestValidateHTTP(t *testing.T) {
	v := NewValidator([]Spec{spec("p", Fp32, 1)}, 4, true)
	good := `{"model_name":"m","outputs":[{"name":"p","datatype":"FP32","shape":[4,1],"parameters":{"binary_data_size":16}}]}`
	n, body := httpResp(good, make([]byte, 16))
	if err := v.ValidateHTTP(n, body); err != nil {
		t.Fatal(err)
	}
	// Extra outputs are fine.
	two := `{"outputs":[{"name":"p","datatype":"FP32","shape":[4,1],"parameters":{"binary_data_size":16}},` +
		`{"name":"q","datatype":"INT32","shape":[4],"parameters":{"binary_data_size":16}}]}`
	n, body = httpResp(two, make([]byte, 32))
	if err := v.ValidateHTTP(n, body); err != nil {
		t.Fatal(err)
	}
	bad := func(js string, bin int, kind MismatchKind) {
		t.Helper()
		n, body := httpResp(js, make([]byte, bin))
		wantKind(t, v.ValidateHTTP(n, body), kind)
	}
	ent := func(name, dt, shape string, size int) string {
		return fmt.Sprintf(`{"outputs":[{"name":%q,"datatype":%q,"shape":%s,"parameters":{"binary_data_size":%d}}]}`, name, dt, shape, size)
	}
	bad(ent("other", "FP32", "[4,1]", 16), 16, MismatchMissingOutput)
	bad(ent("p", "INT32", "[4,1]", 16), 16, MismatchDatatype)
	for _, sh := range []string{"[3,1]", "[4,2]", "[4]", "[4,1,1]"} {
		bad(ent("p", "FP32", sh, 16), 16, MismatchShape)
	}
	bad(ent("p", "FP32", "[4,1]", 12), 12, MismatchRawSize)                                 // wrong size for the shape
	bad(ent("p", "FP32", "[4,1]", 16), 12, MismatchRawSize)                                 // body shorter than declared
	bad(ent("p", "FP32", "[4,1]", 16), 20, MismatchRawSize)                                 // trailing bytes
	bad(`{"outputs":[{"name":"p","datatype":"FP32","shape":[4,1]}]}`, 16, MismatchRawCount) // no binary_data_size
	bad(`{"outputs":[{"name":"p","datatype":"FP32","shape":[4,1],"data":[0,0,0,0]}]}`, 0, MismatchRawCount)
	bad(`{"outputs":[`+strings.Repeat(`{"name":"p","datatype":"FP32","shape":[4,1],"parameters":{"binary_data_size":16}},`, 2)[:], 32, MismatchDecode)
	bad(`{"outputs":[{"name":"p","datatype":"FP32","shape":[4,1],"parameters":{"binary_data_size":16}},{"name":"p","datatype":"FP32","shape":[4,1],"parameters":{"binary_data_size":16}}]}`, 32, MismatchDuplicateOutput)
	bad(`not json`, 0, MismatchDecode)
	wantKind(t, v.ValidateHTTP(10000, []byte(good)), MismatchDecode) // header length past the body
	// Without the header, the whole body is JSON.
	wantKind(t, v.ValidateHTTP(-1, []byte(`{"outputs":[]}`)), MismatchMissingOutput)
}

func TestValidateHTTPBytesAndUnbatched(t *testing.T) {
	v := NewValidator([]Spec{spec("s", Bytes, 2)}, 1, false)
	js := `{"outputs":[{"name":"s","datatype":"BYTES","shape":[2],"parameters":{"binary_data_size":%d}}]}`
	okBin := []byte{1, 0, 0, 0, 'a', 0, 0, 0, 0}
	n, body := httpResp(fmt.Sprintf(js, len(okBin)), okBin)
	if err := v.ValidateHTTP(n, body); err != nil {
		t.Fatal(err)
	}
	badBin := []byte{9, 0, 0, 0, 'a'}
	n, body = httpResp(fmt.Sprintf(js, len(badBin)), badBin)
	wantKind(t, v.ValidateHTTP(n, body), MismatchRawBytes)
}

func TestModelFromJSON(t *testing.T) {
	meta := `{"name":"m","versions":["1"],"platform":"onnxruntime_onnx",
	  "inputs":[{"name":"x","datatype":"INT32","shape":[-1,208]},{"name":"mc","datatype":"INT32","shape":[-1,85]}],
	  "outputs":[{"name":"p","datatype":"FP32","shape":[-1,1]}]}`
	cfg := `{"name":"m","max_batch_size":16,"input":[]}`
	m, err := ModelFromJSON([]byte(meta), []byte(cfg), "m", "1")
	if err != nil {
		t.Fatal(err)
	}
	if !m.Batched || m.MaxBatchSize != 16 || m.Sequence || len(m.Inputs) != 2 || m.Inputs[0].Dims[0] != 208 ||
		m.Outputs[0].Name != "p" || m.Outputs[0].Dims[0] != 1 || m.Version != "1" {
		t.Fatalf("%+v", m)
	}
	if _, err := m.NewTemplate(nil, 17); err == nil {
		t.Error("batch size over max_batch_size accepted")
	}
	// No max_batch_size: the model does not batch and the shapes are its own.
	m, err = ModelFromJSON([]byte(`{"inputs":[{"name":"x","datatype":"FP32","shape":[3]}],"outputs":[]}`), []byte(`{"name":"m"}`), "m", "")
	if err != nil || m.Batched || m.Inputs[0].Dims[0] != 3 {
		t.Fatalf("%+v %v", m, err)
	}
	if _, err := m.NewTemplate(nil, 2); err == nil {
		t.Error("batch size 2 on a non-batching model accepted")
	}
	m, err = ModelFromJSON([]byte(meta), []byte(`{"max_batch_size":4,"sequence_batching":{"max_sequence_idle_microseconds":"5000000"}}`), "m", "")
	if err != nil || !m.Sequence {
		t.Fatalf("%+v %v", m, err)
	}
	if _, err := ModelFromJSON([]byte(`{`), []byte(cfg), "m", ""); err == nil {
		t.Error("bad metadata accepted")
	}
	if _, err := ModelFromJSON([]byte(meta), []byte(`[`), "m", ""); err == nil {
		t.Error("bad config accepted")
	}
}
