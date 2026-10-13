package payload

import (
	"bytes"
	"testing"

	"github.com/criteo/triton-stress/pkg/tensorgen"
	pb "github.com/criteo/triton-stress/pkg/tritonpb"
	"google.golang.org/protobuf/proto"
)

func tpl(batch int, inputs ...tensorgen.Spec) *tensorgen.Template {
	return &tensorgen.Template{Model: "m", Inputs: inputs, BatchSize: batch, Batched: true}
}

func zeros() *tensorgen.Generator {
	return tensorgen.NewGenerator(1, tensorgen.WithDist(tensorgen.Dist{Kind: tensorgen.DistZeros}))
}

func TestPoolBasics(t *testing.T) {
	x := tensorgen.Spec{Name: "x", DType: tensorgen.Int32, Dims: []int64{208}}
	m := tensorgen.Spec{Name: "m", DType: tensorgen.Int32, Dims: []int64{85}}
	p, err := Build(tensorgen.NewGenerator(1), tpl(4, x, m), 8)
	if err != nil {
		t.Fatal(err)
	}
	if p.Len() != 8 || p.TotalBytes() != 8*len(p.Get(0)) {
		t.Fatalf("len %d total %d", p.Len(), p.TotalBytes())
	}
	lo, hi := p.SizeRange()
	if lo != hi || lo < 4*(208+85)*4 {
		t.Fatalf("size range %d..%d", lo, hi)
	}
	var req pb.ModelInferRequest
	if err := proto.Unmarshal(p.Get(3), &req); err != nil || req.Inputs[0].Shape[0] != 4 {
		t.Fatalf("%v %+v", err, &req)
	}
	// Wrap-around, and requests are distinct.
	if !bytes.Equal(p.Get(3), p.Get(11)) || bytes.Equal(p.Get(0), p.Get(1)) {
		t.Fatal("wrap-around or distinctness broken")
	}
}

func TestPoolDeterministic(t *testing.T) {
	x := tensorgen.Spec{Name: "x", DType: tensorgen.Int32, Dims: []int64{16}}
	a, _ := Build(tensorgen.NewGenerator(1), tpl(1, x), 3)
	b, _ := Build(tensorgen.NewGenerator(1), tpl(1, x), 3)
	c, _ := Build(tensorgen.NewGenerator(2), tpl(1, x), 3)
	if !bytes.Equal(a.Get(2), b.Get(2)) || bytes.Equal(a.Get(0), c.Get(0)) {
		t.Fatal("seed handling")
	}
}

func TestPoolBytesInputsVaryInSize(t *testing.T) {
	s := tensorgen.Spec{Name: "s", DType: tensorgen.Bytes, Dims: []int64{50}}
	p, err := Build(tensorgen.NewGenerator(1), tpl(1, s), 16)
	if err != nil {
		t.Fatal(err)
	}
	if lo, hi := p.SizeRange(); lo >= hi {
		t.Fatalf("BYTES requests should differ in size: %d..%d", lo, hi)
	}
}

func TestPoolCapAndErrors(t *testing.T) {
	// 64 MiB per request: 9 of them are over the 512 MiB cap, 7 fit.
	big := tensorgen.Spec{Name: "big", DType: tensorgen.Uint8, Dims: []int64{64 << 20}}
	if _, err := Build(zeros(), tpl(1, big), 9); err == nil {
		t.Error("pool over the cap accepted")
	}
	p, err := Build(zeros(), tpl(1, big), 7)
	if err != nil {
		t.Fatal(err)
	}
	if p.Len() != 7 || p.TotalBytes() > MaxPoolBytes {
		t.Fatalf("%d requests, %d bytes", p.Len(), p.TotalBytes())
	}
	huge := tensorgen.Spec{Name: "huge", DType: tensorgen.Uint8, Dims: []int64{520 << 20}}
	if _, err := Build(zeros(), tpl(1, huge), 1); err == nil {
		t.Error("a single request over the cap accepted")
	}
	v := tensorgen.Spec{Name: "v", DType: tensorgen.Int32, Dims: []int64{-1}}
	if _, err := Build(zeros(), tpl(1, v), 1); err == nil {
		t.Error("variable dim accepted")
	}
	if _, err := Build(zeros(), tpl(1, big), 0); err == nil {
		t.Error("pool size 0 accepted")
	}
}
