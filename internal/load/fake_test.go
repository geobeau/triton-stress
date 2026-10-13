package load

import (
	"context"
	"net"
	"sync/atomic"
	"testing"
	"time"

	"github.com/criteo/triton-stress/internal/grpcclient"
	"github.com/criteo/triton-stress/internal/h2client"
	"github.com/criteo/triton-stress/internal/payload"
	"github.com/criteo/triton-stress/pkg/tensorgen"
	pb "github.com/criteo/triton-stress/pkg/tritonpb"
	"google.golang.org/grpc"
	"google.golang.org/grpc/test/bufconn"
)

// fakeTriton is an in-process Triton for tests (not part of the binary): ModelInfer
// answers after delay, or blocks until release is closed.
type fakeTriton struct {
	pb.UnimplementedGRPCInferenceServiceServer
	delay    atomic.Int64 // ns
	block    chan struct{}
	received atomic.Int64
}

func (f *fakeTriton) ModelInfer(ctx context.Context, req *pb.ModelInferRequest) (*pb.ModelInferResponse, error) {
	f.received.Add(1)
	if f.block != nil {
		select {
		case <-f.block:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	if d := f.delay.Load(); d > 0 {
		time.Sleep(time.Duration(d))
	}
	batch := req.Inputs[0].Shape[0]
	return &pb.ModelInferResponse{
		Outputs:           []*pb.ModelInferResponse_InferOutputTensor{{Name: "p", Datatype: "FP32", Shape: []int64{batch, 1}}},
		RawOutputContents: [][]byte{make([]byte, 4*batch)},
	}, nil
}

func (f *fakeTriton) ModelMetadata(_ context.Context, _ *pb.ModelMetadataRequest) (*pb.ModelMetadataResponse, error) {
	return &pb.ModelMetadataResponse{
		Name:    "m",
		Inputs:  []*pb.ModelMetadataResponse_TensorMetadata{{Name: "x", Datatype: "INT32", Shape: []int64{-1, 8}}},
		Outputs: []*pb.ModelMetadataResponse_TensorMetadata{{Name: "p", Datatype: "FP32", Shape: []int64{-1, 1}}},
	}, nil
}

func (f *fakeTriton) ModelConfig(_ context.Context, r *pb.ModelConfigRequest) (*pb.ModelConfigResponse, error) {
	return &pb.ModelConfigResponse{Config: &pb.ModelConfig{Name: r.Name, MaxBatchSize: 16}}, nil
}

type rig struct {
	f     *fakeTriton
	conns []*h2client.Conn
	meta  *grpcclient.Conn // grpc-go, for the model description only
	pool  *payload.Pool
	val   *tensorgen.Validator
	lis   *bufconn.Listener
	srv   *grpc.Server
}

// newRig starts the fake and n connected Conns. With up false the server is not started,
// so no connection ever becomes Ready.
func newRig(t *testing.T, n int, up bool, f *fakeTriton) *rig {
	t.Helper()
	return newRigOpts(t, n, up, f)
}

// newRigOpts is newRig with options for the server.
func newRigOpts(t *testing.T, n int, up bool, f *fakeTriton, opts ...grpc.ServerOption) *rig {
	t.Helper()
	rg := &rig{f: f, lis: bufconn.Listen(1 << 20), srv: grpc.NewServer(opts...)}
	pb.RegisterGRPCInferenceServiceServer(rg.srv, f)
	if up {
		go func() { _ = rg.srv.Serve(rg.lis) }()
	}
	t.Cleanup(func() {
		for _, c := range rg.conns {
			_ = c.Close()
		}
		if rg.meta != nil {
			_ = rg.meta.Close()
		}
		rg.srv.Stop()
	})
	dial := func(ctx context.Context, _ string) (net.Conn, error) { return rg.lis.DialContext(ctx) }
	for i := 0; i < n; i++ {
		c, err := h2client.New(h2client.Options{Target: "bufnet:80", Dialer: dial,
			BackoffMin: 5 * time.Millisecond, BackoffMax: 50 * time.Millisecond})
		if err != nil {
			t.Fatal(err)
		}
		rg.conns = append(rg.conns, c)
	}
	if up {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		for _, c := range rg.conns {
			if err := c.WaitReady(ctx); err != nil {
				t.Fatal(err)
			}
		}
		var err error
		rg.meta, err = grpcclient.New(grpcclient.Options{Target: "passthrough:///bufnet", Dialer: dial})
		if err != nil {
			t.Fatal(err)
		}
		if err := rg.meta.WaitReady(ctx); err != nil {
			t.Fatal(err)
		}
		m, err := rg.meta.FetchModel(ctx, "m", "")
		if err != nil {
			t.Fatal(err)
		}
		tpl, err := m.NewTemplate(nil, 2)
		if err != nil {
			t.Fatal(err)
		}
		rg.pool, err = payload.Build(tensorgen.NewGenerator(1), tpl, 16)
		if err != nil {
			t.Fatal(err)
		}
		rg.val = tensorgen.NewValidator(m.Outputs, 2, m.Batched)
	} else {
		rg.pool = &payload.Pool{}
	}
	return rg
}
