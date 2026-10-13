package grpcclient

import (
	"context"
	"sync/atomic"
	"time"

	"github.com/criteo/triton-stress/internal/clock"
	"github.com/criteo/triton-stress/pkg/tensorgen"
)

// InferMethod is the full gRPC method name of ModelInfer.
const InferMethod = "/inference.GRPCInferenceService/ModelInfer"

// Result is the outcome of one Infer call. Timestamps are clock.Now values
// (monotonic ns); 0 means the call never got that far.
type Result struct {
	Status Status
	// TReady: the connection was found Ready and the call is being prepared.
	TReady int64
	// TSend: just before the request is handed to grpc-go.
	TSend int64
	// TDone: the response and trailers are complete (before validation).
	TDone int64
	// RespBytes is the size of the response message (0 when none arrived).
	RespBytes int
}

// Infer sends one pre-encoded ModelInferRequest and waits for the response.
//
// timeout is enforced client-side with a timer that cancels the call; it is
// not sent as a grpc-timeout header, so Triton sees no deadline. A Conn that
// is not Ready fails at once as StatusConnect (nothing waits for a reconnect).
// With a non-nil v, an OK response is decoded and validated after TDone, and a
// mismatch is StatusInvalidResponse.
func (c *Conn) Infer(ctx context.Context, body []byte, timeout time.Duration, v *tensorgen.Validator) Result {
	var r Result
	if !c.Ready() {
		r.Status = StatusConnect
		return r
	}
	r.TReady = clock.Now()

	cctx, cancel := context.WithCancel(ctx)
	var fired atomic.Bool
	timer := time.AfterFunc(timeout, func() {
		fired.Store(true)
		cancel()
	})
	req := RawRequest{Body: body}
	resp := RawResponse{Keep: v != nil}
	r.TSend = clock.Now()
	err := c.cc.Invoke(cctx, InferMethod, &req, &resp, c.inferOpts...)
	r.TDone = clock.Now()
	timer.Stop()
	cancel()

	switch {
	case err == nil:
		r.Status = StatusOK
		r.RespBytes = resp.Size
		if v != nil {
			if v.ValidateBytes(resp.Data) != nil {
				r.Status = StatusInvalidResponse
			}
		}
	case fired.Load():
		r.Status = StatusTimeout
	default:
		r.Status = Classify(err)
	}
	return r
}
