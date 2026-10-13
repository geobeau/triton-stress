package h2client

import (
	"time"

	"github.com/criteo/triton-stress/internal/grpcclient"
)

// Request is one unary gRPC call. The caller owns it and may reuse it once Done has
// been called; the transport keeps no reference after that.
type Request struct {
	// Set by the caller before Submit.

	// Body is the pre-encoded request message (without the gRPC framing). It must
	// not change until Done.
	Body []byte
	// Keep asks for the response message to be copied into Data.
	Keep bool
	// Timeout bounds the request from Submit to the end of the response; when it
	// expires the stream is reset and the status is StatusTimeout. No deadline is
	// sent to the server.
	Timeout time.Duration
	// Done is called exactly once, from a transport goroutine, or from Submit itself
	// when the request cannot be sent. It must not block for long.
	Done func(*Request)
	// Owner is for the caller's use.
	Owner any

	// Results, valid in Done.

	Status grpcclient.Status
	// TReady is when the request found capacity (a free stream and window); TSend is
	// when its frames were handed to the connection's write buffer; TDone is when the
	// response was complete. All are clock.Now values; zero means not reached.
	TReady, TSend, TDone int64
	// RespBytes is the size of the response message (0 when none arrived).
	RespBytes int
	// Data is the response message when Keep was set.
	Data []byte

	// Transport state.
	id         uint32
	sent       int // bytes of the framed message (5-byte prefix + Body) written so far
	sendWin    int64
	deadline   int64
	recvAcc    int64 // response bytes since the last stream window update
	msgSeen    int   // response bytes seen, the 5-byte message prefix included
	badMsg     bool  // the response message is compressed: not understood
	httpStatus int
	started    bool
	finished   bool
	inSendQ    bool
	inDL       bool
	sprev      *Request
	snext      *Request
	dprev      *Request
	dnext      *Request
}

func (r *Request) reset() {
	r.Status, r.TReady, r.TSend, r.TDone, r.RespBytes, r.Data = 0, 0, 0, 0, 0, nil
	r.id, r.sent, r.sendWin, r.deadline, r.recvAcc, r.msgSeen = 0, 0, 0, 0, 0, 0
	r.badMsg, r.httpStatus = false, 0
	r.started, r.finished, r.inSendQ, r.inDL = false, false, false, false
	r.sprev, r.snext, r.dprev, r.dnext = nil, nil, nil, nil
}
