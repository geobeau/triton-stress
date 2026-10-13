package h1client

import (
	"time"

	"github.com/criteo/triton-stress/internal/grpcclient"
)

// headerContentLength is the header that gives the JSON part's length of a
// binary-extension body.
const headerContentLength = "Inference-Header-Content-Length"

// Request is one HTTP exchange. The caller owns it and may reuse it once Done has been
// called; the client keeps no reference after that.
type Request struct {
	// Set by the caller before Submit.

	// Body is the complete pre-built HTTP/1.1 request, header and body. It must not change
	// until Done.
	Body []byte
	// Keep asks for the response body to be copied into Data (whose capacity is reused).
	Keep bool
	// Timeout bounds the request from Submit to the end of the response, a wait for a
	// free connection included. When it expires the connection is closed (its state is
	// unknown) and the status is StatusTimeout.
	Timeout time.Duration
	// Done is called exactly once, from a client goroutine, or from Submit itself when the
	// request cannot be sent. It must not block for long.
	Done func(*Request)
	// Owner is for the caller's use.
	Owner any

	// Results, valid in Done.

	// Status is StatusHTTP(code) when a response arrived (StatusHTTP200 is success), or
	// StatusTimeout, StatusTransport, StatusConnect, StatusInvalidResponse.
	Status grpcclient.Status
	// TReady is when a free connection took the request; TSend is just before its bytes
	// were written; TDone is when the response was complete. All are clock.Now values;
	// zero means not reached.
	TReady, TSend, TDone int64
	// RespBytes is the size of the response body (0 when none arrived).
	RespBytes int
	// Conn is the index of the connection that carried the request, -1 when none did.
	Conn int
	// HdrLen is the response's Inference-Header-Content-Length, -1 when absent.
	HdrLen int
	// Data is the response body when Keep was set.
	Data []byte

	// Client state.
	deadline   int64
	c          *conn
	timedOut   bool
	inQ, inDL  bool
	qprev, qnx *Request
	dprev, dnx *Request
}

func (r *Request) reset() {
	r.Status, r.TReady, r.TSend, r.TDone, r.RespBytes, r.Conn, r.HdrLen = 0, 0, 0, 0, 0, -1, -1
	r.Data = r.Data[:0]
	r.deadline, r.c, r.timedOut, r.inQ, r.inDL = 0, nil, false, false, false
	r.qprev, r.qnx, r.dprev, r.dnx = nil, nil, nil, nil
}
