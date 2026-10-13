package grpcclient

import (
	"strconv"
	"strings"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// Status is the outcome of a call. Values 0..16 are the gRPC codes (a status
// the server sent, or a code grpc-go produced that is not recognised as a
// client-side failure); the rest are client-side. Values are dense, so a
// Status can index an array of NumStatuses counters.
type Status uint16

// Client-side statuses, following the 17 gRPC codes.
const (
	StatusOK                Status = Status(codes.OK)
	StatusCanceled          Status = Status(codes.Canceled)
	StatusUnknown           Status = Status(codes.Unknown)
	StatusPermissionDenied  Status = Status(codes.PermissionDenied)
	StatusResourceExhausted Status = Status(codes.ResourceExhausted)
	StatusUnimplemented     Status = Status(codes.Unimplemented)
	StatusInternal          Status = Status(codes.Internal)
	StatusUnavailable       Status = Status(codes.Unavailable)
	StatusUnauthenticated   Status = Status(codes.Unauthenticated)
)

const (
	StatusTimeout Status = 17 + iota
	StatusTransport
	StatusGoaway
	StatusConnect
	StatusInvalidResponse

	// StatusHTTPBase is the Status of HTTP code 100; the code c (100..599) is
	// StatusHTTPBase + c - 100. They exist for --protocol http, where the status
	// label is http_<code>.
	StatusHTTPBase

	// NumStatuses is the number of distinct Status values.
	NumStatuses = int(StatusHTTPBase) + 500
)

// StatusHTTP200 is the successful HTTP status.
const StatusHTTP200 = StatusHTTPBase + 100

// StatusHTTP is the Status of an HTTP response code; a code outside 100..599 is
// not a valid response and gives StatusInvalidResponse.
func StatusHTTP(code int) Status {
	if code < 100 || code > 599 {
		return StatusInvalidResponse
	}
	return StatusHTTPBase + Status(code-100)
}

var codeLabels = [17]string{
	"OK", "CANCELLED", "UNKNOWN", "INVALID_ARGUMENT", "DEADLINE_EXCEEDED", "NOT_FOUND",
	"ALREADY_EXISTS", "PERMISSION_DENIED", "RESOURCE_EXHAUSTED", "FAILED_PRECONDITION",
	"ABORTED", "OUT_OF_RANGE", "UNIMPLEMENTED", "INTERNAL", "UNAVAILABLE", "DATA_LOSS",
	"UNAUTHENTICATED",
}

// IsOK reports a successful call: gRPC OK or HTTP 200.
func (s Status) IsOK() bool { return s == StatusOK || s == StatusHTTP200 }

// Label is the status label used in output and metrics: the gRPC code name
// (OK, UNAVAILABLE, ...), http_<code> (HTTP), or timeout, transport, goaway, connect,
// invalid_response.
func (s Status) Label() string {
	switch {
	case s >= StatusHTTPBase && int(s) < NumStatuses:
		return "http_" + strconv.Itoa(int(s-StatusHTTPBase)+100)
	case int(s) < len(codeLabels):
		return codeLabels[s]
	case s == StatusTimeout:
		return "timeout"
	case s == StatusTransport:
		return "transport"
	case s == StatusGoaway:
		return "goaway"
	case s == StatusConnect:
		return "connect"
	case s == StatusInvalidResponse:
		return "invalid_response"
	}
	return "unknown"
}

func fromCode(c codes.Code) Status {
	if int(c) >= len(codeLabels) {
		return Status(codes.Unknown)
	}
	return Status(c)
}

// Classify maps an error returned by a grpc-go call to a Status. The mapping
// is described in the package documentation; the timeout case is decided by
// the caller, which knows whether its own timer fired.
func Classify(err error) Status {
	if err == nil {
		return StatusOK
	}
	st, ok := status.FromError(err)
	if !ok {
		return StatusTransport
	}
	msg := strings.ToLower(st.Message())
	switch st.Code() {
	case codes.Unavailable:
		switch {
		case strings.Contains(msg, "goaway") || strings.Contains(msg, "the connection is draining"):
			return StatusGoaway
		case containsAny(msg, "error while dialing", "dial tcp", "connection refused", "no route to host",
			"network is unreachable", "no such host", "i/o timeout", "name resolver error", "last connection error"):
			return StatusConnect
		case containsAny(msg, "error reading from server", "transport is closing", "connection error",
			"connection closed", "connection reset", "broken pipe", "stream terminated by rst_stream",
			"error writing", "closed network connection"):
			return StatusTransport
		}
	case codes.Canceled:
		if strings.Contains(msg, "client connection is closing") {
			return StatusTransport
		}
	}
	return fromCode(st.Code())
}

func containsAny(s string, subs ...string) bool {
	for _, sub := range subs {
		if strings.Contains(s, sub) {
			return true
		}
	}
	return false
}
