// Package grpcclient is RTS's Triton gRPC client, built on grpc-go.
//
// The load generator sends its requests through internal/h2client, a minimal HTTP/2
// client that costs a fraction of grpc-go's CPU per request (see its package comment
// and the benchmarks there). This package serves the startup RPCs (FetchModel), defines
// the Status type shared by both clients, and keeps grpc-go's Infer as the reference
// the h2client benchmarks and tests compare against.
//
// A Conn wraps one grpc.ClientConn, and RTS opens one Conn per configured
// connection (a ClientConn with the default pick_first policy and a single
// target address holds one HTTP/2 connection).
//
// # Connection behaviour
//
//   - Plaintext, or TLS with system roots, or TLS without verification (TLSMode).
//     grpc-go's TLS credentials add ALPN "h2".
//   - Fixed HTTP/2 flow-control windows are set, which switches grpc-go's BDP
//     estimator (and its BDP PING frames) off; keepalive is not configured, so
//     the client sends no PINGs at all.
//   - Proxy environment variables are ignored (WithNoProxy), service configs
//     are ignored, and the dial backoff is Options.BackoffMin..BackoffMax
//     (multiplier 2, jitter 0.2). grpc-go returns a ClientConn whose connection
//     was lost to IDLE; the Conn watcher calls Connect at once, so the first
//     reconnect attempt is immediate and later ones back off.
//   - Conn.Ready is a lock-free read of the connectivity state kept by a
//     watcher goroutine. A load dispatcher must send only on a Ready Conn
//     (Infer itself returns "connect" immediately for a Conn that is not Ready,
//     rather than letting grpc-go block the call while it is connecting).
//
// # Pass-through codec
//
// Infer sends a pre-encoded ModelInferRequest through a grpc.ForceCodecV2
// codec that frames the given bytes without copying them, and hands back the
// response bytes undecoded (and not even copied unless the call validates).
// See RawCodec.
//
// # Status labels
//
// Every call ends in a Status, whose Label is, as in the RTS plan: the gRPC
// code name in upper case (OK, UNAVAILABLE, ...) when the server answered, or
// one of the client-side names:
//
//	timeout           --request-timeout expired; the call was cancelled. No
//	                  grpc-timeout header is sent, so Triton sees no deadline.
//	transport         the connection broke while the request was in flight
//	                  (reset, EOF, transport closing, RST_STREAM, ...).
//	goaway            the server sent GOAWAY and the call was refused or cut
//	                  (the connection is draining).
//	connect           the Conn was not Ready when the call was dispatched, or
//	                  grpc-go failed to dial for it.
//	invalid_response  the response did not decode or failed validation.
//
// grpc-go reports client-side failures as ordinary status errors with code
// UNAVAILABLE, indistinguishable by code from an UNAVAILABLE sent by Triton
// (for example a queue rejection). Classify tells them apart by the error
// text that grpc-go's transport produces. That is best effort and depends on
// grpc-go's wording: an unrecognised UNAVAILABLE is reported as UNAVAILABLE.
package grpcclient
