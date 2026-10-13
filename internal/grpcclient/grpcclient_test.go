package grpcclient

import (
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"errors"
	"math/big"
	"net"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/criteo/triton-stress/internal/clock"
	"github.com/criteo/triton-stress/pkg/tensorgen"
	pb "github.com/criteo/triton-stress/pkg/tritonpb"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/test/bufconn"
	"google.golang.org/protobuf/proto"
)

// fake is an in-process Triton: ModelInfer returns reply after delay (or blocks
// when block is set) and records what it received.
type fake struct {
	pb.UnimplementedGRPCInferenceServiceServer
	mu          sync.Mutex
	reply       *pb.ModelInferResponse
	err         error
	delay       time.Duration
	block       chan struct{}
	seen        []*pb.ModelInferRequest
	sawDeadline atomic.Bool
	inflight    chan struct{}
}

func (f *fake) ModelInfer(ctx context.Context, req *pb.ModelInferRequest) (*pb.ModelInferResponse, error) {
	if _, ok := ctx.Deadline(); ok {
		f.sawDeadline.Store(true)
	}
	f.mu.Lock()
	f.seen = append(f.seen, req)
	reply, err, delay, block := f.reply, f.err, f.delay, f.block
	f.mu.Unlock()
	if f.inflight != nil {
		select {
		case f.inflight <- struct{}{}:
		default:
		}
	}
	if block != nil {
		select {
		case <-block:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	if delay > 0 {
		time.Sleep(delay)
	}
	return reply, err
}

func (f *fake) ModelMetadata(_ context.Context, r *pb.ModelMetadataRequest) (*pb.ModelMetadataResponse, error) {
	if r.Name != "m" {
		return nil, status.Error(codes.NotFound, "no such model")
	}
	return &pb.ModelMetadataResponse{
		Name: "m",
		Inputs: []*pb.ModelMetadataResponse_TensorMetadata{
			{Name: "x", Datatype: "INT32", Shape: []int64{-1, 8}},
		},
		Outputs: []*pb.ModelMetadataResponse_TensorMetadata{
			{Name: "p", Datatype: "FP32", Shape: []int64{-1, 1}},
		},
	}, nil
}

func (f *fake) ModelConfig(_ context.Context, r *pb.ModelConfigRequest) (*pb.ModelConfigResponse, error) {
	return &pb.ModelConfigResponse{Config: &pb.ModelConfig{Name: r.Name, MaxBatchSize: 16}}, nil
}

// bufServer serves f on a bufconn listener that tests can stop and replace.
type bufServer struct {
	mu  sync.Mutex
	lis *bufconn.Listener
	srv *grpc.Server
	f   *fake
}

func newBufServer(t *testing.T, f *fake) *bufServer {
	t.Helper()
	b := &bufServer{f: f}
	b.start()
	t.Cleanup(b.stop)
	return b
}

func (b *bufServer) start() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.lis = bufconn.Listen(1 << 20)
	b.srv = grpc.NewServer()
	pb.RegisterGRPCInferenceServiceServer(b.srv, b.f)
	go func() { _ = b.srv.Serve(b.lis) }()
}

func (b *bufServer) stop() {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.srv != nil {
		b.srv.Stop()
		b.srv = nil
	}
}

func (b *bufServer) dial(ctx context.Context, _ string) (net.Conn, error) {
	b.mu.Lock()
	lis := b.lis
	b.mu.Unlock()
	return lis.DialContext(ctx)
}

func newConn(t *testing.T, b *bufServer, mod func(*Options)) *Conn {
	t.Helper()
	o := Options{Target: "passthrough:///bufnet", Dialer: b.dial, BackoffMin: 5 * time.Millisecond, BackoffMax: 50 * time.Millisecond}
	if mod != nil {
		mod(&o)
	}
	c, err := New(o)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = c.Close() })
	return c
}

func waitReady(t *testing.T, c *Conn) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := c.WaitReady(ctx); err != nil {
		t.Fatal(err)
	}
}

func response(batch int64, rawLen int) *pb.ModelInferResponse {
	return &pb.ModelInferResponse{
		Outputs:           []*pb.ModelInferResponse_InferOutputTensor{{Name: "p", Datatype: "FP32", Shape: []int64{batch, 1}}},
		RawOutputContents: [][]byte{make([]byte, rawLen)},
	}
}

func testModelAndBody(t *testing.T, c *Conn, batch int) (*tensorgen.Model, []byte, *tensorgen.Validator) {
	t.Helper()
	m, err := c.FetchModel(context.Background(), "m", "")
	if err != nil {
		t.Fatal(err)
	}
	tpl, err := m.NewTemplate(nil, batch)
	if err != nil {
		t.Fatal(err)
	}
	body, err := tensorgen.NewGenerator(1).Encode(tpl)
	if err != nil {
		t.Fatal(err)
	}
	return m, body, tensorgen.NewValidator(m.Outputs, batch, m.Batched)
}

func TestFetchModelAndCodecRoundTrip(t *testing.T) {
	f := &fake{reply: response(4, 16)}
	c := newConn(t, newBufServer(t, f), nil)
	waitReady(t, c)
	m, body, v := testModelAndBody(t, c, 4)
	if !m.Batched || m.MaxBatchSize != 16 || m.Inputs[0].Dims[0] != 8 || m.Outputs[0].Name != "p" {
		t.Fatalf("%+v", m)
	}
	if _, err := c.FetchModel(context.Background(), "nope", ""); err == nil {
		t.Fatal("unknown model accepted")
	}

	r := c.Infer(context.Background(), body, time.Second, v)
	if r.Status != StatusOK {
		t.Fatalf("status %s", r.Status.Label())
	}
	if r.TReady == 0 || r.TReady > r.TSend || r.TSend > r.TDone {
		t.Fatalf("timestamps %+v", r)
	}
	// The server decoded exactly the pre-encoded bytes.
	var want pb.ModelInferRequest
	if err := proto.Unmarshal(body, &want); err != nil {
		t.Fatal(err)
	}
	f.mu.Lock()
	got := f.seen[0]
	f.mu.Unlock()
	if !proto.Equal(got, &want) {
		t.Fatal("server saw a different request")
	}
	if f.sawDeadline.Load() {
		t.Fatal("a deadline (grpc-timeout) reached the server")
	}

	// The codec returns the raw response bytes.
	var raw RawResponse
	raw.Keep = true
	if err := c.ClientConn().Invoke(context.Background(), InferMethod, &RawRequest{Body: body}, &raw, c.inferOpts...); err != nil {
		t.Fatal(err)
	}
	wantBytes, _ := proto.Marshal(f.reply)
	if raw.Size != len(wantBytes) || len(raw.Data) != len(wantBytes) {
		t.Fatalf("raw response %d/%d bytes, want %d", raw.Size, len(raw.Data), len(wantBytes))
	}
	var back pb.ModelInferResponse
	if err := proto.Unmarshal(raw.Data, &back); err != nil || !proto.Equal(&back, f.reply) {
		t.Fatalf("raw bytes do not decode to the reply: %v", err)
	}
	// Without Keep, only the size is reported.
	raw = RawResponse{}
	if err := c.ClientConn().Invoke(context.Background(), InferMethod, &RawRequest{Body: body}, &raw, c.inferOpts...); err != nil || raw.Data != nil || raw.Size != len(wantBytes) {
		t.Fatalf("%+v %v", raw, err)
	}
}

func TestCodecRejectsWrongTypes(t *testing.T) {
	if _, err := (RawCodec{}).Marshal("x"); err == nil {
		t.Error("Marshal accepted a string")
	}
	if err := (RawCodec{}).Unmarshal(nil, new(string)); err == nil {
		t.Error("Unmarshal accepted *string")
	}
	if (RawCodec{}).Name() != "proto" {
		t.Error("codec name")
	}
}

func TestInferStatuses(t *testing.T) {
	f := &fake{reply: response(1, 4)}
	c := newConn(t, newBufServer(t, f), nil)
	waitReady(t, c)
	_, body, v := testModelAndBody(t, c, 1)
	ctx := context.Background()

	if r := c.Infer(ctx, body, time.Second, nil); r.Status != StatusOK {
		t.Fatal(r.Status.Label())
	}
	// A response failing validation.
	f.mu.Lock()
	f.reply = response(3, 12)
	f.mu.Unlock()
	if r := c.Infer(ctx, body, time.Second, v); r.Status != StatusInvalidResponse || r.TDone == 0 {
		t.Fatalf("%s", r.Status.Label())
	}
	// Without validation the same response is OK.
	if r := c.Infer(ctx, body, time.Second, nil); r.Status != StatusOK {
		t.Fatal(r.Status.Label())
	}
	// Statuses the server sends keep their code name, UNAVAILABLE included.
	for code, label := range map[codes.Code]string{
		codes.Unavailable: "UNAVAILABLE", codes.ResourceExhausted: "RESOURCE_EXHAUSTED", codes.Internal: "INTERNAL", codes.InvalidArgument: "INVALID_ARGUMENT",
	} {
		f.mu.Lock()
		f.reply, f.err = nil, status.Error(code, "Exceeds maximum queue size")
		f.mu.Unlock()
		if r := c.Infer(ctx, body, time.Second, nil); r.Status.Label() != label {
			t.Errorf("server %s reported as %s", label, r.Status.Label())
		}
	}
	// Client-side timeout: the handler blocks, the call is cancelled.
	f.mu.Lock()
	f.err, f.block = nil, make(chan struct{})
	f.mu.Unlock()
	start := time.Now()
	r := c.Infer(ctx, body, 50*time.Millisecond, nil)
	if r.Status != StatusTimeout || time.Since(start) > 2*time.Second || r.TSend == 0 {
		t.Fatalf("timeout: %s after %v", r.Status.Label(), time.Since(start))
	}
	if f.sawDeadline.Load() {
		t.Fatal("a deadline reached the server")
	}
	// A cancelled caller context is CANCELLED, not timeout.
	cctx, cancel := context.WithCancel(ctx)
	time.AfterFunc(20*time.Millisecond, cancel)
	if r := c.Infer(cctx, body, 5*time.Second, nil); r.Status.Label() != "CANCELLED" {
		t.Fatalf("got %s", r.Status.Label())
	}
}

func TestTransportErrorAndConnectWhenDown(t *testing.T) {
	f := &fake{reply: response(1, 4), block: make(chan struct{}), inflight: make(chan struct{}, 1)}
	b := newBufServer(t, f)
	c := newConn(t, b, nil)
	waitReady(t, c)
	_, body, _ := testModelAndBody(t, c, 1)

	res := make(chan Result, 1)
	go func() { res <- c.Infer(context.Background(), body, 10*time.Second, nil) }()
	<-f.inflight
	b.stop() // the connection dies mid-flight
	select {
	case r := <-res:
		if r.Status != StatusTransport {
			t.Fatalf("mid-flight failure reported as %s", r.Status.Label())
		}
	case <-time.After(5 * time.Second):
		t.Fatal("call did not complete after the server stopped")
	}
	// While the server is gone the Conn is not Ready and requests fail at once as connect.
	deadline := time.Now().Add(5 * time.Second)
	for c.Ready() && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	start := time.Now()
	r := c.Infer(context.Background(), body, 10*time.Second, nil)
	if r.Status != StatusConnect || time.Since(start) > 100*time.Millisecond || r.TReady != 0 || r.TSend != 0 {
		t.Fatalf("got %s %+v after %v", r.Status.Label(), r, time.Since(start))
	}
}

func TestReconnectAndStateEvents(t *testing.T) {
	f := &fake{reply: response(1, 4)}
	b := newBufServer(t, f)
	var mu sync.Mutex
	var events []connectivity.State
	c := newConn(t, b, func(o *Options) {
		o.OnState = func(_, to connectivity.State) {
			mu.Lock()
			events = append(events, to)
			mu.Unlock()
		}
	})
	waitReady(t, c)
	b.stop()
	deadline := time.Now().Add(5 * time.Second)
	for c.Ready() && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if c.Ready() {
		t.Fatal("still Ready after the server stopped")
	}
	b.start() // the server comes back: the Conn reconnects by itself
	waitReady(t, c)
	_, body, _ := testModelAndBody(t, c, 1)
	if r := c.Infer(context.Background(), body, time.Second, nil); r.Status != StatusOK {
		t.Fatal(r.Status.Label())
	}
	mu.Lock()
	defer mu.Unlock()
	seenReady := 0
	for _, e := range events {
		if e == connectivity.Ready {
			seenReady++
		}
	}
	if seenReady < 2 {
		t.Fatalf("events %v: expected Ready at least twice", events)
	}
}

func TestNeverReachable(t *testing.T) {
	c, err := New(Options{
		Target:     "passthrough:///x",
		Dialer:     func(context.Context, string) (net.Conn, error) { return nil, errors.New("connection refused") },
		BackoffMin: 5 * time.Millisecond, BackoffMax: 20 * time.Millisecond,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c.Close() }()
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	if err := c.WaitReady(ctx); err == nil {
		t.Fatal("WaitReady succeeded")
	}
	if r := c.Infer(context.Background(), []byte("x"), time.Second, nil); r.Status != StatusConnect {
		t.Fatalf("got %s", r.Status.Label())
	}
}

func TestOptionsValidation(t *testing.T) {
	if _, err := New(Options{Target: "x:1", BackoffMin: time.Second, BackoffMax: time.Millisecond}); err == nil {
		t.Error("max < min accepted")
	}
	if _, err := New(Options{Target: "x:1", TLS: 9}); err == nil {
		t.Error("bad TLS mode accepted")
	}
}

func selfSigned(t *testing.T) tls.Certificate {
	t.Helper()
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	tmpl := &x509.Certificate{
		SerialNumber: big.NewInt(1), Subject: pkix.Name{CommonName: "localhost"},
		NotBefore: time.Now().Add(-time.Hour), NotAfter: time.Now().Add(time.Hour),
		DNSNames: []string{"localhost"}, IPAddresses: []net.IP{net.ParseIP("127.0.0.1")},
		KeyUsage: x509.KeyUsageDigitalSignature, ExtKeyUsage: []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
	}
	der, err := x509.CreateCertificate(rand.Reader, tmpl, tmpl, &key.PublicKey, key)
	if err != nil {
		t.Fatal(err)
	}
	return tls.Certificate{Certificate: [][]byte{der}, PrivateKey: key}
}

func TestTLSModes(t *testing.T) {
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	f := &fake{reply: response(1, 4)}
	srv := grpc.NewServer(grpc.Creds(credentials.NewServerTLSFromCert(ptr(selfSigned(t)))))
	pb.RegisterGRPCInferenceServiceServer(srv, f)
	go func() { _ = srv.Serve(lis) }()
	t.Cleanup(srv.Stop)
	target := lis.Addr().String()

	// Unverified TLS connects and serves calls (ALPN h2 comes from grpc-go's credentials).
	c, err := New(Options{Target: target, TLS: TLSInsecure})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c.Close() }()
	waitReady(t, c)
	_, body, _ := testModelAndBody(t, c, 1)
	if r := c.Infer(context.Background(), body, time.Second, nil); r.Status != StatusOK {
		t.Fatal(r.Status.Label())
	}

	// System roots do not trust the self-signed certificate.
	c2, _ := New(Options{Target: target, TLS: TLSSystem})
	defer func() { _ = c2.Close() }()
	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Millisecond)
	defer cancel()
	if err := c2.WaitReady(ctx); err == nil {
		t.Fatal("self-signed certificate accepted with system roots")
	}
	// Plaintext against a TLS server never gets ready either.
	c3, _ := New(Options{Target: target})
	defer func() { _ = c3.Close() }()
	ctx2, cancel2 := context.WithTimeout(context.Background(), 300*time.Millisecond)
	defer cancel2()
	if err := c3.WaitReady(ctx2); err == nil {
		t.Fatal("plaintext client got ready against a TLS server")
	}
}

func ptr[T any](v T) *T { return &v }

func TestStatusLabels(t *testing.T) {
	seenHTTP := map[Status]bool{}
	want := map[Status]string{
		StatusOK: "OK", Status(codes.Canceled): "CANCELLED", Status(codes.Unknown): "UNKNOWN",
		Status(codes.InvalidArgument): "INVALID_ARGUMENT", Status(codes.DeadlineExceeded): "DEADLINE_EXCEEDED",
		Status(codes.NotFound): "NOT_FOUND", Status(codes.AlreadyExists): "ALREADY_EXISTS",
		Status(codes.PermissionDenied): "PERMISSION_DENIED", Status(codes.ResourceExhausted): "RESOURCE_EXHAUSTED",
		Status(codes.FailedPrecondition): "FAILED_PRECONDITION", Status(codes.Aborted): "ABORTED",
		Status(codes.OutOfRange): "OUT_OF_RANGE", Status(codes.Unimplemented): "UNIMPLEMENTED",
		Status(codes.Internal): "INTERNAL", Status(codes.Unavailable): "UNAVAILABLE", Status(codes.DataLoss): "DATA_LOSS",
		Status(codes.Unauthenticated): "UNAUTHENTICATED",
		StatusTimeout:                 "timeout", StatusTransport: "transport", StatusGoaway: "goaway",
		StatusConnect: "connect", StatusInvalidResponse: "invalid_response",
	}
	if len(want) != int(StatusHTTPBase) || NumStatuses != int(StatusHTTPBase)+500 {
		t.Fatalf("%d labels for %d client and gRPC statuses", len(want), StatusHTTPBase)
	}
	// HTTP codes 100..599 follow, each with its own label.
	for code := 100; code <= 599; code++ {
		s := StatusHTTP(code)
		if int(s) >= NumStatuses || s.Label() != "http_"+strconv.Itoa(code) || seenHTTP[s] {
			t.Fatalf("code %d: status %d label %q", code, s, s.Label())
		}
		seenHTTP[s] = true
	}
	if !StatusHTTP(200).IsOK() || StatusHTTP(201).IsOK() || StatusHTTP(503).IsOK() || StatusHTTP(200) != StatusHTTP200 {
		t.Error("only HTTP 200 is OK")
	}
	if StatusHTTP(99) != StatusInvalidResponse || StatusHTTP(600) != StatusInvalidResponse {
		t.Error("codes outside 100..599 are not valid responses")
	}
	seen := map[string]bool{}
	for s, l := range want {
		if s.Label() != l {
			t.Errorf("%d: %q, want %q", s, s.Label(), l)
		}
		if seen[l] {
			t.Errorf("duplicate label %s", l)
		}
		seen[l] = true
		if int(s) >= NumStatuses {
			t.Errorf("%s outside the dense range", l)
		}
	}
	if !StatusOK.IsOK() || StatusTimeout.IsOK() {
		t.Error("IsOK")
	}
	// Each code name is its code in upper snake case.
	for c := codes.OK; c <= codes.Unauthenticated; c++ {
		if Status(c).Label() == "" || Status(c).Label() == "unknown" {
			t.Errorf("code %d has no label", c)
		}
	}
	if fromCode(codes.Code(99)) != Status(codes.Unknown) {
		t.Error("unknown codes map to UNKNOWN")
	}
}

func TestClassify(t *testing.T) {
	u := func(msg string) error { return status.Error(codes.Unavailable, msg) }
	cases := []struct {
		err  error
		want string
	}{
		{nil, "OK"},
		{u("Exceeds maximum queue size"), "UNAVAILABLE"},
		{u("connection error: desc = \"transport: Error while dialing: dial tcp 127.0.0.1:1: connect: connection refused\""), "connect"},
		{u("last connection error: dial tcp: lookup nohost: no such host"), "connect"},
		{u("connection error: desc = \"transport: Error while dialing: dial tcp 10.0.0.1:8001: i/o timeout\""), "connect"},
		{u("error reading from server: EOF"), "transport"},
		{u("transport is closing"), "transport"},
		{u("connection error: desc = \"transport: read tcp 1.2.3.4:5->6.7.8.9:10: read: connection reset by peer\""), "transport"},
		{u("stream terminated by RST_STREAM with error code: REFUSED_STREAM"), "transport"},
		{u("the connection is draining"), "goaway"},
		{u("error reading from server: connection error: ... received GOAWAY"), "goaway"},
		{u("connection error: desc = \"transport: ... too_many_pings ... GOAWAY\""), "goaway"},
		{status.Error(codes.Canceled, "grpc: the client connection is closing"), "transport"},
		{status.Error(codes.Canceled, "context canceled"), "CANCELLED"},
		{status.Error(codes.DeadlineExceeded, "context deadline exceeded"), "DEADLINE_EXCEEDED"},
		{status.Error(codes.ResourceExhausted, "busy"), "RESOURCE_EXHAUSTED"},
		{errors.New("plain error"), "transport"},
	}
	for _, c := range cases {
		if got := Classify(c.err).Label(); got != c.want {
			t.Errorf("%v: got %s, want %s", c.err, got, c.want)
		}
	}
}

func TestClockMonotonic(t *testing.T) {
	a := clock.Now()
	b := clock.Now()
	if a < 1 || b < a {
		t.Fatalf("%d %d", a, b)
	}
}
