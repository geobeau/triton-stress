package main

import (
	"bytes"
	"context"
	"encoding/json"
	"math"
	"net"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/criteo/triton-stress/internal/arrival"
	pb "github.com/criteo/triton-stress/pkg/tritonpb"
	"google.golang.org/grpc"

	"github.com/criteo/triton-stress/internal/grpcclient"
	"github.com/criteo/triton-stress/pkg/tensorgen"
)

func parse(t *testing.T, args ...string) (*config, error) {
	t.Helper()
	return parseConfig(args, &bytes.Buffer{})
}

func TestDefaults(t *testing.T) {
	c, err := parse(t, "--model", "m")
	if err != nil {
		t.Fatal(err)
	}
	if c.Target != "127.0.0.1:8001" || c.Protocol != "grpc" || c.Connections != 128 || c.BatchSize != 1 ||
		c.PoolSize != 256 || c.Seed != 1 || c.Timeout != 10*time.Second || c.BackoffMin != 10*time.Millisecond ||
		c.BackoffMax != time.Second || c.TLS != grpcclient.TLSOff || c.Validate || c.ModelVersion != "" {
		t.Fatalf("%+v", c)
	}
	if c.Gen != tensorgen.DefaultDist || len(c.Inputs) != 0 {
		t.Fatalf("%+v", c)
	}
}

func TestFlags(t *testing.T) {
	c, err := parse(t, "--model", "m", "--model-version", "2", "--target", "h:1", "--tls-insecure", "--connections", "8",
		"--batch-size", "4", "--input", "a:INT32:3", "--input", "img:FP32:3,224,224", "--gen", "const:7", "--payload-pool", "5",
		"--seed", "9", "--request-timeout", "2s", "--reconnect-backoff-min", "1ms", "--reconnect-backoff-max", "20ms", "--validate")
	if err != nil {
		t.Fatal(err)
	}
	if c.ModelVersion != "2" || c.Target != "h:1" || c.TLS != grpcclient.TLSInsecure || c.Connections != 8 || c.BatchSize != 4 ||
		len(c.Inputs) != 2 || c.Inputs[1].Name != "img" || len(c.Inputs[1].Dims) != 3 || c.Gen.A != 7 || c.PoolSize != 5 ||
		c.Seed != 9 || c.Timeout != 2*time.Second || c.BackoffMax != 20*time.Millisecond || !c.Validate {
		t.Fatalf("%+v", c)
	}
	if c, err = parse(t, "--model", "m", "--tls"); err != nil || c.TLS != grpcclient.TLSSystem {
		t.Fatalf("%+v %v", c, err)
	}
}

func TestFlagErrors(t *testing.T) {
	cases := map[string][]string{
		"missing model":     {},
		"tls and insecure":  {"--model", "m", "--tls", "--tls-insecure"},
		"bad protocol":      {"--model", "m", "--protocol", "udp"},
		"zero connections":  {"--model", "m", "--connections", "0"},
		"zero batch":        {"--model", "m", "--batch-size", "0"},
		"zero pool":         {"--model", "m", "--payload-pool", "0"},
		"bad gen":           {"--model", "m", "--gen", "gaussian"},
		"bad input":         {"--model", "m", "--input", "x:INT32"},
		"bad input dtype":   {"--model", "m", "--input", "x:INT7:1"},
		"backoff max < min": {"--model", "m", "--reconnect-backoff-min", "1s", "--reconnect-backoff-max", "1ms"},
		"zero timeout":      {"--model", "m", "--request-timeout", "0s"},
		"unknown flag":      {"--model", "m", "--nope"},
		"negative seed":     {"--model", "m", "--seed", "-1"},
	}
	for name, args := range cases {
		if _, err := parse(t, args...); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}

func TestHTTPDefaults(t *testing.T) {
	c, err := parse(t, "--model", "m", "--protocol", "http")
	if err != nil {
		t.Fatal(err)
	}
	if c.Target != "127.0.0.1:8000" || c.Connections != 2048 {
		t.Fatalf("%+v", c)
	}
	c, err = parse(t, "--model", "m", "--protocol", "http", "--target", "h:80", "--connections", "7")
	if err != nil || c.Target != "h:80" || c.Connections != 7 {
		t.Fatalf("%+v %v", c, err)
	}
}

func TestRunFatalExitCodes(t *testing.T) {
	var out, errb bytes.Buffer
	dir := filepath.Join(t.TempDir(), "run")
	code := run(context.Background(), []string{"--model", "m", "--target", "127.0.0.1:1", "--request-timeout", "300ms",
		"--out-dir", dir, "--metrics-addr", "off"}, &out, &errb)
	if code != 1 {
		t.Fatalf("unreachable target exited %d", code)
	}
	if !strings.Contains(errb.String(), "no connection") {
		t.Fatalf("stderr: %s", errb.String())
	}
	// the fatal stop event is the last events.jsonl line and the last stdout line
	b, _ := os.ReadFile(filepath.Join(dir, "events.jsonl"))
	lines := strings.Split(strings.TrimSpace(string(b)), "\n")
	var stop map[string]any
	if err := json.Unmarshal([]byte(lines[len(lines)-1]), &stop); err != nil {
		t.Fatal(err)
	}
	if stop["event"] != "stop" || !strings.HasPrefix(stop["reason"].(string), "fatal:no connection") || stop["ts"] == nil || stop["iso"] == nil {
		t.Fatalf("%v", stop)
	}
	if strings.TrimSpace(out.String()) != lines[len(lines)-1] {
		t.Fatalf("stdout %q", out.String())
	}
	errb.Reset()
	if code := run(context.Background(), []string{"--tls", "--tls-insecure", "--model", "m"}, &out, &errb); code != 1 || !strings.Contains(errb.String(), "mutually exclusive") {
		t.Fatalf("%d %s", code, errb.String())
	}
}

func TestLoadFlagDefaults(t *testing.T) {
	c, err := parse(t, "--model", "m")
	if err != nil {
		t.Fatal(err)
	}
	s := c.Schedule
	if s.Warmup != 10*time.Second || s.Start != 1000 || s.Slope != 1000 || s.Max != 1e6 || s.Duration != 0 {
		t.Fatalf("%+v", s)
	}
	g := c.Guards
	if *g.Achieved != 0.95 || *g.ErrorRate != 0.01 || g.Latency.Pct != 99 || g.Latency.Max != 100*time.Millisecond ||
		g.SendDelay.Max != time.Millisecond || *g.MaxInflight != 100000 || g.Consecutive != 3 {
		t.Fatalf("%+v", g)
	}
	if c.Arrival != arrival.Constant || c.Interval != time.Second || c.MetricsAddr != "0.0.0.0:9095" || c.OutDir != "" ||
		c.RequestLogEvery != 0 || c.Workers < 1 || c.Seed != 1 {
		t.Fatalf("%+v", c)
	}
}

func TestLoadFlags(t *testing.T) {
	c, err := parse(t, "--model", "m", "--workers", "3", "--arrival", "poisson", "--seed", "5", "--warmup", "5s",
		"--start-rate", "20k", "--slope", "0", "--max-rate", "2M", "--duration", "30s", "--guard-achieved", "off",
		"--guard-error-rate", "0.5", "--guard-latency", "p99.9:2s", "--guard-send-delay", "off", "--guard-max-inflight", "off",
		"--guard-consecutive", "5", "--interval", "500ms", "--metrics-addr", "off", "--out-dir", "/x/y", "--request-log-every", "100")
	if err != nil {
		t.Fatal(err)
	}
	g := c.Guards
	if c.Workers != 3 || c.Arrival != arrival.Poisson || c.Seed != 5 || c.Schedule.Start != 20000 || c.Schedule.Slope != 0 ||
		c.Schedule.Max != 2e6 || c.Schedule.Duration != 30*time.Second || c.Schedule.Warmup != 5*time.Second ||
		g.Achieved != nil || *g.ErrorRate != 0.5 || g.Latency.Pct != 99.9 || g.Latency.Max != 2*time.Second ||
		g.SendDelay != nil || g.MaxInflight != nil || g.Consecutive != 5 || c.Interval != 500*time.Millisecond ||
		c.MetricsAddr != "" || c.OutDir != "/x/y" || c.RequestLogEvery != 100 {
		t.Fatalf("%+v", c)
	}
}

func TestLoadFlagErrors(t *testing.T) {
	for name, extra := range map[string][]string{
		"bad arrival":      {"--arrival", "uniform"},
		"bad rate":         {"--start-rate", "1m"},
		"zero start rate":  {"--start-rate", "0"},
		"start above max":  {"--start-rate", "2M"},
		"negative slope":   {"--slope", "-1"},
		"bad warmup":       {"--warmup", "soon"},
		"bad duration":     {"--duration", "x"},
		"zero duration":    {"--duration", "0s"},
		"bad guard":        {"--guard-latency", "99:1ms"},
		"bad fraction":     {"--guard-achieved", "2"},
		"bad inflight":     {"--guard-max-inflight", "1.5"},
		"zero consecutive": {"--guard-consecutive", "0"},
		"zero interval":    {"--interval", "0s"},
		"negative workers": {"--workers", "-1"},
		"bad request log":  {"--request-log-every", "x"},
	} {
		if _, err := parse(t, append([]string{"--model", "m"}, extra...)...); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}

// fakeTriton is a localhost Triton stand-in for the end-to-end test (a test fixture only).
type fakeTriton struct {
	pb.UnimplementedGRPCInferenceServiceServer
}

func (f *fakeTriton) ModelInfer(_ context.Context, req *pb.ModelInferRequest) (*pb.ModelInferResponse, error) {
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

func TestEndToEndShortRamp(t *testing.T) {
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	srv := grpc.NewServer()
	pb.RegisterGRPCInferenceServiceServer(srv, &fakeTriton{})
	go func() { _ = srv.Serve(lis) }()
	defer srv.Stop()

	dir := filepath.Join(t.TempDir(), "run")
	var out, errb bytes.Buffer
	code := run(context.Background(), []string{"--model", "m", "--target", lis.Addr().String(), "--connections", "4", "--workers", "2",
		"--batch-size", "2", "--validate", "--warmup", "300ms", "--start-rate", "200", "--slope", "400", "--max-rate", "600",
		"--interval", "200ms", "--metrics-addr", "127.0.0.1:0", "--out-dir", dir, "--request-log-every", "10",
		// Send delay measures the client host, which a loaded CI runner under -race cannot hold to 1 ms.
		"--guard-send-delay", "off", "--guard-achieved", "off", "--guard-latency", "off"}, &out, &errb)
	if code != 0 {
		t.Fatalf("exit %d: %s", code, errb.String())
	}
	b, _ := os.ReadFile(filepath.Join(dir, "events.jsonl"))
	lines := strings.Split(strings.TrimSpace(string(b)), "\n")
	var stop map[string]any
	if err := json.Unmarshal([]byte(lines[len(lines)-1]), &stop); err != nil {
		t.Fatal(err)
	}
	if stop["event"] != "stop" || stop["reason"] != "max_rate_reached" || math.Abs(stop["offered_rps"].(float64)-600) > 1 {
		t.Fatalf("%v\n%s", stop, errb.String())
	}
	// connection events were recorded, and stdout ends with the stop event
	if !strings.Contains(string(b), `"event":"conn"`) || !strings.Contains(string(b), `"to":"READY"`) {
		t.Fatalf("no connection events:\n%s", b)
	}
	outLines := strings.Split(strings.TrimSpace(out.String()), "\n")
	if outLines[len(outLines)-1] != lines[len(lines)-1] || len(outLines) < 6 {
		t.Fatalf("stdout %q", out.String())
	}
	for _, name := range []string{"intervals.csv", "requests.csv"} {
		if st, err := os.Stat(filepath.Join(dir, name)); err != nil || st.Size() == 0 {
			t.Fatalf("%s: %v", name, err)
		}
	}
	// 0.3 s at 200 + 1 s from 200 to 600: 460 requests, all OK (validated)
	var ok float64
	for _, l := range outLines[:len(outLines)-1] {
		var m map[string]any
		if err := json.Unmarshal([]byte(l), &m); err != nil {
			t.Fatal(err)
		}
		st := m["status"].(map[string]any)
		for k, v := range st {
			if k != "OK" {
				t.Fatalf("status %s", k)
			}
			ok += v.(float64)
		}
	}
	if math.Abs(ok-460)/460 > 0.05 {
		t.Fatalf("%v OK, want about 460", ok)
	}
}

func TestGuardTripExitsNonZero(t *testing.T) {
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	srv := grpc.NewServer()
	pb.RegisterGRPCInferenceServiceServer(srv, &fakeTriton{})
	go func() { _ = srv.Serve(lis) }()
	defer srv.Stop()

	dir := filepath.Join(t.TempDir(), "run")
	var out, errb bytes.Buffer
	code := run(context.Background(), []string{"--model", "m", "--target", lis.Addr().String(),
		"--connections", "2", "--workers", "1", "--warmup", "0s", "--start-rate", "100", "--slope", "0",
		"--max-rate", "100", "--duration", "10s", "--interval", "100ms",
		// A latency no request can meet, so the run trips well before --duration.
		"--guard-latency", "p50:1ns", "--guard-consecutive", "1",
		// Send delay measures the client host, which a loaded CI runner cannot hold to 1 ms.
		"--guard-send-delay", "off",
		"--guard-achieved", "off",
		"--metrics-addr", "off", "--out-dir", dir}, &out, &errb)
	if code != 2 {
		t.Fatalf("exit %d, want 2: %s", code, errb.String())
	}
	b, _ := os.ReadFile(filepath.Join(dir, "events.jsonl"))
	lines := strings.Split(strings.TrimSpace(string(b)), "\n")
	var stop map[string]any
	if err := json.Unmarshal([]byte(lines[len(lines)-1]), &stop); err != nil {
		t.Fatal(err)
	}
	if stop["event"] != "stop" || stop["reason"] != "guard:latency" {
		t.Fatalf("%v\n%s", stop, errb.String())
	}
}

func TestSignalDuringStartupIsNormal(t *testing.T) {
	ctx, cancel := context.WithCancelCause(context.Background())
	cancel(context.Canceled)
	var out, errb bytes.Buffer
	dir := filepath.Join(t.TempDir(), "run")
	if code := run(ctx, []string{"--model", "m", "--target", "127.0.0.1:1", "--out-dir", dir, "--metrics-addr", "off"}, &out, &errb); code != 0 {
		t.Fatalf("%d %s", code, errb.String())
	}
	if !strings.Contains(out.String(), `"reason":"interrupted"`) {
		t.Fatal(out.String())
	}
}
