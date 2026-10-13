package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"math"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/criteo/triton-stress/internal/kservefake"
)

func TestHTTPEndToEndShortRamp(t *testing.T) {
	f := kservefake.New(t)
	dir := filepath.Join(t.TempDir(), "run")
	var out, errb bytes.Buffer
	code := run(context.Background(), []string{"--model", "m", "--protocol", "http", "--target", f.Addr, "--connections", "6", "--workers", "2",
		"--batch-size", "2", "--validate", "--warmup", "300ms", "--start-rate", "200", "--slope", "400", "--max-rate", "600",
		"--interval", "200ms", "--metrics-addr", "127.0.0.1:0", "--out-dir", dir, "--request-log-every", "10",
		// Send delay measures the client host, which a loaded CI runner under -race cannot hold to 1 ms.
		"--guard-send-delay", "off"}, &out, &errb)
	if code != 0 {
		t.Fatalf("exit %d: %s", code, errb.String())
	}
	if f.Bad.Load() != 0 {
		t.Fatalf("the server refused requests: %s", f.LastError())
	}
	if f.Metadata.Load() != 2 {
		t.Fatalf("%d metadata requests, want metadata and config", f.Metadata.Load())
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
	if !strings.Contains(string(b), `"event":"conn"`) || !strings.Contains(string(b), `"to":"READY"`) {
		t.Fatalf("no connection events:\n%s", b)
	}
	outLines := strings.Split(strings.TrimSpace(out.String()), "\n")
	var ok float64
	for _, l := range outLines[:len(outLines)-1] {
		var m map[string]any
		if err := json.Unmarshal([]byte(l), &m); err != nil {
			t.Fatal(err)
		}
		for k, v := range m["status"].(map[string]any) {
			if k != "http_200" {
				t.Fatalf("status %s", k)
			}
			ok += v.(float64)
		}
	}
	if math.Abs(ok-460)/460 > 0.05 || int64(ok) != f.Received.Load() {
		t.Fatalf("%v OK, server got %d, want about 460", ok, f.Received.Load())
	}
	if st, err := os.Stat(filepath.Join(dir, "requests.csv")); err != nil || st.Size() == 0 {
		t.Fatal(err)
	}
	csv, _ := os.ReadFile(filepath.Join(dir, "requests.csv"))
	if !strings.Contains(string(csv), ",http_200,") {
		t.Fatalf("requests.csv has no http_200 status:\n%.300s", csv)
	}
	if !strings.Contains(errb.String(), "protocol http") || !strings.Contains(errb.String(), "6 connections") {
		t.Fatal(errb.String())
	}
}

func TestHTTPMetricsLabels(t *testing.T) {
	f := kservefake.New(t)
	dir := filepath.Join(t.TempDir(), "run")
	var out, errb bytes.Buffer
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	maddr := ln.Addr().String()
	_ = ln.Close()
	done := make(chan int)
	go func() {
		done <- run(ctx, []string{"--model", "m", "--protocol", "http", "--target", f.Addr, "--connections", "2", "--workers", "1",
			"--warmup", "0s", "--start-rate", "300", "--slope", "0", "--duration", "1500ms", "--interval", "200ms",
			"--metrics-addr", maddr, "--out-dir", dir}, &out, &errb)
	}()
	var body string
	for i := 0; i < 100 && !strings.Contains(body, `status="http_200"`); i++ {
		resp, err := http.Get("http://" + maddr + "/metrics")
		if err == nil {
			b, _ := io.ReadAll(resp.Body)
			_ = resp.Body.Close()
			body = string(b)
		}
		<-timeAfter()
	}
	if !strings.Contains(body, `protocol="http"`) || !strings.Contains(body, `status="http_200"`) || strings.Contains(body, `status="OK"`) {
		t.Fatalf("metrics:\n%.1500s", body)
	}
	if code := <-done; code != 0 {
		t.Fatalf("%d %s", code, errb.String())
	}
}

func timeAfter() <-chan time.Time { return time.After(50 * time.Millisecond) }
