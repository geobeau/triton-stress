package grpcclient

import (
	"context"
	"os"
	"sort"
	"testing"
	"time"

	"github.com/criteo/triton-stress/pkg/tensorgen"
)

// Live checks against a running Triton:
//
//	RTS_LIVE_TARGET=127.0.0.1:8001 [RTS_LIVE_MODEL=model_2026_07_15] go test -v -run Live ./internal/grpcclient
//
// They are skipped when RTS_LIVE_TARGET is not set.
func TestLiveSequentialValidatedInfers(t *testing.T) {
	target := os.Getenv("RTS_LIVE_TARGET")
	if target == "" {
		t.Skip("RTS_LIVE_TARGET not set")
	}
	model := os.Getenv("RTS_LIVE_MODEL")
	if model == "" {
		model = "model_2026_07_15"
	}
	c, err := New(Options{Target: target})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c.Close() }()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := c.WaitReady(ctx); err != nil {
		t.Fatal(err)
	}
	m, err := c.FetchModel(ctx, model, "")
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("model %s batched=%v max_batch_size=%d inputs=%v outputs=%v", m.Name, m.Batched, m.MaxBatchSize, m.Inputs, m.Outputs)
	if !m.Batched || m.MaxBatchSize < 4 || len(m.Inputs) == 0 || len(m.Outputs) == 0 {
		t.Fatalf("unexpected model %+v", m)
	}

	const requests = 1000
	for _, batch := range []int{1, 4} {
		tpl, err := m.NewTemplate(nil, batch)
		if err != nil {
			t.Fatal(err)
		}
		g := tensorgen.NewGenerator(1)
		bodies := make([][]byte, 64)
		for i := range bodies {
			if bodies[i], err = g.Encode(tpl); err != nil {
				t.Fatal(err)
			}
		}
		v := tensorgen.NewValidator(m.Outputs, batch, m.Batched)
		lat := make([]time.Duration, 0, requests)
		bad := map[string]int{}
		for i := 0; i < requests; i++ {
			r := c.Infer(context.Background(), bodies[i%len(bodies)], 10*time.Second, v)
			if !r.Status.IsOK() {
				bad[r.Status.Label()]++
				continue
			}
			if r.TReady > r.TSend || r.TSend > r.TDone {
				t.Fatalf("timestamps out of order: %+v", r)
			}
			lat = append(lat, time.Duration(r.TDone-r.TSend))
		}
		if len(bad) != 0 || len(lat) != requests {
			t.Fatalf("batch size %d: %d/%d OK, other statuses %v", batch, len(lat), requests, bad)
		}
		sort.Slice(lat, func(i, j int) bool { return lat[i] < lat[j] })
		t.Logf("batch size %d: %d/%d OK, p50 %v p99 %v max %v", batch, len(lat), requests, lat[requests/2], lat[requests*99/100], lat[requests-1])
	}
}
