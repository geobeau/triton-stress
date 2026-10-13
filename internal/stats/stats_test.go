package stats

import (
	"math"
	"sync"
	"testing"

	"github.com/criteo/triton-stress/internal/grpcclient"
)

func TestNativeKeyBoundaries(t *testing.T) {
	// Exact powers of two sit at the top of their bucket: v = 2^(k/8) has key k.
	for _, k := range []int{-100, -17, -8, -1, 0, 1, 7, 8, 9, 40} {
		v := math.Pow(2, float64(k)/8)
		got := nativeKey(v)
		// pow rounding can push an exact boundary either way by one bucket
		if got != k && got != k+1 {
			t.Errorf("key(2^(%d/8)) = %d", k, got)
		}
		if mid := nativeKey(v * 1.04); mid != k+1 {
			t.Errorf("key(2^(%d/8)*1.04) = %d, want %d", k, mid, k+1)
		}
	}
	if nativeKey(1) != 0 || nativeKey(2) != 8 || nativeKey(0.5) != -8 {
		t.Fatal(nativeKey(1), nativeKey(2), nativeKey(0.5))
	}
}

func TestPromHistClassicAndNative(t *testing.T) {
	var h PromHist
	for _, s := range []float64{5e-6, 10e-6, 15e-6, 1e-3, 0.3, 25, 0} {
		h.Observe(s)
	}
	if h.Count != 7 {
		t.Fatal(h.Count)
	}
	cum := h.ClassicCumulative()
	// bounds: 10us, 20us, ..., 5e-6 and 10e-6 land in the first bucket (le is inclusive)
	if cum[0] != 3 || cum[1] != 4 { // zero counts as <= 10us too
		t.Fatalf("%v", cum)
	}
	if h.Above != 1 || cum[len(cum)-1] != 6 {
		t.Fatalf("above %d, cum %v", h.Above, cum)
	}
	if h.Zero != 1 {
		t.Fatal(h.Zero)
	}
	spans, deltas := h.NativePositive()
	var total int64
	var run int64
	for _, d := range deltas {
		run += d
		total += run
	}
	if total != 6 || len(deltas) == 0 || len(spans) == 0 {
		t.Fatalf("spans %+v deltas %v total %d", spans, deltas, total)
	}
	n := uint32(0)
	for _, s := range spans {
		n += s.Length
	}
	if int(n) != len(deltas) {
		t.Fatalf("span lengths %d != %d deltas", n, len(deltas))
	}
}

func TestNativeSpanEncoding(t *testing.T) {
	var h PromHist
	// keys: 0, 1, 2 (contiguous), then 10
	for _, k := range []int{0, 1, 1, 2, 10} {
		h.Native[k-nativeMinKey]++
	}
	spans, deltas := h.NativePositive()
	if len(spans) != 2 || spans[0] != (Span{0, 3}) || spans[1] != (Span{Offset: 7, Length: 1}) {
		t.Fatalf("%+v", spans)
	}
	want := []int64{1, 1, -1, 0} // counts 1,2,1,1
	for i, d := range want {
		if deltas[i] != d {
			t.Fatalf("%v", deltas)
		}
	}
}

func rec(sched, emit, send, done int64, st grpcclient.Status) Record {
	return Record{TSched: sched, TEmit: emit, TReady: emit, TSend: send, TDone: done, Status: st, BytesDown: 10}
}

func TestWorkerRecordsAndDrains(t *testing.T) {
	w := NewWorker(3, 2)
	for i := 0; i < 4; i++ {
		w.Sent(100)
		w.Done(rec(1000, 1500, 2000, 12000, grpcclient.StatusOK), true)
	}
	// a request that never got sent: sched_lag only
	w.Done(Record{TSched: 5000, TEmit: 7000, TDone: 7000, Status: grpcclient.StatusConnect, Conn: -1}, false)
	if w.Inflight() != 0 {
		t.Fatal(w.Inflight())
	}
	if s, b := w.Totals(); s != 4 || b != 400 {
		t.Fatal(s, b)
	}
	tick, log := NewTick(), []ReqLog{}
	w.Drain(tick, &log)
	if tick.Completed() != 5 || tick.OK() != 4 || tick.ByStatus[grpcclient.StatusConnect] != 1 || tick.BytesDown != 40 {
		t.Fatalf("%+v", tick.ByStatus)
	}
	if n := tick.Hist[SchedLag].TotalCount(); n != 5 {
		t.Fatalf("sched_lag samples %d", n)
	}
	for _, h := range []int{LatSched, LatSent, SendDelay} {
		if n := tick.Hist[h].TotalCount(); n != 4 {
			t.Fatalf("hist %d has %d samples; the unsent request must not be in it", h, n)
		}
	}
	if v := tick.Hist[LatSched].ValueAtQuantile(50); math.Abs(float64(v)-11000) > 11000*0.001 {
		t.Fatalf("lat_sched %d", v)
	}
	if v := tick.Hist[LatSent].ValueAtQuantile(50); math.Abs(float64(v)-10000) > 10000*0.001 {
		t.Fatalf("lat_sent %d", v)
	}
	if v := tick.Hist[SendDelay].ValueAtQuantile(50); math.Abs(float64(v)-1000) > 1000*0.001 {
		t.Fatalf("send_delay %d", v)
	}
	if len(log) != 2 || log[0].Worker != 3 {
		t.Fatalf("log %+v", log)
	}
	// a drained worker starts the next tick empty, but its cumulative view keeps counting
	tick2 := NewTick()
	w.Drain(tick2, nil)
	if tick2.Completed() != 0 {
		t.Fatal("tick not reset")
	}
	var c Cum
	w.AddCumulative(&c)
	if c.ByStatus[grpcclient.StatusOK] != 4 || c.Hist[LatSched].Count != 4 || c.Hist[SchedLag].Count != 5 {
		t.Fatalf("%+v", c.ByStatus)
	}
}

func TestWorkerClampsOutOfRange(t *testing.T) {
	w := NewWorker(0, 0)
	w.Sent(1)
	w.Done(rec(1000, 500, 600, 1000+2*histMax, grpcclient.StatusOK), true) // emit before sched, huge latency
	tick := NewTick()
	w.Drain(tick, nil)
	if tick.Hist[SchedLag].TotalCount() != 1 || tick.Hist[LatSched].TotalCount() != 1 {
		t.Fatal("out-of-range values must still be counted")
	}
}

func TestWorkerConcurrent(t *testing.T) {
	w := NewWorker(0, 1)
	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 1000; i++ {
				w.Sent(10)
				w.Done(rec(1, 2, 3, 4+int64(i), grpcclient.StatusOK), true)
			}
		}()
	}
	done := make(chan struct{})
	go func() {
		tick := NewTick()
		for {
			select {
			case <-done:
				return
			default:
				w.Drain(tick, nil)
				var c Cum
				w.AddCumulative(&c)
			}
		}
	}()
	wg.Wait()
	close(done)
	if w.Inflight() != 0 {
		t.Fatal(w.Inflight())
	}
}

func TestOKCountsHTTP200(t *testing.T) {
	var tick Tick
	tick.ByStatus[grpcclient.StatusHTTP200] = 3
	tick.ByStatus[grpcclient.StatusHTTP(503)] = 2
	tick.ByStatus[grpcclient.StatusOK] = 1
	if tick.OK() != 4 || tick.Completed() != 6 {
		t.Fatalf("ok %d completed %d", tick.OK(), tick.Completed())
	}
}
