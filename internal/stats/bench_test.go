package stats

import (
	"sync/atomic"
	"testing"

	"github.com/criteo/triton-stress/internal/grpcclient"
)

// BenchmarkDone is the cost of recording one completed request, from many goroutines
// into one worker (the contention a worker's request goroutines cause each other).
func BenchmarkDone(b *testing.B) {
	w := NewWorker(0, 0)
	var seed atomic.Int64
	b.ReportAllocs()
	b.RunParallel(func(pb *testing.PB) {
		x := seed.Add(1) * 7919
		for pb.Next() {
			x = x*6364136223846793005 + 1442695040888963407
			base := int64(1_000_000 + (x>>40)%5_000_000)
			w.Sent(1300)
			w.Done(Record{TSched: 1, TEmit: 1 + base/100, TReady: 1 + base/100, TSend: 1 + base/50, TDone: 1 + base,
				Status: grpcclient.StatusOK, BytesDown: 40}, true)
		}
	})
}

// BenchmarkDoneSerial is the same from one goroutine (no contention).
func BenchmarkDoneSerial(b *testing.B) {
	w := NewWorker(0, 0)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		base := int64(1_000_000 + (i*7919)%5_000_000)
		w.Sent(1300)
		w.Done(Record{TSched: 1, TEmit: 1 + base/100, TReady: 1 + base/100, TSend: 1 + base/50, TDone: 1 + base,
			Status: grpcclient.StatusOK, BytesDown: 40}, true)
	}
}
