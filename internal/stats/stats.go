// Package stats holds the per-worker measurements: histograms per tick, counters, and
// the per-request records. Each worker owns one Worker, so the hot path takes only a
// per-worker mutex; the controller drains the workers once per tick.
package stats

import (
	"sync"
	"sync/atomic"

	"github.com/HdrHistogram/hdrhistogram-go"
	"github.com/criteo/triton-stress/internal/grpcclient"
)

// NumStatuses is the number of distinct request statuses.
const NumStatuses = grpcclient.NumStatuses

// The four measured histograms.
const (
	LatSched  = iota // t_done - t_sched
	LatSent          // t_done - t_send
	SendDelay        // t_send - t_sched
	SchedLag         // t_emit - t_sched
	NumHists
)

// HistNames name the histograms in column names.
var HistNames = [NumHists]string{"lat_sched", "lat_sent", "send_delay", "sched_lag"}

// Histogram range: 1 ns to one hour with 3 significant digits.
const (
	histMax    = int64(3_600_000_000_000)
	histDigits = 3
)

// NewHist returns an empty tick histogram.
func NewHist() *hdrhistogram.Histogram { return hdrhistogram.New(1, histMax, histDigits) }

// Tick holds what completed during one interval.
type Tick struct {
	Hist      [NumHists]*hdrhistogram.Histogram
	ByStatus  [NumStatuses]uint64
	BytesDown uint64
}

// NewTick returns an empty Tick.
func NewTick() *Tick {
	t := &Tick{}
	for i := range t.Hist {
		t.Hist[i] = NewHist()
	}
	return t
}

// Reset empties t.
func (t *Tick) Reset() {
	for _, h := range t.Hist {
		h.Reset()
	}
	t.ByStatus = [NumStatuses]uint64{}
	t.BytesDown = 0
}

// Merge adds o to t.
func (t *Tick) Merge(o *Tick) {
	for i := range t.Hist {
		t.Hist[i].Merge(o.Hist[i])
	}
	for i, n := range o.ByStatus {
		t.ByStatus[i] += n
	}
	t.BytesDown += o.BytesDown
}

// Completed is the number of requests that completed, whatever the status.
func (t *Tick) Completed() uint64 {
	var n uint64
	for _, c := range t.ByStatus {
		n += c
	}
	return n
}

// OK is the number of successful requests (gRPC OK or HTTP 200).
func (t *Tick) OK() uint64 {
	return t.ByStatus[grpcclient.StatusOK] + t.ByStatus[grpcclient.StatusHTTP200]
}

// Cum is the cumulative (whole-run) view of one or more workers, for Prometheus.
type Cum struct {
	ByStatus  [NumStatuses]uint64
	BytesDown uint64
	Hist      [NumHists]PromHist
}

// Add merges o into c.
func (c *Cum) Add(o *Cum) {
	for i, n := range o.ByStatus {
		c.ByStatus[i] += n
	}
	c.BytesDown += o.BytesDown
	for i := range c.Hist {
		c.Hist[i].Add(&o.Hist[i])
	}
}

// Record is one finished request. Timestamps are clock.Now values; a zero TReady or
// TSend means the request never got that far (it was never sent).
type Record struct {
	TSched, TEmit, TReady, TSend, TDone int64
	Status                              grpcclient.Status
	BytesDown                           int
	Conn                                int // connection index, -1 when none was Ready
}

// ReqLog is a sampled request for requests.csv.
type ReqLog struct {
	Record
	Worker int
}

// Worker is one worker's statistics.
type Worker struct {
	id       int
	logEvery uint64

	sent     atomic.Uint64
	bytesUp  atomic.Uint64
	inflight atomic.Int64

	mu    sync.Mutex
	tick  Tick
	cum   Cum
	nDone uint64
	log   []ReqLog

	_ [64]byte // keep neighbouring workers' counters off this cache line
}

// NewWorker creates worker id's statistics. logEvery > 0 samples one request in that many
// into the request log.
func NewWorker(id int, logEvery uint64) *Worker {
	w := &Worker{id: id, logEvery: logEvery, tick: *NewTick()}
	return w
}

// Sent counts a request about to be handed to the client: it is now in flight.
func (w *Worker) Sent(bytes int) {
	w.sent.Add(1)
	w.bytesUp.Add(uint64(bytes))
	w.inflight.Add(1)
}

// Inflight is the number of requests sent and not yet completed.
func (w *Worker) Inflight() int64 { return w.inflight.Load() }

// Totals are the cumulative requests sent and request bytes sent.
func (w *Worker) Totals() (sent, bytesUp uint64) { return w.sent.Load(), w.bytesUp.Load() }

func clampNs(v int64) int64 { return min(max(v, 0), histMax) }

// record adds v, which the caller has clamped to the histogram range.
func record(h *hdrhistogram.Histogram, v int64) {
	if err := h.RecordValue(v); err != nil {
		panic(err)
	}
}

// Done records a request that completed. wasSent says whether Sent was called for it
// (false for a request that failed with status connect before dispatch).
func (w *Worker) Done(r Record, wasSent bool) {
	if wasSent {
		w.inflight.Add(-1)
	}
	lag := clampNs(r.TEmit - r.TSched)
	var latSched, latSent, delay int64
	hasSend := r.TSend > 0
	if hasSend {
		latSched, latSent, delay = clampNs(r.TDone-r.TSched), clampNs(r.TDone-r.TSend), clampNs(r.TSend-r.TSched)
	}
	w.mu.Lock()
	t, c := &w.tick, &w.cum
	record(t.Hist[SchedLag], lag)
	c.Hist[SchedLag].Observe(float64(lag) * 1e-9)
	if hasSend {
		record(t.Hist[LatSched], latSched)
		record(t.Hist[LatSent], latSent)
		record(t.Hist[SendDelay], delay)
		c.Hist[LatSched].Observe(float64(latSched) * 1e-9)
		c.Hist[LatSent].Observe(float64(latSent) * 1e-9)
		c.Hist[SendDelay].Observe(float64(delay) * 1e-9)
	}
	t.ByStatus[r.Status]++
	c.ByStatus[r.Status]++
	t.BytesDown += uint64(r.BytesDown)
	c.BytesDown += uint64(r.BytesDown)
	if w.logEvery > 0 {
		if w.nDone++; w.nDone%w.logEvery == 0 {
			w.log = append(w.log, ReqLog{Record: r, Worker: w.id})
		}
	}
	w.mu.Unlock()
}

// Drain adds the worker's tick data to t and empties it. Sampled requests are appended
// to *log (when non-nil).
func (w *Worker) Drain(t *Tick, log *[]ReqLog) {
	w.mu.Lock()
	t.Merge(&w.tick)
	w.tick.Reset()
	if log != nil {
		*log = append(*log, w.log...)
	}
	w.log = w.log[:0]
	w.mu.Unlock()
}

// AddCumulative adds the worker's whole-run counts to c.
func (w *Worker) AddCumulative(c *Cum) {
	w.mu.Lock()
	c.Add(&w.cum)
	w.mu.Unlock()
}
