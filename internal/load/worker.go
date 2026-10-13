package load

import (
	"math"
	"sync"
	"sync/atomic"
	"time"

	"github.com/criteo/triton-stress/internal/arrival"
	"github.com/criteo/triton-stress/internal/clock"
	"github.com/criteo/triton-stress/internal/grpcclient"
	"github.com/criteo/triton-stress/internal/h1client"
	"github.com/criteo/triton-stress/internal/h2client"
	"github.com/criteo/triton-stress/internal/plan"
	"github.com/criteo/triton-stress/internal/stats"
	"github.com/criteo/triton-stress/internal/timer"
)

// maxWait caps one timer wait, so a stop or a plan change is noticed within this time
// even when the next arrival is far away.
const maxWait = 10 * time.Millisecond

// worker is one dispatcher. Worker k of N owns connections k, k+N, ... and takes every
// N-th arrival of the plan. It owns its statistics, its round-robin position and its
// offset into the payload pool.
type worker struct {
	id, n   int
	r       *Runner
	conns   []*h2client.Conn
	group   *h1client.Group // HTTP: the worker's connections (conns is then empty)
	connIdx []int           // index of conns[i] among all connections
	rr      int
	pool    int // next payload index
	st      *stats.Worker

	// Free request records. A record travels with its request: the dispatcher fills it
	// in, the connection's reader goroutine completes it and hands it back.
	mu    sync.Mutex
	free  []*job
	hfree []*hjob
}

// job is one request in flight: the transport's request plus what the dispatcher knows.
type job struct {
	req           h2client.Request
	w             *worker
	idx           int
	tSched, tEmit int64
}

func (w *worker) getJob() *job {
	w.mu.Lock()
	if n := len(w.free); n > 0 {
		j := w.free[n-1]
		w.free = w.free[:n-1]
		w.mu.Unlock()
		return j
	}
	w.mu.Unlock()
	j := &job{w: w}
	j.req.Done = j.done
	j.req.Owner = j
	j.req.Timeout = w.r.cfg.Timeout
	j.req.Keep = w.r.deps.Validator != nil
	return j
}

// done records a finished request. It runs on a connection goroutine (or, for a
// request that could not be sent, on the dispatcher).
func (j *job) done(r *h2client.Request) {
	w := j.w
	st := r.Status
	if st == grpcclient.StatusOK && w.r.deps.Validator != nil && w.r.deps.Validator.ValidateBytes(r.Data) != nil {
		st = grpcclient.StatusInvalidResponse
	}
	tDone := r.TDone
	if tDone == 0 {
		tDone = clock.Now()
	}
	w.st.Done(stats.Record{
		TSched: j.tSched, TEmit: j.tEmit, TReady: r.TReady, TSend: r.TSend, TDone: tDone,
		Status: st, BytesDown: r.RespBytes, Conn: j.idx,
	}, true)
	r.Data, r.Body = nil, nil
	w.mu.Lock()
	w.free = append(w.free, j)
	w.mu.Unlock()
}

// hjob is job for the HTTP client.
type hjob struct {
	req           h1client.Request
	w             *worker
	tSched, tEmit int64
}

func (w *worker) getHJob() *hjob {
	w.mu.Lock()
	if n := len(w.hfree); n > 0 {
		j := w.hfree[n-1]
		w.hfree = w.hfree[:n-1]
		w.mu.Unlock()
		return j
	}
	w.mu.Unlock()
	j := &hjob{w: w}
	j.req.Done = j.done
	j.req.Owner = j
	j.req.Timeout = w.r.cfg.Timeout
	j.req.Keep = w.r.deps.Validator != nil
	return j
}

// done records a finished HTTP request (on a connection goroutine, or on the dispatcher
// for a request that could not be sent).
func (j *hjob) done(r *h1client.Request) {
	w := j.w
	st := r.Status
	if st == grpcclient.StatusHTTP200 && w.r.deps.Validator != nil && w.r.deps.Validator.ValidateHTTP(r.HdrLen, r.Data) != nil {
		st = grpcclient.StatusInvalidResponse
	}
	tDone := r.TDone
	if tDone == 0 {
		tDone = clock.Now()
	}
	w.st.Done(stats.Record{
		TSched: j.tSched, TEmit: j.tEmit, TReady: r.TReady, TSend: r.TSend, TDone: tDone,
		Status: st, BytesDown: r.RespBytes, Conn: r.Conn,
	}, true)
	w.mu.Lock()
	w.hfree = append(w.hfree, j)
	w.mu.Unlock()
}

// pick returns the next Ready connection round-robin, or nil when none is Ready.
func (w *worker) pick() (c *h2client.Conn, idx int) {
	for i := 0; i < len(w.conns); i++ {
		j := w.rr
		if w.rr++; w.rr == len(w.conns) {
			w.rr = 0
		}
		if w.conns[j].Ready() {
			return w.conns[j], w.connIdx[j]
		}
	}
	return nil, -1
}

// emit handles one due request: records t_emit, picks a Ready connection and hands the
// request to its own goroutine. With no Ready connection it completes at once as connect.
func (w *worker) emit(tSched int64) {
	r := w.r
	tEmit := clock.Now()
	if w.group != nil {
		if !w.group.Up() { // nothing is queued behind a reconnect
			w.st.Done(stats.Record{TSched: tSched, TEmit: tEmit, TDone: tEmit,
				Status: grpcclient.StatusConnect, Conn: -1}, false)
			return
		}
		body := r.deps.Pool.Get(w.pool)
		w.pool++
		w.st.Sent(len(body))
		j := w.getHJob()
		j.tSched, j.tEmit = tSched, tEmit
		j.req.Body = body
		w.group.Submit(&j.req) // a free connection, the FIFO, or at once connect
		return
	}
	conn, idx := w.pick()
	if conn == nil {
		w.st.Done(stats.Record{TSched: tSched, TEmit: tEmit, TDone: tEmit,
			Status: grpcclient.StatusConnect, Conn: -1}, false)
		return
	}
	body := r.deps.Pool.Get(w.pool)
	w.pool++
	w.st.Sent(len(body))
	j := w.getJob()
	j.idx, j.tSched, j.tEmit = idx, tSched, tEmit
	j.req.Body = body
	conn.Submit(&j.req)
}

// run is the dispatcher loop: it follows the worker's arrival process, parking on a
// timerfd between arrivals, and sends each due request as soon as it is due (late ones
// at once, with their original scheduled time) until the run stops.
func (w *worker) run(planPtr *atomic.Pointer[plan.Plan], stopped *atomic.Bool, t0 float64, tm *timer.Timer) {
	cur := planPtr.Load()
	gen := arrival.New(w.r.cfg.Arrival, w.id, w.n, w.r.cfg.Seed, cur, t0)
	for !stopped.Load() {
		if p := planPtr.Load(); p != cur {
			gen.Replan(cur, p)
			cur = p
		}
		t, ok := gen.Peek()
		if !ok {
			// The plan has no more arrivals for this worker; wait for a change or the stop.
			if _, err := tm.Wait(maxWait, maxWait); err != nil {
				return
			}
			continue
		}
		due := int64(math.Round(t * 1e9))
		if wait := due - clock.Now(); wait > 0 {
			if _, err := tm.Wait(time.Duration(wait), maxWait); err != nil {
				return
			}
			continue
		}
		w.emit(due)
		gen.Advance(cur)
	}
}
