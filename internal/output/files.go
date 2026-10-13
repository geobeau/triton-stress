package output

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"time"

	"github.com/criteo/triton-stress/internal/stats"
)

// Event is one line of events.jsonl. ts and iso come first, then "event", then Fields.
type Event struct {
	TS     time.Time
	Kind   string
	Fields []KV
}

// Field helpers for building events.
func Str(name, v string) KV           { return str(name, v) }
func Int(name string, v int64) KV     { return integer(name, v) }
func Float(name string, v float64) KV { return num(name, v, 6) }
func Rate(name string, v float64) KV  { return num(name, v, 1) }
func (e Event) json() []byte {
	kvs := append([]KV{{Name: "ts", JSON: FormatTS(e.TS)}, str("iso", FormatISO(e.TS)), str("event", e.Kind)}, e.Fields...)
	return append(jsonObject(kvs), '\n')
}

// Files are the run's output files and stdout. All methods are safe for concurrent use.
type Files struct {
	Dir string

	mu        sync.Mutex
	stopped   bool // the stop event was written: it stays the last event
	stdout    io.Writer
	intervals *os.File
	events    *os.File
	requests  *os.File
	reqBuf    *bufio.Writer
}

// RequestsHeader is the requests.csv header.
const RequestsHeader = "t_sched_ns,t_emit_ns,t_ready_ns,t_send_ns,t_done_ns,status,worker,connection\n"

// Open creates dir and its files. intervals.csv and events.jsonl are created exclusively,
// so an existing run is never overwritten. requests.csv is created only when requestLog is set.
func Open(dir string, requestLog bool, stdout io.Writer) (*Files, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("creating the output directory: %w", err)
	}
	f := &Files{Dir: dir, stdout: stdout}
	create := func(name string) (*os.File, error) {
		return os.OpenFile(filepath.Join(dir, name), os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
	}
	var err error
	if f.intervals, err = create("intervals.csv"); err != nil {
		return nil, fmt.Errorf("creating intervals.csv: %w", err)
	}
	if f.events, err = create("events.jsonl"); err != nil {
		_ = f.Close()
		return nil, fmt.Errorf("creating events.jsonl: %w", err)
	}
	if _, err = f.intervals.WriteString(CSVHeader()); err != nil {
		_ = f.Close()
		return nil, err
	}
	if requestLog {
		if f.requests, err = create("requests.csv"); err != nil {
			_ = f.Close()
			return nil, fmt.Errorf("creating requests.csv: %w", err)
		}
		if _, err = f.requests.WriteString(RequestsHeader); err != nil {
			_ = f.Close()
			return nil, err
		}
		f.reqBuf = bufio.NewWriterSize(f.requests, 1<<16)
	}
	return f, nil
}

// WriteInterval appends the tick to intervals.csv and prints its JSON line on stdout.
func (f *Files) WriteInterval(iv *Interval) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	_, err := f.intervals.WriteString(iv.CSVLine())
	if _, e := f.stdout.Write(iv.JSONLine()); err == nil {
		err = e
	}
	return err
}

// WriteEvent appends an event to events.jsonl. Events after the stop event are dropped.
func (f *Files) WriteEvent(e Event) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.stopped {
		return nil // nothing follows the stop event
	}
	_, err := f.events.Write(e.json())
	return err
}

// WriteStop appends the final stop event to events.jsonl and repeats it as the last
// stdout line.
func (f *Files) WriteStop(e Event) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.stopped = true
	b := e.json()
	_, err := f.events.Write(b)
	if _, e := f.stdout.Write(b); err == nil {
		err = e
	}
	return err
}

// WriteRequests appends sampled requests, converting their monotonic timestamps to UTC
// epoch nanoseconds with a. A no-op without --request-log-every.
func (f *Files) WriteRequests(recs []stats.ReqLog, a Anchor) error {
	if f.reqBuf == nil {
		return nil
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	b := make([]byte, 0, 160)
	utc := func(mono int64) []byte {
		if mono <= 0 {
			return nil // the request never got that far
		}
		return strconv.AppendInt(nil, a.At(mono).UnixNano(), 10)
	}
	for _, r := range recs {
		b = b[:0]
		for _, t := range []int64{r.TSched, r.TEmit, r.TReady, r.TSend, r.TDone} {
			b = append(b, utc(t)...)
			b = append(b, ',')
		}
		b = append(b, csvEscape(r.Status.Label())...)
		b = append(b, ',')
		b = strconv.AppendInt(b, int64(r.Worker), 10)
		b = append(b, ',')
		if r.Conn >= 0 {
			b = strconv.AppendInt(b, int64(r.Conn), 10)
		}
		b = append(b, '\n')
		if _, err := f.reqBuf.Write(b); err != nil {
			return err
		}
	}
	return f.reqBuf.Flush()
}

// Close closes the files.
func (f *Files) Close() error {
	f.mu.Lock()
	defer f.mu.Unlock()
	var err error
	for _, c := range []*os.File{f.intervals, f.events, f.requests} {
		if c != nil {
			if e := c.Close(); err == nil {
				err = e
			}
		}
	}
	return err
}
