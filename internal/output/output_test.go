package output

import (
	"bytes"
	"encoding/csv"
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/criteo/triton-stress/internal/grpcclient"
	"github.com/criteo/triton-stress/internal/health"
	"github.com/criteo/triton-stress/internal/stats"
)

var tsRe = regexp.MustCompile(`^\d{10}\.\d{3}$`)

func TestTimestampFormats(t *testing.T) {
	tm := time.Date(2026, 10, 9, 14, 23, 6, 968_000_000+123_456, time.UTC)
	if got := FormatTS(tm); got != strconvTS(tm) || !tsRe.MatchString(got) || !strings.HasSuffix(got, ".968") {
		t.Fatal(got)
	}
	if got := FormatISO(tm); got != "2026-10-09T14:23:06.968Z" {
		t.Fatal(got)
	}
	// milliseconds are zero-padded, and a non-UTC zone is converted
	z := time.Date(2026, 10, 9, 16, 0, 0, 5_000_000, time.FixedZone("x", 7200))
	if got := FormatTS(z); !strings.HasSuffix(got, ".005") {
		t.Fatal(got)
	}
	if got := FormatISO(z); got != "2026-10-09T14:00:00.005Z" {
		t.Fatal(got)
	}
	if got := FormatTS(time.UnixMilli(1785922486968)); got != "1785922486.968" {
		t.Fatal(got)
	}
	if RunDirName(tm) != "20261009T142306Z" {
		t.Fatal(RunDirName(tm))
	}
}

func strconvTS(tm time.Time) string { return FormatTS(tm) }

func sample() *Interval {
	iv := &Interval{
		TS: time.UnixMilli(1785922486968), Phase: "ramp",
		OfferedRPS: 5000, SentRPS: 4999.5, OKRPS: 4900, ErrorRPS: 99.5, Inflight: 42,
		Health:    health.Health{CPUUserPct: 120.5, CPUSysPct: 30.25, BusiestThreadPct: 80, NonvolCtxSwitches: 7},
		HasHealth: true,
		Status:    map[string]uint64{"OK": 4900, "timeout": 99},
	}
	iv.Hist[stats.LatSched] = HistSummary{Valid: true, Ns: [5]int64{1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000}}
	return iv
}

func TestIntervalCSVAndJSON(t *testing.T) {
	iv := sample()
	cols := IntervalColumns()
	if cols[0] != "ts" || cols[1] != "iso" || cols[2] != "phase" || cols[3] != "offered_rps" {
		t.Fatal(cols)
	}
	if len(cols) != 8+4*5+6 {
		t.Fatalf("%d columns: %v", len(cols), cols)
	}
	r := csv.NewReader(strings.NewReader(CSVHeader() + iv.CSVLine()))
	recs, err := r.ReadAll()
	if err != nil || len(recs) != 2 || len(recs[0]) != len(recs[1]) || len(recs[0]) != len(cols) {
		t.Fatal(err, recs)
	}
	byName := map[string]string{}
	for i, c := range recs[0] {
		byName[c] = recs[1][i]
	}
	if byName["ts"] != "1785922486.968" || byName["iso"] != "2026-08-05T09:34:46.968Z" || byName["phase"] != "ramp" ||
		byName["offered_rps"] != "5000.0" || byName["inflight"] != "42" || byName["lat_sched_p99_us"] != "3000.0" ||
		byName["lat_sched_max_us"] != "5000.0" || byName["lat_sent_p50_us"] != "" || byName["cpu_user_pct"] != "120.5" ||
		byName["cgroup_nr_throttled"] != "" {
		t.Fatalf("%v", byName)
	}
	var j map[string]any
	line := iv.JSONLine()
	if !bytes.HasSuffix(line, []byte("\n")) || bytes.Count(line, []byte("\n")) != 1 {
		t.Fatal("one line")
	}
	if err := json.Unmarshal(line, &j); err != nil {
		t.Fatalf("%v: %s", err, line)
	}
	if j["ts"].(float64) != 1785922486.968 || j["phase"] != "ramp" || j["lat_sent_p50_us"] != nil {
		t.Fatalf("%v", j)
	}
	st := j["status"].(map[string]any)
	if st["OK"].(float64) != 4900 || st["timeout"].(float64) != 99 || len(st) != 2 {
		t.Fatalf("%v", st)
	}
	if strings.Index(string(line), `"ts"`) != 1 || strings.Index(string(line), `"iso"`) > strings.Index(string(line), `"phase"`) {
		t.Fatalf("ts first, then iso: %s", line)
	}
}

func TestEventJSON(t *testing.T) {
	e := Event{TS: time.UnixMilli(1785922486968), Kind: "breach", Fields: []KV{Str("guard", "latency"), Float("observed", 0.25), Float("threshold", 0.1)}}
	var j map[string]any
	if err := json.Unmarshal(e.json(), &j); err != nil {
		t.Fatal(err)
	}
	if j["ts"].(float64) != 1785922486.968 || j["iso"] != "2026-08-05T09:34:46.968Z" || j["event"] != "breach" ||
		j["guard"] != "latency" || j["observed"].(float64) != 0.25 {
		t.Fatalf("%v", j)
	}
}

func TestFilesRoundTrip(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "run")
	var out bytes.Buffer
	f, err := Open(dir, true, &out)
	if err != nil {
		t.Fatal(err)
	}
	if err := f.WriteInterval(sample()); err != nil {
		t.Fatal(err)
	}
	if err := f.WriteEvent(Event{TS: time.UnixMilli(1000), Kind: "phase", Fields: []KV{Str("phase", "ramp")}}); err != nil {
		t.Fatal(err)
	}
	if err := f.WriteStop(Event{TS: time.UnixMilli(2000), Kind: "stop", Fields: []KV{Str("reason", "duration")}}); err != nil {
		t.Fatal(err)
	}
	a := Anchor{Mono: 1000, Wall: time.Unix(100, 0)}
	err = f.WriteRequests([]stats.ReqLog{
		{Record: stats.Record{TSched: 1000, TEmit: 1500, TReady: 1600, TSend: 1700, TDone: 3000, Status: grpcclient.StatusOK, Conn: 4}, Worker: 2},
		{Record: stats.Record{TSched: 1000, TEmit: 1500, TDone: 1500, Status: grpcclient.StatusConnect, Conn: -1}, Worker: 1},
	}, a)
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(out.String()), "\n")
	if len(lines) != 2 || !strings.Contains(lines[1], `"event":"stop"`) {
		t.Fatalf("stdout: %q", out.String())
	}
	ev, _ := os.ReadFile(filepath.Join(dir, "events.jsonl"))
	evLines := strings.Split(strings.TrimSpace(string(ev)), "\n")
	if len(evLines) != 2 || evLines[1] != lines[1] {
		t.Fatalf("the last stdout line must repeat the stop event: %q vs %q", evLines, lines)
	}
	rq, _ := os.ReadFile(filepath.Join(dir, "requests.csv"))
	want := RequestsHeader +
		"100000000000,100000000500,100000000600,100000000700,100000002000,OK,2,4\n" +
		"100000000000,100000000500,,,100000000500,connect,1,\n"
	if string(rq) != want {
		t.Fatalf("requests.csv:\n%q\nwant\n%q", rq, want)
	}
	// Opening the same directory again must not overwrite the run.
	if _, err := Open(dir, false, &out); err == nil {
		t.Fatal("existing run overwritten")
	}
	ic, _ := os.ReadFile(filepath.Join(dir, "intervals.csv"))
	if !strings.HasPrefix(string(ic), "ts,iso,phase,") || strings.Count(string(ic), "\n") != 2 {
		t.Fatalf("%q", ic)
	}
}

func TestNoRequestLogByDefault(t *testing.T) {
	dir := t.TempDir()
	f, err := Open(filepath.Join(dir, "r"), false, &bytes.Buffer{})
	if err != nil {
		t.Fatal(err)
	}
	if err := f.WriteRequests([]stats.ReqLog{{}}, Anchor{}); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(dir, "r", "requests.csv")); err == nil {
		t.Fatal("requests.csv must not exist without --request-log-every")
	}
}

func TestAnchor(t *testing.T) {
	a := Anchor{Mono: 5_000_000_000, Wall: time.Unix(1000, 0)}
	if got := a.At(5_000_000_000 + 1500); got.UnixNano() != 1000_000_001_500 {
		t.Fatal(got.UnixNano())
	}
	if got := a.At(4_000_000_000); got.Unix() != 999 {
		t.Fatal(got)
	}
}
