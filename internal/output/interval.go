package output

import (
	"encoding/json"
	"sort"
	"strings"
	"time"

	"github.com/criteo/triton-stress/internal/health"
	"github.com/criteo/triton-stress/internal/stats"
)

// Percentile columns of every histogram.
var pctNames = [5]string{"p50", "p90", "p99", "p999", "max"}

// Percentiles are the quantiles written for each histogram (the last column is the max).
var Percentiles = [4]float64{50, 90, 99, 99.9}

// HistSummary is the percentiles of one histogram in nanoseconds; Valid is false when
// the histogram had no samples in the tick.
type HistSummary struct {
	Valid bool
	Ns    [5]int64 // p50, p90, p99, p99.9, max
}

// Interval is one tick's record.
type Interval struct {
	TS    time.Time
	Phase string
	// Rates are requests per second over the tick.
	OfferedRPS, SentRPS, OKRPS, ErrorRPS float64
	Inflight                             int64
	Hist                                 [stats.NumHists]HistSummary
	Health                               health.Health
	HasHealth                            bool
	// Status is the tick's completions per status label (nonzero entries only).
	Status map[string]uint64
}

// IntervalColumns are the intervals.csv column names, in order.
func IntervalColumns() []string {
	var iv Interval
	kvs := iv.fields()
	cols := make([]string, len(kvs))
	for i, kv := range kvs {
		cols[i] = kv.Name
	}
	return cols
}

func us(ns int64) float64 { return float64(ns) / 1e3 }

func (iv *Interval) fields() []KV {
	kvs := []KV{
		{Name: "ts", JSON: FormatTS(iv.TS)},
		str("iso", FormatISO(iv.TS)),
		str("phase", iv.Phase),
		num("offered_rps", iv.OfferedRPS, 1),
		num("sent_rps", iv.SentRPS, 1),
		num("ok_rps", iv.OKRPS, 1),
		num("error_rps", iv.ErrorRPS, 1),
		integer("inflight", iv.Inflight),
	}
	for h := 0; h < stats.NumHists; h++ {
		for p, name := range pctNames {
			col := stats.HistNames[h] + "_" + name + "_us"
			if iv.Hist[h].Valid {
				kvs = append(kvs, num(col, us(iv.Hist[h].Ns[p]), 1))
			} else {
				kvs = append(kvs, null(col))
			}
		}
	}
	if iv.HasHealth {
		h := iv.Health
		kvs = append(kvs,
			num("cpu_user_pct", h.CPUUserPct, 1),
			num("cpu_sys_pct", h.CPUSysPct, 1),
			num("busiest_thread_cpu_pct", h.BusiestThreadPct, 1),
			integer("nonvol_ctx_switches", h.NonvolCtxSwitches))
		if h.Cgroup {
			kvs = append(kvs, integer("cgroup_nr_throttled", h.NrThrottled), integer("cgroup_throttled_us", h.ThrottledUS))
		} else {
			kvs = append(kvs, null("cgroup_nr_throttled"), null("cgroup_throttled_us"))
		}
	} else {
		kvs = append(kvs, null("cpu_user_pct"), null("cpu_sys_pct"), null("busiest_thread_cpu_pct"),
			null("nonvol_ctx_switches"), null("cgroup_nr_throttled"), null("cgroup_throttled_us"))
	}
	return kvs
}

// CSVHeader is the intervals.csv header line (with the newline).
func CSVHeader() string { return strings.Join(IntervalColumns(), ",") + "\n" }

// CSVLine is the intervals.csv line (with the newline).
func (iv *Interval) CSVLine() string {
	kvs := iv.fields()
	f := make([]string, len(kvs))
	for i, kv := range kvs {
		f[i] = csvField(kv)
	}
	return strings.Join(f, ",") + "\n"
}

// JSONLine is the stdout line: the interval record plus the per-status breakdown.
func (iv *Interval) JSONLine() []byte {
	labels := make([]string, 0, len(iv.Status))
	for l := range iv.Status {
		labels = append(labels, l)
	}
	sort.Strings(labels)
	st := []byte(`"status":{`)
	for i, l := range labels {
		if i > 0 {
			st = append(st, ',')
		}
		k, _ := json.Marshal(l)
		st = append(st, k...)
		st = append(st, ':')
		st = append(st, []byte(num("", float64(iv.Status[l]), 0).JSON)...)
	}
	st = append(st, '}')
	return append(jsonObject(iv.fields(), string(st)), '\n')
}
