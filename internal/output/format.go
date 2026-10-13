// Package output writes triton-stress's records: intervals.csv, events.jsonl, the optional
// requests.csv and the JSON lines on stdout.
package output

import (
	"encoding/json"
	"math"
	"strconv"
	"strings"
	"time"
)

// Anchor maps the monotonic clock (clock.Now values) to UTC. It is sampled afresh
// every tick, so a stepped wall clock only affects the tick in which it happens.
type Anchor struct {
	Mono int64
	Wall time.Time
}

// At converts a monotonic timestamp to UTC.
func (a Anchor) At(mono int64) time.Time { return a.Wall.Add(time.Duration(mono - a.Mono)).UTC() }

// FormatTS is the ts column: epoch seconds with millisecond precision, as tbench writes it
// (1785922486.968).
func FormatTS(t time.Time) string {
	ms := t.UnixMilli()
	sign := ""
	if ms < 0 {
		sign, ms = "-", -ms
	}
	frac := strconv.FormatInt(ms%1000, 10)
	return sign + strconv.FormatInt(ms/1000, 10) + "." + strings.Repeat("0", 3-len(frac)) + frac
}

// FormatISO is the iso column: RFC 3339 in UTC with millisecond precision.
func FormatISO(t time.Time) string { return t.UTC().Format("2006-01-02T15:04:05.000Z") }

// RunDirName is the default run directory name for a start time: 20261009T142300Z.
func RunDirName(t time.Time) string { return t.UTC().Format("20060102T150405Z") }

// KV is one field of a record: a name and its value already rendered as JSON.
// CSV output uses the same rendering (strings unquoted).
type KV struct {
	Name string
	// JSON is the JSON text of the value ("null" when the value is missing).
	JSON string
	// Str marks a string value, which the CSV writes without its quotes.
	Str bool
}

func num(name string, v float64, prec int) KV {
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return KV{Name: name, JSON: "null"}
	}
	return KV{Name: name, JSON: strconv.FormatFloat(v, 'f', prec, 64)}
}

func integer(name string, v int64) KV { return KV{Name: name, JSON: strconv.FormatInt(v, 10)} }

func str(name, v string) KV {
	b, _ := json.Marshal(v)
	return KV{Name: name, JSON: string(b), Str: true}
}

func null(name string) KV { return KV{Name: name, JSON: "null"} }

// jsonObject renders fields as one JSON object, in order.
func jsonObject(kvs []KV, extra ...string) []byte {
	b := make([]byte, 0, 512)
	b = append(b, '{')
	for i, kv := range kvs {
		if i > 0 {
			b = append(b, ',')
		}
		k, _ := json.Marshal(kv.Name)
		b = append(b, k...)
		b = append(b, ':')
		b = append(b, kv.JSON...)
	}
	for _, e := range extra { // pre-rendered `"name":value` members
		b = append(b, ',')
		b = append(b, e...)
	}
	return append(b, '}')
}

func csvField(kv KV) string {
	if kv.JSON == "null" && !kv.Str {
		return ""
	}
	if kv.Str {
		var s string
		_ = json.Unmarshal([]byte(kv.JSON), &s)
		return csvEscape(s)
	}
	return kv.JSON
}

func csvEscape(s string) string {
	if strings.ContainsAny(s, ",\"\n\r") {
		return `"` + strings.ReplaceAll(s, `"`, `""`) + `"`
	}
	return s
}
