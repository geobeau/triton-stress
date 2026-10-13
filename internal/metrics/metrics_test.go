package metrics

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/criteo/triton-stress/internal/grpcclient"
	"github.com/criteo/triton-stress/internal/stats"
	dto "github.com/prometheus/client_model/go"
	"github.com/prometheus/common/expfmt"
)

const protoAccept = "application/vnd.google.protobuf;proto=io.prometheus.client.MetricFamily;encoding=delimited"

func fixture(t *testing.T) (*Metrics, *httptest.Server) {
	t.Helper()
	ws := []*stats.Worker{stats.NewWorker(0, 0), stats.NewWorker(1, 0)}
	for i, w := range ws {
		for j := 0; j < 50; j++ {
			w.Sent(100)
			// latencies from 200 us to ~20 ms
			done := int64(1000 + 200_000*(j+1)*(i+1))
			w.Done(stats.Record{TSched: 1000, TEmit: 1200, TReady: 1300, TSend: 1400, TDone: done,
				Status: grpcclient.StatusOK, BytesDown: 8}, true)
		}
		w.Done(stats.Record{TSched: 1000, TEmit: 1100, TDone: 1100, Status: grpcclient.StatusConnect, Conn: -1}, false)
	}
	ws[0].Sent(100) // one still in flight
	m := New("grpc", "mymodel", ws)
	m.SetOffered(1234)
	m.SetPhase("ramp")
	m.Breach("latency")
	m.Trip("latency")
	srv := httptest.NewServer(m.Handler())
	t.Cleanup(srv.Close)
	return m, srv
}

func scrape(t *testing.T, url, accept string) (*http.Response, []byte) {
	t.Helper()
	req, _ := http.NewRequest("GET", url, nil)
	if accept != "" {
		req.Header.Set("Accept", accept)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = resp.Body.Close() }()
	b, _ := io.ReadAll(resp.Body)
	return resp, b
}

var histNames = []string{
	"triton_stress_latency_from_scheduled_seconds", "triton_stress_latency_from_sent_seconds",
	"triton_stress_send_delay_seconds", "triton_stress_sched_lag_seconds",
}

func TestProtobufScrapeHasClassicAndNativeBuckets(t *testing.T) {
	_, srv := fixture(t)
	resp, body := scrape(t, srv.URL, protoAccept)
	if ct := resp.Header.Get("Content-Type"); !strings.Contains(ct, "protobuf") {
		t.Fatalf("content type %q", ct)
	}
	dec := expfmt.NewDecoder(strings.NewReader(string(body)), expfmt.ResponseFormat(resp.Header))
	fams := map[string]*dto.MetricFamily{}
	for {
		var mf dto.MetricFamily
		if err := dec.Decode(&mf); err != nil {
			break
		}
		fams[mf.GetName()] = &mf
	}
	for _, name := range histNames {
		mf := fams[name]
		if mf == nil || len(mf.Metric) != 1 {
			t.Fatalf("%s missing", name)
		}
		h := mf.Metric[0].Histogram
		if h == nil {
			t.Fatalf("%s: no histogram", name)
		}
		if h.GetSchema() != 3 {
			t.Errorf("%s: schema %d, want 3", name, h.GetSchema())
		}
		if len(h.PositiveSpan) == 0 || len(h.PositiveDelta) == 0 {
			t.Errorf("%s: no native buckets", name)
		}
		if len(h.Bucket) != 20 || h.Bucket[0].GetUpperBound() != 10e-6 || h.Bucket[19].GetUpperBound() != 20 {
			t.Errorf("%s: classic buckets %d", name, len(h.Bucket))
		}
		// the native deltas must add up to the sample count (less the zero bucket)
		var run, total int64
		for _, d := range h.PositiveDelta {
			run += d
			total += run
		}
		if uint64(total)+h.GetZeroCount() != h.GetSampleCount() {
			t.Errorf("%s: native %d + zero %d != count %d", name, total, h.GetZeroCount(), h.GetSampleCount())
		}
		labels := map[string]string{}
		for _, l := range mf.Metric[0].Label {
			labels[l.GetName()] = l.GetValue()
		}
		if labels["protocol"] != "grpc" || labels["model"] != "mymodel" || len(labels) != 2 {
			t.Errorf("%s labels %v", name, labels)
		}
	}
	lat := fams["triton_stress_latency_from_scheduled_seconds"].Metric[0].Histogram
	if lat.GetSampleCount() != 100 {
		t.Fatalf("latency samples %d (the unsent requests must not be counted)", lat.GetSampleCount())
	}
	if fams["triton_stress_sched_lag_seconds"].Metric[0].Histogram.GetSampleCount() != 102 {
		t.Fatal("sched_lag counts every request, sent or not")
	}
	gauge := func(name string) float64 { return fams[name].Metric[0].Gauge.GetValue() }
	counter := func(name string) float64 { return fams[name].Metric[0].Counter.GetValue() }
	if gauge("triton_stress_offered_rate_rps") != 1234 || gauge("triton_stress_inflight_requests") != 1 {
		t.Fatal("gauges")
	}
	if counter("triton_stress_requests_sent_total") != 101 || counter("triton_stress_bytes_sent_total") != 10100 || counter("triton_stress_bytes_received_total") != 800 {
		t.Fatalf("counters %v %v %v", counter("triton_stress_requests_sent_total"), counter("triton_stress_bytes_sent_total"), counter("triton_stress_bytes_received_total"))
	}
	byStatus := map[string]float64{}
	for _, m := range fams["triton_stress_requests_completed_total"].Metric {
		for _, l := range m.Label {
			if l.GetName() == "status" {
				byStatus[l.GetValue()] = m.Counter.GetValue()
			}
		}
	}
	if byStatus["OK"] != 100 || byStatus["connect"] != 2 || len(byStatus) != 2 {
		t.Fatalf("%v", byStatus)
	}
	if len(fams["triton_stress_phase"].Metric) != 4 {
		t.Fatal("triton_stress_phase has one series per phase")
	}
	for _, m := range fams["triton_stress_phase"].Metric {
		var phase string
		for _, l := range m.Label {
			if l.GetName() == "phase" {
				phase = l.GetValue()
			}
		}
		if want := map[bool]float64{true: 1, false: 0}[phase == "ramp"]; m.Gauge.GetValue() != want {
			t.Errorf("phase %s = %v", phase, m.Gauge.GetValue())
		}
	}
	if len(fams["triton_stress_guard_breaches_total"].Metric) != 5 || len(fams["triton_stress_guard_trips_total"].Metric) != 5 {
		t.Fatal("guard counters carry a guard label, one per guard")
	}
}

func TestTextScrapeHasClassicBucketsOnly(t *testing.T) {
	_, srv := fixture(t)
	resp, body := scrape(t, srv.URL, "text/plain;version=0.0.4")
	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "text/plain") {
		t.Fatalf("content type %q", ct)
	}
	text := string(body)
	for _, want := range []string{
		`triton_stress_latency_from_scheduled_seconds_bucket{model="mymodel",protocol="grpc",le="1e-05"}`,
		`triton_stress_latency_from_scheduled_seconds_bucket{model="mymodel",protocol="grpc",le="20"}`,
		`triton_stress_latency_from_scheduled_seconds_bucket{model="mymodel",protocol="grpc",le="+Inf"} 100`,
		`triton_stress_latency_from_scheduled_seconds_count{model="mymodel",protocol="grpc"} 100`,
		`triton_stress_requests_completed_total{model="mymodel",protocol="grpc",status="connect"} 2`,
		`triton_stress_offered_rate_rps{model="mymodel",protocol="grpc"} 1234`,
		`triton_stress_phase{model="mymodel",phase="ramp",protocol="grpc"} 1`,
		`triton_stress_guard_breaches_total{guard="latency",model="mymodel",protocol="grpc"} 1`,
	} {
		if !strings.Contains(text, want) {
			t.Errorf("missing %q", want)
		}
	}
	if strings.Contains(strings.ToLower(text), "native") || strings.Contains(text, "span") {
		t.Error("text format must not carry native buckets")
	}
	// the default (no Accept header) is text too
	if _, b := scrape(t, srv.URL, ""); !strings.Contains(string(b), "# TYPE triton_stress_sched_lag_seconds histogram") {
		t.Error("default scrape")
	}
}

func TestServe(t *testing.T) {
	m := New("grpc", "m", nil)
	s, err := Serve("127.0.0.1:0", m)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = s.Close() }()
	resp, body := scrape(t, "http://"+s.Addr()+"/metrics", "")
	if resp.StatusCode != 200 || !strings.Contains(string(body), "triton_stress_inflight_requests") {
		t.Fatal(resp.StatusCode)
	}
	if _, err := Serve(s.Addr(), m); err == nil {
		t.Fatal("binding a taken address must fail")
	}
}
