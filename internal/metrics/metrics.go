// Package metrics exposes RTS's Prometheus metrics. The histograms and counters are read
// at scrape time from the workers' own cumulative statistics, so the request hot path
// never touches a shared metric. Each histogram carries classic buckets (1-2-5, 10 us to
// 20 s) and native buckets (schema 3); promhttp picks the format from the Accept header
// (protobuf includes both, text only the classic buckets).
package metrics

import (
	"context"
	"errors"
	"net"
	"net/http"
	"time"

	"github.com/criteo/triton-stress/internal/grpcclient"
	"github.com/criteo/triton-stress/internal/guard"
	"github.com/criteo/triton-stress/internal/stats"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	dto "github.com/prometheus/client_model/go"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/timestamppb"
)

// NativeBucketFactor is the native histograms' growth factor, 2^(1/8) (schema 3).
const NativeBucketFactor = 1.0905077326652577

// zeroThreshold is the width of the native zero bucket: 2^-32 s.
const zeroThreshold = 2.3283064365386963e-10

// Phases are the values of the rts_phase gauge's phase label.
var Phases = []string{"warmup", "ramp", "drain", "done"}

// Metrics is the registry and the controller-fed series.
type Metrics struct {
	reg        *prometheus.Registry
	offered    prometheus.Gauge
	phase      *prometheus.GaugeVec
	breaches   *prometheus.CounterVec
	trips      *prometheus.CounterVec
	phaseNames []string
}

// New builds the metrics for a run. protocol and model become labels of every series.
func New(protocol, model string, workers []*stats.Worker) *Metrics {
	reg := prometheus.NewRegistry()
	w := prometheus.WrapRegistererWith(prometheus.Labels{"protocol": protocol, "model": model}, reg)
	m := &Metrics{
		reg: reg,
		offered: prometheus.NewGauge(prometheus.GaugeOpts{
			Name: "rts_offered_rate_rps", Help: "Offered request rate of the last tick, in requests per second."}),
		phase: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: "rts_phase", Help: "1 for the current phase, 0 for the others."}, []string{"phase"}),
		breaches: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "rts_guard_breaches_total", Help: "Ticks (or in-flight checks) in which a guard was breached."}, []string{"guard"}),
		trips: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "rts_guard_trips_total", Help: "Guards that stopped the run."}, []string{"guard"}),
		phaseNames: Phases,
	}
	for _, p := range Phases {
		m.phase.WithLabelValues(p).Set(0)
	}
	for _, g := range guard.Names {
		m.breaches.WithLabelValues(g)
		m.trips.WithLabelValues(g)
	}
	w.MustRegister(m.offered, m.phase, m.breaches, m.trips, newLoadCollector(workers, protocol))
	return m
}

// SetOffered sets the offered rate gauge.
func (m *Metrics) SetOffered(rps float64) { m.offered.Set(rps) }

// SetPhase marks phase as the current one.
func (m *Metrics) SetPhase(phase string) {
	for _, p := range m.phaseNames {
		v := 0.0
		if p == phase {
			v = 1
		}
		m.phase.WithLabelValues(p).Set(v)
	}
}

// Breach counts a guard breach.
func (m *Metrics) Breach(guardName string) { m.breaches.WithLabelValues(guardName).Inc() }

// Trip counts a guard trip.
func (m *Metrics) Trip(guardName string) { m.trips.WithLabelValues(guardName).Inc() }

// Handler serves /metrics, negotiating protobuf or text from the Accept header.
func (m *Metrics) Handler() http.Handler {
	return promhttp.HandlerFor(m.reg, promhttp.HandlerOpts{})
}

// Server is a running metrics endpoint.
type Server struct {
	srv *http.Server
	ln  net.Listener
}

// Serve listens on addr and serves /metrics in the background.
func Serve(addr string, m *Metrics) (*Server, error) {
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return nil, err
	}
	mux := http.NewServeMux()
	mux.Handle("/metrics", m.Handler())
	s := &Server{srv: &http.Server{Handler: mux, ReadHeaderTimeout: 10 * time.Second}, ln: ln}
	go func() {
		if err := s.srv.Serve(ln); err != nil && !errors.Is(err, http.ErrServerClosed) {
			// the listener is gone; nothing else to do for a metrics endpoint
			_ = err
		}
	}()
	return s, nil
}

// Addr is the address being served.
func (s *Server) Addr() string { return s.ln.Addr().String() }

// Close stops the server.
func (s *Server) Close() error {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	return s.srv.Shutdown(ctx)
}

// loadCollector exports the workers' cumulative statistics.
type loadCollector struct {
	workers   []*stats.Worker
	okStatus  int // the success status, always exported
	created   time.Time
	hists     [stats.NumHists]*prometheus.Desc
	sent      *prometheus.Desc
	completed *prometheus.Desc
	bytesUp   *prometheus.Desc
	bytesDown *prometheus.Desc
	inflight  *prometheus.Desc
}

func newLoadCollector(workers []*stats.Worker, protocol string) *loadCollector {
	c := &loadCollector{workers: workers, created: time.Now(), okStatus: int(grpcclient.StatusOK)}
	if protocol == "http" {
		c.okStatus = int(grpcclient.StatusHTTP200)
	}
	help := [stats.NumHists]string{
		"Latency from the scheduled send time to the completed response.",
		"Latency from the actual send time to the completed response.",
		"Delay between the scheduled and the actual send time.",
		"Delay between the scheduled time and the dispatcher handling the request.",
	}
	names := [stats.NumHists]string{
		"rts_latency_from_scheduled_seconds", "rts_latency_from_sent_seconds",
		"rts_send_delay_seconds", "rts_sched_lag_seconds",
	}
	for i := range c.hists {
		c.hists[i] = prometheus.NewDesc(names[i], help[i], nil, nil)
	}
	c.sent = prometheus.NewDesc("rts_requests_sent_total", "Requests sent.", nil, nil)
	c.completed = prometheus.NewDesc("rts_requests_completed_total", "Requests completed, by status.", []string{"status"}, nil)
	c.bytesUp = prometheus.NewDesc("rts_bytes_sent_total", "Request message bytes sent.", nil, nil)
	c.bytesDown = prometheus.NewDesc("rts_bytes_received_total", "Response message bytes received.", nil, nil)
	c.inflight = prometheus.NewDesc("rts_inflight_requests", "Requests sent and not yet completed.", nil, nil)
	return c
}

// Describe sends nothing: this is an unchecked collector.
func (c *loadCollector) Describe(chan<- *prometheus.Desc) {}

// Collect implements prometheus.Collector.
func (c *loadCollector) Collect(ch chan<- prometheus.Metric) {
	var cum stats.Cum
	var sent, bytesUp uint64
	var inflight int64
	for _, w := range c.workers {
		w.AddCumulative(&cum)
		s, b := w.Totals()
		sent += s
		bytesUp += b
		inflight += w.Inflight()
	}
	ch <- prometheus.MustNewConstMetric(c.sent, prometheus.CounterValue, float64(sent))
	ch <- prometheus.MustNewConstMetric(c.bytesUp, prometheus.CounterValue, float64(bytesUp))
	ch <- prometheus.MustNewConstMetric(c.bytesDown, prometheus.CounterValue, float64(cum.BytesDown))
	ch <- prometheus.MustNewConstMetric(c.inflight, prometheus.GaugeValue, float64(max(inflight, 0)))
	for s := 0; s < stats.NumStatuses; s++ {
		// The success status (OK, or http_200) is always exported; the others once seen.
		if n := cum.ByStatus[s]; n > 0 || s == c.okStatus {
			ch <- prometheus.MustNewConstMetric(c.completed, prometheus.CounterValue, float64(n), grpcclient.Status(s).Label())
		}
	}
	for i := range c.hists {
		ch <- &histMetric{desc: c.hists[i], h: cum.Hist[i], created: c.created}
	}
}

// histMetric is a histogram snapshot with classic and native buckets, which client_golang's
// own const histograms cannot carry together.
type histMetric struct {
	desc    *prometheus.Desc
	h       stats.PromHist
	created time.Time
}

func (m *histMetric) Desc() *prometheus.Desc { return m.desc }

func (m *histMetric) Write(out *dto.Metric) error {
	cum := m.h.ClassicCumulative()
	buckets := make([]*dto.Bucket, len(cum))
	for i, n := range cum {
		buckets[i] = &dto.Bucket{CumulativeCount: proto.Uint64(n), UpperBound: proto.Float64(stats.ClassicBounds[i])}
	}
	spans, deltas := m.h.NativePositive()
	pspans := make([]*dto.BucketSpan, len(spans))
	for i, s := range spans {
		pspans[i] = &dto.BucketSpan{Offset: proto.Int32(s.Offset), Length: proto.Uint32(s.Length)}
	}
	out.Histogram = &dto.Histogram{
		SampleCount:      proto.Uint64(m.h.Count),
		SampleSum:        proto.Float64(m.h.Sum),
		Bucket:           buckets,
		CreatedTimestamp: timestamppb.New(m.created),
		Schema:           proto.Int32(stats.NativeSchema),
		ZeroThreshold:    proto.Float64(zeroThreshold),
		ZeroCount:        proto.Uint64(m.h.Zero),
		PositiveSpan:     pspans,
		PositiveDelta:    deltas,
	}
	return nil
}
