package main

import (
	"errors"
	"fmt"
	"io"
	"runtime"
	"time"

	"github.com/alecthomas/kingpin/v2"
	"github.com/criteo/triton-stress/internal/arrival"
	"github.com/criteo/triton-stress/internal/grpcclient"
	"github.com/criteo/triton-stress/internal/guard"
	"github.com/criteo/triton-stress/internal/load"
	"github.com/criteo/triton-stress/internal/units"
	"github.com/criteo/triton-stress/pkg/tensorgen"
)

// config is the validated command line.
type config struct {
	Model        string
	ModelVersion string
	Target       string
	Protocol     string
	TLS          grpcclient.TLSMode
	Connections  int
	BatchSize    int
	Inputs       []tensorgen.Spec
	Gen          tensorgen.Dist
	PoolSize     int
	Seed         uint64
	Timeout      time.Duration
	BackoffMin   time.Duration
	BackoffMax   time.Duration
	Validate     bool

	Workers         int
	Arrival         arrival.Mode
	Schedule        load.Schedule
	Guards          guard.Config
	Interval        time.Duration
	MetricsAddr     string // empty: disabled
	OutDir          string // empty: runs/<UTC start>
	RequestLogEvery uint64
}

const (
	defaultGRPCTarget      = "127.0.0.1:8001"
	defaultHTTPTarget      = "127.0.0.1:8000"
	defaultGRPCConnections = 128
	defaultHTTPConnections = 2048
)

// buildVersion is set at build time with -ldflags "-X main.buildVersion=<tag>".
var buildVersion = "dev"

// parseConfig parses and validates args (without the program name). Usage text and
// parse errors go to stderr.
func parseConfig(args []string, stderr io.Writer) (*config, error) {
	app := kingpin.New("rts", "Open-loop load generator for NVIDIA Triton Inference Server.")
	app.UsageWriter(stderr).ErrorWriter(stderr)
	app.Version(buildVersion)

	model := app.Flag("model", "Model name.").Required().String()
	version := app.Flag("model-version", "Model version (empty is Triton's latest).").Default("").String()
	target := app.Flag("target", "Triton endpoint, host:port (default 127.0.0.1:8001 for grpc, 127.0.0.1:8000 for http).").Default("").String()
	protocol := app.Flag("protocol", "grpc or http.").Default("grpc").Enum("grpc", "http")
	useTLS := app.Flag("tls", "Use TLS, verifying the server against the system roots.").Bool()
	tlsInsecure := app.Flag("tls-insecure", "Use TLS without verifying the server.").Bool()
	var connectionsSet bool
	connections := app.Flag("connections", "Connections (default 128 for grpc, 2048 for http).").IsSetByUser(&connectionsSet).Int()
	batch := app.Flag("batch-size", "Client-side batch size.").Default("1").Int()
	inputs := app.Flag("input", "Input override or extra input, name:DTYPE:d1,d2 (dims exclude the batch dimension). Repeatable.").Strings()
	gen := app.Flag("gen", "Value distribution for every input: zeros | const:<v> | uniform:<lo>..<hi> | random-bytes.").Default("uniform:0..1000").String()
	pool := app.Flag("payload-pool", "Number of distinct pre-encoded requests.").Default("256").Int()
	seed := app.Flag("seed", "Random seed.").Default("1").Uint64()
	timeout := app.Flag("request-timeout", "Client-side timeout per request.").Default("10s").Duration()
	bmin := app.Flag("reconnect-backoff-min", "Reconnect backoff start.").Default("10ms").Duration()
	bmax := app.Flag("reconnect-backoff-max", "Reconnect backoff ceiling.").Default("1s").Duration()
	validate := app.Flag("validate", "Decode and check every response.").Bool()

	workers := app.Flag("workers", "Dispatcher goroutines (default: GOMAXPROCS, which follows the CPU limit of the container).").Int()
	arrivalMode := app.Flag("arrival", "Spacing between requests: constant or poisson.").Default("constant").Enum("constant", "poisson")
	warmup := app.Flag("warmup", "Warmup: hold --start-rate for this long (for example 10s).").Default("10s").String()
	startRate := app.Flag("start-rate", "Start rate in req/s (for example 1k).").Default("1k").String()
	slope := app.Flag("slope", "Ramp slope in req/s per second (for example 1k/s); 0 keeps the rate flat.").Default("1k/s").String()
	maxRate := app.Flag("max-rate", "Rate at which the ramp ends the run (for example 1M).").Default("1M").String()
	duration := app.Flag("duration", "Total run time including the warmup; default none.").Default("").String()
	guardAchieved := app.Flag("guard-achieved", "Breach when ok/s is below this fraction of offered/s, or off.").Default("0.95").String()
	guardErr := app.Flag("guard-error-rate", "Breach when the error fraction is above this, or off.").Default("0.01").String()
	guardLat := app.Flag("guard-latency", "Breach when pNN of latency from scheduled is above the limit (pNN:duration), or off.").Default("p99:100ms").String()
	guardDelay := app.Flag("guard-send-delay", "Breach when pNN of send delay is above the limit (pNN:duration), or off.").Default("p99:1ms").String()
	guardInflight := app.Flag("guard-max-inflight", "Trip at once when in-flight requests exceed this, or off.").Default("100k").String()
	guardConsecutive := app.Flag("guard-consecutive", "Consecutive breaching ticks that trip a guard.").Default("3").Int()
	interval := app.Flag("interval", "Tick length.").Default("1s").Duration()
	metricsAddr := app.Flag("metrics-addr", "Prometheus listen address, or off.").Default("0.0.0.0:9095").String()
	outDir := app.Flag("out-dir", "Output directory (default runs/<UTC start>).").Default("").String()
	requestLog := app.Flag("request-log-every", "Write one request in N to requests.csv (default off).").Default("0").String()

	if _, err := app.Parse(args); err != nil {
		return nil, err
	}
	c := &config{
		Model: *model, ModelVersion: *version, Target: *target, Protocol: *protocol,
		Connections: *connections, BatchSize: *batch, PoolSize: *pool, Seed: *seed,
		Timeout: *timeout, BackoffMin: *bmin, BackoffMax: *bmax, Validate: *validate,
	}
	if *useTLS && *tlsInsecure {
		return nil, errors.New("--tls and --tls-insecure are mutually exclusive")
	}
	switch {
	case *useTLS:
		c.TLS = grpcclient.TLSSystem
	case *tlsInsecure:
		c.TLS = grpcclient.TLSInsecure
	}
	if c.Target == "" {
		c.Target = defaultGRPCTarget
		if c.Protocol == "http" {
			c.Target = defaultHTTPTarget
		}
	}
	if !connectionsSet {
		c.Connections = defaultGRPCConnections
		if c.Protocol == "http" {
			c.Connections = defaultHTTPConnections
		}
	}
	if c.Connections < 1 {
		return nil, errors.New("--connections must be at least 1")
	}
	if c.BatchSize < 1 {
		return nil, errors.New("--batch-size must be at least 1")
	}
	if c.PoolSize < 1 {
		return nil, errors.New("--payload-pool must be at least 1")
	}
	if c.Timeout <= 0 {
		return nil, errors.New("--request-timeout must be positive")
	}
	if c.BackoffMin <= 0 || c.BackoffMax < c.BackoffMin {
		return nil, errors.New("--reconnect-backoff-min must be positive and not above --reconnect-backoff-max")
	}
	var err error
	if err := c.parseLoad(*workers, *arrivalMode, *warmup, *startRate, *slope, *maxRate, *duration,
		*guardAchieved, *guardErr, *guardLat, *guardDelay, *guardInflight, *guardConsecutive,
		*interval, *metricsAddr, *outDir, *requestLog); err != nil {
		return nil, err
	}
	if c.Gen, err = tensorgen.ParseDist(*gen); err != nil {
		return nil, fmt.Errorf("--gen: %w", err)
	}
	for _, s := range *inputs {
		spec, err := tensorgen.ParseSpec(s)
		if err != nil {
			return nil, fmt.Errorf("--input: %w", err)
		}
		c.Inputs = append(c.Inputs, spec)
	}
	return c, nil
}

// parseLoad fills in and validates the load flags.
func (c *config) parseLoad(workers int, arrivalMode, warmup, startRate, slope, maxRate, duration,
	gAchieved, gErr, gLat, gDelay, gInflight string, gConsecutive int, interval time.Duration,
	metricsAddr, outDir, requestLog string) error {
	if workers < 0 {
		return errors.New("--workers must not be negative")
	}
	c.Workers = workers
	if c.Workers == 0 {
		c.Workers = runtime.GOMAXPROCS(0)
	}
	if arrivalMode == "poisson" {
		c.Arrival = arrival.Poisson
	}
	var err error
	s := &c.Schedule
	if s.Warmup, err = units.ParseDuration(warmup); err != nil {
		return fmt.Errorf("--warmup: %w", err)
	}
	if s.Start, err = units.ParseRate(startRate); err != nil {
		return fmt.Errorf("--start-rate: %w", err)
	}
	if s.Slope, err = units.ParseSlope(slope); err != nil {
		return fmt.Errorf("--slope: %w", err)
	}
	if s.Max, err = units.ParseRate(maxRate); err != nil {
		return fmt.Errorf("--max-rate: %w", err)
	}
	if duration != "" {
		if s.Duration, err = units.ParseDuration(duration); err != nil {
			return fmt.Errorf("--duration: %w", err)
		}
		if s.Duration == 0 {
			return errors.New("--duration must be above 0 (leave it out for none)")
		}
	}
	if err := s.Validate(); err != nil {
		return err
	}
	g := &c.Guards
	if g.Achieved, err = units.ParseFractionOrOff(gAchieved); err != nil {
		return fmt.Errorf("--guard-achieved: %w", err)
	}
	if g.ErrorRate, err = units.ParseFractionOrOff(gErr); err != nil {
		return fmt.Errorf("--guard-error-rate: %w", err)
	}
	if g.Latency, err = units.ParsePctThreshold(gLat); err != nil {
		return fmt.Errorf("--guard-latency: %w", err)
	}
	if g.SendDelay, err = units.ParsePctThreshold(gDelay); err != nil {
		return fmt.Errorf("--guard-send-delay: %w", err)
	}
	if g.MaxInflight, err = units.ParseCountOrOff(gInflight); err != nil {
		return fmt.Errorf("--guard-max-inflight: %w", err)
	}
	if gConsecutive < 1 {
		return errors.New("--guard-consecutive must be at least 1")
	}
	g.Consecutive = gConsecutive
	if interval <= 0 {
		return errors.New("--interval must be positive")
	}
	c.Interval = interval
	if metricsAddr != "off" {
		c.MetricsAddr = metricsAddr
	}
	c.OutDir = outDir
	if c.RequestLogEvery, err = units.ParseCount(requestLog); err != nil {
		return fmt.Errorf("--request-log-every: %w", err)
	}
	return nil
}
