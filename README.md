# triton-stress

triton-stress is an open-loop load generator for NVIDIA Triton Inference Server, written in Go. It drives a
Triton over gRPC or HTTP at a configured request rate, measures latency without coordinated
omission, stops itself when the system under test (or triton-stress) falls behind, and writes one record per
second to CSV, JSON lines and Prometheus.

- **Open loop.** Requests go out on a schedule whether or not earlier ones have returned. A slow
  Triton does not slow the offered load down; it shows up as latency, in-flight requests and errors.
- **Arrivals.** Constant spacing (default) or Poisson.
- **Rate plan.** A warmup at a fixed rate, then a linear ramp (or a flat rate with `--slope 0`).
- **Stops on guards.** Achieved rate, error rate, latency, send delay and in-flight count, each with
  a threshold. There is no hold and no ramp-down: triton-stress stops and drains.
- **Protocols.** gRPC (KServe v2 `ModelInfer`) and HTTP/1.1 (KServe v2 binary tensor extension).
- **Client-side batch size.** `--batch-size N` packs N batch elements into every request.
- **Generated tensors.** Input and output tensors come from the model's metadata and config,
  filled from a seeded distribution and pre-encoded once into a pool, so nothing is serialised per
  request.
- **Linux only** (timerfd scheduling, cgroup v2 health counters).

Rates in this document are always requests per second (req/s).

## Build

```
make build   # CGO_ENABLED=0 go build -o bin/triton-stress ./cmd/triton-stress   (static binary)
make test    # go test ./... in the main module and in pkg/tensorgen
make race    # go test -race on internal/... and cmd/..., and on pkg/tensorgen
make lint    # gofmt -s, go vet in both modules, and a check that pkg/tensorgen has no protobuf/grpc dependency
make proto   # regenerate pkg/tritonpb from proto/ (needs protoc, protoc-gen-go, protoc-gen-go-grpc)
```

- Go 1.27 for the main module (`go 1.27` in `go.mod`); `pkg/tensorgen` is a separate module at
  `go 1.24`. The Makefile puts `$HOME/.local/go/bin` and `$HOME/go/bin` on `PATH`.
- `bin/triton-stress` is a statically linked executable with no runtime dependencies: copy it into any image.
- Go 1.25 or later is what makes `GOMAXPROCS` follow the container CPU limit (see
  [Sizing](#sizing-and-performance)).

## Quick start

All examples use `model_2026_07_15` (inputs `x` INT32[208] and `multicategorical` INT32[85] per batch
element, output `PSale25H_ClickedDisplay` FP32[1] per batch element).

**A flat run**: 20k req/s for 70 s (10 s warmup plus 60 s guarded), batch size 4, over gRPC.

```
./bin/triton-stress --model model_2026_07_15 --target triton:8001 --batch-size 4 \
  --start-rate 20k --slope 0 --warmup 10s --duration 70s
```

**A ramp to saturation**: from 10k req/s, rising by 5k req/s every second, until a guard trips or
400k req/s is reached.

```
./bin/triton-stress --model model_2026_07_15 --target triton:8001 --batch-size 4 \
  --start-rate 10k --slope 5k/s --max-rate 400k --warmup 10s --out-dir runs/ramp-grpc
```

The run ends with a `stop` event (`guard:send_delay`, `guard:latency`, `max_rate_reached`, ...). triton-stress
does not compute a "maximum sustainable rate": read it from `intervals.csv` (see
[Outputs](#outputs)). A guard trips after `--guard-consecutive` breaching ticks, so the rate where
things first degraded is a few seconds before the stop; the `breach` events give the first one.

**HTTP instead of gRPC**: same flags, plus `--protocol http` and Triton's HTTP port.

```
./bin/triton-stress --model model_2026_07_15 --protocol http --target triton:8000 --batch-size 4 \
  --start-rate 10k --slope 5k/s --max-rate 400k
```

**Checking correctness with `--validate`**: decode every response and check output names,
datatypes, shapes and binary sizes. A mismatch counts as `invalid_response`, which is an error for
the error-rate guard.

```
./bin/triton-stress --model model_2026_07_15 --target triton:8001 --batch-size 4 --validate \
  --start-rate 1k --slope 0 --warmup 0s --duration 20s
```

Then compare the `OK` (or `http_200`) count in the stdout JSON lines with the delta of Triton's
`nv_inference_request_success`.

On start, triton-stress prints to stderr the model, input shapes, request size and pool size. `./bin/triton-stress --help`
lists the flags.

## How a run works

### Startup

1. Open `--out-dir` (it must not already contain `intervals.csv` or `events.jsonl`: an existing run
   is never overwritten).
2. Open the `--connections` load connections and wait up to `--request-timeout` for them. At least
   one must come up (otherwise: fatal, exit 1); the others keep reconnecting.
3. Fetch the model's metadata and config once (gRPC `ModelMetadata`/`ModelConfig`, or the HTTP
   `/v2/models/...` endpoints), apply `--input` overrides and `--batch-size`, build the pool of
   `--payload-pool` distinct pre-encoded requests (refused above 512 MiB in total), start the
   Prometheus endpoint, then start sending.

### Phases

`warmup -> ramp -> drain -> done`, each written to `events.jsonl` as a `phase` event.

| Phase | What happens |
|---|---|
| `warmup` | Holds `--start-rate` for `--warmup`. Guards are **report-only**: breaches are logged, never trip. With `--warmup 0s` the run starts in `ramp`. |
| `ramp` | The rate rises by `--slope` req/s every second. **Reaching `--max-rate` ends the run** (`max_rate_reached`); there is no hold at the maximum. With `--slope 0` the phase is still named `ramp`, but the rate is flat until `--duration`, a guard trip or a signal. |
| `drain` | Sending has stopped; in-flight requests finish, for at most `--request-timeout`. Due but unsent arrivals are dropped, so later ticks show `offered_rps` 0. |
| `done` | The final partial tick has been written, then the `stop` event. |

- `--duration` counts from the start of the warmup and also caps a ramp: the run ends at whichever
  comes first of the ramp's end (`warmup + (max-rate - start-rate) / slope`) and `--duration`. The
  defaults (10 s, 1k, 1k/s, 1M) therefore run about 1,009 s unless something trips.
- `--start-rate` must be above 0 and not above `--max-rate`.

### The tick

Every `--interval` (default 1 s) the controller drains the per-worker statistics and writes one
record (`intervals.csv` line, stdout JSON line, histograms for the guards). Percentiles are per
tick, never cumulative. A tick's phase is the phase it began in.

### Guards

| Flag | Default | Breach when (per tick) |
|---|---|---|
| `--guard-achieved` | `0.95` | `ok_rps < 0.95 x offered_rps` |
| `--guard-error-rate` | `0.01` | errors / completed above 1% (every non-OK status is an error, `connect` included) |
| `--guard-latency` | `p99:100ms` | p99 of latency from scheduled above 100 ms |
| `--guard-send-delay` | `p99:1ms` | p99 of send delay above 1 ms |
| `--guard-max-inflight` | `100k` | in-flight requests above 100,000 (checked every 10 ms) |

- Each guard accepts `off`. Percentile thresholds are `pNN:duration` (`p99.9:5ms`).
- **Debounce.** A per-tick guard trips after `--guard-consecutive` (default 3) consecutive breaching
  ticks. A clean tick resets its streak. Ticks without samples do not breach the latency guards.
- **Max in-flight is instant.** It is checked every 10 ms and trips at once, without debounce.
- **Warmup is report-only** for every guard, in-flight included (reported at most once per tick).
  Warmup breaches also reset the streaks, so a trip needs `--guard-consecutive` breaching ticks after
  the warmup. A tick that began in warmup counts as warmup even if it ended in `ramp`.
- Guards are not evaluated while draining or on the final partial tick.
- Every breach, including report-only ones, is a `breach` event and increments
  `triton_stress_guard_breaches_total{guard}`; a trip also increments `triton_stress_guard_trips_total{guard}`.

### Stopping

- **Stop only.** On a trip, `max_rate_reached`, `--duration` or a signal, triton-stress stops sending, drains
  (up to `--request-timeout`), writes a last partial tick and the `stop` event. There is no hold, no
  ramp-down and no run summary document.
- **Signals.** SIGINT and SIGTERM stop gracefully with reason `interrupted` or `terminated`. A second
  signal gets the default action and kills the process. A signal during startup is also a normal end
  (stop event written, exit 0).
- **Exit codes.** `0` when the run completed without a guard tripping, including signals. `2` when a
  guard tripped: the load was offered and the server did not keep up. `1` for fatal errors: bad flags,
  output directory not creatable or already used, no connection, metadata failure, metrics address not
  bindable. A fatal error after the output directory was opened writes a `stop` event with reason
  `fatal:<message>`.

## Timing definitions

Every request carries five monotonic timestamps (written to `requests.csv` as UTC epoch ns):

| Timestamp | Meaning |
|---|---|
| `t_sched` | When the arrival schedule says the request is due |
| `t_emit` | When the worker's dispatcher picked it up |
| `t_ready` | gRPC: the request got a free stream and flow-control window. HTTP: a free connection took it |
| `t_send` | gRPC: its frames were handed to the connection's write buffer. HTTP: just before its bytes were written |
| `t_done` | The response is complete (gRPC: trailers received) |

```
t_sched      t_emit         t_ready  t_send                       t_done
   |-sched_lag-|--backpressure--|------|                              |
   |------ send delay ----------------|------ latency from sent -----|
   |------------------- latency from scheduled ---------------------|
```

| Quantity | Definition | Column / metric stem |
|---|---|---|
| latency from scheduled | `t_done - t_sched` | `lat_sched` / `triton_stress_latency_from_scheduled_seconds` |
| latency from sent | `t_done - t_send` | `lat_sent` / `triton_stress_latency_from_sent_seconds` |
| send delay | `t_send - t_sched` = `sched_lag` + backpressure | `send_delay` / `triton_stress_send_delay_seconds` |
| sched lag | `t_emit - t_sched` | `sched_lag` / `triton_stress_sched_lag_seconds` |

- **`sched_lag` is triton-stress being late**: its own timer wake-up, scheduler, GC or CPU starvation. If it
  rises, the load generator is the bottleneck: give the pod more CPU, and check
  `cgroup_nr_throttled` and `cpu_*_pct`.
- **Backpressure (`t_send - t_emit`) is the transport saying no**: no free gRPC stream or
  flow-control window, or (HTTP) no free connection, with the request waiting in the worker's FIFO.
  It points at Triton, the network or too few `--connections`.
- The send-delay guard catches both; `sched_lag` tells them apart.
- **No coordinated omission.** A late request is never skipped or re-timed: it is sent as soon as
  possible and keeps its original `t_sched`. Latency from scheduled therefore includes the time it was
  late; latency from sent does not. Use `lat_sched` for what a client with that arrival process
  would have seen, `lat_sent` for the service time.
- Requests that never reached the wire (status `connect`, or a request that timed out waiting for a
  stream or an HTTP connection) record only `sched_lag`. Every sent request records its latencies,
  whatever its status, timeouts included.

## Flags

Rates take SI suffixes `k`/`K`, `M`, `G` (`1500`, `2.5k`, `1M`; a lowercase `m` is rejected) and may
end in `/s`. `--warmup`, `--duration` and the guard durations accept `ns`, `us`, `ms`, `s`, `m`, `h`
(a bare number is seconds); `--request-timeout`, `--reconnect-backoff-*` and `--interval` take Go
durations. Boolean flags also have a `--no-` form.

| Flag | Default | Meaning |
|---|---|---|
| `--model` | required | Model name |
| `--model-version` | empty | Model version (empty is Triton's latest) |
| `--target` | `127.0.0.1:8001` (grpc), `127.0.0.1:8000` (http) | Triton endpoint, `host:port`. gRPC also takes `dns:///host:port` and `passthrough:///host:port`; no port means 443 (gRPC) or 80, or 443 with TLS (http). HTTP rejects URL schemes |
| `--protocol` | `grpc` | `grpc` or `http` |
| `--tls` | off | TLS, verifying the server against the system roots |
| `--tls-insecure` | off | TLS without verification. Giving both is an error |
| `--connections` | 128 (grpc), 2048 (http) | Number of load connections. HTTP has one request in flight per connection |
| `--batch-size` | `1` | Client-side batch size (at most the model's `max_batch_size`; a model with `max_batch_size` 0 takes only 1) |
| `--input` | none | Repeatable `name:DTYPE:d1,d2`. Overrides a model input or adds one; dims exclude the batch dimension and must be >= 1 |
| `--gen` | `uniform:0..1000` | Value distribution for every input: `zeros`, `const:<v>`, `uniform:<lo>..<hi>`, `random-bytes` |
| `--payload-pool` | `256` | Number of distinct pre-encoded requests, cycled. Total capped at 512 MiB |
| `--seed` | `1` | Random seed (payloads and Poisson arrivals) |
| `--request-timeout` | `10s` | Client-side timeout per request, also the drain limit and the startup wait |
| `--reconnect-backoff-min` | `10ms` | Reconnect backoff start |
| `--reconnect-backoff-max` | `1s` | Reconnect backoff ceiling |
| `--validate` | off | Decode and check every response |
| `--workers` | `GOMAXPROCS` | Dispatcher goroutines; capped at `--connections` |
| `--arrival` | `constant` | `constant` or `poisson` |
| `--warmup` | `10s` | Hold `--start-rate` for this long |
| `--start-rate` | `1k` | Start rate |
| `--slope` | `1k/s` | Ramp slope in req/s per second; `0` keeps the rate flat |
| `--max-rate` | `1M` | Rate at which the ramp ends the run |
| `--duration` | none | Total run time including the warmup |
| `--guard-achieved` | `0.95` | Fraction of offered rate that must be OK, or `off` |
| `--guard-error-rate` | `0.01` | Maximum error fraction, or `off` |
| `--guard-latency` | `p99:100ms` | Latency from scheduled limit, or `off` |
| `--guard-send-delay` | `p99:1ms` | Send delay limit, or `off` |
| `--guard-max-inflight` | `100k` | Maximum in-flight requests, or `off` |
| `--guard-consecutive` | `3` | Consecutive breaching ticks that trip a guard |
| `--interval` | `1s` | Tick length |
| `--metrics-addr` | `0.0.0.0:9095` | Prometheus listen address, or `off` |
| `--control-addr` | `off` | Control API listen address (`GET /ready`, `POST /start`); when set, the load waits for `POST /start`. `off` disables it |
| `--out-dir` | `runs/<UTC start>` | Output directory, relative to the working directory; start looks like `20261012T093837Z` |
| `--request-log-every` | `0` (off) | Write one request in N to `requests.csv` (SI suffixes allowed) |

Payload notes:

- `uniform:<lo>..<hi>` is inclusive for integer types and BOOL, half-open (`lo < hi`) for floats.
  A value outside the datatype's range is a startup error. The default `0..1000` does not fit
  INT8, UINT8 or BOOL: pass `--gen`.
- FP16 and BF16 accept only `zeros` and `random-bytes`. BYTES inputs are decimal strings of integers
  drawn from the distribution.
- Datatype names are Triton's, upper case: `BOOL UINT8 UINT16 UINT32 UINT64 INT8 INT16 INT32 INT64 FP16
  FP32 FP64 BF16 BYTES`.
- Requests select no outputs, so Triton returns all of them; over HTTP the request sets
  `binary_data_output: true`.
- A model with sequence batching gets `sequence_id` (a counter from 1) and `sequence_start=true` on
  every request.

## Outputs

Everything goes under `--out-dir`; in a container, mount a volume there.

| File | Content |
|---|---|
| `intervals.csv` | One line per tick (the run's main result) |
| `events.jsonl` | Phase changes, connection state changes, guard breaches, and the final `stop` event |
| `requests.csv` | Only with `--request-log-every N`: one request in N |
| stdout | One JSON line per tick, then the `stop` event repeated. Human-readable messages go to stderr |

### `intervals.csv`

Rates are req/s over the tick. Latencies are in microseconds with one decimal. Empty cells (JSON
`null`) mean no data.

| Column | Meaning |
|---|---|
| `ts` | End of the tick, UTC epoch seconds with milliseconds (`1785922486.968`) |
| `iso` | The same instant, RFC 3339 UTC |
| `phase` | Phase the tick began in: `warmup`, `ramp` or `drain` |
| `offered_rps` | Scheduled rate (the integral of the plan over the tick) |
| `sent_rps` | Requests handed to a connection (excludes `connect` failures) |
| `ok_rps` | Completions with status `OK` (gRPC) or `http_200` |
| `error_rps` | All other completions |
| `inflight` | Requests sent and not yet completed, at the end of the tick |
| `lat_sched_{p50,p90,p99,p999,max}_us` | Latency from scheduled, in us |
| `lat_sent_{p50,p90,p99,p999,max}_us` | Latency from sent |
| `send_delay_{p50,p90,p99,p999,max}_us` | Send delay |
| `sched_lag_{p50,p90,p99,p999,max}_us` | Scheduling lag |
| `cpu_user_pct`, `cpu_sys_pct` | triton-stress process CPU, in percent of one core (200 is two busy cores) |
| `busiest_thread_cpu_pct` | CPU of the busiest OS thread, same unit |
| `nonvol_ctx_switches` | Non-voluntary context switches over the tick, all threads |
| `cgroup_nr_throttled`, `cgroup_throttled_us` | CFS throttling counters over the tick; empty when the cgroup v2 `cpu.stat` has none |

The health columns are recorded, never used to judge the run (empty on a very short final tick).
The stdout line has the same fields plus `"status": {"OK": 12345, ...}`, the tick's completions per
[status label](#status-labels).

### `events.jsonl`

One JSON object per line, starting with `ts`, `iso` and `event`.

| `event` | Other fields |
|---|---|
| `phase` | `phase` (`warmup`, `ramp`, `drain`, `done`) |
| `conn` | `conn` (connection index), `from`, `to` (`IDLE`, `CONNECTING`, `READY`, `TRANSIENT_FAILURE`, `SHUTDOWN`) |
| `breach` | `guard`, `observed`, `threshold` |
| `stop` | `reason`, `offered_rps`, `inflight` |

- `breach` units: a ratio for `achieved` (observed is ok/offered, threshold as configured), a
  fraction for `error_rate`, seconds for `latency` and `send_delay`, a count for `max_inflight`.
- `stop` is always the last line of `events.jsonl` and the last line on stdout. `reason` is
  `guard:<name>` (`achieved`, `error_rate`, `latency`, `send_delay`, `max_inflight`),
  `max_rate_reached`, `duration`, `interrupted`, `terminated` or `fatal:<message>`.
  `offered_rps` and `inflight` are the values at the moment sending stopped.

### `requests.csv`

Header: `t_sched_ns,t_emit_ns,t_ready_ns,t_send_ns,t_done_ns,status,worker,connection`.
Timestamps are UTC epoch nanoseconds; a timestamp the request never reached is empty, and
`connection` is empty when no connection was ready. The sample is every N-th completion per worker.

### Joining with server-side telemetry

`ts` is UTC epoch seconds with millisecond precision, the same format as the `tbench` collector, so
tick `ts` values join directly with Triton-side samples. `ts` is the end of the tick, which covers the
interval before it. The monotonic clock is re-anchored to UTC every tick, so a stepped wall clock only
affects the tick in which it happens.

## Control API

`--control-addr` (default `off`) serves a small HTTP API so that several processes started against one server
can begin together. When it is set, triton-stress finishes its startup (connections, model metadata, payload
pool, metrics endpoint), then waits for `POST /start` before the warmup and the ramp begin. It prints
`waiting for POST /start` on stderr. A signal while waiting is a normal end (reason `interrupted` or
`terminated`, exit 0).

| Endpoint | Meaning |
|---|---|
| `GET /ready` | `503` until the startup has finished, then `200`, including while the load runs. Use it as the Kubernetes readiness probe |
| `POST /start` | `503` until the startup has finished (retry), then `200` and the load begins. A second call also returns `200` and does nothing. Other methods get `405` |

## Prometheus

`--metrics-addr` (default `0.0.0.0:9095`) serves `GET /metrics`; `off` disables it. Values are
cumulative since the start of the process. Every series carries the constant labels `protocol`
(`grpc` or `http`) and `model`. There are no Go runtime or process collectors.

| Metric | Type | Extra labels | Meaning |
|---|---|---|---|
| `triton_stress_latency_from_scheduled_seconds` | histogram | | `t_done - t_sched`, sent requests |
| `triton_stress_latency_from_sent_seconds` | histogram | | `t_done - t_send` |
| `triton_stress_send_delay_seconds` | histogram | | `t_send - t_sched` |
| `triton_stress_sched_lag_seconds` | histogram | | `t_emit - t_sched`, every completion |
| `triton_stress_requests_sent_total` | counter | | Requests handed to a connection |
| `triton_stress_requests_completed_total` | counter | `status` | Completions by [status label](#status-labels); `OK`/`http_200` always present, others once seen |
| `triton_stress_bytes_sent_total` | counter | | Request bytes (gRPC: protobuf message; HTTP: the whole pre-built request) |
| `triton_stress_bytes_received_total` | counter | | Response bytes (gRPC: message; HTTP: body) |
| `triton_stress_inflight_requests` | gauge | | Sent and not yet completed |
| `triton_stress_offered_rate_rps` | gauge | | Offered rate of the last tick |
| `triton_stress_phase` | gauge | `phase` | 1 for the current phase, 0 for the others |
| `triton_stress_guard_breaches_total` | counter | `guard` | Breaches, report-only ones included |
| `triton_stress_guard_trips_total` | counter | `guard` | Guards that stopped the run |

**Histograms.** Each carries classic buckets and native buckets in the same metric, and the format
follows the `Accept` header: the protobuf exposition format has both, the text format only the
classic buckets.

- Classic `le` bounds (seconds): 10 us, 20 us, 50 us, 100 us, 200 us, 500 us, 1 ms, 2 ms, 5 ms, 10 ms,
  20 ms, 50 ms, 100 ms, 200 ms, 500 ms, 1 s, 2 s, 5 s, 10 s, 20 s, plus `+Inf`.
- Native: schema 3 (bucket growth factor 2^(1/8), about 9%), zero threshold 2^-32 s.
- To use the native histograms, scrape with the protobuf format (Prometheus `scrape_protocols`
  listing `PrometheusProto` first) and enable native histograms on your Prometheus (the setting
  depends on its version).

```
sum(rate(triton_stress_requests_completed_total{status=~"OK|http_200"}[30s]))
histogram_quantile(0.99, sum by (le) (rate(triton_stress_latency_from_scheduled_seconds_bucket[30s])))   # classic
histogram_quantile(0.99, sum(rate(triton_stress_latency_from_scheduled_seconds[30s])))                   # native
```

## Status labels

The `status` label, the `status` object on stdout and the `status` column of `requests.csv` all use
these names.

| Label | Meaning |
|---|---|
| `OK` | gRPC success |
| `http_200` | HTTP success. Any other 2xx is an error |
| gRPC code names (`UNAVAILABLE`, `RESOURCE_EXHAUSTED`, `INTERNAL`, `CANCELLED`, `DEADLINE_EXCEEDED`, ...) | The server answered with that gRPC status. A queue rejection from Triton is typically `UNAVAILABLE` |
| `http_<code>` | The server answered with that HTTP status (for example `http_503`) |
| `timeout` | `--request-timeout` expired (the gRPC stream is reset; the HTTP connection is closed and re-dialled). No deadline is sent to Triton. For HTTP the timeout includes the wait in the FIFO |
| `transport` | The connection broke while the request was in flight |
| `goaway` | gRPC: the server sent GOAWAY and the request was refused or never processed |
| `connect` | No usable connection when the request was due: it failed at once, nothing waits for a reconnect |
| `invalid_response` | Malformed response, a response failing `--validate`, an HTTP code outside 100-599, or an HTTP output returned as JSON instead of binary |

Every status other than `OK`/`http_200` is an error for the error-rate and achieved-rate guards.
Requests are never retried.

## Connections and reconnects

- **Requests never wait for a reconnect.** A request goes only to a ready connection. A request
  whose connection dies in flight completes at once as `transport`. If none of the worker's
  connections is ready when a request is due, it completes at once as `connect`. Connections are
  split between workers (worker k of N owns k, k+N, ...), and the check is per worker.
- **Reconnection runs in the background.** The first retry after a lost connection is immediate; after
  failed attempts the delay starts at `--reconnect-backoff-min` and doubles up to
  `--reconnect-backoff-max` (with +/-20% jitter), and resets after a successful connect. Every state
  change is a `conn` event and a stderr line.
- **gRPC.** `--connections` HTTP/2 connections, spread over the workers, with fixed flow-control
  windows (1 MiB per stream, 16 MiB per connection). **triton-stress never sends HTTP/2 PINGs** (it answers the
  server's). On GOAWAY, streams the server never processed complete as `goaway`, the others finish,
  and the connection is replaced.
- **HTTP.** `--connections` persistent HTTP/1.1 connections with one request in flight each. Due
  requests wait in the worker's FIFO for a free connection; that wait is backpressure. Because the
  concurrency is the connection count, the sustainable rate is at most `connections / latency`:
  at 300k req/s and 10 ms, you need 3,000 connections.
- **TLS.** `--tls` verifies the server against the system roots (server name taken from `--target`);
  `--tls-insecure` skips verification. Minimum TLS 1.2; ALPN `h2` for gRPC, `http/1.1` for HTTP. No
  custom CA, client certificate or SNI override.

## Sizing and performance

Measured during development (indicative; your Triton, network and pod differ):

- **triton-stress's own ceiling:** about 1M req/s at about 15 us of CPU per request on 16 CPUs (send-delay p99
  about 455 us); 1.2M req/s at about 12.6 us was CPU-saturated. These runs were against a throwaway
  sink, not a Triton.
- **CPU per request depends on the rate.** Around 50k to 200k req/s it was 50 to 60 us per request;
  only near 1M req/s does it fall to 15 us. Do not extrapolate the 15 us to
  low rates.
- **Real Triton limits observed** (Triton was the limit, not triton-stress): gRPC about 270k req/s at batch
  size 1 and about 300k req/s at batch size 4. HTTP at batch size 4 reaches 300k req/s cleanly
  without `--validate`; at batch size 1 it tops out around 275k req/s.
- **Loopback figures include the kernel cost on both sides**: client and Triton share the host, so
  the CPU per request includes both TCP stacks. A pod talking to a remote Triton pays only its half
  in the pod.
- **`--validate` costs CPU** (every response is copied and decoded). Use it for correctness checks,
  not for peak-rate runs.
- **Pod size: at most 16 CPUs.** Above that, Go scheduler overhead makes send delay worse (at 42 CPUs
  the send-delay p99 was 1 to 3 ms at 100k to 250k req/s). One pod at 16 CPUs is the unit of scale.
- **`--workers` defaults to `GOMAXPROCS`**, which Go 1.25+ derives from the container CPU limit
  (rounded up, minimum 2; the `GOMAXPROCS` environment variable overrides it). It is capped at
  `--connections`.
- **Network:** `model_2026_07_15` sends 1,172 bytes of tensor data per batch element, so about 1.3 KB
  x batch size per request, 1.3 GB/s upstream at batch size 1 and 1M req/s. The startup description
  prints the exact request size.

## Running in k8s

No manifests or chart ship with triton-stress. Each GitHub release publishes the image
`ghcr.io/geobeau/triton-stress:<tag>` (distroless, runs as nonroot in `/home/nonroot`, so the default
`runs/` directory is writable) and a linux/amd64 binary. Set:

- **A CPU limit** (at most 16). `GOMAXPROCS`, and with it `--workers`, follows the limit; there is no
  need to pass `--workers`. A request without a limit gives triton-stress every CPU of the node, which is slower
  above 16.
- **The metrics port**: `--metrics-addr` defaults to `0.0.0.0:9095`; expose it and add it to your
  Prometheus scrape configuration (protobuf format for native histograms).
- **The control port**, to start several pods together: set `--control-addr 0.0.0.0:9096`, use
  `GET /ready` as the readiness probe, and send `POST /start` to every pod once all are ready (see
  [Control API](#control-api)).
- **A volume for `--out-dir`**, so `intervals.csv` and `events.jsonl` survive the pod. Give each run
  its own directory: triton-stress refuses to reuse one that holds a previous run.
- **Log shipping**: stdout is only JSON lines (one per tick, then the `stop` event), so it can be
  shipped as is; stderr holds the human-readable messages.
- **SIGTERM** stops the run gracefully (`terminated`): triton-stress drains for up to `--request-timeout`, then
  writes the last tick and the `stop` event. Set `terminationGracePeriodSeconds` above
  `--request-timeout`. A second signal kills it.
- **Exit code 2** when a guard trip ends the run, so a Kubernetes Job fails when the server cannot hold
  the offered load, and succeeds when the plan completes. The `stop` event carries which guard tripped.
- **`--target`**: one endpoint per run; use the gRPC (8001) or HTTP (8000) port for `--protocol`.

## `pkg/tensorgen` for the blackbox-prober probe

`github.com/criteo/triton-stress/pkg/tensorgen` is the random request generator triton-stress uses, written to
be importable by the blackbox-prober Triton probe in place of its own generator.

- **Separate Go module** (`go 1.24`, only dependency `google.golang.org/protobuf` for `protowire`).
- **No generated protobuf types**, because two generated copies of Triton's package in one binary
  (triton-stress's `pkg/tritonpb` and the probe's own) panic at init with "file ... is already registered".
  It works on plain Go values and encodes the wire format by hand with `protowire`.

Main API:

| Symbol | Purpose |
|---|---|
| `NewGenerator(seed, opts...)` | Seeded, deterministic, not safe for concurrent use. Options: `WithDist`, `WithSequenceIDStart`, `WithRequestedOutputs` |
| `Generator.BuildInferRequest(meta, cfg, model, version, batchSize)` | One `*Request` (plain values), the shape of the probe's method of the same name |
| `NewModel`, `Model.NewTemplate(overrides, batchSize)`, `Generator.Build`, `Generator.Encode` | Lower level: build many requests; `Encode` returns `ModelInferRequest` wire bytes |
| `Generator.EncodeHTTP`, `HTTPBody`, `HTTPRequest` | HTTP/1.1 binary-extension requests |
| `ParseDist`, `ProbeDist` | Value distributions; `ProbeDist` reproduces the probe's current values |
| `Probeable(meta)` | Reports models that cannot be probed blindly (BYTES[1] inputs, dynamic non-batch dimensions) |
| `NewValidator(outputs, batchSize, batched)` | `ValidateBytes`, `Validate`, `ValidateHTTP` |

```go
import "github.com/criteo/triton-stress/pkg/tensorgen"

g := tensorgen.NewGenerator(42, tensorgen.WithRequestedOutputs())
req, err := g.BuildInferRequest(
    tensorgen.Metadata{Inputs: inputs, Outputs: outputs}, // []tensorgen.TensorMeta from ModelMetadata
    &tensorgen.Config{MaxBatchSize: 8},                   // from ModelConfig; nil is allowed
    "model_2026_07_15", "", 4)                            // model, version, batch size
if err != nil { /* ... */ }

// Fill the probe's own generated request from plain values.
for _, in := range req.Inputs {
    pbReq.Inputs = append(pbReq.Inputs, &pb.ModelInferRequest_InferInputTensor{
        Name: in.Name, Datatype: in.Datatype, Shape: in.Shape})
    pbReq.RawInputContents = append(pbReq.RawInputContents, in.Raw)
}
// req.Outputs holds the output names; req.Parameters (int64, bool, string) the sequence parameters.
```

`pkg/tritonpbadapter` is for programs that use **our own** generated types (`pkg/tritonpb`), as triton-stress
does: `NewModel`, `Metadata`, `Config`, `Probeable`, `BuildInferRequest` returning a
`*pb.ModelInferRequest`, `Request`, and a `Validator` that checks a `*pb.ModelInferResponse`. A
program with its own generated Triton types, like the probe, must not import it.

## Limitations

- **Generated payloads only.** No replay of captured traffic.
- **One endpoint per run.** No load balancing across targets, no Triton-side metrics scraping.
- **No gRPC compression**, no retries, and no deadline sent to Triton.
- **HTTP is HTTP/1.1 with the KServe v2 binary tensor extension only.**
- **Not tested end to end:** HTTP TLS on the load path, and real-world GOAWAY handling.
- **Dynamic non-batch dimensions need `--input`.** triton-stress refuses to start when an input still has a
  variable dimension; pass `--input name:DTYPE:dims`.
- **One `--gen` distribution for all inputs.**
- **No maximum-sustainable-rate result and no run summary file:** the per-tick records and the `stop`
  event are the output.
