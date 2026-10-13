module github.com/criteo/triton-stress

go 1.27

require (
	github.com/HdrHistogram/hdrhistogram-go v1.4.0
	github.com/alecthomas/kingpin/v2 v2.4.0
	github.com/criteo/triton-stress/pkg/tensorgen v0.0.0
	github.com/prometheus/client_golang v1.25.0
	github.com/prometheus/client_model v0.6.3
	github.com/prometheus/common v0.72.0
	golang.org/x/net v0.59.0
	golang.org/x/sys v0.48.0
	google.golang.org/grpc v1.84.0
	google.golang.org/protobuf v1.36.12
)

// pkg/tensorgen is its own module (go 1.24, only protobuf for protowire) so that other
// projects, such as blackbox-prober, can require it.
replace github.com/criteo/triton-stress/pkg/tensorgen => ./pkg/tensorgen

require (
	github.com/alecthomas/units v0.0.0-20240927000941-0f3dac36c52b // indirect
	github.com/beorn7/perks v1.0.1 // indirect
	github.com/cespare/xxhash/v2 v2.3.0 // indirect
	github.com/munnerz/goautoneg v0.0.0-20191010083416-a7dc8b61c822 // indirect
	github.com/prometheus/procfs v0.22.0 // indirect
	github.com/xhit/go-str2duration/v2 v2.1.0 // indirect
	golang.org/x/text v0.42.0 // indirect
	google.golang.org/genproto/googleapis/rpc v0.0.0-20260706201446-f0a921348800 // indirect
)
