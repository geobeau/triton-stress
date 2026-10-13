// Package tensorgen builds random Triton inference requests.
//
// It turns a model's metadata and config (as the KServe v2 calls ModelMetadata and
// ModelConfig return them, here as plain Metadata and Config values) into a Request
// filled with data drawn from a seeded random distribution, or into the request's
// protobuf wire bytes. It was written for the triton-stress load generator, and a prober can
// import it in place of a hand-written generator.
//
// # No generated protobuf
//
// The package is its own Go module (go 1.24; its only dependency is
// google.golang.org/protobuf for protowire). Everything works on plain Go values:
// specs (name, datatype string, shape), tensors (name, datatype, shape, raw bytes) and
// parameters (int64, bool or string). Requests are pre-encoded to ModelInferRequest
// wire bytes with protowire, and gRPC responses are validated by decoding the
// ModelInferResponse the same way. Programs that have generated types convert at the
// edge: triton-stress uses pkg/tritonpbadapter, and a prober fills its own generated request
// from Request.
//
// This matters because two generated copies of package inference (triton-stress's
// pkg/tritonpb and a prober's own client) cannot live in one binary: the second
// registration of grpc_service.proto makes protobuf-go panic at init.
//
// # Quick use
//
//	g := tensorgen.NewGenerator(seed)
//	req, err := g.BuildInferRequest(metadata, config, "model", "", 4)
//
// BuildInferRequest has the shape of blackbox-prober's Generator method of the same
// name. By default requests select no outputs, so Triton returns all of them;
// WithRequestedOutputs names them instead, as blackbox-prober does.
//
// # Lower level
//
// Model describes a model per batch element; Model.NewTemplate applies
// overrides (extra or replaced inputs) and a batch size and returns a
// Template; Generator.Build and Generator.Encode produce requests from it.
// Encode returns wire bytes for a pass-through gRPC codec, so a load generator
// can pre-encode a pool of requests once.
//
// # Data
//
// One Dist applies to every input (ParseDist: zeros, const:V, uniform:LO..HI,
// random-bytes; the default is uniform:0..1000):
//
//   - uniform:LO..HI is inclusive [LO, HI] for integer types and BOOL (LO == HI
//     is allowed and always gives that value), and half-open [LO, HI) with
//     LO < HI for floats;
//   - a value outside the datatype's range is an error (Dist.Check);
//   - random-bytes fills the tensor with random bytes; BOOL is masked to 0/1;
//   - FP16 and BF16 accept only zeros and random-bytes;
//   - BYTES elements are the decimal strings of integers drawn from the
//     distribution (random-bytes gives a random value in [0, 2^32)), encoded
//     as a 4-byte little-endian length plus the text.
//
// ProbeDist is blackbox-prober's own value choice (small integers, floats in [0, 1)),
// available to callers that want to keep what they send; ParseDist does not accept it.
//
// A whole batch is generated in one go: a tensor of batch size B holds B
// independent batch elements. The same seed and the same sequence of calls
// give identical bytes (the RNG is PCG from math/rand/v2).
//
// # Sequence batching
//
// For a model whose config uses sequence_batching, every request gets the
// parameters sequence_id (a counter, unique per request from one Generator,
// starting at 1 unless WithSequenceIDStart says otherwise) and
// sequence_start=true, as blackbox-prober does. sequence_end is not set.
//
// # Models that cannot be probed blindly
//
// Probeable mirrors blackbox-prober's CanProbe: it reports models with BYTES
// inputs of one dimension (likely JSON or image payloads) or with a dynamic
// dimension after the first. Building a request for such a model fails
// unless the dynamic dimensions are fixed by an override.
//
// # Response validation
//
// Validator checks a ModelInferResponse against the model's outputs: names,
// datatypes, shapes and raw content sizes. It takes the wire bytes
// (Validator.ValidateBytes) or the decoded outputs (Validator.Validate), and the HTTP
// binary-extension body (Validator.ValidateHTTP).
package tensorgen
