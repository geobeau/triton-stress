package tensorgen

import (
	"fmt"
	"math/rand/v2"
)

// Tensor is a tensor of a request, with its data: name, Triton datatype name, the full
// shape (batch dimension included) and the raw little-endian content (BYTES: 4-byte
// length-prefixed elements).
type Tensor struct {
	Name     string
	Datatype string
	Shape    []int64
	Raw      []byte
}

// Request is one generated inference request, as plain values: the same fields as
// ModelInferRequest, with the raw input contents attached to their inputs.
type Request struct {
	Model   string
	Version string
	Inputs  []Tensor
	// Outputs are the requested output names; none means Triton returns all of them.
	Outputs []string
	// Parameters are request parameters. A value is an int64, a bool or a string.
	Parameters map[string]any
}

// Generator builds random requests. It is deterministic for a seed and is not
// safe for concurrent use.
type Generator struct {
	rng        *rand.Rand
	dist       Dist
	nextID     int64
	wantOutput bool
}

// Option configures a Generator.
type Option func(*Generator)

// WithDist sets the value distribution (default DefaultDist, uniform:0..1000).
func WithDist(d Dist) Option { return func(g *Generator) { g.dist = d } }

// WithSequenceIDStart sets the first sequence_id handed out (default 1; 0 is
// not a valid Triton sequence id).
func WithSequenceIDStart(id int64) Option { return func(g *Generator) { g.nextID = id } }

// WithRequestedOutputs makes requests name the outputs they want (Template.Outputs)
// instead of selecting none, which makes Triton return every output.
func WithRequestedOutputs() Option { return func(g *Generator) { g.wantOutput = true } }

// NewGenerator returns a Generator seeded with seed.
func NewGenerator(seed uint64, opts ...Option) *Generator {
	g := &Generator{
		rng:    rand.New(rand.NewPCG(seed, 0x9e3779b97f4a7c15)),
		dist:   DefaultDist,
		nextID: 1,
	}
	for _, o := range opts {
		o(g)
	}
	return g
}

// BuildInferRequest is the convenience entry point, close to blackbox-prober's
// Generator method of the same name: it reads the model from metadata and
// config (config may be nil, see NewModel), builds a template for batchSize
// and returns one request. Use NewModel and Build in a loop when building many requests.
func (g *Generator) BuildInferRequest(meta Metadata, cfg *Config, modelName, modelVersion string, batchSize int64) (*Request, error) {
	m, err := NewModel(meta, cfg, modelName, modelVersion)
	if err != nil {
		return nil, err
	}
	t, err := m.NewTemplate(nil, int(batchSize))
	if err != nil {
		return nil, err
	}
	return g.Build(t)
}

// Build generates one request for t.
func (g *Generator) Build(t *Template) (*Request, error) {
	req := &Request{
		Model:   t.Model,
		Version: t.Version,
		Inputs:  make([]Tensor, 0, len(t.Inputs)),
	}
	for _, s := range t.Inputs {
		if err := g.dist.Check(s.DType, s.Name); err != nil {
			return nil, err
		}
		perElem, ok := s.NumElements()
		if !ok {
			return nil, fmt.Errorf("input %s has a variable dimension %v", s.Name, s.Dims)
		}
		n := int(perElem) * t.BatchSize
		var raw []byte
		if s.DType == Bytes {
			raw = fillBytes(g.dist, n, g.rng)
		} else {
			raw = fill(g.dist, s.DType, n, g.rng)
		}
		req.Inputs = append(req.Inputs, Tensor{
			Name:     s.Name,
			Datatype: s.DType.String(),
			Shape:    s.Shape(int64(t.BatchSize), t.Batched),
			Raw:      raw,
		})
	}
	if g.wantOutput {
		req.Outputs = append([]string(nil), t.Outputs...)
	}
	if t.Sequence {
		req.Parameters = map[string]any{
			"sequence_id":    g.nextID,
			"sequence_start": true,
		}
		g.nextID++
	}
	return req, nil
}

// Encode generates one request for t and returns its protobuf wire bytes
// (deterministic encoding), ready for a pass-through gRPC codec.
func (g *Generator) Encode(t *Template) ([]byte, error) {
	req, err := g.Build(t)
	if err != nil {
		return nil, err
	}
	return req.MarshalWire()
}
