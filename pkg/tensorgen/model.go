package tensorgen

import "fmt"

// TensorMeta is one tensor as a server describes it (KServe v2 model metadata): the
// datatype is the Triton name ("INT32", "BYTES", ...) and the shape includes the batch
// dimension (usually -1) of a batching model.
type TensorMeta struct {
	Name     string
	Datatype string
	Shape    []int64
}

// Metadata is a model's inputs and outputs, as in ModelMetadata.
type Metadata struct {
	Inputs  []TensorMeta
	Outputs []TensorMeta
}

// Config is the part of the model config that matters here.
type Config struct {
	// MaxBatchSize is the config's max_batch_size; 0 means the model does not batch.
	MaxBatchSize int32
	// Sequence is true when the config uses sequence_batching.
	Sequence bool
}

// Model is a model's tensors per batch element, as read from its metadata and config.
type Model struct {
	Name    string
	Version string // as requested; empty is Triton's latest
	// Batched is true when the model batches (max_batch_size > 0): requests carry a
	// leading batch dimension. A non-batching model takes batch size 1 and its own shapes.
	Batched bool
	// MaxBatchSize from the config; 0 for a non-batching model or when no config was given.
	MaxBatchSize int32
	// Sequence is true when the config uses sequence_batching.
	Sequence bool
	Inputs   []Spec
	Outputs  []Spec
}

// NewModel reads a Model from the metadata and config. cfg may be nil: the model is
// then taken as batching when every input's first dimension is -1, no max batch size
// is enforced, and it is not a sequence model.
func NewModel(meta Metadata, cfg *Config, name, version string) (*Model, error) {
	m := &Model{Name: name, Version: version}
	if cfg != nil {
		m.MaxBatchSize = cfg.MaxBatchSize
		m.Batched = m.MaxBatchSize > 0
		m.Sequence = cfg.Sequence
	} else if len(meta.Inputs) > 0 {
		m.Batched = true
		for _, in := range meta.Inputs {
			if s := in.Shape; len(s) == 0 || s[0] != -1 {
				m.Batched = false
			}
		}
	}
	var err error
	if m.Inputs, err = specsOf(meta.Inputs, m.Batched); err != nil {
		return nil, err
	}
	if m.Outputs, err = specsOf(meta.Outputs, m.Batched); err != nil {
		return nil, err
	}
	return m, nil
}

// OutputNames lists the model's output names.
func (m *Model) OutputNames() []string {
	names := make([]string, len(m.Outputs))
	for i, o := range m.Outputs {
		names[i] = o.Name
	}
	return names
}

func specsOf(ts []TensorMeta, batched bool) ([]Spec, error) {
	specs := make([]Spec, 0, len(ts))
	for _, t := range ts {
		dt, err := ParseDType(t.Datatype)
		if err != nil {
			return nil, fmt.Errorf("tensor %s: %w", t.Name, err)
		}
		dims := append([]int64(nil), t.Shape...)
		if batched {
			if len(dims) == 0 {
				return nil, fmt.Errorf("tensor %s: batching model but empty shape", t.Name)
			}
			dims = dims[1:]
		}
		specs = append(specs, Spec{Name: t.Name, DType: dt, Dims: dims})
	}
	return specs, nil
}

// Template is a fully resolved request shape: every input has fixed dims.
type Template struct {
	Model     string
	Version   string
	Inputs    []Spec
	BatchSize int
	Batched   bool
	Sequence  bool
	// Outputs are the model's output names; a Generator built WithRequestedOutputs lists
	// them in the request. NewTemplate fills it with every output.
	Outputs []string
}

// NewTemplate applies overrides and a client-side batch size to the model.
// An override naming a model input replaces it; any other name is appended as
// an extra input. The batch size must be >= 1 and, for a batching model with a
// known limit, <= max_batch_size; a non-batching model takes only 1. Inputs
// that still have a variable dimension are an error.
func (m *Model) NewTemplate(overrides []Spec, batchSize int) (*Template, error) {
	if batchSize < 1 {
		return nil, fmt.Errorf("batch size must be at least 1")
	}
	if !m.Batched {
		if batchSize != 1 {
			return nil, fmt.Errorf("model %s does not batch (max_batch_size 0): batch size must be 1", m.Name)
		}
	} else if m.MaxBatchSize > 0 && int64(batchSize) > int64(m.MaxBatchSize) {
		return nil, fmt.Errorf("batch size %d exceeds the model's max_batch_size %d", batchSize, m.MaxBatchSize)
	}
	inputs := append([]Spec(nil), m.Inputs...)
	for _, o := range overrides {
		replaced := false
		for i := range inputs {
			if inputs[i].Name == o.Name {
				inputs[i], replaced = o, true
				break
			}
		}
		if !replaced {
			inputs = append(inputs, o)
		}
	}
	for _, s := range inputs {
		if _, ok := s.NumElements(); !ok {
			return nil, fmt.Errorf("input %s has a variable dimension %v: give its dims with --input", s.Name, s.Dims)
		}
	}
	return &Template{
		Model: m.Name, Version: m.Version, Inputs: inputs,
		BatchSize: batchSize, Batched: m.Batched, Sequence: m.Sequence, Outputs: m.OutputNames(),
	}, nil
}

// Probeable mirrors blackbox-prober's CanProbe. It reports false, with the
// reason, for a model with a BYTES input of a single dimension (it likely
// expects JSON or image data; BYTES with several dimensions are string arrays
// and fine) or with a dynamic dimension (-1) after the first one.
func Probeable(meta Metadata) (ok bool, reason string) {
	for _, in := range meta.Inputs {
		shape := in.Shape
		if in.Datatype == "BYTES" && len(shape) == 1 {
			return false, fmt.Sprintf("input %q has BYTES[1] (likely expects JSON or image)", in.Name)
		}
		for i, d := range shape {
			if i > 0 && d == -1 {
				return false, fmt.Sprintf("input %q has dynamic shape at index %d", in.Name, i)
			}
		}
	}
	return true, ""
}
