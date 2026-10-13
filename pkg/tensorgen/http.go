package tensorgen

import (
	"encoding/json"
	"fmt"
	"net/url"
	"strconv"
)

// This file is the KServe v2 REST side, binary tensor extension only: a request is a
// JSON header followed by the raw tensors (announced by the Inference-Header-Content-Length
// header), and the response comes back the same way. The tensors are the ones
// Generator.Build draws, so a seed gives the same data over gRPC and over HTTP.

// HeaderContentLength is the HTTP header that gives the length of the JSON part of a
// binary-extension request or response body.
const HeaderContentLength = "Inference-Header-Content-Length"

// InferPath is the path of a model's infer endpoint; an empty version is Triton's latest.
func InferPath(model, version string) string {
	return modelPath(model, version) + "/infer"
}

// ModelPath is the path of a model's metadata endpoint ("/config" below it is its config).
func ModelPath(model, version string) string { return modelPath(model, version) }

func modelPath(model, version string) string {
	p := "/v2/models/" + url.PathEscape(model)
	if version != "" {
		p += "/versions/" + url.PathEscape(version)
	}
	return p
}

type httpInput struct {
	Name       string       `json:"name"`
	Shape      []int64      `json:"shape"`
	Datatype   string       `json:"datatype"`
	Parameters httpBinaryIn `json:"parameters"`
}

type httpBinaryIn struct {
	Size int `json:"binary_data_size"`
}

type httpOutput struct {
	Name string `json:"name"`
}

type httpHeader struct {
	Inputs     []httpInput    `json:"inputs"`
	Outputs    []httpOutput   `json:"outputs,omitempty"`
	Parameters map[string]any `json:"parameters"`
}

// HTTPBody turns a request into the body of a binary-extension HTTP request: the JSON
// header (input names, shapes, datatypes and binary_data_size, the requested outputs if
// any, plus the request parameters, with binary_data_output set so outputs come back
// binary) followed by the raw tensors. headerLen is the JSON part's length. Without
// requested outputs Triton returns all of them.
func HTTPBody(req *Request) (body []byte, headerLen int, err error) {
	h := httpHeader{Inputs: make([]httpInput, len(req.Inputs)), Parameters: map[string]any{"binary_data_output": true}}
	total := 0
	for i, in := range req.Inputs {
		h.Inputs[i] = httpInput{Name: in.Name, Shape: in.Shape, Datatype: in.Datatype, Parameters: httpBinaryIn{Size: len(in.Raw)}}
		total += len(in.Raw)
	}
	for _, name := range req.Outputs {
		h.Outputs = append(h.Outputs, httpOutput{Name: name})
	}
	for k, p := range req.Parameters {
		switch p.(type) {
		case int64, bool, string:
			h.Parameters[k] = p
		default:
			return nil, 0, fmt.Errorf("parameter %s: unsupported type %T", k, p)
		}
	}
	js, err := json.Marshal(h) // struct fields in order, map keys sorted: deterministic
	if err != nil {
		return nil, 0, err
	}
	body = make([]byte, 0, len(js)+total)
	body = append(body, js...)
	for _, in := range req.Inputs {
		body = append(body, in.Raw...)
	}
	return body, len(js), nil
}

// HTTPRequest assembles the complete HTTP/1.1 request, ready to write to a connection:
// POST to the model's infer path with Host, Content-Type application/octet-stream,
// Inference-Header-Content-Length and Content-Length, then body.
func HTTPRequest(host, model, version string, body []byte, headerLen int) []byte {
	b := make([]byte, 0, 256+len(body))
	b = append(b, "POST "...)
	b = append(b, InferPath(model, version)...)
	b = append(b, " HTTP/1.1\r\nHost: "...)
	b = append(b, host...)
	b = append(b, "\r\nContent-Type: application/octet-stream\r\n"+HeaderContentLength+": "...)
	b = strconv.AppendInt(b, int64(headerLen), 10)
	b = append(b, "\r\nContent-Length: "...)
	b = strconv.AppendInt(b, int64(len(body)), 10)
	b = append(b, "\r\n\r\n"...)
	return append(b, body...)
}

// EncodeHTTP generates one request for t, as complete HTTP/1.1 request bytes for a
// server at host (the Host header). It draws the same tensors as Build and Encode.
func (g *Generator) EncodeHTTP(t *Template, host string) ([]byte, error) {
	req, err := g.Build(t)
	if err != nil {
		return nil, err
	}
	body, n, err := HTTPBody(req)
	if err != nil {
		return nil, err
	}
	return HTTPRequest(host, t.Model, t.Version, body, n), nil
}

// httpResponseOutput is one entry of a response's outputs.
type httpResponseOutput struct {
	Name       string          `json:"name"`
	Datatype   string          `json:"datatype"`
	Shape      []int64         `json:"shape"`
	Data       json.RawMessage `json:"data"`
	Parameters struct {
		Size *int64 `json:"binary_data_size"`
	} `json:"parameters"`
}

// ValidateHTTP checks the body of a binary-extension response against the model's
// outputs: names, datatypes, shapes, a binary_data_size for every output, and that the
// raw contents add up to the bytes after the JSON header. headerLen is the value of
// Inference-Header-Content-Length, or -1 when the header was absent (then the whole body
// is JSON and there is no binary part).
func (v *Validator) ValidateHTTP(headerLen int, body []byte) error {
	if headerLen < 0 {
		headerLen = len(body)
	}
	if headerLen > len(body) {
		return &Mismatch{Kind: MismatchDecode, Detail: fmt.Sprintf("header length %d is past the %d-byte body", headerLen, len(body))}
	}
	var resp struct {
		Outputs []httpResponseOutput `json:"outputs"`
	}
	if err := json.Unmarshal(body[:headerLen], &resp); err != nil {
		return &Mismatch{Kind: MismatchDecode, Detail: err.Error()}
	}
	bin := body[headerLen:]
	views := make([]outView, len(resp.Outputs))
	for i, o := range resp.Outputs {
		views[i] = outView{name: o.Name, dtype: o.Datatype, shape: o.Shape}
		if o.Parameters.Size == nil {
			if len(o.Data) > 0 {
				return &Mismatch{Kind: MismatchRawCount, Output: o.Name, Detail: "output is JSON data, not binary"}
			}
			return &Mismatch{Kind: MismatchRawCount, Output: o.Name, Detail: "no binary_data_size"}
		}
		n := *o.Parameters.Size
		if n < 0 || n > int64(len(bin)) {
			return &Mismatch{Kind: MismatchRawSize, Output: o.Name,
				Detail: fmt.Sprintf("binary_data_size %d, but %d binary bytes remain", n, len(bin))}
		}
		views[i].raw, bin = bin[:n], bin[n:]
	}
	if len(bin) != 0 {
		return &Mismatch{Kind: MismatchRawSize, Detail: fmt.Sprintf("%d body bytes past the declared binary outputs", len(bin))}
	}
	return v.check(views)
}

// ModelFromJSON reads a Model from the KServe v2 REST answers: the model metadata
// (GET /v2/models/{m}[/versions/{v}]) and the model config (GET .../config). It maps
// them to the same description NewModel builds from the gRPC messages (max_batch_size
// and sequence batching come from the config).
func ModelFromJSON(metaJSON, configJSON []byte, name, version string) (*Model, error) {
	var meta struct {
		Inputs  []httpTensorMeta `json:"inputs"`
		Outputs []httpTensorMeta `json:"outputs"`
	}
	if err := json.Unmarshal(metaJSON, &meta); err != nil {
		return nil, fmt.Errorf("model metadata: %w", err)
	}
	var cfg struct {
		MaxBatchSize *int32          `json:"max_batch_size"`
		Sequence     json.RawMessage `json:"sequence_batching"`
	}
	if err := json.Unmarshal(configJSON, &cfg); err != nil {
		return nil, fmt.Errorf("model config: %w", err)
	}
	var md Metadata
	for _, t := range meta.Inputs {
		md.Inputs = append(md.Inputs, TensorMeta(t))
	}
	for _, t := range meta.Outputs {
		md.Outputs = append(md.Outputs, TensorMeta(t))
	}
	var c Config // an omitted max_batch_size is 0, as in the protobuf config
	if cfg.MaxBatchSize != nil {
		c.MaxBatchSize = *cfg.MaxBatchSize
	}
	c.Sequence = len(cfg.Sequence) > 0 && string(cfg.Sequence) != "null"
	return NewModel(md, &c, name, version)
}

type httpTensorMeta struct {
	Name     string  `json:"name"`
	Datatype string  `json:"datatype"`
	Shape    []int64 `json:"shape"`
}
