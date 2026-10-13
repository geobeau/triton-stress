package h1client

import (
	"context"
	"crypto/tls"
	"fmt"
	"io"
	"net/http"

	"github.com/criteo/triton-stress/internal/grpcclient"
	"github.com/criteo/triton-stress/pkg/tensorgen"
)

// maxMeta bounds a metadata or config answer.
const maxMeta = 16 << 20

// FetchModel reads the model's metadata (GET /v2/models/{m}[/versions/{v}]) and config
// (GET .../config) from target, over a plain HTTP connection of its own (net/http is fine
// here: this is startup, not the load path), and maps them to the model description the
// gRPC path builds. TLS follows tlsMode.
func FetchModel(ctx context.Context, target string, tlsMode grpcclient.TLSMode, model, version string) (*tensorgen.Model, error) {
	addr, err := Address(target, tlsMode != grpcclient.TLSOff)
	if err != nil {
		return nil, err
	}
	scheme := "http"
	tr := &http.Transport{Proxy: nil, DisableKeepAlives: true, TLSHandshakeTimeout: connectTimeout}
	switch tlsMode {
	case grpcclient.TLSSystem:
		scheme = "https"
	case grpcclient.TLSInsecure:
		scheme = "https"
		tr.TLSClientConfig = &tls.Config{MinVersion: tls.VersionTLS12, InsecureSkipVerify: true}
	}
	defer tr.CloseIdleConnections()
	cl := &http.Client{Transport: tr, Timeout: 0}
	get := func(path string) ([]byte, error) {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, scheme+"://"+addr+path, nil)
		if err != nil {
			return nil, err
		}
		resp, err := cl.Do(req)
		if err != nil {
			return nil, err
		}
		defer func() { _ = resp.Body.Close() }()
		b, err := io.ReadAll(io.LimitReader(resp.Body, maxMeta))
		if err != nil {
			return nil, err
		}
		if resp.StatusCode != http.StatusOK {
			msg := string(b)
			if len(msg) > 200 {
				msg = msg[:200]
			}
			return nil, fmt.Errorf("GET %s: HTTP %d: %s", path, resp.StatusCode, msg)
		}
		return b, nil
	}
	meta, err := get(tensorgen.ModelPath(model, version))
	if err != nil {
		return nil, fmt.Errorf("fetching model metadata: %w", err)
	}
	cfg, err := get(tensorgen.ModelPath(model, version) + "/config")
	if err != nil {
		return nil, fmt.Errorf("fetching model config: %w", err)
	}
	return tensorgen.ModelFromJSON(meta, cfg, model, version)
}
