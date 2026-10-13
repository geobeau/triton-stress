package h1client

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/criteo/triton-stress/internal/grpcclient"
)

func TestFetchModel(t *testing.T) {
	var paths []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		paths = append(paths, r.Method+" "+r.URL.Path)
		switch r.URL.Path {
		case "/v2/models/m/versions/3":
			_, _ = w.Write([]byte(`{"name":"m","inputs":[{"name":"x","datatype":"INT32","shape":[-1,8]}],"outputs":[{"name":"p","datatype":"FP32","shape":[-1,1]}]}`))
		case "/v2/models/m/versions/3/config":
			_, _ = w.Write([]byte(`{"name":"m","max_batch_size":16}`))
		default:
			http.Error(w, `{"error":"unknown model"}`, http.StatusBadRequest)
		}
	}))
	defer srv.Close()
	addr := strings.TrimPrefix(srv.URL, "http://")
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	m, err := FetchModel(ctx, addr, grpcclient.TLSOff, "m", "3")
	if err != nil {
		t.Fatal(err)
	}
	if !m.Batched || m.MaxBatchSize != 16 || m.Inputs[0].Name != "x" || m.Inputs[0].Dims[0] != 8 || m.Outputs[0].Name != "p" {
		t.Fatalf("%+v", m)
	}
	if strings.Join(paths, ",") != "GET /v2/models/m/versions/3,GET /v2/models/m/versions/3/config" {
		t.Fatal(paths)
	}
	_, err = FetchModel(ctx, addr, grpcclient.TLSOff, "nope", "")
	if err == nil || !strings.Contains(err.Error(), "HTTP 400") || !strings.Contains(err.Error(), "unknown model") {
		t.Fatal(err)
	}
	// TLS against a TLS server, with and without verification.
	tsrv := httptest.NewTLSServer(srv.Config.Handler)
	defer tsrv.Close()
	taddr := strings.TrimPrefix(tsrv.URL, "https://")
	if _, err := FetchModel(ctx, taddr, grpcclient.TLSInsecure, "m", "3"); err != nil {
		t.Fatal(err)
	}
	if _, err := FetchModel(ctx, taddr, grpcclient.TLSSystem, "m", "3"); err == nil {
		t.Fatal("untrusted certificate accepted")
	}
}
