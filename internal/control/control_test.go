package control

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func do(t *testing.T, method, url string) int {
	t.Helper()
	req, err := http.NewRequest(method, url, nil)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	_ = resp.Body.Close()
	return resp.StatusCode
}

func TestNotReady(t *testing.T) {
	s := &Server{start: make(chan struct{})}
	srv := httptest.NewServer(s.Handler())
	defer srv.Close()
	if c := do(t, http.MethodGet, srv.URL+"/ready"); c != http.StatusServiceUnavailable {
		t.Fatalf("/ready before startup: %d", c)
	}
	if c := do(t, http.MethodPost, srv.URL+"/start"); c != http.StatusServiceUnavailable {
		t.Fatalf("/start before startup: %d", c)
	}
	select {
	case <-s.start:
		t.Fatal("a refused /start released the load")
	default:
	}
}

func TestReadyAndStart(t *testing.T) {
	s := &Server{start: make(chan struct{})}
	srv := httptest.NewServer(s.Handler())
	defer srv.Close()
	s.SetReady()
	if c := do(t, http.MethodGet, srv.URL+"/ready"); c != http.StatusOK {
		t.Fatalf("/ready: %d", c)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	if err := s.Wait(ctx); err == nil {
		t.Fatal("Wait returned before /start")
	}
	if c := do(t, http.MethodPost, srv.URL+"/start"); c != http.StatusOK {
		t.Fatalf("/start: %d", c)
	}
	if err := s.Wait(context.Background()); err != nil {
		t.Fatal(err)
	}
	if c := do(t, http.MethodPost, srv.URL+"/start"); c != http.StatusOK {
		t.Fatalf("second /start: %d", c)
	}
	if c := do(t, http.MethodGet, srv.URL+"/ready"); c != http.StatusOK {
		t.Fatalf("/ready after /start: %d", c)
	}
}

func TestMethods(t *testing.T) {
	s := &Server{start: make(chan struct{})}
	srv := httptest.NewServer(s.Handler())
	defer srv.Close()
	s.SetReady()
	for _, m := range []string{http.MethodGet, http.MethodPut, http.MethodDelete} {
		if c := do(t, m, srv.URL+"/start"); c != http.StatusMethodNotAllowed {
			t.Errorf("%s /start: %d", m, c)
		}
	}
	if c := do(t, http.MethodPost, srv.URL+"/ready"); c != http.StatusMethodNotAllowed {
		t.Errorf("POST /ready: %d", c)
	}
	if c := do(t, http.MethodGet, srv.URL+"/other"); c != http.StatusNotFound {
		t.Errorf("GET /other: %d", c)
	}
	select {
	case <-s.start:
		t.Fatal("a refused method released the load")
	default:
	}
}

func TestWaitCancelled(t *testing.T) {
	s := &Server{start: make(chan struct{})}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := s.Wait(ctx); err != context.Canceled {
		t.Fatalf("%v", err)
	}
}

func TestServe(t *testing.T) {
	s, err := Serve("127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = s.Close() }()
	if c := do(t, http.MethodGet, "http://"+s.Addr()+"/ready"); c != http.StatusServiceUnavailable {
		t.Fatalf("%d", c)
	}
	s.SetReady()
	if c := do(t, http.MethodPost, "http://"+s.Addr()+"/start"); c != http.StatusOK {
		t.Fatalf("%d", c)
	}
	if err := s.Wait(context.Background()); err != nil {
		t.Fatal(err)
	}
	if _, err := Serve(s.Addr()); err == nil {
		t.Fatal("binding a taken address must fail")
	}
}
