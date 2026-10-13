package timer

import (
	"sort"
	"testing"
	"time"
)

func TestWaitFiresOnTime(t *testing.T) {
	tm, err := New()
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = tm.Close() }()
	const n = 200
	var late []time.Duration
	for i := 0; i < n; i++ {
		want := 500 * time.Microsecond
		start := time.Now()
		full, err := tm.Wait(want, 10*time.Millisecond)
		got := time.Since(start)
		if err != nil || !full {
			t.Fatal(full, err)
		}
		if got < want {
			t.Fatalf("fired early: %v < %v", got, want)
		}
		late = append(late, got-want)
	}
	sort.Slice(late, func(i, j int) bool { return late[i] < late[j] })
	// "Within a few hundred microseconds": the median is the robust figure on a shared machine.
	if med := late[n/2]; med > 400*time.Microsecond {
		t.Fatalf("median lateness %v", med)
	}
	t.Logf("lateness median %v p90 %v max %v", late[n/2], late[n*9/10], late[n-1])
}

func TestWaitRespectsCap(t *testing.T) {
	tm, err := New()
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = tm.Close() }()
	start := time.Now()
	full, err := tm.Wait(5*time.Second, 10*time.Millisecond)
	got := time.Since(start)
	if err != nil || full {
		t.Fatal(full, err)
	}
	if got < 10*time.Millisecond || got > 60*time.Millisecond {
		t.Fatalf("capped wait took %v", got)
	}
}

func TestWaitNonPositiveReturnsAtOnce(t *testing.T) {
	tm, err := New()
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = tm.Close() }()
	start := time.Now()
	for _, d := range []time.Duration{0, -time.Second} {
		if full, err := tm.Wait(d, time.Second); !full || err != nil {
			t.Fatal(full, err)
		}
	}
	if time.Since(start) > 5*time.Millisecond {
		t.Fatal("a due wait must not sleep")
	}
}

func TestCloseUnblocksWait(t *testing.T) {
	tm, err := New()
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() { _, err := tm.Wait(5*time.Second, 5*time.Second); done <- err }()
	time.Sleep(20 * time.Millisecond)
	if err := tm.Close(); err != nil {
		t.Fatal(err)
	}
	select {
	case err := <-done:
		if err == nil {
			t.Fatal("expected an error after Close")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("Wait stayed parked after Close")
	}
}
