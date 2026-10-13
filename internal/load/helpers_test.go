package load

import (
	"testing"

	"github.com/criteo/triton-stress/internal/stats"
	"github.com/criteo/triton-stress/internal/timer"
)

func newTimer(t *testing.T) *timer.Timer {
	t.Helper()
	tm, err := timer.New()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = tm.Close() })
	return tm
}

// drainLog empties the sampled requests of every worker.
func drainLog(r *Runner) []stats.ReqLog {
	var log []stats.ReqLog
	tick := stats.NewTick()
	for _, w := range r.stats {
		w.Drain(tick, &log)
	}
	return log
}
