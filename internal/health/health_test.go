package health

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

// A real stat line, including a command name with spaces and a parenthesis.
const statFixture = `1234 (rts (go) x) S 1 1234 1234 0 -1 4194560 1000 0 0 0 250 75 0 0 20 0 12 0 5000 1000000 500 18446744073709551615 1 1 0 0 0 0 0 0 0 0 0 0 17 3 0 0 0 0 0`

const cpuStatFixture = `usage_usec 5000000
user_usec 3000000
system_usec 2000000
nr_periods 100
nr_throttled 7
throttled_usec 123456
nr_bursts 0
burst_usec 0
`

const statusFixture = "Name:\trts\nvoluntary_ctxt_switches:\t11\nnonvoluntary_ctxt_switches:\t42\n"

func TestParseProcStat(t *testing.T) {
	u, s, err := ParseProcStat(statFixture)
	if err != nil || u != 250 || s != 75 {
		t.Fatal(u, s, err)
	}
	for _, bad := range []string{"", "1 (x) S 1 2", "1 (x) S " + "0 0 0 0 0 0 0 0 0 0 0 abc 1"} {
		if _, _, err := ParseProcStat(bad); err == nil {
			t.Errorf("%q should fail", bad)
		}
	}
}

func TestParseCPUStat(t *testing.T) {
	nr, us, ok := ParseCPUStat(cpuStatFixture)
	if !ok || nr != 7 || us != 123456 {
		t.Fatal(nr, us, ok)
	}
	if _, _, ok := ParseCPUStat("usage_usec 1\nuser_usec 1\n"); ok {
		t.Fatal("no throttling counters: not ok")
	}
}

func TestParseNonvoluntaryAndCgroup(t *testing.T) {
	if n, ok := ParseNonvoluntary(statusFixture); !ok || n != 42 {
		t.Fatal(n, ok)
	}
	if _, ok := ParseNonvoluntary("Name:\tx\n"); ok {
		t.Fatal("missing field")
	}
	if p, ok := ParseCgroupPath("0::/kubepods/pod1/ctr\n"); !ok || p != "/kubepods/pod1/ctr" {
		t.Fatal(p, ok)
	}
	if _, ok := ParseCgroupPath("12:cpu:/x\n"); ok {
		t.Fatal("cgroup v1 line has no 0:: entry")
	}
}

type fixtureTree struct{ root string }

func (f fixtureTree) write(t *testing.T, rel, content string) {
	t.Helper()
	p := filepath.Join(f.root, rel)
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func stat(u, s int) string {
	return "1 (rts) S 1 1 1 0 -1 0 0 0 0 0 " + itoa(u) + " " + itoa(s) + " 0 0 20 0 1 0 1 1 1 1"
}

func itoa(n int) string {
	return string(rune('0'+n/1000%10)) + string(rune('0'+n/100%10)) + string(rune('0'+n/10%10)) + string(rune('0'+n%10))
}

func TestSampler(t *testing.T) {
	root := t.TempDir()
	f := fixtureTree{root}
	proc, cg := filepath.Join(root, "proc"), filepath.Join(root, "cg")
	fixture := func(procU, procS, t1, t2, nv1, nv2, throttled, usec int) {
		f.write(t, "proc/self/stat", stat(procU, procS))
		f.write(t, "proc/self/cgroup", "0::/pod/a\n")
		f.write(t, "proc/self/task/1/stat", stat(t1, 0))
		f.write(t, "proc/self/task/2/stat", stat(t2, 0))
		f.write(t, "proc/self/task/1/status", "nonvoluntary_ctxt_switches:\t"+itoa(nv1)+"\n")
		f.write(t, "proc/self/task/2/status", "nonvoluntary_ctxt_switches:\t"+itoa(nv2)+"\n")
		f.write(t, "cg/pod/a/cpu.stat", "nr_throttled "+itoa(throttled)+"\nthrottled_usec "+itoa(usec)+"\n")
	}
	s := NewAt(proc, cg)
	t0 := time.Now()
	fixture(100, 20, 60, 40, 5, 6, 1, 100)
	if _, ok := s.Sample(t0); ok {
		t.Fatal("first sample only primes")
	}
	// One second later: 150 user ticks (150%), 50 system ticks; thread 1 used 100 ticks.
	fixture(250, 70, 160, 90, 8, 9, 4, 600)
	h, ok := s.Sample(t0.Add(time.Second))
	if !ok {
		t.Fatal("no sample")
	}
	if h.CPUUserPct != 150 || h.CPUSysPct != 50 || h.BusiestThreadPct != 100 || h.NonvolCtxSwitches != 6 ||
		!h.Cgroup || h.NrThrottled != 3 || h.ThrottledUS != 500 {
		t.Fatalf("%+v", h)
	}
}

func TestSamplerWithoutCgroupOrTasks(t *testing.T) {
	root := t.TempDir()
	fixtureTree{root}.write(t, "proc/self/stat", stat(1, 1))
	s := NewAt(filepath.Join(root, "proc"), filepath.Join(root, "cg"))
	t0 := time.Now()
	s.Sample(t0)
	h, ok := s.Sample(t0.Add(time.Second))
	if !ok || h.Cgroup {
		t.Fatalf("%+v %v", h, ok)
	}
}

func TestSamplerRealProc(t *testing.T) {
	s := New()
	t0 := time.Now()
	s.Sample(t0)
	x := 0
	for time.Since(t0) < 60*time.Millisecond {
		x++
	}
	h, ok := s.Sample(time.Now())
	if !ok {
		t.Skip("no /proc")
	}
	if h.CPUUserPct+h.CPUSysPct <= 0 || h.BusiestThreadPct <= 0 {
		t.Fatalf("%+v", h)
	}
}
