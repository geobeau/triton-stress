// Package health samples the client's own CPU use, cgroup throttling and context
// switches once per tick. The values are recorded only; nothing judges them.
package health

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// userHZ is the unit of utime/stime in /proc/*/stat (USER_HZ is 100 on Linux).
const userHZ = 100

// Health is one tick's client-health record. Percentages are of one CPU (200 means
// two busy cores). Counter fields are deltas over the tick.
type Health struct {
	CPUUserPct float64
	CPUSysPct  float64
	// BusiestThreadPct is the CPU use of the busiest OS thread (Go has no worker
	// threads; the runtime's threads share the goroutines).
	BusiestThreadPct float64
	// NonvolCtxSwitches is the non-voluntary context switches summed over all threads.
	NonvolCtxSwitches int64
	// Cgroup is false when the process's cgroup v2 cpu.stat has no throttling counters.
	Cgroup      bool
	NrThrottled int64
	ThrottledUS int64
}

// ParseProcStat extracts utime and stime (clock ticks) from the content of a
// /proc/<pid>/stat or /proc/<pid>/task/<tid>/stat file. The command name may hold
// spaces and parentheses, so fields are counted after the last ')'.
func ParseProcStat(data string) (utime, stime uint64, err error) {
	i := strings.LastIndexByte(data, ')')
	if i < 0 {
		return 0, 0, fmt.Errorf("no command name in stat")
	}
	f := strings.Fields(data[i+1:])
	// f[0] is the state (field 3), so utime (field 14) is f[11] and stime f[12].
	if len(f) < 13 {
		return 0, 0, fmt.Errorf("stat has %d fields after the command name", len(f))
	}
	if utime, err = strconv.ParseUint(f[11], 10, 64); err != nil {
		return 0, 0, fmt.Errorf("utime: %w", err)
	}
	if stime, err = strconv.ParseUint(f[12], 10, 64); err != nil {
		return 0, 0, fmt.Errorf("stime: %w", err)
	}
	return utime, stime, nil
}

// ParseCPUStat extracts nr_throttled and throttled_usec from a cgroup v2 cpu.stat.
// ok is false when either is missing (the cpu controller is not enabled there).
func ParseCPUStat(data string) (nrThrottled, throttledUS int64, ok bool) {
	var gotNr, gotUS bool
	for _, line := range strings.Split(data, "\n") {
		k, v, found := strings.Cut(strings.TrimSpace(line), " ")
		if !found {
			continue
		}
		n, err := strconv.ParseInt(strings.TrimSpace(v), 10, 64)
		if err != nil {
			continue
		}
		switch k {
		case "nr_throttled":
			nrThrottled, gotNr = n, true
		case "throttled_usec":
			throttledUS, gotUS = n, true
		}
	}
	return nrThrottled, throttledUS, gotNr && gotUS
}

// ParseNonvoluntary extracts nonvoluntary_ctxt_switches from a /proc/.../status file.
func ParseNonvoluntary(data string) (int64, bool) {
	for _, line := range strings.Split(data, "\n") {
		if v, ok := strings.CutPrefix(line, "nonvoluntary_ctxt_switches:"); ok {
			n, err := strconv.ParseInt(strings.TrimSpace(v), 10, 64)
			return n, err == nil
		}
	}
	return 0, false
}

// ParseCgroupPath returns the cgroup v2 path (the "0::" line) of a /proc/<pid>/cgroup file.
func ParseCgroupPath(data string) (string, bool) {
	sc := bufio.NewScanner(strings.NewReader(data))
	for sc.Scan() {
		if p, ok := strings.CutPrefix(sc.Text(), "0::"); ok {
			return p, true
		}
	}
	return "", false
}

type taskCPU struct {
	ticks   uint64
	nonvol  int64
	hasStat bool
}

type snapshot struct {
	at           time.Time
	utime, stime uint64
	tasks        map[string]taskCPU
	nrThrottled  int64
	throttledUS  int64
	cgroup       bool
}

// Sampler takes Health samples. The first Sample only primes the baseline.
type Sampler struct {
	proc, cgroupRoot string
	prev             *snapshot
}

// New samples the running process (/proc/self, /sys/fs/cgroup).
func New() *Sampler { return NewAt("/proc", "/sys/fs/cgroup") }

// NewAt samples the process tree rooted at proc (which holds a "self" entry) and the
// cgroup hierarchy mounted at cgroupRoot; tests point both at fixtures.
func NewAt(proc, cgroupRoot string) *Sampler {
	return &Sampler{proc: proc, cgroupRoot: cgroupRoot}
}

func readFile(path string) (string, error) {
	b, err := os.ReadFile(path)
	return string(b), err
}

func (s *Sampler) snapshot(at time.Time) (*snapshot, error) {
	self := filepath.Join(s.proc, "self")
	stat, err := readFile(filepath.Join(self, "stat"))
	if err != nil {
		return nil, err
	}
	ut, st, err := ParseProcStat(stat)
	if err != nil {
		return nil, err
	}
	snap := &snapshot{at: at, utime: ut, stime: st, tasks: map[string]taskCPU{}}
	if ents, err := os.ReadDir(filepath.Join(self, "task")); err == nil {
		for _, e := range ents {
			dir := filepath.Join(self, "task", e.Name())
			var tc taskCPU
			if b, err := readFile(filepath.Join(dir, "stat")); err == nil {
				if u, s2, err := ParseProcStat(b); err == nil {
					tc.ticks, tc.hasStat = u+s2, true
				}
			}
			if b, err := readFile(filepath.Join(dir, "status")); err == nil {
				tc.nonvol, _ = ParseNonvoluntary(b)
			}
			snap.tasks[e.Name()] = tc // a thread that exited meanwhile reads as zeros
		}
	}
	if cg, err := readFile(filepath.Join(self, "cgroup")); err == nil {
		if p, ok := ParseCgroupPath(cg); ok {
			if b, err := readFile(filepath.Join(s.cgroupRoot, p, "cpu.stat")); err == nil {
				snap.nrThrottled, snap.throttledUS, snap.cgroup = ParseCPUStat(b)
			}
		}
	}
	return snap, nil
}

// Sample reads the counters and returns the deltas since the previous call; the first
// call returns ok=false. at is the sample time (monotonic readings are used).
func (s *Sampler) Sample(at time.Time) (h Health, ok bool) {
	cur, err := s.snapshot(at)
	if err != nil {
		return Health{}, false
	}
	prev := s.prev
	s.prev = cur
	if prev == nil {
		return Health{}, false
	}
	secs := cur.at.Sub(prev.at).Seconds()
	if secs <= 0 {
		return Health{}, false
	}
	pct := func(ticks uint64) float64 { return float64(ticks) / userHZ / secs * 100 }
	h.CPUUserPct = pct(sub(cur.utime, prev.utime))
	h.CPUSysPct = pct(sub(cur.stime, prev.stime))
	var busiest uint64
	for tid, c := range cur.tasks {
		p := prev.tasks[tid] // a new thread starts from zero
		busiest = max(busiest, sub(c.ticks, p.ticks))
		if d := c.nonvol - p.nonvol; d > 0 {
			h.NonvolCtxSwitches += d
		}
	}
	h.BusiestThreadPct = pct(busiest)
	if cur.cgroup && prev.cgroup {
		h.Cgroup = true
		h.NrThrottled = max(cur.nrThrottled-prev.nrThrottled, 0)
		h.ThrottledUS = max(cur.throttledUS-prev.throttledUS, 0)
	}
	return h, true
}

func sub(a, b uint64) uint64 {
	if a < b {
		return 0
	}
	return a - b
}
