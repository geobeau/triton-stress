// Package timer waits on a Linux timerfd that is registered with Go's netpoller, so a
// goroutine waiting for the next send time parks (nothing spins, no OS thread is
// blocked) and wakes through epoll when the kernel's hrtimer fires.
package timer

import (
	"fmt"
	"os"
	"syscall"
	"time"

	"golang.org/x/sys/unix"
)

// Timer is a CLOCK_MONOTONIC timerfd. A Timer serves one goroutine at a time.
type Timer struct {
	f   *os.File
	rc  syscall.RawConn
	buf [8]byte
}

// New creates a Timer.
func New() (*Timer, error) {
	fd, err := unix.TimerfdCreate(unix.CLOCK_MONOTONIC, unix.TFD_NONBLOCK|unix.TFD_CLOEXEC)
	if err != nil {
		return nil, fmt.Errorf("timerfd_create: %w", err)
	}
	f := os.NewFile(uintptr(fd), "timerfd")
	if f == nil {
		_ = unix.Close(fd)
		return nil, fmt.Errorf("timerfd: os.NewFile failed")
	}
	rc, err := f.SyscallConn()
	if err != nil {
		_ = f.Close()
		return nil, fmt.Errorf("timerfd: %w", err)
	}
	return &Timer{f: f, rc: rc}, nil
}

// Wait arms the timer d from now, capped at limit, and parks until it fires. It
// reports whether the whole of d elapsed (false when limit cut the wait short). A
// zero or negative d returns at once without a system call.
func (t *Timer) Wait(d, limit time.Duration) (full bool, err error) {
	if d <= 0 {
		return true, nil
	}
	full = true
	if limit > 0 && d > limit {
		d, full = limit, false
	}
	// An all-zero it_value would disarm the timer, so d is at least 1 ns (guaranteed by d > 0).
	spec := unix.ItimerSpec{Value: unix.NsecToTimespec(int64(d))}
	// The timer is armed inside the poller callback, after the poller's readiness flag
	// was reset for this wait, so an expiry between the arming and the park is not lost,
	// and no read is wasted on a timer that cannot have fired yet.
	armed := false
	var ferr error
	err = t.rc.Read(func(fd uintptr) bool {
		if !armed {
			armed = true
			if ferr = unix.TimerfdSettime(int(fd), 0, &spec, nil); ferr != nil {
				return true
			}
			return false // park until the timerfd is readable
		}
		for {
			_, e := unix.Read(int(fd), t.buf[:])
			switch e {
			case nil:
				return true
			case unix.EINTR:
				continue
			case unix.EAGAIN: // a stale readiness: wait for the real expiry
				return false
			}
			ferr = e
			return true
		}
	})
	if err != nil {
		return false, fmt.Errorf("timerfd read: %w", err)
	}
	if ferr != nil {
		return false, fmt.Errorf("timerfd: %w", ferr)
	}
	return full, nil
}

// Close releases the descriptor; a goroutine parked in Wait returns an error.
func (t *Timer) Close() error { return t.f.Close() }
