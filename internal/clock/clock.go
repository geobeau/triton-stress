// Package clock is the process-wide monotonic time base: timestamps are
// nanoseconds since the first use of the package, read with time.Since (which
// uses the monotonic clock, vDSO-backed and ~20-30 ns).
package clock

import "time"

var base = time.Now()

// Now is monotonic nanoseconds since the process base, always >= 1 so that 0
// can mean "not reached".
func Now() int64 { return int64(time.Since(base)) + 1 }

// Base is the instant Now counts from.
func Base() time.Time { return base }
