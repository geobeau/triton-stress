// Package payload builds the pool of pre-encoded requests that load workers cycle through.
package payload

import (
	"fmt"

	"github.com/criteo/triton-stress/pkg/tensorgen"
)

// MaxPoolBytes caps the pool's total pre-encoded bytes.
const MaxPoolBytes = 512 << 20

// Pool is a set of pre-encoded requests (ModelInferRequest bodies, or HTTP requests), cycled by index. The
// bodies are shared and must never be modified.
type Pool struct {
	bodies [][]byte
	total  int
}

// Build encodes size distinct requests of t with g. It fails when size times the
// encoded request size is over MaxPoolBytes (checked after the first request,
// before the rest are generated).
func Build(g *tensorgen.Generator, t *tensorgen.Template, size int) (*Pool, error) {
	return BuildFunc(func() ([]byte, error) { return g.Encode(t) }, size)
}

// BuildFunc is Build with any request encoder (for HTTP, complete request bytes).
func BuildFunc(encode func() ([]byte, error), size int) (*Pool, error) {
	if size < 1 {
		return nil, fmt.Errorf("--payload-pool must be at least 1")
	}
	p := &Pool{bodies: make([][]byte, 0, min(size, 1<<16))}
	for len(p.bodies) < size {
		b, err := encode()
		if err != nil {
			return nil, err
		}
		if len(p.bodies) == 0 && len(b) > MaxPoolBytes/size {
			return nil, tooBig(size, len(b))
		}
		p.total += len(b)
		if p.total > MaxPoolBytes { // BYTES requests can grow past the first one
			return nil, tooBig(size, len(b))
		}
		p.bodies = append(p.bodies, b)
	}
	return p, nil
}

func tooBig(size, reqBytes int) error {
	return fmt.Errorf("--payload-pool %d x %d bytes per request is over the %d MiB cap: lower --payload-pool",
		size, reqBytes, MaxPoolBytes>>20)
}

// Len is the number of requests.
func (p *Pool) Len() int { return len(p.bodies) }

// Get is request i, wrapping around the pool.
func (p *Pool) Get(i int) []byte { return p.bodies[i%len(p.bodies)] }

// TotalBytes is the sum of all encoded sizes.
func (p *Pool) TotalBytes() int { return p.total }

// SizeRange is the smallest and largest encoded request. They differ only for
// BYTES inputs, whose strings vary in length.
func (p *Pool) SizeRange() (lo, hi int) {
	lo, hi = len(p.bodies[0]), len(p.bodies[0])
	for _, b := range p.bodies {
		lo, hi = min(lo, len(b)), max(hi, len(b))
	}
	return lo, hi
}
