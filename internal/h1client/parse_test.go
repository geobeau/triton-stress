package h1client

import (
	"bufio"
	"errors"
	"io"
	"strings"
	"testing"
)

// splitAt delivers its data in two reads, cut at the given offset.
type splitAt struct {
	data []byte
	cut  int
	pos  int
}

func (s *splitAt) Read(p []byte) (int, error) {
	if s.pos >= len(s.data) {
		return 0, io.EOF
	}
	end := len(s.data)
	if s.pos < s.cut {
		end = s.cut
	}
	n := copy(p, s.data[s.pos:end])
	s.pos += n
	return n, nil
}

func parseAll(raw string, keep bool, cut int) (response, string, int, error) {
	br := bufio.NewReaderSize(&splitAt{data: []byte(raw), cut: cut}, readBufSize)
	var dst []byte
	var p *[]byte
	if keep {
		p = &dst
	}
	resp, n, err := readResponse(br, p)
	return resp, string(dst), n, err
}

func TestReadResponse(t *testing.T) {
	cases := []struct {
		name   string
		raw    string
		code   int
		hdrLen int
		close  bool
		body   string
	}{
		{"content-length", "HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nhello", 200, -1, false, "hello"},
		{"triton like", "HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nInference-Header-Content-Length: 3\r\nContent-Length: 7\r\n\r\nabcdefg", 200, 3, false, "abcdefg"},
		{"lower-case names", "HTTP/1.1 200 OK\r\ncontent-length: 2\r\ninference-header-content-length: 1\r\n\r\nxy", 200, 1, false, "xy"},
		{"no reason phrase", "HTTP/1.1 200\r\nContent-Length: 0\r\n\r\n", 200, -1, false, ""},
		{"503 json", "HTTP/1.1 503 Service Unavailable\r\nContent-Length: 13\r\n\r\n{\"error\":\"x\"}", 503, -1, false, `{"error":"x"}`},
		{"connection close", "HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Length: 1\r\n\r\nz", 200, -1, true, "z"},
		{"connection token list", "HTTP/1.1 200 OK\r\nConnection: Keep-Alive, Close\r\nContent-Length: 0\r\n\r\n", 200, -1, true, ""},
		{"http/1.0 closes", "HTTP/1.0 200 OK\r\nContent-Length: 1\r\n\r\nz", 200, -1, true, "z"},
		{"http/1.0 keep-alive", "HTTP/1.0 200 OK\r\nConnection: keep-alive\r\nContent-Length: 1\r\n\r\nz", 200, -1, false, "z"},
		{"to end of connection", "HTTP/1.1 200 OK\r\n\r\nuntil eof", 200, -1, true, "until eof"},
		{"204", "HTTP/1.1 204 No Content\r\n\r\n", 204, -1, false, ""},
		{"100 continue first", "HTTP/1.1 100 Continue\r\n\r\nHTTP/1.1 200 OK\r\nContent-Length: 1\r\n\r\nq", 200, -1, false, "q"},
		{"chunked", "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nInference-Header-Content-Length: 2\r\n\r\n3\r\nabc\r\n2;ext=1\r\nde\r\n0\r\nTrailer: x\r\n\r\n", 200, 2, false, "abcde"},
		{"chunked uppercase hex", "HTTP/1.1 200 OK\r\nTransfer-Encoding: Chunked\r\n\r\nA\r\n0123456789\r\n0\r\n\r\n", 200, -1, false, "0123456789"},
		{"bare LF", "HTTP/1.1 200 OK\nContent-Length: 2\n\nab", 200, -1, false, "ab"},
	}
	for _, c := range cases {
		for _, keep := range []bool{true, false} {
			// every cut position, including 0 (everything at once) and len (one read)
			for cut := 0; cut <= len(c.raw); cut++ {
				resp, body, n, err := parseAll(c.raw, keep, cut)
				if err != nil {
					t.Fatalf("%s keep=%v cut=%d: %v", c.name, keep, cut, err)
				}
				if resp.code != c.code || resp.hdrLen != c.hdrLen || resp.close != c.close || n != len(c.body) {
					t.Fatalf("%s keep=%v cut=%d: %+v n=%d", c.name, keep, cut, resp, n)
				}
				if keep && body != c.body {
					t.Fatalf("%s cut=%d: body %q, want %q", c.name, cut, body, c.body)
				}
			}
		}
	}
}

func TestReadResponseOneByteAtATime(t *testing.T) {
	raw := "HTTP/1.1 200 OK\r\nInference-Header-Content-Length: 3\r\nContent-Length: 9\r\n\r\nabcdefghi"
	br := bufio.NewReaderSize(oneByte{strings.NewReader(raw)}, readBufSize)
	var dst []byte
	resp, n, err := readResponse(br, &dst)
	if err != nil || resp.code != 200 || resp.hdrLen != 3 || n != 9 || string(dst) != "abcdefghi" {
		t.Fatalf("%+v %d %q %v", resp, n, dst, err)
	}
}

type oneByte struct{ r io.Reader }

func (o oneByte) Read(p []byte) (int, error) { return o.r.Read(p[:min(1, len(p))]) }

func TestReadResponseTwoInARow(t *testing.T) {
	raw := "HTTP/1.1 200 OK\r\nContent-Length: 1\r\n\r\nAHTTP/1.1 404 Not Found\r\nContent-Length: 2\r\n\r\nBC"
	br := bufio.NewReaderSize(strings.NewReader(raw), readBufSize)
	var dst []byte
	if r, _, err := readResponse(br, &dst); err != nil || r.code != 200 || string(dst) != "A" {
		t.Fatal(r, err, string(dst))
	}
	if r, _, err := readResponse(br, &dst); err != nil || r.code != 404 || string(dst) != "BC" {
		t.Fatal(r, err, string(dst))
	}
}

func TestReadResponseBodyBufferIsReused(t *testing.T) {
	dst := make([]byte, 0, 64)
	br := bufio.NewReaderSize(strings.NewReader("HTTP/1.1 200 OK\r\nContent-Length: 4\r\n\r\nwxyz"), readBufSize)
	if _, _, err := readResponse(br, &dst); err != nil || string(dst) != "wxyz" || cap(dst) != 64 {
		t.Fatalf("%q cap %d %v", dst, cap(dst), err)
	}
}

func TestReadResponseBad(t *testing.T) {
	cases := map[string]string{
		"not http":            "SSH-2.0-x\r\n\r\n",
		"short status line":   "HTTP/1.1 20\r\n\r\n",
		"non numeric code":    "HTTP/1.1 2x0 OK\r\n\r\n",
		"http/2":              "HTTP/2.0 200 OK\r\n\r\n",
		"no space after code": "HTTP/1.1 200OK\r\n\r\n",
		"bad content-length":  "HTTP/1.1 200 OK\r\nContent-Length: abc\r\n\r\n",
		"negative length":     "HTTP/1.1 200 OK\r\nContent-Length: -1\r\n\r\n",
		"conflicting lengths": "HTTP/1.1 200 OK\r\nContent-Length: 1\r\nContent-Length: 2\r\n\r\nab",
		"header without :":    "HTTP/1.1 200 OK\r\nbroken\r\n\r\n",
		"bad header length":   "HTTP/1.1 200 OK\r\nInference-Header-Content-Length: x\r\nContent-Length: 0\r\n\r\n",
		"huge body":           "HTTP/1.1 200 OK\r\nContent-Length: 99999999999\r\n\r\n",
		"bad chunk size":      "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\nzz\r\n",
		"chunk without crlf":  "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n1\r\nabXX0\r\n\r\n",
		"line too long":       "HTTP/1.1 200 OK\r\nX: " + strings.Repeat("a", 2*readBufSize) + "\r\n\r\n",
		"too many headers":    "HTTP/1.1 200 OK\r\n" + strings.Repeat("X: y\r\n", maxHeaders+2) + "\r\n",
	}
	for name, raw := range cases {
		for _, cut := range []int{0, 5, len(raw) / 2} {
			_, _, _, err := parseAll(raw, true, cut)
			var b *badResponse
			if !errors.As(err, &b) {
				t.Errorf("%s cut=%d: got %v, want an invalid response", name, cut, err)
			}
		}
	}
}

func TestReadResponseTruncated(t *testing.T) {
	full := "HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\n0123456789"
	for n := 0; n < len(full); n++ {
		_, _, _, err := parseAll(full[:n], true, 0)
		if err == nil {
			t.Fatalf("truncated at %d parsed", n)
		}
		var b *badResponse
		if errors.As(err, &b) {
			t.Fatalf("truncated at %d is a bad response, not a transport error: %v", n, err)
		}
	}
	// A chunked body cut short too.
	if _, _, _, err := parseAll("HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nab", true, 0); !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatal(err)
	}
}

func TestHasToken(t *testing.T) {
	if !hasToken([]byte("keep-alive, Close"), "close") || hasToken([]byte("closed"), "close") || hasToken(nil, "close") {
		t.Fatal("hasToken")
	}
}
