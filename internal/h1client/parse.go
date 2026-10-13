package h1client

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"slices"
	"strconv"
)

// maxBody bounds a response body, so a broken server cannot make the client allocate
// without limit.
const maxBody = 64 << 20

// maxHeaders bounds the header lines of one response.
const maxHeaders = 200

// badResponse is a response that is not valid HTTP/1.1 (the request completes as
// invalid_response and the connection is closed, since its state is unknown).
type badResponse struct{ msg string }

func (e *badResponse) Error() string { return "invalid HTTP response: " + e.msg }

func bad(format string, a ...any) error { return &badResponse{fmt.Sprintf(format, a...)} }

// response is the part of a response the client needs.
type response struct {
	code   int  // status code
	hdrLen int  // Inference-Header-Content-Length, -1 when absent
	close  bool // the server ends the connection after this response
}

// readLine returns one line without its terminator, valid until the next read.
func readLine(br *bufio.Reader) ([]byte, error) {
	line, err := br.ReadSlice('\n')
	switch {
	case err == bufio.ErrBufferFull:
		return nil, bad("line longer than %d bytes", br.Size())
	case err != nil:
		if err == io.EOF && len(line) > 0 {
			err = io.ErrUnexpectedEOF
		}
		return nil, err
	}
	line = line[:len(line)-1]
	if n := len(line); n > 0 && line[n-1] == '\r' {
		line = line[:n-1]
	}
	return line, nil
}

// readResponse reads one response from br. When dst is not nil the body replaces its
// content, otherwise the body is read and dropped; bodyLen is its length either way. A
// 1xx interim response (other than 101) is skipped. It returns a *badResponse for
// malformed input, and the read error for a connection that ended or failed.
func readResponse(br *bufio.Reader, dst *[]byte) (resp response, bodyLen int, err error) {
	for {
		resp = response{hdrLen: -1}
		line, err := readLine(br)
		if err != nil {
			return resp, 0, err
		}
		// "HTTP/1.x SP code SP reason"
		if len(line) < 12 || !bytes.HasPrefix(line, []byte("HTTP/1.")) || line[8] != ' ' || (line[7] != '0' && line[7] != '1') {
			return resp, 0, bad("bad status line %q", clip(line))
		}
		http10 := line[7] == '0'
		code := 0
		for _, c := range line[9:12] {
			if c < '0' || c > '9' {
				return resp, 0, bad("bad status code in %q", clip(line))
			}
			code = code*10 + int(c-'0')
		}
		if len(line) > 12 && line[12] != ' ' {
			return resp, 0, bad("bad status line %q", clip(line))
		}
		resp.code = code

		contentLen := int64(-1)
		chunked, keepAlive := false, false
		for n := 0; ; n++ {
			if n > maxHeaders {
				return resp, 0, bad("more than %d headers", maxHeaders)
			}
			h, err := readLine(br)
			if err != nil {
				return resp, 0, err
			}
			if len(h) == 0 {
				break
			}
			colon := bytes.IndexByte(h, ':')
			if colon <= 0 {
				return resp, 0, bad("bad header line %q", clip(h))
			}
			name, val := h[:colon], bytes.TrimSpace(h[colon+1:])
			switch {
			case bytes.EqualFold(name, []byte("content-length")):
				v, err := strconv.ParseUint(string(val), 10, 63)
				if err != nil {
					return resp, 0, bad("bad Content-Length %q", clip(val))
				}
				if contentLen >= 0 && contentLen != int64(v) {
					return resp, 0, bad("conflicting Content-Length headers")
				}
				contentLen = int64(v)
			case bytes.EqualFold(name, []byte("transfer-encoding")):
				if hasToken(val, "chunked") {
					chunked = true
				}
			case bytes.EqualFold(name, []byte("connection")):
				if hasToken(val, "close") {
					resp.close = true
				}
				if hasToken(val, "keep-alive") {
					keepAlive = true
				}
			case bytes.EqualFold(name, []byte(headerContentLength)):
				v, err := strconv.ParseUint(string(val), 10, 31)
				if err != nil {
					return resp, 0, bad("bad %s %q", headerContentLength, clip(val))
				}
				resp.hdrLen = int(v)
			}
		}
		if http10 && !keepAlive {
			resp.close = true
		}
		if code >= 100 && code < 200 && code != 101 {
			continue // interim response: the real one follows
		}
		if chunked {
			resp.close = resp.close || contentLen >= 0 // both: ambiguous framing, do not reuse
			n, err := readChunked(br, dst)
			return resp, n, err
		}
		switch {
		case code == 204 || code == 304 || code == 101:
			if dst != nil {
				*dst = (*dst)[:0]
			}
			return resp, 0, nil
		case contentLen >= 0:
			n, err := readN(br, contentLen, dst)
			return resp, n, err
		default: // the body runs to the end of the connection
			resp.close = true
			n, err := readToEOF(br, dst)
			return resp, n, err
		}
	}
}

func clip(b []byte) []byte {
	if len(b) > 64 {
		return b[:64]
	}
	return b
}

// hasToken reports whether the comma-separated header value has the token (any case).
func hasToken(val []byte, token string) bool {
	for len(val) > 0 {
		var item []byte
		if i := bytes.IndexByte(val, ','); i >= 0 {
			item, val = val[:i], val[i+1:]
		} else {
			item, val = val, nil
		}
		if bytes.EqualFold(bytes.TrimSpace(item), []byte(token)) {
			return true
		}
	}
	return false
}

// readN reads exactly n body bytes into dst (or drops them).
func readN(br *bufio.Reader, n int64, dst *[]byte) (int, error) {
	if n > maxBody {
		return 0, bad("body of %d bytes is over the %d limit", n, maxBody)
	}
	if dst == nil {
		d, err := br.Discard(int(n))
		return d, unexpected(err)
	}
	*dst = slices.Grow((*dst)[:0], int(n))[:n]
	_, err := io.ReadFull(br, *dst)
	return int(n), unexpected(err)
}

// unexpected turns an EOF inside a message into io.ErrUnexpectedEOF.
func unexpected(err error) error {
	if err == io.EOF {
		return io.ErrUnexpectedEOF
	}
	return err
}

func readToEOF(br *bufio.Reader, dst *[]byte) (int, error) {
	var local []byte
	if dst == nil {
		dst = &local
	}
	*dst = (*dst)[:0]
	for {
		if len(*dst) > maxBody {
			return 0, bad("body is over the %d limit", maxBody)
		}
		*dst = slices.Grow(*dst, 4096)
		n, err := br.Read((*dst)[len(*dst):cap(*dst)])
		*dst = (*dst)[:len(*dst)+n]
		if err == io.EOF {
			return len(*dst), nil
		}
		if err != nil {
			return 0, err
		}
	}
}

// readChunked reads a chunked body (defensively: Triton answers with Content-Length).
func readChunked(br *bufio.Reader, dst *[]byte) (int, error) {
	total := 0
	if dst != nil {
		*dst = (*dst)[:0]
	}
	for {
		line, err := readLine(br)
		if err != nil {
			return 0, unexpected(err)
		}
		if i := bytes.IndexByte(line, ';'); i >= 0 {
			line = line[:i]
		}
		size, err := strconv.ParseUint(string(bytes.TrimSpace(line)), 16, 63)
		if err != nil {
			return 0, bad("bad chunk size %q", clip(line))
		}
		if size == 0 {
			for { // trailers, then the blank line
				t, err := readLine(br)
				if err != nil {
					return 0, unexpected(err)
				}
				if len(t) == 0 {
					return total, nil
				}
			}
		}
		if size > maxBody || int64(total)+int64(size) > maxBody {
			return 0, bad("chunked body is over the %d limit", maxBody)
		}
		if dst == nil {
			_, err = br.Discard(int(size))
		} else {
			old := len(*dst)
			*dst = slices.Grow(*dst, int(size))[:old+int(size)]
			_, err = io.ReadFull(br, (*dst)[old:])
		}
		if err != nil {
			return 0, unexpected(err)
		}
		total += int(size)
		var crlf [2]byte
		if _, err := io.ReadFull(br, crlf[:]); err != nil {
			return 0, unexpected(err)
		}
		if crlf != [2]byte{'\r', '\n'} {
			return 0, bad("chunk not followed by CRLF")
		}
	}
}

func newReader(r io.Reader) *bufio.Reader { return bufio.NewReaderSize(r, readBufSize) }
