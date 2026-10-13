package h2client

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"net"
	"strconv"
	"sync"
	"time"

	"github.com/criteo/triton-stress/internal/clock"
	"github.com/criteo/triton-stress/internal/grpcclient"
	"golang.org/x/net/http2"
	"golang.org/x/net/http2/hpack"
)

// Receive windows advertised to the server: fixed, so nothing here ever needs a
// bandwidth-delay estimate (and no PING).
const (
	streamRecvWindow = 1 << 20
	connRecvWindow   = 16 << 20
	// A window update is sent once this much has been consumed.
	connUpdateAt   = connRecvWindow / 4
	streamUpdateAt = streamRecvWindow / 2
)

// Defaults from RFC 9113 until the server's SETTINGS say otherwise.
const (
	defaultWindow   = 65535
	defaultFrameMax = 16384
	maxWindow       = 1<<31 - 1
)

// idSlack is how many stream ids are kept in reserve when the ids run out, for the
// requests already queued.
const idSlack = 1 << 20

const clientPreface = "PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n"

func appendFrameHeader(b []byte, length int, typ, flags byte, stream uint32) []byte {
	return append(b, byte(length>>16), byte(length>>8), byte(length), typ, flags,
		byte(stream>>24)&0x7f, byte(stream>>16), byte(stream>>8), byte(stream))
}

// streamTable maps stream ids to requests. Ids grow by 2 and requests finish roughly in
// order, so a ring indexed by id/2 almost never collides; it doubles when it does.
type streamTable struct {
	slots []*Request
	mask  uint32
}

func newStreamTable() streamTable { return streamTable{slots: make([]*Request, 1024), mask: 1023} }

func (s *streamTable) get(id uint32) *Request {
	if r := s.slots[(id>>1)&s.mask]; r != nil && r.id == id {
		return r
	}
	return nil
}

func (s *streamTable) put(r *Request) {
	for {
		i := (r.id >> 1) & s.mask
		if s.slots[i] == nil {
			s.slots[i] = r
			return
		}
		s.grow()
	}
}

func (s *streamTable) del(r *Request) {
	if i := (r.id >> 1) & s.mask; s.slots[i] == r {
		s.slots[i] = nil
	}
}

func (s *streamTable) grow() {
	old := s.slots
	for size := 2 * len(old); ; size *= 2 {
		slots := make([]*Request, size)
		mask := uint32(size - 1)
		ok := true
		for _, r := range old {
			if r == nil {
				continue
			}
			i := (r.id >> 1) & mask
			if slots[i] != nil {
				ok = false
				break
			}
			slots[i] = r
		}
		if ok {
			s.slots, s.mask = slots, mask
			return
		}
	}
}

// transport is one HTTP/2 connection to the server.
type transport struct {
	nc       net.Conn
	hdrBlock []byte // pre-encoded request header block, identical for every request

	mu     sync.Mutex
	wbuf   []byte // frames waiting for the writer
	closed bool   // no request is accepted any more and everything was failed
	goaway bool   // the server sent GOAWAY: no new streams

	nextID   uint32
	streams  streamTable
	sendHead *Request // FIFO of requests with frames still to write
	sendTail *Request
	dlHead   *Request // every unfinished request, in submit order (so in deadline order)
	dlTail   *Request
	active   int // streams opened and not finished

	connWin  int64 // connection-level send window
	initWin  int64 // server's initial stream window
	maxConc  uint32
	maxFrame int
	recvAcc  int64 // bytes received and not yet credited to the server

	kicked bool // the writer goroutine has been woken and has not yet found wbuf empty

	wake     chan struct{}
	reapWake chan struct{}
	done     chan struct{} // closed when the transport is shut down
	ready    chan struct{} // closed on the server's first SETTINGS
	lost     chan struct{} // closed when new requests can no longer be sent (closed or GOAWAY)
	lostOnce sync.Once
	err      error

	rdone []*Request // reader-owned scratch list of finished requests

	// Response header block being decoded (reader goroutine only).
	dec   *hpack.Decoder
	hdrSt int
	hdrGS int
}

func newTransport(nc net.Conn, hdrBlock []byte) *transport {
	t := &transport{
		nc: nc, hdrBlock: hdrBlock,
		nextID: 1, streams: newStreamTable(),
		connWin: defaultWindow, initWin: defaultWindow, maxConc: 1<<32 - 1, maxFrame: defaultFrameMax,
		wake: make(chan struct{}, 1), reapWake: make(chan struct{}, 1),
		done: make(chan struct{}), ready: make(chan struct{}), lost: make(chan struct{}),
		wbuf: make([]byte, 0, 64<<10),
	}
	t.dec = hpack.NewDecoder(4096, func(f hpack.HeaderField) {
		switch f.Name {
		case ":status":
			t.hdrSt, _ = strconv.Atoi(f.Value)
		case "grpc-status":
			if n, err := strconv.Atoi(f.Value); err == nil {
				t.hdrGS = n
			} else {
				t.hdrGS = int(grpcclient.StatusUnknown)
			}
		}
	})
	return t
}

// start sends the connection preface and starts the goroutines.
func (t *transport) start() {
	b := make([]byte, 0, 128)
	b = append(b, clientPreface...)
	b = appendFrameHeader(b, 3*6, 0x4, 0, 0)
	b = appendSetting(b, 0x2, 0)                // ENABLE_PUSH
	b = appendSetting(b, 0x4, streamRecvWindow) // INITIAL_WINDOW_SIZE
	b = appendSetting(b, 0x6, 64<<10)           // MAX_HEADER_LIST_SIZE
	b = appendFrameHeader(b, 4, 0x8, 0, 0)      // WINDOW_UPDATE on the connection
	inc := uint32(connRecvWindow - defaultWindow)
	b = append(b, byte(inc>>24), byte(inc>>16), byte(inc>>8), byte(inc))
	t.mu.Lock()
	t.wbuf = append(t.wbuf, b...)
	t.unlockFlush()
	go t.writeLoop()
	go t.readLoop()
	go t.reapLoop()
}

func appendSetting(b []byte, id uint16, v uint32) []byte {
	return append(b, byte(id>>8), byte(id), byte(v>>24), byte(v>>16), byte(v>>8), byte(v))
}

// unlockFlush releases t.mu and wakes the writer goroutine if frames are waiting and it
// is not already on its way to them. (The wake is a non-blocking send made under the
// lock, so it cannot race with fail closing the channel.)
func (t *transport) unlockFlush() {
	if len(t.wbuf) > 0 && !t.kicked && !t.closed {
		t.kicked = true
		select {
		case t.wake <- struct{}{}:
		default:
		}
	}
	t.mu.Unlock()
}

// deliver calls Done on finished requests; the transport lock must not be held.
func deliver(ds []*Request) {
	for i, r := range ds {
		ds[i] = nil
		r.Done(r)
	}
}

// submit queues r; false means the transport takes no more requests (r is untouched).
func (t *transport) submit(r *Request, now int64) bool {
	t.mu.Lock()
	if t.closed || t.goaway {
		t.mu.Unlock()
		return false
	}
	if t.nextID >= maxWindow-idSlack { // stream ids nearly used up: finish on a fresh connection
		t.goaway = true
		t.mu.Unlock()
		t.markLost()
		return false
	}
	r.deadline = now + int64(r.Timeout)
	t.insertDeadline(r)
	// send queue (tail)
	r.sprev, r.snext, r.inSendQ = t.sendTail, nil, true
	if t.sendTail != nil {
		t.sendTail.snext = r
	} else {
		t.sendHead = r
	}
	t.sendTail = r
	t.pump()
	t.unlockFlush()
	return true
}

// insertDeadline links r into the deadline list. Requests normally share one timeout,
// so r goes at the tail; a shorter one is moved forward past the longer ones.
func (t *transport) insertDeadline(r *Request) {
	after := t.dlTail
	for after != nil && after.deadline > r.deadline {
		after = after.dprev
	}
	r.inDL = true
	r.dprev = after
	if after == nil {
		r.dnext = t.dlHead
		t.dlHead = r
	} else {
		r.dnext = after.dnext
		after.dnext = r
	}
	if r.dnext != nil {
		r.dnext.dprev = r
	} else {
		t.dlTail = r
	}
	if r.dprev == nil { // new head: the reaper may be sleeping on a later deadline
		select {
		case t.reapWake <- struct{}{}:
		default:
		}
	}
}

// pump writes as much of the send queue as the windows allow. Called with t.mu held.
func (t *transport) pump() {
	for r := t.sendHead; r != nil; r = t.sendHead {
		if !r.started {
			if uint32(t.active) >= t.maxConc || t.nextID > maxWindow-2 {
				return
			}
			r.TReady = clock.Now()
			r.started = true
			r.id = t.nextID
			t.nextID += 2
			r.sendWin = t.initWin
			t.streams.put(r)
			t.active++
			t.wbuf = appendFrameHeader(t.wbuf, len(t.hdrBlock), 0x1, 0x4, r.id) // HEADERS, END_HEADERS
			t.wbuf = append(t.wbuf, t.hdrBlock...)
		}
		total := 5 + len(r.Body)
		for r.sent < total {
			n := min(total-r.sent, t.maxFrame)
			n = int(min(int64(n), t.connWin, r.sendWin))
			if n <= 0 {
				return // out of window: wait for a WINDOW_UPDATE
			}
			var flags byte
			if r.sent+n == total {
				flags = 0x1 // END_STREAM
			}
			t.wbuf = appendFrameHeader(t.wbuf, n, 0x0, flags, r.id)
			if r.sent < 5 {
				var prefix [5]byte
				prefix[1], prefix[2], prefix[3], prefix[4] = byte(len(r.Body)>>24), byte(len(r.Body)>>16), byte(len(r.Body)>>8), byte(len(r.Body))
				k := min(5-r.sent, n)
				t.wbuf = append(t.wbuf, prefix[r.sent:r.sent+k]...)
				t.wbuf = append(t.wbuf, r.Body[:n-k]...)
			} else {
				t.wbuf = append(t.wbuf, r.Body[r.sent-5:r.sent-5+n]...)
			}
			r.sent += n
			t.connWin -= int64(n)
			r.sendWin -= int64(n)
			if r.TSend == 0 {
				r.TSend = clock.Now()
			}
		}
		// fully written: leave the send queue
		t.sendHead = r.snext
		if t.sendHead != nil {
			t.sendHead.sprev = nil
		} else {
			t.sendTail = nil
		}
		r.snext, r.inSendQ = nil, false
	}
}

// finishLocked ends r with st. If the stream is still open on the wire and the
// connection is up, the server is told to cancel it. r is appended to ds, and the
// caller delivers it after unlocking.
func (t *transport) finishLocked(r *Request, st grpcclient.Status, reset bool, ds *[]*Request) {
	if r.finished {
		return
	}
	r.finished = true
	r.Status = st
	if r.inSendQ {
		if r.sprev != nil {
			r.sprev.snext = r.snext
		} else {
			t.sendHead = r.snext
		}
		if r.snext != nil {
			r.snext.sprev = r.sprev
		} else {
			t.sendTail = r.sprev
		}
		r.sprev, r.snext, r.inSendQ = nil, nil, false
	}
	if r.inDL {
		if r.dprev != nil {
			r.dprev.dnext = r.dnext
		} else {
			t.dlHead = r.dnext
		}
		if r.dnext != nil {
			r.dnext.dprev = r.dprev
		} else {
			t.dlTail = r.dprev
		}
		r.dprev, r.dnext, r.inDL = nil, nil, false
	}
	if r.started {
		t.streams.del(r)
		t.active--
		if (reset || r.sent < 5+len(r.Body)) && !t.closed { // the server is told to drop what we did not finish sending
			t.wbuf = appendFrameHeader(t.wbuf, 4, 0x3, 0, r.id) // RST_STREAM
			t.wbuf = append(t.wbuf, 0, 0, 0, 8)                 // CANCEL
		}
	}
	*ds = append(*ds, r)
	if t.goaway && t.active == 0 && t.sendHead == nil && !t.closed {
		go t.fail(errors.New("connection drained after GOAWAY"), grpcclient.StatusTransport)
	}
}

// fail shuts the transport down and ends every unfinished request with st.
func (t *transport) fail(err error, st grpcclient.Status) {
	t.mu.Lock()
	if t.closed {
		t.mu.Unlock()
		return
	}
	t.closed = true
	t.err = err
	var ds []*Request
	for r := t.dlHead; r != nil; r = t.dlHead {
		t.finishLocked(r, st, false, &ds)
	}
	close(t.wake) // ends the writer
	t.mu.Unlock()
	t.markLost()
	close(t.done)
	_ = t.nc.Close()
	deliver(ds)
}

func (t *transport) markLost() { t.lostOnce.Do(func() { close(t.lost) }) }

// close is fail for a shutdown requested from outside.
func (t *transport) close() {
	t.fail(errors.New("client connection is closing"), grpcclient.StatusTransport)
}

// writeLoop is the only goroutine that writes to the connection. Everything queued
// while it was in a write goes out in the next one, so under load a write carries many
// requests.
func (t *transport) writeLoop() {
	var spare []byte
	for range t.wake {
		for {
			t.mu.Lock()
			if len(t.wbuf) == 0 {
				t.kicked = false
				t.mu.Unlock()
				break
			}
			buf := t.wbuf
			t.wbuf = spare[:0]
			t.mu.Unlock()
			if _, err := t.nc.Write(buf); err != nil {
				t.fail(fmt.Errorf("error writing: %w", err), grpcclient.StatusTransport)
				return
			}
			if cap(buf) <= 4<<20 {
				spare = buf
			} else {
				spare = nil
			}
		}
	}
}

func (t *transport) readLoop() {
	br := bufio.NewReaderSize(t.nc, 64<<10)
	fr := http2.NewFramer(io.Discard, br)
	fr.SetReuseFrames()
	fr.SetMaxReadFrameSize(1 << 20)
	var (
		hdrStream uint32
		hdrEnd    bool // END_STREAM was set on the HEADERS frame that began the block
		seenFirst bool
	)
	for {
		f, err := fr.ReadFrame()
		if err != nil {
			var se http2.StreamError
			if errors.As(err, &se) { // a malformed frame on one stream: not fatal for the others
				continue
			}
			if err == io.EOF || err == io.ErrUnexpectedEOF {
				err = errors.New("error reading from server: EOF")
			}
			t.fail(err, grpcclient.StatusTransport)
			return
		}
		now := clock.Now()
		ds := t.rdone[:0]
		switch f := f.(type) {
		case *http2.DataFrame:
			t.onData(f, now, &ds)
		case *http2.HeadersFrame:
			hdrStream, hdrEnd = f.StreamID, f.StreamEnded()
			t.hdrSt, t.hdrGS = 0, -1
			if _, err := t.dec.Write(f.HeaderBlockFragment()); err != nil {
				t.fail(fmt.Errorf("connection error: %w", err), grpcclient.StatusTransport)
				return
			}
			if f.HeadersEnded() {
				if err := t.dec.Close(); err != nil {
					t.fail(fmt.Errorf("connection error: %w", err), grpcclient.StatusTransport)
					return
				}
				t.onHeaders(hdrStream, hdrEnd, now, &ds)
			}
		case *http2.ContinuationFrame:
			if _, err := t.dec.Write(f.HeaderBlockFragment()); err != nil {
				t.fail(fmt.Errorf("connection error: %w", err), grpcclient.StatusTransport)
				return
			}
			if f.HeadersEnded() {
				if err := t.dec.Close(); err != nil {
					t.fail(fmt.Errorf("connection error: %w", err), grpcclient.StatusTransport)
					return
				}
				t.onHeaders(hdrStream, hdrEnd, now, &ds)
			}
		case *http2.SettingsFrame:
			if f.IsAck() {
				break
			}
			t.onSettings(f)
			if !seenFirst {
				seenFirst = true
				close(t.ready)
			}
		case *http2.WindowUpdateFrame:
			t.onWindowUpdate(f)
		case *http2.PingFrame:
			if !f.IsAck() {
				t.mu.Lock()
				t.wbuf = appendFrameHeader(t.wbuf, 8, 0x6, 0x1, 0)
				t.wbuf = append(t.wbuf, f.Data[:]...)
				t.unlockFlush()
			}
		case *http2.RSTStreamFrame:
			t.mu.Lock()
			if r := t.streams.get(f.StreamID); r != nil {
				r.TDone = now
				t.finishLocked(r, rstStatus(f.ErrCode), false, &ds)
				t.pump()
			}
			t.unlockFlush()
		case *http2.GoAwayFrame:
			t.onGoAway(f, now, &ds)
		case *http2.PushPromiseFrame:
			t.fail(errors.New("connection error: unexpected PUSH_PROMISE"), grpcclient.StatusTransport)
			return
		}
		deliver(ds)
		t.rdone = ds[:0]
	}
}

// onData handles a DATA frame.
func (t *transport) onData(f *http2.DataFrame, now int64, ds *[]*Request) {
	data := f.Data()
	n := int64(f.Length) // the whole frame counts against flow control, padding included
	t.mu.Lock()
	r := t.streams.get(f.StreamID)
	if r != nil {
		if r.msgSeen == 0 && len(data) > 0 && data[0] != 0 {
			r.badMsg = true // compressed: not understood
		}
		r.msgSeen += len(data)
		if r.Keep {
			r.Data = append(r.Data, data...)
		}
		r.recvAcc += n
		if r.recvAcc >= streamUpdateAt {
			t.wbuf = appendFrameHeader(t.wbuf, 4, 0x8, 0, r.id)
			t.wbuf = append(t.wbuf, byte(r.recvAcc>>24), byte(r.recvAcc>>16), byte(r.recvAcc>>8), byte(r.recvAcc))
			r.recvAcc = 0
		}
		if f.StreamEnded() { // a response with no trailers
			r.TDone = now
			t.endResponse(r, -1, ds)
		}
	}
	t.recvAcc += n
	if t.recvAcc >= connUpdateAt {
		t.wbuf = appendFrameHeader(t.wbuf, 4, 0x8, 0, 0)
		t.wbuf = append(t.wbuf, byte(t.recvAcc>>24), byte(t.recvAcc>>16), byte(t.recvAcc>>8), byte(t.recvAcc))
		t.recvAcc = 0
	}
	t.unlockFlush()
}

// onHeaders handles a complete response header block (the first headers or the
// trailers). The HPACK state has been updated whether or not the stream is still known.
func (t *transport) onHeaders(id uint32, end bool, now int64, ds *[]*Request) {
	t.mu.Lock()
	r := t.streams.get(id)
	if r != nil {
		if t.hdrSt != 0 && r.httpStatus == 0 {
			r.httpStatus = t.hdrSt
		}
		if end {
			r.TDone = now
			t.endResponse(r, t.hdrGS, ds)
		}
	}
	t.unlockFlush()
}

// endResponse decides the status of a response that ended (END_STREAM seen), with the
// grpc-status of the last header block (-1 when it carried none).
func (t *transport) endResponse(r *Request, grpcStatus int, ds *[]*Request) {
	st := grpcclient.StatusOK
	switch {
	case grpcStatus > 0:
		st = statusFromCode(grpcStatus)
	case grpcStatus == 0:
		if r.badMsg || r.msgSeen < 5 {
			st = grpcclient.StatusInvalidResponse
		} else {
			r.RespBytes = r.msgSeen - 5
			if r.Keep && len(r.Data) >= 5 {
				r.Data = r.Data[5:]
			}
		}
	default: // no grpc-status: derive it from the HTTP status as the gRPC spec says
		st = statusFromHTTP(r.httpStatus)
	}
	t.finishLocked(r, st, false, ds)
	if t.sendHead != nil {
		t.pump()
	}
}

func (t *transport) onSettings(f *http2.SettingsFrame) {
	t.mu.Lock()
	_ = f.ForeachSetting(func(s http2.Setting) error { // the callback never fails
		switch s.ID {
		case http2.SettingInitialWindowSize:
			delta := int64(s.Val) - t.initWin
			t.initWin = int64(s.Val)
			for _, r := range t.streams.slots {
				if r != nil {
					r.sendWin += delta
				}
			}
		case http2.SettingMaxConcurrentStreams:
			t.maxConc = s.Val
		case http2.SettingMaxFrameSize:
			t.maxFrame = int(s.Val)
		}
		return nil
	})
	t.wbuf = appendFrameHeader(t.wbuf, 0, 0x4, 0x1, 0) // SETTINGS ack
	t.pump()
	t.unlockFlush()
}

func (t *transport) onWindowUpdate(f *http2.WindowUpdateFrame) {
	t.mu.Lock()
	if f.StreamID == 0 {
		t.connWin += int64(f.Increment)
	} else if r := t.streams.get(f.StreamID); r != nil {
		r.sendWin += int64(f.Increment)
	}
	t.pump()
	t.unlockFlush()
}

func (t *transport) onGoAway(f *http2.GoAwayFrame, now int64, ds *[]*Request) {
	t.mu.Lock()
	t.goaway = true
	for _, r := range t.streams.slots {
		if r != nil && r.id > f.LastStreamID { // never processed by the server
			r.TDone = now
			t.finishLocked(r, grpcclient.StatusGoaway, false, ds)
		}
	}
	for r := t.sendHead; r != nil; r = t.sendHead { // not started: refused
		r.TDone = now
		t.finishLocked(r, grpcclient.StatusGoaway, false, ds)
	}
	if t.active == 0 && !t.closed {
		go t.fail(errors.New("connection drained after GOAWAY"), grpcclient.StatusTransport)
	}
	t.unlockFlush()
	t.markLost()
}

// reapLoop times requests out: they are in submit order, so only the head can be the
// next to expire.
func (t *transport) reapLoop() {
	timer := time.NewTimer(time.Hour)
	defer timer.Stop()
	var ds []*Request
	for {
		t.mu.Lock()
		head := t.dlHead
		var wait time.Duration
		if head != nil {
			wait = time.Duration(head.deadline - clock.Now())
		}
		if head != nil && wait <= 0 {
			now := clock.Now()
			for h := t.dlHead; h != nil && h.deadline <= now; h = t.dlHead {
				h.TDone = now
				t.finishLocked(h, grpcclient.StatusTimeout, true, &ds)
			}
			t.pump()
			t.unlockFlush()
			deliver(ds)
			ds = ds[:0]
			continue
		}
		t.mu.Unlock()
		if head == nil {
			select {
			case <-t.reapWake:
			case <-t.done:
				return
			}
			continue
		}
		if !timer.Stop() {
			select {
			case <-timer.C:
			default:
			}
		}
		timer.Reset(wait)
		select {
		case <-timer.C:
		case <-t.reapWake: // a request with an earlier deadline joined
		case <-t.done:
			return
		}
	}
}

func statusFromCode(c int) grpcclient.Status {
	if c < 0 || c > 16 {
		return grpcclient.StatusUnknown
	}
	return grpcclient.Status(c)
}

// statusFromHTTP is the gRPC status for a response that carried no grpc-status.
func statusFromHTTP(code int) grpcclient.Status {
	switch code {
	case 200:
		return grpcclient.StatusInternal // 200 without grpc-status: a protocol violation
	case 400:
		return grpcclient.StatusInternal
	case 401:
		return grpcclient.StatusUnauthenticated
	case 403:
		return grpcclient.StatusPermissionDenied
	case 404:
		return grpcclient.StatusUnimplemented
	case 429, 502, 503, 504:
		return grpcclient.StatusUnavailable
	}
	return grpcclient.StatusUnknown
}

// rstStatus maps an RST_STREAM error code received mid-request.
func rstStatus(code http2.ErrCode) grpcclient.Status {
	switch code {
	case http2.ErrCodeRefusedStream:
		return grpcclient.StatusTransport
	case http2.ErrCodeCancel:
		return grpcclient.StatusCanceled
	case http2.ErrCodeEnhanceYourCalm:
		return grpcclient.StatusResourceExhausted
	case http2.ErrCodeInadequateSecurity:
		return grpcclient.StatusPermissionDenied
	}
	return grpcclient.StatusInternal
}
