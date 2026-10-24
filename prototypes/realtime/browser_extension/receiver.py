"""Local receiver for the browser-extension prototype: a tiny WebSocket server (standard library only) that writes WAV files.

    python receiver.py [--port 8765] [--out ../_local/ext_run] [--secret-file ../_local/receiver_secret.txt] [--extension-id <id>]

Security model (a web page must not be able to feed or read anything here):
  * binds to 127.0.0.1 only;
  * the Host header must be 127.0.0.1:<port> or localhost:<port> (DNS-rebinding guard);
  * the Origin header must be chrome-extension://<id> (a normal web page sends https://...; with --extension-id it must be that very id);
  * the first message must be a JSON hello with the shared secret (compared in constant time), within 5 s; the secret is never in a URL;
  * messages are capped at 1 MiB; at most --max-conns connections at once.
The secret is created on first start in --secret-file (not printed); paste it into the extension's options page.

Wire format after the hello:
  audio (binary): u32 seq | f64 context time in ms | int16 little-endian mono samples   (one packet = 50 ms at 48 kHz)
  dom   (text)  : one JSON object per message (events of the DOM probe)
Per connection it writes <out>/<source>_<n>.wav (+ <source>_<n>.stats.json: packets, gaps, bytes, arrival jitter, audio seconds vs wall clock).
"""
import argparse, asyncio, base64, hashlib, hmac, json, os, secrets, struct, sys, time, wave

sys.stdout.reconfigure(encoding="utf-8")
GUID = b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
MAX_MSG = 1 << 20


class Sink:
    count = {}

    def __init__(self, out, source, rate):
        self.source, self.rate = source, rate
        Sink.count[source] = Sink.count.get(source, 0) + 1
        self.name = f"{source}_{Sink.count[source]}"                # every connection gets its own file: tab_1.wav, tab_2.wav ...
        self.w = wave.open(os.path.join(out, f"{self.name}.wav"), "wb")
        self.w.setnchannels(1); self.w.setsampwidth(2); self.w.setframerate(rate)
        self.stats_path = os.path.join(out, f"{self.name}.stats.json")
        self.t0 = time.perf_counter(); self.last_arrival = None
        self.packets = self.frames = self.gaps = self.missing = self.bytes = 0
        self.expect = None; self.jitter = []; self.first_ctx = None; self.last_ctx = None

    def packet(self, data):
        if len(data) < 12:
            return
        seq, ctx_ms = struct.unpack("<Id", data[:12])
        pcm = data[12:]
        now = time.perf_counter()
        if self.last_arrival is not None:
            self.jitter.append(now - self.last_arrival)
        self.last_arrival = now
        if self.expect is not None and seq != self.expect:
            self.gaps += 1; self.missing += (seq - self.expect) & 0xFFFFFFFF
        self.expect = (seq + 1) & 0xFFFFFFFF
        self.first_ctx = ctx_ms if self.first_ctx is None else self.first_ctx
        self.last_ctx = ctx_ms
        self.w.writeframes(pcm)
        self.packets += 1; self.frames += len(pcm) // 2; self.bytes += len(data)

    def close(self):
        self.w.close()
        j = sorted(self.jitter) or [0.0]
        wall = time.perf_counter() - self.t0
        st = {"file": self.name, "source": self.source, "rate": self.rate, "packets": self.packets, "audio_s": round(self.frames / self.rate, 3),
              "wall_s": round(wall, 3), "seq_gaps": self.gaps, "missing_packets": self.missing, "bytes": self.bytes,
              "arrival_interval_ms": {"median": round(1000 * j[len(j) // 2], 1), "p99": round(1000 * j[int(len(j) * 0.99) - 1], 1) if len(j) > 1 else 0,
                                      "max": round(1000 * j[-1], 1)},
              "audio_minus_wall_ms": round(1000 * (self.frames / self.rate - wall), 1),
              "context_clock_span_s": round((self.last_ctx - self.first_ctx) / 1000, 3) if self.first_ctx is not None else None}
        json.dump(st, open(self.stats_path, "w", encoding="utf-8"), indent=1)
        print("closed", json.dumps(st), flush=True)


async def read_frame(reader):
    """Returns (opcode, payload) of one complete message (continuations joined), or None on close."""
    msg, opcode = b"", None
    while True:
        h = await reader.readexactly(2)
        fin, op, masked, ln = h[0] >> 7, h[0] & 15, h[1] >> 7, h[1] & 127
        if ln == 126:
            ln = struct.unpack(">H", await reader.readexactly(2))[0]
        elif ln == 127:
            ln = struct.unpack(">Q", await reader.readexactly(8))[0]
        if ln > MAX_MSG or len(msg) + ln > MAX_MSG:
            raise ValueError("message too large")
        if not masked:
            raise ValueError("client frames must be masked")
        mask = await reader.readexactly(4)
        data = bytearray(await reader.readexactly(ln))
        for i in range(ln):
            data[i] ^= mask[i & 3]
        if op in (8, 9, 10):
            return op, bytes(data)
        if op in (1, 2):
            opcode = op
        msg += bytes(data)
        if fin:
            return opcode, msg


def frame(op, payload=b""):
    return bytes([0x80 | op, len(payload)]) + payload      # control frames only (< 126 bytes)


class Server:
    def __init__(self, a):
        self.a, self.conns, self.sinks = a, 0, {}
        os.makedirs(a.out, exist_ok=True)
        if not os.path.exists(a.secret_file):
            os.makedirs(os.path.dirname(os.path.abspath(a.secret_file)), exist_ok=True)
            open(a.secret_file, "w").write(secrets.token_urlsafe(24))
        self.secret = open(a.secret_file).read().strip().encode()
        self.dom = open(os.path.join(a.out, "dom.jsonl"), "a", encoding="utf-8")

    async def handle(self, reader, writer):
        peer = writer.get_extra_info("peername")
        try:
            await self._handle(reader, writer, peer)
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        except Exception as e:
            print("connection error:", type(e).__name__, e, flush=True)
        finally:
            self.conns -= 1
            try:
                writer.close()
            except Exception:
                pass

    async def _handle(self, reader, writer, peer):
        self.conns += 1
        head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 5)
        lines = head.decode("latin-1").split("\r\n")
        hdr = {k.strip().lower(): v.strip() for k, _, v in (ln.partition(":") for ln in lines[1:] if ":" in ln)}
        def refuse(code, why):
            print(f"refused {peer}: {why}", flush=True)
            writer.write(f"HTTP/1.1 {code} {why}\r\nConnection: close\r\nContent-Length: 0\r\n\r\n".encode())
        if self.conns > self.a.max_conns:
            return refuse(503, "too many connections")
        if hdr.get("host", "") not in (f"127.0.0.1:{self.a.port}", f"localhost:{self.a.port}"):
            return refuse(403, "bad host")
        origin = hdr.get("origin", "")
        if not origin.startswith("chrome-extension://") or (self.a.extension_id and origin != f"chrome-extension://{self.a.extension_id}"):
            return refuse(403, "bad origin")
        if hdr.get("upgrade", "").lower() != "websocket" or "sec-websocket-key" not in hdr:
            return refuse(400, "not a websocket request")
        acc = base64.b64encode(hashlib.sha1(hdr["sec-websocket-key"].encode() + GUID).digest()).decode()
        writer.write(f"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: {acc}\r\n\r\n".encode())
        await writer.drain()
        op, data = await asyncio.wait_for(read_frame(reader), 5)
        try:
            hello = json.loads(data.decode())
            ok = op == 1 and hmac.compare_digest(str(hello.get("secret", "")).encode(), self.secret)
        except Exception:
            ok = False
        if not ok:
            print(f"bad secret from {peer}", flush=True)
            writer.write(frame(8, struct.pack(">H", 4401) + b"unauthorized")); await writer.drain()
            return
        source = str(hello.get("source", "x"))
        if source not in ("tab", "mic", "dom"):
            return
        sink = Sink(self.a.out, source, int(hello.get("sampleRate", 48000))) if source != "dom" else None
        print(f"connected: {source} {hello.get('sampleRate')} Hz from {origin}", flush=True)
        try:
            while True:
                op, data = await read_frame(reader)
                if op == 8:
                    break
                if op == 9:
                    writer.write(frame(10, data)); await writer.drain()
                elif op == 2 and sink:
                    sink.packet(data)
                elif op == 1 and source == "dom":
                    self.dom.write(data.decode("utf-8", "replace") + "\n"); self.dom.flush()
        finally:
            if sink:
                sink.close()
            print(f"disconnected: {source}", flush=True)


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--out", default=os.path.join("..", "_local", "ext_run"))
    ap.add_argument("--secret-file", default=os.path.join("..", "_local", "receiver_secret.txt"))
    ap.add_argument("--extension-id", default="")
    ap.add_argument("--max-conns", type=int, default=4)
    a = ap.parse_args()
    s = Server(a)
    srv = await asyncio.start_server(s.handle, "127.0.0.1", a.port)
    print(f"listening on 127.0.0.1:{a.port}, writing to {os.path.abspath(a.out)}", flush=True)
    async with srv:
        await srv.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
