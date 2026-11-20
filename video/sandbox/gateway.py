"""Model gateway for the sandbox (our stand-in for the Kaggle model proxy).

The agent container has no internet route; its only reachable host is this gateway. The gateway
  * accepts Anthropic Messages API calls at /c/<call_id>/v1/messages,
  * forces the model to an allowlist (anything else is rewritten to GATEWAY_MODEL and logged),
  * adds the upstream credential (OLLAMA_API_KEY from its own environment; the agent only has a dummy token),
  * streams the response back unchanged and appends one usage record per request to /logs/usage.jsonl.
Nothing else is forwarded (telemetry etc. get a local 404).  Stdlib only.
"""
import http.client
import json
import os
import re
import select
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = os.environ.get("GATEWAY_UPSTREAM", "ollama.com")
UP_PREFIX = os.environ.get("GATEWAY_UPSTREAM_PREFIX", "")          # e.g. "/anthropic" for api.deepseek.com
UP_MODEL = os.environ.get("GATEWAY_UPSTREAM_MODEL", "")            # upstream name of the allowed model (e.g. deepseek-flash)
MAX_STEPS = int(os.environ.get("GATEWAY_MAX_STEPS", "0"))          # per-run tool-step budget (0 = off)
WARN_STEPS = int(os.environ.get("GATEWAY_WARN_STEPS", "8"))        # announce the remaining budget for the last N steps
RETRY_S = float(os.environ.get("GATEWAY_RETRY_S", "600"))          # give up on an upstream request after this much retrying
TIME_BUDGET = float(os.environ.get("GATEWAY_TIME_BUDGET_S", "0"))  # per-run wall budget (0 = off): remaining minutes are
                                                                   # announced after 50%, the answer is forced at 90%
FIRST = {}                                                         # call id -> time of its first request
FATAL = {}                                                         # set when upstream says quota/balance is exhausted
STALL = {}                                                         # call id -> seconds spent waiting in retry backoff
FATAL_MARKERS = ("weekly usage limit", "insufficient balance", "insufficient_balance", "usage credits", "billing")
MODEL = os.environ.get("GATEWAY_MODEL", "deepseek-v4.1-flash")
ALLOW = set(filter(None, os.environ.get("GATEWAY_ALLOW", MODEL).split(",")))
KEY = ""  # read from stdin at startup (never from env/argv/files)
LOG = os.environ.get("GATEWAY_LOG", "/logs/usage.jsonl")
DUMP = os.environ.get("GATEWAY_DUMP_PREFIX", "")  # debug: dump request bodies of calls whose id starts with this
_lock = threading.Lock()
_vdm_bad = {}  # vdm upstream -> time of last failure
# Ollama cloud rejects too many concurrent requests (429) and drops TLS under load: cap in-flight upstream calls
# and retry transparently (the agent only sees a slower response).
_upstream = threading.BoundedSemaphore(int(os.environ.get("GATEWAY_MAX_INFLIGHT", "6")))
RETRY_STATUS = {429, 500, 502, 503, 504, 529}
CONNECT_ALLOW = set(filter(None, os.environ.get("GATEWAY_CONNECT_ALLOW", "").split(",")))  # e.g. api.anthropic.com:443
VDM_UPSTREAM = os.environ.get("VDM_UPSTREAM", "")  # e.g. http://172.17.0.1:18080 ; empty = vdm disabled
MODE = os.environ.get("GATEWAY_MODE", "anthropic")  # "google": Gemini API (generativelanguage.googleapis.com)
GOOGLE_RE = re.compile(r"^/c/([A-Za-z0-9_.^-]+)(/v1(?:beta|alpha)?/.*)$")
PATH_RE = re.compile(r"^/c/([A-Za-z0-9_.^-]+)(/v1/messages(?:/count_tokens)?)(\?.*)?$")


def log(rec):
    with _lock, open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")


def budget(body, rec, call=""):
    """Stateless per-run tool-step budget: the step count is the number of assistant turns already in the request.
    Only agent turns (requests offering the StructuredOutput tool) are budgeted. Near the end the remaining budget is
    appended to the newest user turn; at the cap the request may only call StructuredOutput (thinking off, since
    DeepSeek's thinking mode rejects a forced tool_choice)."""
    if not MAX_STEPS:
        return
    tools = body.get("tools") or []
    so = [t for t in tools if t.get("name") == "StructuredOutput"]
    msgs = body.get("messages") or []
    rec["tools"] = [t.get("name") for t in tools][:12]
    rec["last_role"] = msgs[-1].get("role") if msgs else None
    # Claude Code 2.1.300 ends the history with a role "system" reminder turn (in-history system prompt updates)
    if not so or not msgs or msgs[-1].get("role") not in ("user", "system"):
        return
    n = sum(1 for m in msgs if m.get("role") == "assistant")
    rec["step"] = n
    left = MAX_STEPS - n
    el = time.time() - FIRST.setdefault(call, time.time()) if call else 0.0
    rec["elapsed_s"] = round(el)
    t_warn = TIME_BUDGET and el >= 0.5 * TIME_BUDGET
    t_force = TIME_BUDGET and el >= 0.9 * TIME_BUDGET
    if left > WARN_STEPS and not t_warn:
        return
    last = msgs[-1]
    if isinstance(last.get("content"), str):
        last["content"] = [{"type": "text", "text": last["content"]}]
    if left > 0 and not t_force:
        note = (f"[harness] Budget: {left} of {MAX_STEPS} tool steps left"
                + (f", {max(0, TIME_BUDGET - el) / 60:.0f} of {TIME_BUDGET / 60:.0f} minutes left (long commands count)" if t_warn else "")
                + ". Use them on what decides the answer; when you are confident, answer with StructuredOutput.")
    else:
        note = ("[harness] Tool budget exhausted. Do not call any other tool: answer now with StructuredOutput, choosing "
                "the best-supported option from what you have already seen.")
        body["tools"] = so
        body["tool_choice"] = {"type": "tool", "name": "StructuredOutput"}
        body["thinking"] = {"type": "disabled"}
        body.pop("output_config", None)
        rec["forced"] = True
    last["content"] = list(last["content"]) + [{"type": "text", "text": note}]


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # keep stdout quiet; never log headers
        pass

    def _reply(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_CONNECT(self):
        """HTTPS proxy for an allowlist of host:port (used for the Opus/subscription condition)."""
        t0, target = time.time(), self.path
        if target not in CONNECT_ALLOW:
            log({"call": "connect", "ts": t0, "target": target, "status": 403})
            return self._reply(403, {"error": "blocked by sandbox"})
        host, port = target.rsplit(":", 1)
        try:
            up = socket.create_connection((host, int(port)), timeout=30)
        except Exception as e:
            log({"call": "connect", "ts": t0, "target": target, "status": 502, "error": str(e)[:200]})
            return self._reply(502, {"error": "upstream unreachable"})
        self.send_response(200, "Connection established")
        self.end_headers()
        cl, nbytes = self.connection, [0, 0]
        try:
            while True:
                r, _, _ = select.select([cl, up], [], [], 900)
                if not r:
                    break
                for src, dst, k in ((cl, up, 0), (up, cl, 1)):
                    if src in r:
                        b = src.recv(65536)
                        if not b:
                            raise EOFError
                        dst.sendall(b)
                        nbytes[k] += len(b)
        except Exception:
            pass
        finally:
            up.close()
        self.close_connection = True
        log({"call": "connect", "ts": t0, "target": target, "status": 200, "up": nbytes[0], "down": nbytes[1], "dt": time.time() - t0})

    def _vdm(self, method):
        """Forward /vdm/<path> to the vdm servers (the only other endpoint the sandbox may reach).
        VDM_UPSTREAM may list several servers (comma-separated); a server that failed in the last 60 s is tried
        last; on connection errors / 5xx the next one is tried. If none answers: 503 + "retryable"."""
        if not VDM_UPSTREAM:
            return self._reply(503, {"error": "vdm not configured"})
        from urllib.parse import urlsplit
        n = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(n) if n else None
        t0 = time.time()
        ups = [u.strip() for u in VDM_UPSTREAM.split(",") if u.strip()]
        ups.sort(key=lambda u: _vdm_bad.get(u, 0) > t0 - 60)  # healthy first (stable order otherwise)
        tried = []
        for up in ups:
            u = urlsplit(up)
            try:
                conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=1800)
                conn.request(method, self.path[len("/vdm"):] or "/", body,
                             {"content-type": self.headers.get("content-type", "application/json")})
                r = conn.getresponse()
                data = r.read()
                if r.status >= 500:
                    raise RuntimeError(f"status {r.status}")
            except Exception as e:
                _vdm_bad[up] = time.time()
                tried.append(f"{up}: {str(e)[:80]}")
                continue
            log({"call": "vdm", "ts": t0, "path": self.path, "status": r.status, "upstream": up, "failed_over": tried,
                 "req_bytes": n, "resp_bytes": len(data), "dt": time.time() - t0})
            self.send_response(r.status)
            self.send_header("content-type", r.getheader("content-type", "application/json"))
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        log({"call": "vdm", "ts": t0, "path": self.path, "status": 599, "failed": tried, "dt": time.time() - t0})
        return self._reply(503, {"error": "vdm temporarily unavailable (all servers down)", "retryable": True})

    def _google(self, method):
        """Gemini API pass-through: only allowlisted models, key injected here, usage logged."""
        m = GOOGLE_RE.match(self.path)
        n = int(self.headers.get("content-length") or 0)
        raw = self.rfile.read(n) if n else None
        if not m:
            return self._reply(404, {"error": "sandbox gateway"})
        call, path = m.group(1), m.group(2)
        path = re.sub(r"([?&])key=[^&]*&?", r"\1", path).rstrip("?&")
        mm = re.search(r"/models/([^:/?]+)", path)
        t0 = time.time()
        rec = {"call": call, "ts": t0, "method": method, "path": path.split("?")[0][-80:],
               "model": mm.group(1) if mm else None}
        if mm and mm.group(1) not in ALLOW:
            rec.update(status=403, blocked=True)
            log(rec)
            return self._reply(403, {"error": {"code": 403, "message": f"model {mm.group(1)} not allowed in sandbox"}})
        hdr = {"content-type": self.headers.get("content-type", "application/json"), "x-goog-api-key": KEY}
        r, errs, attempts = None, [], 0
        with _upstream:
            while attempts < 14:
                attempts += 1
                try:
                    conn = http.client.HTTPSConnection("generativelanguage.googleapis.com", timeout=900)
                    conn.request(method, path, raw, hdr)
                    r = conn.getresponse()
                    if r.status not in RETRY_STATUS:
                        break
                    errs.append(r.status)
                    r.read()
                except Exception as e:
                    errs.append(type(e).__name__)
                r = None
                time.sleep(min(60, 3 * 2 ** min(attempts, 5)))
            rec["attempts"], rec["upstream_errors"] = attempts, errs[-5:]
            if r is None:
                rec.update(status=599, dt=time.time() - t0)
                log(rec)
                return self._reply(503, {"error": {"code": 503, "message": "upstream unreachable"}})
            self.send_response(r.status)
            self.send_header("content-type", r.getheader("content-type", "application/json"))
            self.send_header("transfer-encoding", "chunked")
            self.end_headers()
            buf, first = b"", None
            try:
                while True:
                    chunk = r.read1(65536)
                    if not chunk:
                        break
                    first = first or time.time() - t0
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                    self.wfile.flush()
                    buf += chunk
                self.wfile.write(b"0\r\n\r\n")
            except (BrokenPipeError, ConnectionResetError):
                rec["client_disconnected"] = True
        usage = {}
        for mt in re.finditer(rb'"usageMetadata"\s*:\s*(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})', buf):
            try:
                usage = json.loads(mt.group(1))  # the last one is the final usage of the response
            except Exception:
                pass
        if r.status >= 400:
            rec["error"] = buf[:400].decode("utf-8", "ignore")
        rec.update(status=r.status, usage={k: v for k, v in usage.items() if isinstance(v, (int, float))},
                   ttfb=first, dt=time.time() - t0)
        log(rec)

    def do_GET(self):
        if self.path.rstrip("/") == "/gw/status":
            return self._reply(200, {"fatal": FATAL, "stall": STALL, "max_steps": MAX_STEPS, "upstream": UPSTREAM})
        if self.path.startswith("/vdm"):
            return self._vdm("GET")
        if MODE == "google":
            return self._google("GET")
        self._reply(404, {"type": "error", "error": {"type": "not_found_error", "message": "sandbox gateway"}})

    def do_POST(self):
        if self.path.startswith("/vdm"):
            return self._vdm("POST")
        if MODE == "google":
            return self._google("POST")
        n = int(self.headers.get("content-length") or 0)
        raw = self.rfile.read(n)
        m = PATH_RE.match(self.path)
        if not m:
            return self.do_GET()
        call, path = m.group(1), m.group(2)
        if path.endswith("count_tokens"):
            return self._reply(404, {"type": "error", "error": {"type": "not_found_error", "message": "count_tokens unsupported"}})
        try:
            body = json.loads(raw)
        except Exception:
            return self._reply(400, {"type": "error", "error": {"type": "invalid_request_error", "message": "bad json"}})
        if DUMP and call.startswith(DUMP):
            with open(os.path.join(os.path.dirname(LOG), f"dump_{call}_{time.time():.0f}.json"), "w") as f:
                json.dump(body, f)
        req_model = body.get("model")
        if req_model not in ALLOW:
            body["model"] = MODEL
        t0 = time.time()
        rec = {"call": call, "ts": t0, "requested_model": req_model, "model": body["model"],
               "stream": bool(body.get("stream")), "n_messages": len(body.get("messages", []))}
        if UP_MODEL:
            body["model"] = UP_MODEL
        budget(body, rec, call)
        hdr = {"content-type": "application/json", "authorization": "Bearer " + KEY,
               "anthropic-version": self.headers.get("anthropic-version", "2023-06-01")}
        if self.headers.get("anthropic-beta"):
            hdr["anthropic-beta"] = self.headers["anthropic-beta"]
        if FATAL:
            rec.update(status=598, fatal=True, dt=0)
            log(rec)
            return self._reply(529, {"type": "error", "error": {"type": "overloaded_error", "message": "upstream quota exhausted (harness)"}})
        return self._forward(UP_PREFIX + path, body, hdr, rec, t0, call)

    def _forward(self, path, body, hdr, rec, t0, call=""):
        """retry 429/5xx/connection errors with capped exponential backoff + jitter, WITHOUT holding an in-flight
        slot while sleeping; quota/balance exhaustion is fatal (fail fast, flag it); stall time is booked per call."""
        import random
        r, attempts, errs, waited = None, 0, [], 0.0
        payload = json.dumps(body).encode()
        _upstream.acquire()
        try:
            while True:
                attempts += 1
                try:
                    conn = http.client.HTTPSConnection(UPSTREAM, timeout=900)
                    conn.request("POST", path, payload, hdr)
                    r = conn.getresponse()
                    if r.status not in RETRY_STATUS:
                        break
                    errs.append(r.status)
                    txt = r.read()[:600].decode("utf-8", "ignore")
                    if any(m in txt.lower() for m in FATAL_MARKERS) or r.status == 402:
                        FATAL.update(ts=time.time(), status=r.status, message=txt[:300])
                        r = None
                        break
                except Exception as e:
                    errs.append(type(e).__name__)
                r = None
                if waited >= RETRY_S:
                    break
                d = min(60.0, 2.0 * 2 ** min(attempts, 5)) * (0.5 + random.random())
                _upstream.release()
                time.sleep(d)
                waited += d
                with _lock:
                    STALL[call] = STALL.get(call, 0.0) + d
                _upstream.acquire()
            rec["attempts"], rec["upstream_errors"], rec["stall_s"] = attempts, errs[-5:], round(waited, 1)
            if r is None:
                rec.update(status=599, dt=time.time() - t0, fatal=bool(FATAL))
                log(rec)
                return self._reply(529, {"type": "error", "error": {"type": "overloaded_error", "message": "upstream unreachable"}})
            return self._stream(r, rec, t0)
        finally:
            _upstream.release()

    def _stream(self, r, rec, t0):
        self.send_response(r.status)
        ctype = r.getheader("content-type", "application/json")
        self.send_header("content-type", ctype)
        self.send_header("transfer-encoding", "chunked")
        self.end_headers()
        usage, buf, first = {}, b"", None
        try:
            while True:
                chunk = r.read1(65536) if hasattr(r, "read1") else r.read(65536)
                if not chunk:
                    break
                if first is None:
                    first = time.time() - t0
                self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                self.wfile.flush()
                buf += chunk
            self.wfile.write(b"0\r\n\r\n")
        except (BrokenPipeError, ConnectionResetError):
            rec["client_disconnected"] = True
        # usage accounting
        try:
            if "event-stream" in ctype:
                for line in buf.decode("utf-8", "ignore").splitlines():
                    if line.startswith("data:"):
                        try:
                            ev = json.loads(line[5:])
                        except Exception:
                            continue
                        # Ollama: message_start carries a rough pre-count (base64 images counted as text);
                        # the final message_delta carries the real usage and overrides it.
                        if ev.get("type") == "message_start":
                            u = (ev.get("message") or {}).get("usage") or {}
                            rec["start_estimate_input_tokens"] = u.get("input_tokens")
                            usage.update({k: v for k, v in u.items() if isinstance(v, (int, float))})
                        elif ev.get("usage"):
                            usage.update({k: v for k, v in ev["usage"].items() if isinstance(v, (int, float))})
            else:
                d = json.loads(buf)
                usage = {k: v for k, v in (d.get("usage") or {}).items() if isinstance(v, (int, float))}
                if r.status >= 400:
                    rec["error"] = json.dumps(d)[:500]
        except Exception as e:
            rec["parse_error"] = str(e)[:200]
        rec.update(status=r.status, usage=usage, ttfb=first, dt=time.time() - t0)
        log(rec)


if __name__ == "__main__":
    KEY = sys.stdin.readline().strip()  # may be empty for a gateway that only proxies CONNECT/vdm
    if not KEY and not (CONNECT_ALLOW or VDM_UPSTREAM):
        sys.exit("key missing on stdin")
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    ThreadingHTTPServer(("0.0.0.0", 8080), H).serve_forever()
