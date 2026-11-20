#!/usr/bin/env python3
"""GPU whisper-small transcription service (evaluation helper; nothing is stored). POST /transcribe with raw audio bytes
(any format ffmpeg reads) -> text/tab-separated-values: start_s<TAB>end_s<TAB>text per segment."""
import io, json, subprocess, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np, torch
from transformers import pipeline

PIPE = pipeline("automatic-speech-recognition", model="openai/whisper-small", torch_dtype=torch.float16, device="cuda:0")
LOCK = threading.Lock()


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def _send(self, code, b, ct="text/plain"):
        self.send_response(code); self.send_header("content-type", ct); self.send_header("content-length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        self._send(200, b'{"ok": true, "model": "openai/whisper-small"}', "application/json")
    def do_POST(self):
        try:
            raw = self.rfile.read(int(self.headers.get("content-length") or 0))
            pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", "pipe:0", "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1"],
                                 input=raw, capture_output=True).stdout
            a = np.frombuffer(pcm, np.float32)
            t = time.time()
            with LOCK:
                out = PIPE({"raw": a, "sampling_rate": 16000}, chunk_length_s=30, batch_size=32, return_timestamps=True,
                           generate_kwargs={"task": "transcribe"})
            lines = []
            for c in out.get("chunks", []):
                s, e = c["timestamp"]
                txt = c["text"].strip()
                if txt:
                    lines.append(f"{(s or 0):.1f}\t{(e if e is not None else len(a) / 16000):.1f}\t{txt}")
            self._send(200, ("\n".join(lines) + ("\n" if lines else "")).encode())
            print(f"{len(a)/16000:.0f}s audio in {time.time()-t:.0f}s", flush=True)
        except Exception as e:
            self._send(500, f"{type(e).__name__}: {e}".encode()[:300])


if __name__ == "__main__":
    print("whisper ready", flush=True)
    ThreadingHTTPServer(("127.0.0.1", 8090), H).serve_forever()
