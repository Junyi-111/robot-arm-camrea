#!/usr/bin/env python3
"""One-model HTTP server for score + hidden feature in one forward pass."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .artimuse_backend import ArtiMuseAnalyzer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="/home/hjy/robot_aesthetic_rl")
    parser.add_argument(
        "--model", default="/datasets/hjy/robot_aesthetic_models/ArtiMuse"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-gpu-memory", default="20GiB")
    parser.add_argument("--max-cpu-memory", default="32GiB")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    analyzer = ArtiMuseAnalyzer(
        args.project,
        args.model,
        device=args.device,
        max_gpu_memory=args.max_gpu_memory,
        max_cpu_memory=args.max_cpu_memory,
    )
    model_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def reply(self, code: int, payload: dict):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self.reply(
                    200,
                    {
                        "ready": True,
                        "service": "artimuse-analyze",
                        "device": args.device,
                        "pid": os.getpid(),
                    },
                )
            else:
                self.reply(404, {"error": "use /health or /analyze"})

        def do_POST(self):
            if self.path != "/analyze":
                self.reply(404, {"error": "use /analyze"})
                return
            temporary: Path | None = None
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 1 <= size <= 12 * 1024 * 1024:
                    raise ValueError("image size must be between 1 byte and 12 MiB")
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as handle:
                    handle.write(self.rfile.read(size))
                    temporary = Path(handle.name)
                with model_lock:
                    payload = analyzer.analyze(temporary)
                self.reply(200, payload)
            except Exception as exc:
                traceback.print_exc()
                self.reply(500, {"error": f"{type(exc).__name__}: {exc}"})
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)

        def log_message(self, fmt, *values):
            print(f"[ARTIMUSE-HTTP] {fmt % values}", flush=True)

    print(
        f"[ARTIMUSE] ready at http://{args.host}:{args.port}/analyze "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}",
        flush=True,
    )
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
