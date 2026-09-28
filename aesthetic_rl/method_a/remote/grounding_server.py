#!/usr/bin/env python3
"""Grounding DINO flower-box service with a serialized GPU model call."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--prompt", default="flower")
    parser.add_argument("--box-threshold", type=float, default=0.35)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--min-area-ratio", type=float, default=0.005)
    parser.add_argument("--edge-margin-ratio", type=float, default=0.05)
    args = parser.parse_args()
    from groundingdino.util.inference import load_image, load_model, predict

    model = load_model(args.config, args.checkpoint, device=args.device)
    model_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def reply(self, code, payload):
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
                        "service": "grounding-dino",
                        "prompt": args.prompt,
                        "device": args.device,
                    },
                )
            else:
                self.reply(404, {"error": "use /health or /detect"})

        def do_POST(self):
            if self.path != "/detect":
                self.reply(404, {"error": "use /detect"})
                return
            path = None
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 1 <= size <= 12 * 1024 * 1024:
                    raise ValueError("image size must be between 1 byte and 12 MiB")
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as handle:
                    handle.write(self.rfile.read(size))
                    path = handle.name
                source, image = load_image(path)
                height, width = source.shape[:2]
                started = time.monotonic()
                with model_lock:
                    boxes, logits, phrases = predict(
                        model,
                        image,
                        args.prompt,
                        args.box_threshold,
                        args.text_threshold,
                        device=args.device,
                    )
                detections = []
                for box, score, phrase in zip(boxes, logits, phrases):
                    cx, cy, box_width, box_height = [float(value) for value in box]
                    xyxy = [
                        (cx - box_width / 2.0) * width,
                        (cy - box_height / 2.0) * height,
                        (cx + box_width / 2.0) * width,
                        (cy + box_height / 2.0) * height,
                    ]
                    area = max(0.0, xyxy[2] - xyxy[0]) * max(
                        0.0, xyxy[3] - xyxy[1]
                    )
                    margin = max(
                        0.0,
                        min(xyxy[0], xyxy[1], width - xyxy[2], height - xyxy[3]),
                    )
                    margin_ratio = margin / max(1.0, min(width, height))
                    detections.append(
                        {
                            "score": float(score),
                            "phrase": phrase,
                            "xyxy": xyxy,
                            "area_ratio": area / float(width * height),
                            "margin_ratio": margin_ratio,
                            "cropped": margin_ratio < args.edge_margin_ratio,
                        }
                    )
                visible = [
                    detection
                    for detection in detections
                    if detection["score"] >= args.box_threshold
                    and detection["area_ratio"] >= args.min_area_ratio
                ]
                valid = [detection for detection in visible if not detection["cropped"]]
                self.reply(
                    200,
                    {
                        "target_valid": bool(valid),
                        "width": width,
                        "height": height,
                        "detections": detections,
                        # Return a visible edge-touching box as `best` too, while
                        # target_valid stays false. The actor can display it and
                        # the state preserves its margin instead of collapsing
                        # every invalid case to all zeros.
                        "best": max(visible, key=lambda item: item["score"]) if visible else None,
                        "inference_seconds": time.monotonic() - started,
                    },
                )
            except Exception as exc:
                self.reply(500, {"error": f"{type(exc).__name__}: {exc}"})
            finally:
                if path:
                    try:
                        os.unlink(path)
                    except OSError:
                        pass

        def log_message(self, fmt, *values):
            print(f"[GROUNDING-HTTP] {fmt % values}", flush=True)

    print(
        f"[GROUNDING] ready at http://{args.host}:{args.port}/detect "
        f"prompt={args.prompt!r} CUDA_VISIBLE_DEVICES="
        f"{os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}",
        flush=True,
    )
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
