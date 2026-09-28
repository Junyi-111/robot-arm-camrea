"""RealSense capture with no import-time hardware side effects."""

from __future__ import annotations

import threading
import time

import numpy as np


class CameraError(RuntimeError):
    """Raised when a fresh RGB frame cannot be acquired."""


class D435Camera:
    def __init__(
        self,
        *,
        serial: str,
        width: int,
        height: int,
        fps: int,
    ) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise CameraError("pyrealsense2 is not installed in this environment") from exc
        self._rs = rs
        self._pipeline = rs.pipeline()
        stream = rs.config()
        if serial:
            stream.enable_device(serial)
        stream.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        try:
            self._pipeline.start(stream)
        except Exception as exc:
            raise CameraError(f"cannot start D435 color stream: {exc}") from exc
        self._lock = threading.Lock()
        self._closed = False
        # Discard auto-exposure transients.
        for _ in range(10):
            self._pipeline.wait_for_frames(timeout_ms=3000)

    def capture_rgb(self, timeout_seconds: float = 3.0) -> np.ndarray:
        deadline = time.monotonic() + float(timeout_seconds)
        with self._lock:
            if self._closed:
                raise CameraError("camera is closed")
            while time.monotonic() < deadline:
                timeout_ms = max(1, int((deadline - time.monotonic()) * 1000))
                try:
                    frames = self._pipeline.wait_for_frames(timeout_ms=timeout_ms)
                except RuntimeError as exc:
                    raise CameraError(f"D435 frame timeout: {exc}") from exc
                frame = frames.get_color_frame()
                if frame:
                    bgr = np.asanyarray(frame.get_data()).copy()
                    return bgr[..., ::-1].copy()
        raise CameraError("D435 returned no color frame")

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._pipeline.stop()
                self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        self.close()
