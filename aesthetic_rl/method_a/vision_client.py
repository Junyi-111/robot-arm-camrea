"""HTTP clients for the tunneled ArtiMuse and Grounding DINO services."""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np


class VisionServiceError(RuntimeError):
    """Raised when a remote vision service is unavailable or malformed."""


@dataclass(frozen=True)
class VisionAnalysis:
    score: float
    feature: np.ndarray
    detection: dict
    analyze_seconds: float
    detection_seconds: float


class RemoteVisionClient:
    def __init__(
        self,
        analyze_url: str,
        grounding_url: str,
        *,
        feature_dim: int = 3584,
        timeout_seconds: float = 180.0,
    ) -> None:
        self.analyze_url = self._validate_url(analyze_url)
        self.grounding_url = self._validate_url(grounding_url)
        self.feature_dim = int(feature_dim)
        self.timeout_seconds = float(timeout_seconds)
        if self.feature_dim <= 0 or self.timeout_seconds <= 0.0:
            raise ValueError("feature_dim and timeout_seconds must be positive")

    @staticmethod
    def _validate_url(url: str) -> str:
        url = url.strip()
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"service URL must start with http:// or https://: {url!r}")
        return url

    @staticmethod
    def _jpeg(rgb: np.ndarray, quality: int = 95) -> bytes:
        rgb = np.asarray(rgb, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError(f"RGB image must have shape HxWx3, got {rgb.shape}")
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise VisionServiceError("failed to encode JPEG")
        return encoded.tobytes()

    def _request(self, url: str, body: bytes | None, method: str) -> tuple[dict, float]:
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Accept": "application/json", "Content-Type": "image/jpeg"},
            method=method,
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            raise VisionServiceError(f"{url} returned HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise VisionServiceError(f"cannot use {url}: {exc}") from exc
        if not isinstance(payload, dict) or "error" in payload:
            raise VisionServiceError(f"invalid response from {url}: {payload}")
        return payload, time.monotonic() - started

    def health(self) -> dict[str, dict]:
        analyze_health = self.analyze_url.rsplit("/", 1)[0] + "/health"
        grounding_health = self.grounding_url.rsplit("/", 1)[0] + "/health"
        a, _ = self._request(analyze_health, None, "GET")
        g, _ = self._request(grounding_health, None, "GET")
        if not a.get("ready") or not g.get("ready"):
            raise VisionServiceError(f"vision services are not ready: analyze={a}, grounding={g}")
        return {"artimuse": a, "grounding": g}

    def analyze(self, rgb: np.ndarray) -> VisionAnalysis:
        body = self._jpeg(rgb)
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="vision-http") as pool:
            a_future = pool.submit(self._request, self.analyze_url, body, "POST")
            d_future = pool.submit(self._request, self.grounding_url, body, "POST")
            aesthetic, aesthetic_seconds = a_future.result()
            detection, detection_seconds = d_future.result()
        try:
            score = float(aesthetic["score"])
            feature = np.asarray(aesthetic["multimodal_hidden"], dtype=np.float32)
        except (KeyError, TypeError, ValueError) as exc:
            raise VisionServiceError("ArtiMuse response lacks score/feature") from exc
        if not math.isfinite(score) or feature.shape != (self.feature_dim,):
            raise VisionServiceError(
                f"invalid ArtiMuse output: score={score}, feature_shape={feature.shape}"
            )
        if not np.isfinite(feature).all() or "target_valid" not in detection:
            raise VisionServiceError("vision service returned non-finite/malformed output")
        return VisionAnalysis(score, feature, detection, aesthetic_seconds, detection_seconds)
