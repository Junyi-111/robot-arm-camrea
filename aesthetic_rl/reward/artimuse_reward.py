import gc
import sys
import os
import time

# HIL-SERL installs JAX in the same environment. Transformers otherwise probes
# the Flax backend while importing this PyTorch-only reward, which can fail on
# Isaac servers where JAX cannot discover a CUDA toolkit/NVCC path.
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("USE_TF", "0")

import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoTokenizer, BitsAndBytesConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIMUSE_SRC = PROJECT_ROOT / "ArtiMuse" / "src"
ARTIMUSE_PKG = ARTIMUSE_SRC / "artimuse"

for p in [PROJECT_ROOT, ARTIMUSE_SRC, ARTIMUSE_PKG]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eval.eval_image import load_image
from artimuse.internvl.model.internvl_chat.modeling_artimuse import InternVLChatModel

class ArtiMuseReward:
    def __init__(self, model_path, device="cuda:0", max_gpu_memory="12GiB", max_cpu_memory="16GiB"):
        self.device = device

        requested_device = torch.device(device)
        if requested_device.type != "cuda":
            raise ValueError(f"ArtiMuseReward requires a CUDA device, got {device!r}.")
        device_index = requested_device.index
        if device_index is None:
            device_index = torch.cuda.current_device()

        free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
        gib = float(1024**3)
        print(
            f"[ARTIMUSE] loading on {device} with budget={max_gpu_memory}; "
            f"cuda_free={free_bytes / gib:.2f}GiB cuda_total={total_bytes / gib:.2f}GiB",
            flush=True,
        )

        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        torch.backends.cuda.matmul.allow_tf32 = True

        print("[ARTIMUSE] loading tokenizer", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False
        )

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )

        print("[ARTIMUSE] loading quantized model", flush=True)
        model_started_at = time.monotonic()
        self.model = InternVLChatModel.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map="auto",
            max_memory={
                device_index: max_gpu_memory,
                "cpu": max_cpu_memory,
            },
            trust_remote_code=True,
            low_cpu_mem_usage=True
        ).eval()
        print(
            f"[ARTIMUSE] model ready in "
            f"{time.monotonic() - model_started_at:.1f}s",
            flush=True,
        )

        # Scoring performs one forward pass, so generation caches only increase memory pressure.
        self.model.config.use_cache = False
        if hasattr(self.model, "language_model"):
            self.model.language_model.config.use_cache = False

    @torch.inference_mode()
    def score_image(self, image_path):
        score_started_at = time.monotonic()
        print(f"[ARTIMUSE] scoring {image_path}", flush=True)
        pixel_values = load_image(image_path)
        pixel_values = pixel_values.to(self.device, dtype=torch.float16)

        generation_config = dict(
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id
        )

        try:
            score = self.model.score(
                self.device,
                self.tokenizer,
                pixel_values,
                generation_config
            )
            result = float(score)
            print(
                f"[ARTIMUSE] score={result:.4f} "
                f"elapsed={time.monotonic() - score_started_at:.1f}s",
                flush=True,
            )
            return result
        finally:
            del pixel_values
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @staticmethod
    def _pool_visual_features(visual_features, spatial_grid=4):
        """Keep global appearance and coarse spatial composition in a fixed vector."""
        if visual_features.ndim != 3 or visual_features.shape[0] != 1:
            raise ValueError(
                "Expected visual features with shape [1, tokens, channels], "
                f"got {tuple(visual_features.shape)}."
            )
        token_count = int(visual_features.shape[1])
        side = int(round(token_count**0.5))
        if side * side != token_count:
            raise ValueError(
                f"Visual token count {token_count} is not a square spatial grid."
            )
        if not 1 <= spatial_grid <= side:
            raise ValueError(
                f"spatial_grid must be between 1 and {side}, got {spatial_grid}."
            )

        spatial = visual_features.reshape(
            1, side, side, visual_features.shape[-1]
        ).permute(0, 3, 1, 2)
        pooled = F.adaptive_avg_pool2d(
            spatial.float(), (spatial_grid, spatial_grid)
        ).flatten()
        global_std = visual_features.float().std(dim=1, unbiased=False).flatten()
        feature = torch.cat((pooled, global_std), dim=0)
        return F.normalize(feature, dim=0)

    @torch.inference_mode()
    def score_and_extract(self, image_path, spatial_grid=4):
        """Return the ArtiMuse score and a reusable composition feature."""
        started_at = time.monotonic()
        print(f"[ARTIMUSE] scoring and extracting {image_path}", flush=True)
        pixel_values = load_image(image_path)
        pixel_values = pixel_values.to(self.device, dtype=torch.float16)
        generation_config = dict(
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        visual_features = None
        try:
            visual_features = self.model.extract_feature(pixel_values)
            score = self.model.score(
                self.device,
                self.tokenizer,
                pixel_values,
                generation_config,
                visual_features=visual_features,
            )
            pooled = self._pool_visual_features(
                visual_features, spatial_grid=spatial_grid
            )
            result = float(score), pooled.cpu()
            print(
                f"[ARTIMUSE] score={result[0]:.4f} feature_dim={pooled.numel()} "
                f"elapsed={time.monotonic() - started_at:.1f}s",
                flush=True,
            )
            return result
        finally:
            del pixel_values
            if visual_features is not None:
                del visual_features
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def close(self):
        model = getattr(self, "model", None)
        self.model = None
        self.tokenizer = None
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
