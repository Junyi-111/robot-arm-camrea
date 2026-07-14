import sys
import os
import torch
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
    def __init__(self, model_path, device="cuda:0", max_gpu_memory="7GiB", max_cpu_memory="32GiB"):
        self.device = device

        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_grad_enabled(False)

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

        self.model = InternVLChatModel.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map="auto",
            max_memory={
                0: max_gpu_memory,
                "cpu": max_cpu_memory,
            },
            trust_remote_code=True,
            low_cpu_mem_usage=True
        ).eval()

    def score_image(self, image_path):
        pixel_values = load_image(image_path)
        pixel_values = pixel_values.to(self.device, dtype=torch.float16)

        generation_config = dict(
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id
        )

        score = self.model.score(
            self.device,
            self.tokenizer,
            pixel_values,
            generation_config
        )

        return float(score)
