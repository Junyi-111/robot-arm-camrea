"""Frozen ArtiMuse backend returning score and the final multimodal state."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("USE_TF", "0")

import torch


class ArtiMuseAnalyzer:
    def __init__(
        self,
        project_root: str,
        model_path: str,
        *,
        device: str = "cuda:0",
        max_gpu_memory: str = "20GiB",
        max_cpu_memory: str = "32GiB",
    ) -> None:
        project = Path(project_root).expanduser().resolve()
        model = Path(model_path).expanduser().resolve()
        if not (project / "ArtiMuse/src").is_dir():
            raise FileNotFoundError(f"ArtiMuse source missing under {project}")
        if not model.is_dir():
            raise FileNotFoundError(f"ArtiMuse checkpoint missing: {model}")
        for path in (project, project / "ArtiMuse/src"):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        from aesthetic_rl.reward.artimuse_reward import ArtiMuseReward
        from artimuse.internvl.model.internvl_chat.aes_tokens import (
            AESTHETICS_TOKEN_LIST,
        )
        from eval.eval_image import load_image

        self.device = device
        self.load_image = load_image
        self.aesthetic_tokens = AESTHETICS_TOKEN_LIST
        self.scorer = ArtiMuseReward(
            model_path=str(model),
            device=device,
            max_gpu_memory=max_gpu_memory,
            max_cpu_memory=max_cpu_memory,
        )
        self.model = self.scorer.model
        self.tokenizer = self.scorer.tokenizer
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def _multimodal_forward(self, pixel_values, input_ids, attention_mask):
        """Run ArtiMuse's logits path while retaining the final LLM state.

        ArtiMuse's ``generate_logits`` currently accepts an
        ``output_hidden_states`` argument but hard-codes it to ``False`` when
        invoking the language model. Reproduce that short path here so score
        logits and the train-time aesthetic latent come from the same frozen
        multimodal forward pass without patching the upstream model source.
        """
        visual_features = self.model.extract_feature(pixel_values)
        input_embeds = self.model.language_model.get_input_embeddings()(input_ids)
        batch, sequence_length, channels = input_embeds.shape
        flat_embeds = input_embeds.reshape(batch * sequence_length, channels)
        selected = (
            input_ids.reshape(batch * sequence_length)
            == self.model.img_context_token_id
        )
        flat_visual = visual_features.reshape(-1, channels).to(flat_embeds.device)
        selected_count = int(selected.sum().item())
        if selected_count != int(flat_visual.shape[0]):
            raise RuntimeError(
                "ArtiMuse image-token/visual-token mismatch: "
                f"image_tokens={selected_count}, visual_tokens={flat_visual.shape[0]}"
            )
        flat_embeds[selected] = flat_visual
        input_embeds = flat_embeds.reshape(batch, sequence_length, channels)
        outputs = self.model.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        if not outputs.hidden_states:
            raise RuntimeError("ArtiMuse language model returned no hidden states")
        return outputs

    @torch.inference_mode()
    def analyze(self, image_path: str | Path) -> dict:
        started = time.monotonic()
        pixel_values = self.load_image(str(image_path)).to(
            self.device, dtype=torch.float16
        )
        question = """Rate the aesthetics score of the image in 0-100.
In the output format, numbers are replaced by 2 corresponding letters,
and the mapping relationship is: score 0 to 25: 0-aa, 1-ab, 2-ac, ...,
25-az, score 26 to 50: 26-ca, 27-cb, ..., 50-cy, score 51 to 75:
51-da, 52-db, ..., 75-dy, score 76 to 100: 76-ea, ..., 100-ey.
The answer only outputs 2 corresponding letters."""
        try:
            from artimuse.internvl.conversation import get_conv_template

            context_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
            self.model.img_context_token_id = context_id
            template = get_conv_template(self.model.template)
            template.system_message = self.model.system_message
            template.append_message(template.roles[0], "<image>\n" + question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()
            image_tokens = (
                "<img>"
                + "<IMG_CONTEXT>" * self.model.num_image_token * int(pixel_values.shape[0])
                + "</img>"
            )
            query = query.replace("<image>", image_tokens, 1)
            inputs = self.tokenizer(query, return_tensors="pt")
            outputs = self._multimodal_forward(
                pixel_values,
                inputs["input_ids"].to(self.device),
                inputs["attention_mask"].to(self.device),
            )
            hidden = outputs.hidden_states[-1][:, -1, :]
            token_ids = [
                self.tokenizer.convert_tokens_to_ids(token)
                for token in self.aesthetic_tokens
            ]
            logits = outputs.logits[:, -1, token_ids]
            weights = torch.arange(101, device=logits.device, dtype=logits.dtype)
            score = (torch.softmax(logits, dim=-1) @ weights).item()
            return {
                "score": float(score),
                "multimodal_hidden": hidden[0].float().cpu().tolist(),
                "hidden_dim": int(hidden.shape[-1]),
                "inference_seconds": time.monotonic() - started,
            }
        finally:
            del pixel_values
