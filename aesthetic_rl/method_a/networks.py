"""Flax networks for RGB + frozen ArtiMuse feature fusion."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core import freeze, unfreeze

from .jax_setup import ensure_hil_serl_importable


class TinyRGBEncoder(nn.Module):
    """Small test encoder; production uses the pretrained HIL-SERL ResNet."""

    output_dim: int = 64

    @nn.compact
    def __call__(self, image: jnp.ndarray, *, train: bool = False) -> jnp.ndarray:
        x = image.astype(jnp.float32) / 255.0
        for channels in (16, 32, 64):
            x = nn.Conv(channels, (3, 3), strides=(2, 2), padding="SAME")(x)
            x = nn.relu(x)
        x = jnp.mean(x, axis=(-3, -2))
        return nn.Dense(self.output_dim)(x)


class FusionEncoder(nn.Module):
    aesthetic_feature_dim: int = 3584
    aesthetic_latent_dim: int = 256
    image_encoder_type: str = "resnet-pretrained"

    @nn.compact
    def __call__(self, observations: dict[str, jnp.ndarray], *, train: bool = False):
        image = observations["wrist_view"]
        robot_state = observations["robot_state"].astype(jnp.float32)
        target_state = observations["target_state"].astype(jnp.float32)
        aesthetic = observations["aesthetic_feature"].astype(jnp.float32)
        if aesthetic.shape[-1] != self.aesthetic_feature_dim:
            raise ValueError(
                f"expected aesthetic feature dim {self.aesthetic_feature_dim}, "
                f"got {aesthetic.shape[-1]}"
            )

        if self.image_encoder_type == "tiny":
            image_encoding = TinyRGBEncoder(name="rgb_encoder")(image, train=train)
        elif self.image_encoder_type == "resnet-pretrained":
            ensure_hil_serl_importable()
            from serl_launcher.vision.resnet_v1 import (
                PreTrainedResNetEncoder,
                resnetv1_configs,
            )

            frozen_resnet = resnetv1_configs["resnetv1-10-frozen"](
                pre_pooling=True,
                name="pretrained_encoder",
            )
            image_encoding = PreTrainedResNetEncoder(
                pooling_method="spatial_learned_embeddings",
                num_spatial_blocks=8,
                bottleneck_dim=256,
                pretrained_encoder=frozen_resnet,
                name="rgb_encoder",
            )(image, encode=True, train=train)
        else:
            raise ValueError(f"unknown image encoder: {self.image_encoder_type}")

        robot_encoding = nn.Dense(64, name="robot_dense")(robot_state)
        robot_encoding = nn.tanh(nn.LayerNorm(name="robot_norm")(robot_encoding))
        target_encoding = nn.Dense(32, name="target_dense")(target_state)
        target_encoding = nn.tanh(nn.LayerNorm(name="target_norm")(target_encoding))

        # This is the sole shared trainable Feature Projector. ArtiMuse itself
        # runs remotely under inference_mode and never enters this parameter tree.
        aesthetic = nn.LayerNorm(name="aesthetic_input_norm")(aesthetic)
        aesthetic = nn.Dense(512, name="aesthetic_projector_1")(aesthetic)
        aesthetic = nn.gelu(aesthetic)
        aesthetic = nn.Dense(
            self.aesthetic_latent_dim, name="aesthetic_projector_2"
        )(aesthetic)
        aesthetic = nn.LayerNorm(name="aesthetic_latent_norm")(aesthetic)
        return jnp.concatenate(
            [image_encoding, robot_encoding, target_encoding, aesthetic], axis=-1
        )


class ActorNetwork(nn.Module):
    action_dim: int = 4

    @nn.compact
    def __call__(self, encoding: jnp.ndarray, *, train: bool = False):
        del train
        x = encoding
        for index in range(2):
            x = nn.Dense(256, name=f"dense_{index}")(x)
            x = nn.LayerNorm(name=f"norm_{index}")(x)
            x = nn.tanh(x)
        mean = nn.Dense(self.action_dim, name="mean")(x)
        log_std = nn.Dense(self.action_dim, name="log_std")(x)
        log_std = jnp.clip(log_std, -5.0, 1.6)
        return mean, log_std


class CriticNetwork(nn.Module):
    ensemble_size: int = 2

    @nn.compact
    def __call__(self, encoding: jnp.ndarray, action: jnp.ndarray, *, train=False):
        del train
        values = []
        inputs = jnp.concatenate([encoding, action], axis=-1)
        for critic_index in range(self.ensemble_size):
            x = inputs
            for layer_index in range(2):
                x = nn.Dense(
                    256, name=f"critic_{critic_index}_dense_{layer_index}"
                )(x)
                x = nn.LayerNorm(
                    name=f"critic_{critic_index}_norm_{layer_index}"
                )(x)
                x = nn.tanh(x)
            q = nn.Dense(1, name=f"critic_{critic_index}_q")(x)
            values.append(jnp.squeeze(q, axis=-1))
        return jnp.stack(values, axis=0)


def replace_pretrained_resnet_params(
    encoder_params: Any,
    weights_path: str | Path,
):
    """Load the local ImageNet ResNet-10 convolution weights by exact key."""
    path = Path(weights_path).expanduser().resolve()
    with path.open("rb") as handle:
        source = pickle.load(handle)
    params = unfreeze(encoder_params)
    try:
        # Flax lifts the explicitly named submodule to the FusionEncoder root.
        destination = params["pretrained_encoder"]
    except KeyError as exc:
        raise ValueError("encoder parameter tree has no pretrained ResNet") from exc
    replaced = 0
    for key in list(destination):
        if key in source:
            destination[key] = source[key]
            replaced += 1
    if replaced == 0:
        raise ValueError(f"no matching ResNet parameters found in {path}")
    return freeze(params), replaced


def count_parameters(tree: Any) -> int:
    return int(sum(value.size for value in jax.tree_util.tree_leaves(tree)))
