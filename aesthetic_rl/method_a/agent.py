"""A compact SAC agent with one shared trainable aesthetic projector."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import flax
from flax import struct
from flax.core import freeze, unfreeze
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .networks import (
    ActorNetwork,
    CriticNetwork,
    FusionEncoder,
    replace_pretrained_resnet_params,
)


def _sample_tanh_normal(mean, log_std, rng):
    noise = jax.random.normal(rng, mean.shape)
    pre_tanh = mean + jnp.exp(log_std) * noise
    action = jnp.tanh(pre_tanh)
    gaussian_log_prob = -0.5 * (
        jnp.square(noise) + 2.0 * log_std + jnp.log(2.0 * jnp.pi)
    )
    log_prob = jnp.sum(
        gaussian_log_prob - jnp.log(1.0 - jnp.square(action) + 1e-6), axis=-1
    )
    return action, log_prob


def _action_log_prob(mean, log_std, action):
    clipped = jnp.clip(action, -0.999999, 0.999999)
    pre_tanh = jnp.arctanh(clipped)
    z = (pre_tanh - mean) / jnp.exp(log_std)
    gaussian = -0.5 * (jnp.square(z) + 2.0 * log_std + jnp.log(2.0 * jnp.pi))
    return jnp.sum(
        gaussian - jnp.log(1.0 - jnp.square(clipped) + 1e-6), axis=-1
    )


@struct.dataclass
class AestheticSACAgent:
    encoder_params: Any
    actor_params: Any
    critic_params: Any
    target_encoder_params: Any
    target_critic_params: Any
    log_temperature: jax.Array
    encoder_opt_state: Any
    actor_opt_state: Any
    critic_opt_state: Any
    temperature_opt_state: Any
    rng: jax.Array
    update_step: jax.Array
    encoder_def: Any = struct.field(pytree_node=False)
    actor_def: Any = struct.field(pytree_node=False)
    critic_def: Any = struct.field(pytree_node=False)
    encoder_tx: Any = struct.field(pytree_node=False)
    actor_tx: Any = struct.field(pytree_node=False)
    critic_tx: Any = struct.field(pytree_node=False)
    temperature_tx: Any = struct.field(pytree_node=False)
    discount: float = struct.field(pytree_node=False, default=0.97)
    tau: float = struct.field(pytree_node=False, default=0.005)
    target_entropy: float = struct.field(pytree_node=False, default=-2.0)

    @classmethod
    def create(
        cls,
        seed: int,
        sample_observation: dict[str, np.ndarray],
        *,
        image_encoder_type: str = "resnet-pretrained",
        aesthetic_feature_dim: int = 3584,
        aesthetic_latent_dim: int = 256,
        action_dim: int = 4,
        learning_rate: float = 3e-4,
        temperature_learning_rate: float = 3e-4,
        initial_temperature: float = 0.01,
        discount: float = 0.97,
        tau: float = 0.005,
        pretrained_resnet_path: str | Path | None = None,
    ) -> "AestheticSACAgent":
        if initial_temperature <= 0.0:
            raise ValueError("initial_temperature must be positive")
        rng = jax.random.PRNGKey(seed)
        rng, encoder_rng, actor_rng, critic_rng = jax.random.split(rng, 4)
        encoder_def = FusionEncoder(
            aesthetic_feature_dim=aesthetic_feature_dim,
            aesthetic_latent_dim=aesthetic_latent_dim,
            image_encoder_type=image_encoder_type,
        )
        actor_def = ActorNetwork(action_dim=action_dim)
        critic_def = CriticNetwork(ensemble_size=2)
        sample = jax.tree_util.tree_map(jnp.asarray, sample_observation)
        encoder_params = encoder_def.init(encoder_rng, sample, train=False)["params"]
        if image_encoder_type == "resnet-pretrained":
            if pretrained_resnet_path is None:
                raise ValueError("pretrained_resnet_path is required for production encoder")
            encoder_params, _ = replace_pretrained_resnet_params(
                encoder_params, pretrained_resnet_path
            )
        encoding = encoder_def.apply({"params": encoder_params}, sample, train=False)
        actor_params = actor_def.init(actor_rng, encoding, train=False)["params"]
        zero_action = jnp.zeros((*encoding.shape[:-1], action_dim), dtype=jnp.float32)
        critic_params = critic_def.init(
            critic_rng, encoding, zero_action, train=False
        )["params"]
        encoder_tx = optax.adam(learning_rate)
        actor_tx = optax.adam(learning_rate)
        critic_tx = optax.adam(learning_rate)
        temperature_tx = optax.adam(temperature_learning_rate)
        log_temperature = jnp.asarray(np.log(initial_temperature), dtype=jnp.float32)
        return cls(
            encoder_params=encoder_params,
            actor_params=actor_params,
            critic_params=critic_params,
            target_encoder_params=encoder_params,
            target_critic_params=critic_params,
            log_temperature=log_temperature,
            encoder_opt_state=encoder_tx.init(encoder_params),
            actor_opt_state=actor_tx.init(actor_params),
            critic_opt_state=critic_tx.init(critic_params),
            temperature_opt_state=temperature_tx.init(log_temperature),
            rng=rng,
            update_step=jnp.asarray(0, dtype=jnp.int32),
            encoder_def=encoder_def,
            actor_def=actor_def,
            critic_def=critic_def,
            encoder_tx=encoder_tx,
            actor_tx=actor_tx,
            critic_tx=critic_tx,
            temperature_tx=temperature_tx,
            discount=float(discount),
            tau=float(tau),
            target_entropy=-float(action_dim) / 2.0,
        )

    def _encode(self, observations, params, train, rng=None):
        variables = {"params": params}
        kwargs = {"train": train}
        if train:
            return self.encoder_def.apply(variables, observations, rngs={"dropout": rng}, **kwargs)
        return self.encoder_def.apply(variables, observations, **kwargs)

    @partial(jax.jit, static_argnames=("argmax",))
    def sample_actions(self, observations, *, seed=None, argmax=False):
        encoding = self._encode(observations, self.encoder_params, False)
        mean, log_std = self.actor_def.apply(
            {"params": self.actor_params}, encoding, train=False
        )
        if argmax:
            return jnp.tanh(mean)
        if seed is None:
            seed = self.rng
        action, _ = _sample_tanh_normal(mean, log_std, seed)
        return action

    @jax.jit
    def bc_update(self, batch):
        rng, dropout_rng = jax.random.split(self.rng)

        def loss_fn(encoder_params, actor_params):
            encoding = self._encode(
                batch["observations"], encoder_params, True, dropout_rng
            )
            mean, log_std = self.actor_def.apply(
                {"params": actor_params}, encoding, train=True
            )
            log_prob = _action_log_prob(mean, log_std, batch["actions"])
            mode = jnp.tanh(mean)
            return -jnp.mean(log_prob), {
                "bc_mse": jnp.mean(jnp.square(mode - batch["actions"])),
                "bc_log_prob": jnp.mean(log_prob),
            }

        (loss, info), (encoder_grads, actor_grads) = jax.value_and_grad(
            loss_fn, argnums=(0, 1), has_aux=True
        )(self.encoder_params, self.actor_params)
        encoder_updates, encoder_opt_state = self.encoder_tx.update(
            encoder_grads, self.encoder_opt_state, self.encoder_params
        )
        actor_updates, actor_opt_state = self.actor_tx.update(
            actor_grads, self.actor_opt_state, self.actor_params
        )
        agent = self.replace(
            encoder_params=optax.apply_updates(self.encoder_params, encoder_updates),
            actor_params=optax.apply_updates(self.actor_params, actor_updates),
            encoder_opt_state=encoder_opt_state,
            actor_opt_state=actor_opt_state,
            rng=rng,
            update_step=self.update_step + 1,
        )
        return agent, {"bc_loss": loss, **info}

    @jax.jit
    def sac_update(self, batch):
        rng, next_action_rng, encoder_rng = jax.random.split(self.rng, 3)
        temperature = jnp.exp(self.log_temperature)

        def critic_loss_fn(encoder_params, critic_params):
            next_encoding = self._encode(
                batch["next_observations"], self.target_encoder_params, False
            )
            next_mean, next_log_std = self.actor_def.apply(
                {"params": self.actor_params}, next_encoding, train=False
            )
            next_action, next_log_prob = _sample_tanh_normal(
                next_mean, next_log_std, next_action_rng
            )
            target_qs = self.critic_def.apply(
                {"params": self.target_critic_params},
                next_encoding,
                next_action,
                train=False,
            )
            target_q = jnp.min(target_qs, axis=0) - temperature * next_log_prob
            target = batch["rewards"] + self.discount * batch["masks"] * target_q
            target = jax.lax.stop_gradient(target)
            encoding = self._encode(
                batch["observations"], encoder_params, True, encoder_rng
            )
            predicted = self.critic_def.apply(
                {"params": critic_params}, encoding, batch["actions"], train=True
            )
            loss = jnp.mean(jnp.square(predicted - target[None, :]))
            return loss, {
                "critic_q": jnp.mean(predicted),
                "target_q": jnp.mean(target),
            }

        (critic_loss, critic_info), (encoder_grads, critic_grads) = jax.value_and_grad(
            critic_loss_fn, argnums=(0, 1), has_aux=True
        )(self.encoder_params, self.critic_params)
        encoder_updates, encoder_opt_state = self.encoder_tx.update(
            encoder_grads, self.encoder_opt_state, self.encoder_params
        )
        critic_updates, critic_opt_state = self.critic_tx.update(
            critic_grads, self.critic_opt_state, self.critic_params
        )
        encoder_params = optax.apply_updates(self.encoder_params, encoder_updates)
        critic_params = optax.apply_updates(self.critic_params, critic_updates)

        rng, actor_rng = jax.random.split(rng)
        actor_encoding = jax.lax.stop_gradient(
            self._encode(batch["observations"], encoder_params, False)
        )

        def actor_loss_fn(actor_params):
            mean, log_std = self.actor_def.apply(
                {"params": actor_params}, actor_encoding, train=True
            )
            action, log_prob = _sample_tanh_normal(mean, log_std, actor_rng)
            qs = self.critic_def.apply(
                {"params": critic_params}, actor_encoding, action, train=False
            )
            q = jnp.mean(qs, axis=0)
            loss = jnp.mean(temperature * log_prob - q)
            return loss, {"entropy": -jnp.mean(log_prob), "actor_q": jnp.mean(q)}

        (actor_loss, actor_info), actor_grads = jax.value_and_grad(
            actor_loss_fn, has_aux=True
        )(self.actor_params)
        actor_updates, actor_opt_state = self.actor_tx.update(
            actor_grads, self.actor_opt_state, self.actor_params
        )
        actor_params = optax.apply_updates(self.actor_params, actor_updates)

        rng, temperature_rng = jax.random.split(rng)
        del temperature_rng
        mean, log_std = self.actor_def.apply(
            {"params": actor_params}, actor_encoding, train=False
        )
        _, log_prob = _sample_tanh_normal(mean, log_std, rng)

        def temperature_loss_fn(log_temperature):
            return -jnp.mean(
                log_temperature
                * jax.lax.stop_gradient(log_prob + self.target_entropy)
            )

        temperature_loss, temperature_grads = jax.value_and_grad(
            temperature_loss_fn
        )(self.log_temperature)
        temperature_updates, temperature_opt_state = self.temperature_tx.update(
            temperature_grads, self.temperature_opt_state, self.log_temperature
        )
        log_temperature = optax.apply_updates(
            self.log_temperature, temperature_updates
        )

        target_encoder = jax.tree_util.tree_map(
            lambda p, t: self.tau * p + (1.0 - self.tau) * t,
            encoder_params,
            self.target_encoder_params,
        )
        target_critic = jax.tree_util.tree_map(
            lambda p, t: self.tau * p + (1.0 - self.tau) * t,
            critic_params,
            self.target_critic_params,
        )
        agent = self.replace(
            encoder_params=encoder_params,
            actor_params=actor_params,
            critic_params=critic_params,
            target_encoder_params=target_encoder,
            target_critic_params=target_critic,
            log_temperature=log_temperature,
            encoder_opt_state=encoder_opt_state,
            actor_opt_state=actor_opt_state,
            critic_opt_state=critic_opt_state,
            temperature_opt_state=temperature_opt_state,
            rng=rng,
            update_step=self.update_step + 1,
        )
        return agent, {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "temperature_loss": temperature_loss,
            "temperature": jnp.exp(log_temperature),
            **critic_info,
            **actor_info,
        }

    def policy_payload(self) -> dict[str, Any]:
        encoder = unfreeze(self.encoder_params)
        omitted_frozen_resnet = "pretrained_encoder" in encoder
        encoder.pop("pretrained_encoder", None)
        return {
            # Both local processes initialize the identical read-only ResNet
            # checkpoint. Omitting its 4.9M frozen parameters cuts each policy
            # broadcast substantially without changing actor behavior.
            "encoder_params": jax.device_get(freeze(encoder)),
            "actor_params": jax.device_get(self.actor_params),
            "update_step": int(jax.device_get(self.update_step)),
            "omitted_frozen_resnet": omitted_frozen_resnet,
        }

    def with_policy_payload(self, payload: dict[str, Any]):
        if "encoder_params" not in payload or "actor_params" not in payload:
            raise ValueError("policy payload is missing encoder/actor params")
        received = unfreeze(payload["encoder_params"])
        if payload.get("omitted_frozen_resnet"):
            local = unfreeze(self.encoder_params)
            if "pretrained_encoder" not in local:
                raise ValueError("policy payload expects a locally initialized frozen ResNet")
            local.update(received)
            received = local
        return self.replace(
            encoder_params=freeze(received),
            actor_params=payload["actor_params"],
            update_step=jnp.asarray(payload.get("update_step", 0), dtype=jnp.int32),
        )


def save_checkpoint(agent: AestheticSACAgent, path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(flax.serialization.to_bytes(agent))
    temporary.replace(destination)
    return destination


def load_checkpoint(agent: AestheticSACAgent, path: str | Path) -> AestheticSACAgent:
    source = Path(path).expanduser().resolve()
    return flax.serialization.from_bytes(agent, source.read_bytes())
