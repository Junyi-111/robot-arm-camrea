"""Train SAC on the IsaacLab Franka wrist-camera aesthetic environment."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[2]

parser = argparse.ArgumentParser(description="Train SAC in the Isaac wrist-camera aesthetic scene.")
parser.add_argument("--total_timesteps", type=int, default=300)
parser.add_argument("--episode_steps", type=int, default=20)
parser.add_argument("--learning_starts", type=int, default=60)
parser.add_argument("--buffer_size", type=int, default=20_000)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--train_freq", type=int, default=1)
parser.add_argument("--gradient_steps", type=int, default=1)
parser.add_argument("--checkpoint_freq", type=int, default=100)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--eval_episodes", type=int, default=1)
parser.add_argument("--rl_device", type=str, default="cuda:0")
parser.add_argument("--resume", type=str, default="", help="Path to an SB3 SAC .zip checkpoint.")
parser.add_argument("--resume_replay_buffer", type=str, default="", help="Matching replay-buffer .pkl file.")
parser.add_argument(
    "--output_dir",
    type=str,
    default=str(PROJECT_ROOT / "aesthetic_rl/output/isaac_sac"),
)
parser.add_argument(
    "--scene_variant",
    type=str,
    default="residential_living_room",
    choices=("residential_living_room", "office_lounge", "tabletop_aesthetic", "minimal"),
)
parser.add_argument(
    "--artimuse_model_path",
    type=str,
    default=str(PROJECT_ROOT / "ArtiMuse/checkpoints/ArtiMuse"),
)
parser.add_argument("--artimuse_max_gpu_memory", type=str, default="5GiB")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
from PIL import Image
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aesthetic_rl.envs.isaac_franka_wrist_env import (
    IsaacFrankaWristCameraEnv,
    IsaacFrankaWristCameraEnvCfg,
)


class SACObservationWrapper(gym.ObservationWrapper):
    """Expose channel-first RGB plus target mask for SB3 MultiInputPolicy."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        height, width, _ = env.observation_space["image"].shape
        self.observation_space = gym.spaces.Dict(
            {
                "visual": gym.spaces.Box(0, 255, shape=(4, height, width), dtype=np.uint8),
                "state": env.observation_space["state"],
            }
        )

    def observation(self, observation: dict) -> dict:
        rgb = np.asarray(observation["image"], dtype=np.uint8)
        mask = np.asarray(observation["target_mask"], dtype=np.uint8) * 255
        visual = np.concatenate((rgb, mask), axis=-1).transpose(2, 0, 1)
        return {
            "visual": np.ascontiguousarray(visual),
            "state": np.asarray(observation["state"], dtype=np.float32),
        }

    def set_phase(self, phase: str):
        self.env.set_phase(phase)


class TrainingRecorder(gym.Wrapper):
    """Record transition metrics and initial/final/best images per episode."""

    STEP_FIELDS = (
        "phase", "episode", "step", "reward", "aesthetic_score", "task_aesthetic_score",
        "total_penalty", "target_visibility", "ee_x", "ee_y", "ee_z",
    )
    EPISODE_FIELDS = (
        "phase", "episode", "steps", "return", "initial_score", "final_score",
        "best_score", "score_delta", "best_reward", "final_visibility",
    )

    def __init__(self, env: gym.Env, output_dir: Path):
        super().__init__(env)
        self.output_dir = output_dir
        self.image_dir = output_dir / "episode_images"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.phase = "train"
        self.phase_episode = {"pretrain": 0, "train": 0, "posttrain": 0}
        self.current_episode = 0
        self.step_index = 0
        self.episode_return = 0.0
        self.initial_score = 0.0
        self.best_score = -np.inf
        self.best_reward = -np.inf
        self.best_image = None
        self._initialize_csv(self.output_dir / "steps.csv", self.STEP_FIELDS)
        self._initialize_csv(self.output_dir / "episodes.csv", self.EPISODE_FIELDS)

    @staticmethod
    def _initialize_csv(path: Path, fields: tuple[str, ...]):
        if path.exists() and path.stat().st_size > 0:
            return
        with path.open("w", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=fields).writeheader()

    @staticmethod
    def _append_csv(path: Path, fields: tuple[str, ...], row: dict):
        with path.open("a", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=fields).writerow(row)

    @staticmethod
    def _save_image(path: Path, image: np.ndarray):
        Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path)

    def set_phase(self, phase: str):
        if phase not in self.phase_episode:
            raise ValueError(f"Unsupported recorder phase: {phase}")
        self.phase = phase

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_episode = self.phase_episode[self.phase]
        self.phase_episode[self.phase] += 1
        self.step_index = 0
        self.episode_return = 0.0
        self.initial_score = float(info["aesthetic_score"])
        self.best_score = self.initial_score
        self.best_reward = float(info["reward"])
        self.best_image = obs["image"].copy()
        prefix = f"{self.phase}_episode_{self.current_episode:04d}"
        self._save_image(self.image_dir / f"{prefix}_initial.png", obs["image"])
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = float(reward)
        self.episode_return += reward
        image = obs["image"].copy()
        score = float(info["aesthetic_score"])
        if score > self.best_score:
            self.best_score = score
            self.best_image = image.copy()
        self.best_reward = max(self.best_reward, reward)

        ee_pos = np.asarray(info.get("ee_pos_b", [np.nan, np.nan, np.nan]), dtype=np.float32)
        self._append_csv(
            self.output_dir / "steps.csv",
            self.STEP_FIELDS,
            {
                "phase": self.phase,
                "episode": self.current_episode,
                "step": self.step_index,
                "reward": reward,
                "aesthetic_score": score,
                "task_aesthetic_score": float(info["task_aesthetic_score"]),
                "total_penalty": float(info["total_penalty"]),
                "target_visibility": float(info["target_visibility"]),
                "ee_x": float(ee_pos[0]),
                "ee_y": float(ee_pos[1]),
                "ee_z": float(ee_pos[2]),
            },
        )
        self.step_index += 1

        if terminated or truncated:
            prefix = f"{self.phase}_episode_{self.current_episode:04d}"
            self._save_image(self.image_dir / f"{prefix}_final.png", image)
            self._save_image(self.image_dir / f"{prefix}_best.png", self.best_image)
            self._append_csv(
                self.output_dir / "episodes.csv",
                self.EPISODE_FIELDS,
                {
                    "phase": self.phase,
                    "episode": self.current_episode,
                    "steps": self.step_index,
                    "return": self.episode_return,
                    "initial_score": self.initial_score,
                    "final_score": score,
                    "best_score": self.best_score,
                    "score_delta": score - self.initial_score,
                    "best_reward": self.best_reward,
                    "final_visibility": float(info["target_visibility"]),
                },
            )
        return obs, reward, terminated, truncated, info


def evaluate(model: SAC, env: SACObservationWrapper, phase: str, episodes: int) -> list[dict]:
    env.set_phase(phase)
    summaries = []
    for episode in range(episodes):
        obs, info = env.reset(seed=args_cli.seed + episode)
        initial_score = float(info["aesthetic_score"])
        done = False
        final_info = info
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, final_info = env.step(action)
            done = terminated or truncated
        summaries.append(
            {
                "episode": episode,
                "initial_score": initial_score,
                "final_score": float(final_info["aesthetic_score"]),
                "score_delta": float(final_info["aesthetic_score"]) - initial_score,
                "target_visibility": float(final_info["target_visibility"]),
            }
        )
    return summaries


def main():
    output_dir = Path(args_cli.output_dir).expanduser().resolve()
    cfg = IsaacFrankaWristCameraEnvCfg(
        image_height=128,
        image_width=128,
        max_episode_steps=args_cli.episode_steps,
        output_dir=str(output_dir / "reward"),
        device=args_cli.device,
        reward_mode="artimuse",
        action_mode="ee_delta_xyz",
        scene_variant=args_cli.scene_variant,
        artimuse_model_path=args_cli.artimuse_model_path,
        artimuse_max_gpu_memory=args_cli.artimuse_max_gpu_memory,
    )
    base_env = IsaacFrankaWristCameraEnv(cfg)
    recorder = TrainingRecorder(base_env, output_dir)
    env = SACObservationWrapper(recorder)

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args_cli.checkpoint_freq, 1),
        save_path=str(checkpoint_dir),
        name_prefix="sac_aesthetic",
        save_replay_buffer=True,
    )

    if args_cli.resume:
        model = SAC.load(args_cli.resume, env=env, device=args_cli.rl_device)
        if args_cli.resume_replay_buffer:
            model.load_replay_buffer(args_cli.resume_replay_buffer)
    else:
        model = SAC(
            "MultiInputPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=args_cli.buffer_size,
            learning_starts=args_cli.learning_starts,
            batch_size=args_cli.batch_size,
            train_freq=args_cli.train_freq,
            gradient_steps=args_cli.gradient_steps,
            gamma=0.95,
            tau=0.005,
            ent_coef="auto",
            policy_kwargs={"net_arch": [256, 256]},
            verbose=1,
            seed=args_cli.seed,
            device=args_cli.rl_device,
        )

    pretrain = evaluate(model, env, "pretrain", args_cli.eval_episodes)
    recorder.set_phase("train")
    model.learn(
        total_timesteps=args_cli.total_timesteps,
        callback=checkpoint_callback,
        reset_num_timesteps=not bool(args_cli.resume),
        progress_bar=False,
    )
    model.save(output_dir / "sac_aesthetic_final")
    model.save_replay_buffer(output_dir / "sac_aesthetic_final_replay_buffer.pkl")
    posttrain = evaluate(model, env, "posttrain", args_cli.eval_episodes)

    comparison = {
        "config": vars(args_cli),
        "pretrain": pretrain,
        "posttrain": posttrain,
        "mean_pretrain_delta": float(np.mean([row["score_delta"] for row in pretrain])),
        "mean_posttrain_delta": float(np.mean([row["score_delta"] for row in posttrain])),
        "mean_final_score_improvement": float(
            np.mean([row["final_score"] for row in posttrain])
            - np.mean([row["final_score"] for row in pretrain])
        ),
    }
    with (output_dir / "comparison.json").open("w", encoding="utf-8") as stream:
        json.dump(comparison, stream, indent=2, ensure_ascii=True)
    print(json.dumps(comparison, indent=2, ensure_ascii=True))
    print(f"[OK] SAC training finished: {output_dir}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
