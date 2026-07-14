"""Run first-stage IsaacLab aesthetic photography baselines.

Run from the IsaacLab repository:

    conda activate isaaclab51
    TERM=xterm PYTHONUNBUFFERED=1 ./isaaclab.sh \
      -p /home/junyi/robot_aesthetic_rl/aesthetic_rl/scripts/run_isaac_aesthetic_baselines.py \
      --headless --enable_cameras --reward_mode artimuse
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

PROJECT_ROOT = Path(__file__).resolve().parents[2]

parser = argparse.ArgumentParser(description="Evaluate fixed/random/grid baselines in the Isaac aesthetic env.")
parser.add_argument("--episodes", type=int, default=3, help="Episodes per baseline.")
parser.add_argument("--steps", type=int, default=20, help="Steps per episode.")
parser.add_argument(
    "--policies",
    type=str,
    default="fixed,random,grid",
    help="Comma-separated subset of fixed, random, grid.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=str(PROJECT_ROOT / "aesthetic_rl/output/isaac_aesthetic_baselines"),
    help="Directory for metrics, best images, and trajectories.",
)
parser.add_argument(
    "--reward_mode",
    type=str,
    default="dummy",
    choices=("dummy", "artimuse"),
    help="Reward backend. Use artimuse for final baseline numbers.",
)
parser.add_argument(
    "--action_mode",
    type=str,
    default="joint_delta_7d",
    choices=("ee_delta_xyz", "camera_xyz", "camera_pose_5d", "joint_delta_7d"),
    help="Control space used by all baselines.",
)
parser.add_argument(
    "--scene_variant",
    type=str,
    default="residential_living_room",
    choices=("residential_living_room", "office_lounge", "tabletop_aesthetic", "minimal"),
    help="Scene layout to evaluate.",
)
parser.add_argument(
    "--grid_levels",
    type=int,
    default=3,
    help="Number of values per action dimension for the deterministic grid policy.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=0,
    help="Random seed.",
)
parser.add_argument(
    "--artimuse_model_path",
    type=str,
    default=str(PROJECT_ROOT / "ArtiMuse/checkpoints/ArtiMuse"),
    help="Path to the ArtiMuse checkpoint directory.",
)
parser.add_argument(
    "--artimuse_max_gpu_memory",
    type=str,
    default="5GiB",
    help="GPU memory budget passed to ArtiMuse device_map='auto'.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
from PIL import Image

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aesthetic_rl.envs.isaac_franka_wrist_env import (
    IsaacFrankaWristCameraEnv,
    IsaacFrankaWristCameraEnvCfg,
    make_grid_actions,
)


def save_image(path: Path, image: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def save_target_debug(path_prefix: Path, image: np.ndarray, target_mask: np.ndarray):
    mask = target_mask[..., 0].astype(bool)
    save_image(path_prefix.with_name(f"{path_prefix.name}_mask.png"), mask.astype(np.uint8) * 255)
    overlay = image.astype(np.float32).copy()
    overlay[mask] = 0.45 * overlay[mask] + 0.55 * np.array([40.0, 255.0, 80.0], dtype=np.float32)
    save_image(path_prefix.with_name(f"{path_prefix.name}_overlay.png"), np.clip(overlay, 0, 255).astype(np.uint8))


def serializable_info(info: dict) -> dict:
    result = {}
    for key, value in info.items():
        if isinstance(value, np.ndarray):
            result[key] = value.tolist()
        elif isinstance(value, np.generic):
            result[key] = value.item()
        else:
            result[key] = value
    return result


def make_env() -> IsaacFrankaWristCameraEnv:
    cfg = IsaacFrankaWristCameraEnvCfg(
        image_height=128,
        image_width=128,
        max_episode_steps=args_cli.steps,
        output_dir=args_cli.output_dir,
        device=args_cli.device,
        reward_mode=args_cli.reward_mode,
        action_mode=args_cli.action_mode,
        scene_variant=args_cli.scene_variant,
        artimuse_model_path=args_cli.artimuse_model_path,
        artimuse_max_gpu_memory=args_cli.artimuse_max_gpu_memory,
    )
    return IsaacFrankaWristCameraEnv(cfg)


def action_for_policy(policy: str, env: IsaacFrankaWristCameraEnv, rng: np.random.Generator, step: int):
    if policy == "fixed":
        return np.zeros(env.action_space.shape, dtype=np.float32)
    if policy == "random":
        return rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)
    if policy == "grid":
        if not hasattr(env, "_baseline_grid_actions"):
            env._baseline_grid_actions = make_grid_actions(
                env.action_space,
                levels=args_cli.grid_levels,
                max_actions=args_cli.steps,
            )
        return env._baseline_grid_actions[step % len(env._baseline_grid_actions)]
    raise ValueError(f"Unsupported baseline policy: {policy}")


def transition_row(policy: str, episode: int, step: int, action: np.ndarray, info: dict) -> dict:
    camera_pos = info.get("camera_local_pos", np.zeros(3, dtype=np.float32))
    camera_angles = info.get("camera_pitch_yaw", np.zeros(2, dtype=np.float32))
    ee_pos = info.get("ee_pos_b", np.zeros(3, dtype=np.float32))
    ee_target_pos = info.get("ee_target_pos_b", np.zeros(3, dtype=np.float32))
    grid_action = info.get("absolute_grid_action", np.full(3, np.nan, dtype=np.float32))
    return {
        "policy": policy,
        "episode": episode,
        "step": step,
        "reward": float(info["reward"]),
        "aesthetic_score": float(info["aesthetic_score"]),
        "normalized_aesthetic_score": float(info["normalized_aesthetic_score"]),
        "task_aesthetic_score": float(info["task_aesthetic_score"]),
        "total_penalty": float(info["total_penalty"]),
        "collision_penalty": float(info["collision_penalty"]),
        "out_of_view_penalty": float(info["out_of_view_penalty"]),
        "action_smoothness_penalty": float(info["action_smoothness_penalty"]),
        "target_visibility": float(info["target_visibility"]),
        "target_visibility_ratio": float(info["target_visibility_ratio"]),
        "action_x": float(action[0]) if len(action) > 0 else 0.0,
        "action_y": float(action[1]) if len(action) > 1 else 0.0,
        "action_z": float(action[2]) if len(action) > 2 else 0.0,
        "grid_x": float(grid_action[0]),
        "grid_y": float(grid_action[1]),
        "grid_z": float(grid_action[2]),
        "camera_x": float(camera_pos[0]),
        "camera_y": float(camera_pos[1]),
        "camera_z": float(camera_pos[2]),
        "camera_pitch": float(camera_angles[0]),
        "camera_yaw": float(camera_angles[1]),
        "ee_x": float(ee_pos[0]),
        "ee_y": float(ee_pos[1]),
        "ee_z": float(ee_pos[2]),
        "ee_target_x": float(ee_target_pos[0]),
        "ee_target_y": float(ee_target_pos[1]),
        "ee_target_z": float(ee_target_pos[2]),
        "ee_position_error": float(info.get("ee_position_error", 0.0)),
        "ee_orientation_error": float(info.get("ee_orientation_error", 0.0)),
        "joint_tracking_error": float(info.get("joint_tracking_error", 0.0)),
    }


def run_episode(env: IsaacFrankaWristCameraEnv, policy: str, episode: int, rng: np.random.Generator):
    obs, reset_info = env.reset(seed=args_cli.seed + episode)
    rows = []
    best_reward = {
        "reward": float(reset_info["reward"]),
        "aesthetic_score": float(reset_info["aesthetic_score"]),
        "step": -1,
        "image": obs["image"].copy(),
        "target_mask": obs["target_mask"].copy(),
        "info": serializable_info(reset_info),
    }
    highest_aesthetic = dict(best_reward)

    total_reward = 0.0
    for step in range(args_cli.steps):
        action = action_for_policy(policy, env, rng, step)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)

        candidate = {
            "reward": float(reward),
            "aesthetic_score": float(info["aesthetic_score"]),
            "step": step,
            "image": obs["image"].copy(),
            "target_mask": obs["target_mask"].copy(),
            "info": serializable_info(info),
        }
        if candidate["reward"] > best_reward["reward"]:
            best_reward = candidate
        if candidate["aesthetic_score"] > highest_aesthetic["aesthetic_score"]:
            highest_aesthetic = candidate

        rows.append(transition_row(policy, episode, step, action, info))

        if terminated or truncated:
            break

    scores = [row["aesthetic_score"] for row in rows]
    summary = {
        "policy": policy,
        "episode": episode,
        "steps": len(rows),
        "total_reward": total_reward,
        "mean_reward": total_reward / max(len(rows), 1),
        "initial_aesthetic_score": float(reset_info["aesthetic_score"]),
        "final_aesthetic_score": float(rows[-1]["aesthetic_score"]) if rows else float(reset_info["aesthetic_score"]),
        "mean_aesthetic_score": float(np.mean(scores)) if scores else float(reset_info["aesthetic_score"]),
        "best_reward": best_reward["reward"],
        "best_reward_aesthetic_score": best_reward["aesthetic_score"],
        "best_reward_step": best_reward["step"],
        "best_reward_info": best_reward["info"],
        "highest_aesthetic_score": highest_aesthetic["aesthetic_score"],
        "highest_aesthetic_reward": highest_aesthetic["reward"],
        "highest_aesthetic_step": highest_aesthetic["step"],
    }
    return summary, rows, best_reward, highest_aesthetic


def run_grid_episode(env: IsaacFrankaWristCameraEnv, episode: int):
    obs, reset_info = env.reset(seed=args_cli.seed + episode)
    best_reward = {
        "reward": float(reset_info["reward"]),
        "aesthetic_score": float(reset_info["aesthetic_score"]),
        "step": -1,
        "image": obs["image"].copy(),
        "target_mask": obs["target_mask"].copy(),
        "info": serializable_info(reset_info),
    }
    highest_aesthetic = dict(best_reward)
    actions = make_grid_actions(env.action_space, levels=args_cli.grid_levels, max_actions=args_cli.steps)
    rows = []
    total_reward = 0.0

    for step, action in enumerate(actions):
        env.reset(seed=args_cli.seed + episode, options={"skip_reward": True})
        obs, reward, _, _, info = env.step_absolute_ee(action)
        total_reward += float(reward)
        candidate = {
            "reward": float(reward),
            "aesthetic_score": float(info["aesthetic_score"]),
            "step": step,
            "image": obs["image"].copy(),
            "target_mask": obs["target_mask"].copy(),
            "info": serializable_info(info),
        }
        if candidate["reward"] > best_reward["reward"]:
            best_reward = candidate
        if candidate["aesthetic_score"] > highest_aesthetic["aesthetic_score"]:
            highest_aesthetic = candidate
        rows.append(transition_row("grid", episode, step, action, info))

    scores = [row["aesthetic_score"] for row in rows]
    summary = {
        "policy": "grid",
        "episode": episode,
        "grid_evaluation": "independent_absolute_workspace_points",
        "steps": len(rows),
        "total_reward": total_reward,
        "mean_reward": total_reward / max(len(rows), 1),
        "initial_aesthetic_score": float(reset_info["aesthetic_score"]),
        "final_aesthetic_score": float(rows[-1]["aesthetic_score"]) if rows else float(reset_info["aesthetic_score"]),
        "mean_aesthetic_score": float(np.mean(scores)) if scores else float(reset_info["aesthetic_score"]),
        "best_reward": best_reward["reward"],
        "best_reward_aesthetic_score": best_reward["aesthetic_score"],
        "best_reward_step": best_reward["step"],
        "best_reward_info": best_reward["info"],
        "highest_aesthetic_score": highest_aesthetic["aesthetic_score"],
        "highest_aesthetic_reward": highest_aesthetic["reward"],
        "highest_aesthetic_step": highest_aesthetic["step"],
    }
    return summary, rows, best_reward, highest_aesthetic


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    output_dir = Path(args_cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args_cli.seed)
    policies = [policy.strip() for policy in args_cli.policies.split(",") if policy.strip()]

    env = make_env()
    all_summaries = []
    all_rows = []
    try:
        for policy in policies:
            for episode in range(args_cli.episodes):
                if policy == "grid" and args_cli.action_mode == "ee_delta_xyz":
                    summary, rows, best_reward, highest_aesthetic = run_grid_episode(env, episode)
                else:
                    summary, rows, best_reward, highest_aesthetic = run_episode(env, policy, episode, rng)
                all_summaries.append(summary)
                all_rows.extend(rows)
                best_path = output_dir / "best_images" / f"{policy}_episode_{episode:02d}_best.png"
                save_image(best_path, best_reward["image"])
                save_target_debug(best_path.with_suffix(""), best_reward["image"], best_reward["target_mask"])
                highest_aesthetic_path = (
                    output_dir / "best_images" / f"{policy}_episode_{episode:02d}_highest_aesthetic.png"
                )
                save_image(highest_aesthetic_path, highest_aesthetic["image"])
                save_target_debug(
                    highest_aesthetic_path.with_suffix(""),
                    highest_aesthetic["image"],
                    highest_aesthetic["target_mask"],
                )
                print(
                    f"[RESULT] policy={policy} episode={episode} "
                    f"initial={summary['initial_aesthetic_score']:.4f} "
                    f"final={summary['final_aesthetic_score']:.4f} "
                    f"best_reward={summary['best_reward']:.4f} "
                    f"highest_aesthetic={summary['highest_aesthetic_score']:.4f} "
                    f"mean_reward={summary['mean_reward']:.4f}",
                    flush=True,
                )
    finally:
        env.close()

    summary_path = output_dir / "summary.json"
    metrics_path = output_dir / "metrics.csv"
    with summary_path.open("w") as f:
        json.dump(
            {
                "config": {
                    "episodes": args_cli.episodes,
                    "steps": args_cli.steps,
                    "reward_mode": args_cli.reward_mode,
                    "action_mode": args_cli.action_mode,
                    "scene_variant": args_cli.scene_variant,
                    "grid_levels": args_cli.grid_levels,
                    "seed": args_cli.seed,
                    "grid_evaluation": (
                        "independent_absolute_workspace_points"
                        if args_cli.action_mode == "ee_delta_xyz"
                        else "sequential_actions"
                    ),
                },
                "episodes": all_summaries,
            },
            f,
            indent=2,
        )
    write_csv(metrics_path, all_rows)
    print(f"[OK] wrote {summary_path}", flush=True)
    print(f"[OK] wrote {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
