"""Smoke test for the IsaacLab Franka wrist-camera Gymnasium env.

Run from the IsaacLab repository:

    conda activate isaaclab51
    TERM=xterm PYTHONUNBUFFERED=1 ./isaaclab.sh \
      -p /home/junyi/robot_aesthetic_rl/aesthetic_rl/scripts/test_isaac_franka_wrist_env.py \
      --headless --enable_cameras
"""

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

PROJECT_ROOT = Path(__file__).resolve().parents[2]

parser = argparse.ArgumentParser(description="Random rollout smoke test for IsaacFrankaWristCameraEnv.")
parser.add_argument("--rollout_steps", type=int, default=8, help="Number of random env steps.")
parser.add_argument(
    "--rollout_policy",
    type=str,
    default="random",
    choices=("random", "fixed", "raise"),
    help="Deterministic fixed/raise controls are useful for Differential IK validation.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=str(PROJECT_ROOT / "aesthetic_rl/output/isaac_franka_wrist_env"),
    help="Directory for saved RGB frames.",
)
parser.add_argument(
    "--reward_mode",
    type=str,
    default="dummy",
    choices=("dummy", "artimuse"),
    help="Reward backend. Use dummy for fast smoke tests, artimuse for model scoring.",
)
parser.add_argument(
    "--action_mode",
    type=str,
    default="joint_delta_7d",
    choices=("ee_delta_xyz", "camera_xyz", "camera_pose_5d", "joint_delta_7d"),
    help="First-stage control space for the camera or robot joints.",
)
parser.add_argument(
    "--scene_variant",
    type=str,
    default="residential_living_room",
    choices=("residential_living_room", "office_lounge", "tabletop_aesthetic", "minimal"),
    help="Scene layout to spawn.",
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

from aesthetic_rl.envs.isaac_franka_wrist_env import IsaacFrankaWristCameraEnv, IsaacFrankaWristCameraEnvCfg


def save_image(path: str, image: np.ndarray):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)
    print(f"[INFO]: saved {path}, shape={image.shape}, mean={image.mean():.2f}", flush=True)


def save_target_debug(output_dir: str, stem: str, image: np.ndarray, target_mask: np.ndarray):
    mask = target_mask[..., 0].astype(bool)
    save_image(os.path.join(output_dir, f"{stem}_mask.png"), mask.astype(np.uint8) * 255)
    overlay = image.astype(np.float32).copy()
    overlay[mask] = 0.45 * overlay[mask] + 0.55 * np.array([40.0, 255.0, 80.0], dtype=np.float32)
    save_image(os.path.join(output_dir, f"{stem}_overlay.png"), np.clip(overlay, 0, 255).astype(np.uint8))


def main():
    cfg = IsaacFrankaWristCameraEnvCfg(
        image_height=128,
        image_width=128,
        max_episode_steps=args_cli.rollout_steps,
        output_dir=args_cli.output_dir,
        device=args_cli.device,
        reward_mode=args_cli.reward_mode,
        action_mode=args_cli.action_mode,
        scene_variant=args_cli.scene_variant,
        artimuse_model_path=args_cli.artimuse_model_path,
        artimuse_max_gpu_memory=args_cli.artimuse_max_gpu_memory,
    )
    env = IsaacFrankaWristCameraEnv(cfg)

    obs, info = env.reset(seed=0)
    print("[INFO]: observation_space:", env.observation_space, flush=True)
    print("[INFO]: action_space:", env.action_space, flush=True)
    print("[INFO]: reset state shape:", obs["state"].shape, flush=True)
    print("[INFO]: reset image shape:", obs["image"].shape, flush=True)
    print("[INFO]: reset target mask shape:", obs["target_mask"].shape, flush=True)
    print("[INFO]: reset info:", info, flush=True)
    save_image(os.path.join(args_cli.output_dir, "reset.png"), obs["image"])
    save_target_debug(args_cli.output_dir, "reset", obs["image"], obs["target_mask"])

    rng = np.random.default_rng(0)
    total_reward = 0.0
    for step in range(args_cli.rollout_steps):
        if args_cli.rollout_policy == "fixed":
            action = np.zeros(env.action_space.shape, dtype=np.float32)
        elif args_cli.rollout_policy == "raise":
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            action[2] = 0.5
        else:
            action = rng.uniform(low=-0.5, high=0.5, size=env.action_space.shape).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        ee_pos = np.asarray(info.get("ee_pos_b", np.zeros(3)), dtype=np.float32)
        ee_target = np.asarray(info.get("ee_target_pos_b", np.zeros(3)), dtype=np.float32)
        print(
            f"[INFO]: step={step:03d} reward={reward:.4f} mode={info['reward_mode']} "
            f"aesthetic={info['aesthetic_score']:.4f} penalty={info['total_penalty']:.4f} "
            f"visible={info['target_visibility']:.4f} terminated={terminated} "
            f"truncated={truncated} image_mean={info['image_mean']:.2f} "
            f"ee={np.round(ee_pos, 4).tolist()} target={np.round(ee_target, 4).tolist()} "
            f"pos_err={info.get('ee_position_error', 0.0):.4f} "
            f"rot_err={info.get('ee_orientation_error', 0.0):.4f} "
            f"joint_err={info.get('joint_tracking_error', 0.0):.4f}",
            flush=True,
        )
        if terminated or truncated:
            break

    save_image(os.path.join(args_cli.output_dir, "last.png"), obs["image"])
    save_target_debug(args_cli.output_dir, "last", obs["image"], obs["target_mask"])
    print(f"[OK]: rollout finished, total_reward={total_reward:.4f}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
