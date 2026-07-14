"""Capture third-person overview images of the IsaacLab aesthetic training scene.

Run from /home/junyi/robot_aesthetic_rl:

    TERM=xterm PYTHONUNBUFFERED=1 /home/junyi/IsaacLab_232_sim51/isaaclab.sh \
      -p aesthetic_rl/scripts/capture_isaac_aesthetic_overview.py \
      --headless --enable_cameras
"""

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Save overview-camera images for the tabletop aesthetic scene.")
parser.add_argument("--num_steps", type=int, default=24, help="Number of env steps to roll out.")
parser.add_argument("--save_every", type=int, default=8, help="Save one overview image every N env steps.")
parser.add_argument(
    "--output_dir",
    type=str,
    default="/home/junyi/robot_aesthetic_rl/aesthetic_rl/output/isaac_aesthetic_overview",
    help="Directory for saved overview images.",
)
parser.add_argument(
    "--action_mode",
    type=str,
    default="joint_delta_7d",
    choices=("camera_xyz", "camera_pose_5d", "joint_delta_7d"),
    help="Action mode used for the short rollout.",
)
parser.add_argument(
    "--scene_variant",
    type=str,
    default="residential_living_room",
    choices=("residential_living_room", "office_lounge", "tabletop_aesthetic", "minimal"),
    help="Scene layout to capture.",
)
parser.add_argument(
    "--policy",
    type=str,
    default="fixed",
    choices=("fixed", "random"),
    help="Use fixed zero actions or random actions while capturing overview frames.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
from PIL import Image

PROJECT_ROOT = Path("/home/junyi/robot_aesthetic_rl")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aesthetic_rl.envs.isaac_franka_wrist_env import IsaacFrankaWristCameraEnv, IsaacFrankaWristCameraEnvCfg

OVERVIEW_VIEWS = {
    "front": ((2.60, -1.55, 1.25), (1.45, 0.0, 0.46)),
    "side": ((2.80, 0.20, 1.20), (1.45, 0.0, 0.46)),
    "top": ((1.55, -0.55, 1.80), (1.45, 0.0, 0.40)),
}


def save_image(path: str | Path, image: np.ndarray):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)
    print(f"[INFO]: saved {path}, shape={image.shape}, mean={image.mean():.2f}", flush=True)


def save_overview_views(env: IsaacFrankaWristCameraEnv, output_dir: str | Path, prefix: str):
    for view_name, (eye, target) in OVERVIEW_VIEWS.items():
        env.cfg.overview_eye = eye
        env.cfg.overview_target = target
        save_image(Path(output_dir) / f"{prefix}_{view_name}.png", env.get_overview_rgb())


def main():
    cfg = IsaacFrankaWristCameraEnvCfg(
        image_height=128,
        image_width=128,
        max_episode_steps=args_cli.num_steps,
        output_dir=args_cli.output_dir,
        device=args_cli.device,
        reward_mode="dummy",
        action_mode=args_cli.action_mode,
        scene_variant=args_cli.scene_variant,
        enable_overview_camera=True,
    )
    env = IsaacFrankaWristCameraEnv(cfg)
    rng = np.random.default_rng(0)

    obs, info = env.reset(seed=0)
    save_overview_views(env, args_cli.output_dir, "overview_reset")
    save_image(Path(args_cli.output_dir) / "wrist_reset.png", obs["image"])

    for step in range(args_cli.num_steps):
        if args_cli.policy == "fixed":
            action = np.zeros(env.action_space.shape, dtype=np.float32)
        else:
            action = rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        if step % args_cli.save_every == 0 or step == args_cli.num_steps - 1:
            save_overview_views(env, args_cli.output_dir, f"overview_step_{step:04d}")
            print(
                f"[INFO]: step={step:04d} reward={reward:.4f} "
                f"visible={info['target_visibility']:.4f} image_mean={info['image_mean']:.2f}",
                flush=True,
            )
        if terminated or truncated:
            break

    save_image(Path(args_cli.output_dir) / "wrist_last.png", obs["image"])
    env.close()
    print(f"[OK]: overview capture finished, output_dir={args_cli.output_dir}", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
