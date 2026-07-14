"""Save candidate Franka photo-home poses for visual selection.

Run from /home/junyi/robot_aesthetic_rl:

    TERM=xterm PYTHONUNBUFFERED=1 /home/junyi/IsaacLab_232_sim51/isaaclab.sh \
      -p aesthetic_rl/scripts/scan_franka_photo_home_pose.py \
      --headless --enable_cameras
"""

import argparse
import csv
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Render candidate Franka photography home poses.")
parser.add_argument(
    "--output_dir",
    type=str,
    default="/home/junyi/robot_aesthetic_rl/aesthetic_rl/output/franka_photo_home_scan",
)
parser.add_argument(
    "--scene_variant",
    default="residential_living_room",
    choices=("residential_living_room", "office_lounge", "tabletop_aesthetic", "minimal"),
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


CANDIDATE_POSES = {
    "current": (0.0, -0.78, 0.0, -2.43, 0.0, 2.78, 0.741),
    "lower_2cm": (0.0, -0.78, 0.0, -2.49, 0.0, 2.84, 0.741),
    "raise_2cm": (0.0, -0.78, 0.0, -2.37, 0.0, 2.72, 0.741),
    "raise_4cm": (0.0, -0.78, 0.0, -2.31, 0.0, 2.66, 0.741),
    "raise_back": (0.0, -0.88, 0.0, -2.43, 0.0, 2.68, 0.741),
}


def save_image(path: Path, image: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)
    print(f"[INFO]: saved {path}, shape={image.shape}, mean={image.mean():.2f}", flush=True)


def main():
    output_dir = Path(args_cli.output_dir)
    cfg = IsaacFrankaWristCameraEnvCfg(
        image_height=128,
        image_width=128,
        max_episode_steps=1,
        output_dir=str(output_dir),
        device=args_cli.device,
        reward_mode="dummy",
        action_mode="joint_delta_7d",
        scene_variant=args_cli.scene_variant,
        enable_overview_camera=True,
    )
    env = IsaacFrankaWristCameraEnv(cfg)

    rows = []
    for name, pose in CANDIDATE_POSES.items():
        env.cfg.photo_home_joint_pos = pose
        obs, info = env.reset(seed=0)
        save_image(output_dir / f"{name}_overview.png", env.get_overview_rgb())
        save_image(output_dir / f"{name}_wrist.png", obs["image"])
        rows.append(
            {
                "name": name,
                "pose": ",".join(f"{value:.4f}" for value in pose),
                "image_mean": float(obs["image"].mean()),
                "target_visibility": float(info["target_visibility"]),
                "target_center_quality": float(info["target_center_quality"]),
            }
        )

    with (output_dir / "poses.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK]: wrote candidate poses to {output_dir}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
