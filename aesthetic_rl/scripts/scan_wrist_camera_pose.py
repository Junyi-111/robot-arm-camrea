"""Sweep wrist-camera local pitch/yaw offsets and save the rendered views.

Run from /home/junyi/robot_aesthetic_rl:

    TERM=xterm PYTHONUNBUFFERED=1 /home/junyi/IsaacLab_232_sim51/isaaclab.sh \
      -p aesthetic_rl/scripts/scan_wrist_camera_pose.py \
      --headless --enable_cameras
"""

import argparse
import csv
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Save a pitch/yaw sweep for the wrist camera.")
parser.add_argument(
    "--output_dir",
    type=str,
    default="/home/junyi/robot_aesthetic_rl/aesthetic_rl/output/wrist_camera_pose_scan",
)
parser.add_argument(
    "--pitch_values",
    type=str,
    default="-1.80,-1.40,-1.00,-0.60,-0.20,0.20,0.60,1.00,1.40,1.80",
    help="Comma-separated pitch offsets in radians.",
)
parser.add_argument(
    "--yaw_values",
    type=str,
    default="-3.14,-2.40,-1.60,-0.80,0.00,0.80,1.60,2.40,3.14",
    help="Comma-separated yaw offsets in radians.",
)
parser.add_argument(
    "--camera_pos",
    type=str,
    default="0.13,0.00,-0.15",
    help="Camera local xyz offset, comma-separated.",
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


def parse_floats(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def save_image(path: Path, image: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)
    print(f"[INFO]: saved {path}, shape={image.shape}, mean={image.mean():.2f}", flush=True)


def main():
    output_dir = Path(args_cli.output_dir)
    pitch_values = parse_floats(args_cli.pitch_values)
    yaw_values = parse_floats(args_cli.yaw_values)
    camera_pos = np.array(parse_floats(args_cli.camera_pos), dtype=np.float32)

    cfg = IsaacFrankaWristCameraEnvCfg(
        image_height=128,
        image_width=128,
        max_episode_steps=1,
        output_dir=str(output_dir),
        device=args_cli.device,
        reward_mode="dummy",
        action_mode="camera_pose_5d",
        scene_variant="tabletop_aesthetic",
        enable_overview_camera=True,
    )
    env = IsaacFrankaWristCameraEnv(cfg)
    obs, info = env.reset(seed=0)
    save_image(output_dir / "overview_front.png", env.get_overview_rgb())

    rows = []
    for pitch in pitch_values:
        for yaw in yaw_values:
            env.camera_local_pos = camera_pos.copy()
            env.camera_pitch_yaw = np.array([pitch, yaw], dtype=np.float32)
            env._warmup_camera()
            image = env._get_rgb()
            filename = f"wrist_pitch_{pitch:+.2f}_yaw_{yaw:+.2f}.png".replace("+", "p").replace("-", "m")
            save_image(output_dir / filename, image)
            visibility, center_quality = env._target_visibility(image)
            rows.append(
                {
                    "filename": filename,
                    "pitch": pitch,
                    "yaw": yaw,
                    "camera_x": float(camera_pos[0]),
                    "camera_y": float(camera_pos[1]),
                    "camera_z": float(camera_pos[2]),
                    "image_mean": float(image.mean()),
                    "target_visibility": visibility,
                    "target_center_quality": center_quality,
                }
            )

    with (output_dir / "scan.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best = max(rows, key=lambda row: (row["target_visibility"], row["target_center_quality"]))
    print("[BEST]:", best, flush=True)
    print(f"[OK]: wrote scan to {output_dir}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
