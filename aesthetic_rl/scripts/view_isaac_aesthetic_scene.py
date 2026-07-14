"""Open the Isaac aesthetic scene in the Isaac Sim GUI for visual inspection."""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Interactively view the Isaac aesthetic photography scene.")
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

PROJECT_ROOT = Path("/home/junyi/robot_aesthetic_rl")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aesthetic_rl.envs.isaac_franka_wrist_env import IsaacFrankaWristCameraEnv, IsaacFrankaWristCameraEnvCfg


def main():
    cfg = IsaacFrankaWristCameraEnvCfg(
        device=args_cli.device,
        reward_mode="dummy",
        action_mode="joint_delta_7d",
        scene_variant=args_cli.scene_variant,
        max_episode_steps=1000,
    )
    env = IsaacFrankaWristCameraEnv(cfg)
    env.reset(seed=0)
    zero_action = np.zeros(env.action_space.shape, dtype=np.float32)
    print(f"[INFO]: viewing scene_variant={args_cli.scene_variant}. Close Isaac Sim to exit.", flush=True)

    while simulation_app.is_running():
        _, _, terminated, truncated, _ = env.step(zero_action)
        if terminated or truncated:
            env.reset(seed=0)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
