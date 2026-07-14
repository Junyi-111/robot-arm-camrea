"""Minimal IsaacLab Franka + fixed RGB camera smoke test.

Run from the IsaacLab repository with:

    conda activate isaaclab51
    TERM=xterm PYTHONUNBUFFERED=1 ./isaaclab.sh \
      -p /home/junyi/robot_aesthetic_rl/aesthetic_rl/scripts/isaac_franka_camera_smoke.py \
      --headless --enable_cameras
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Move Franka while rendering RGB frames from a fixed camera.")
parser.add_argument("--num_steps", type=int, default=160, help="Number of simulation steps before exiting.")
parser.add_argument("--save_every", type=int, default=40, help="Save one RGB frame every N simulation steps.")
parser.add_argument(
    "--output_dir",
    type=str,
    default="/home/junyi/robot_aesthetic_rl/aesthetic_rl/output/franka_camera_smoke",
    help="Directory for saved RGB frames.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
from PIL import Image

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sim import SimulationContext
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG


def design_scene() -> tuple[Articulation, Camera]:
    """Create a simple scene with one Franka, a few colored props, and a fixed camera."""
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=3500.0, color=(0.85, 0.85, 0.8))
    light_cfg.func("/World/Light", light_cfg)

    sim_utils.create_prim("/World/Origin", "Xform", translation=(0.0, 0.0, 0.0))
    robot = Articulation(cfg=FRANKA_PANDA_CFG.replace(prim_path="/World/Origin/Robot"))

    cube_cfg = sim_utils.CuboidCfg(
        size=(0.18, 0.18, 0.18),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.18, 0.10), metallic=0.0),
    )
    cube_cfg.func("/World/RedCube", cube_cfg, translation=(0.55, -0.20, 0.09))

    cube_cfg = sim_utils.CuboidCfg(
        size=(0.14, 0.14, 0.14),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.10, 0.40, 0.95), metallic=0.0),
    )
    cube_cfg.func("/World/BlueCube", cube_cfg, translation=(0.45, 0.22, 0.07))

    camera_cfg = CameraCfg(
        prim_path="/World/FixedCamera",
        update_period=0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=28.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 100.0),
        ),
    )
    camera = Camera(cfg=camera_cfg)
    return robot, camera


def reset_robot(robot: Articulation):
    root_state = robot.data.default_root_state.clone()
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()
    return joint_pos


def save_rgb(camera: Camera, step: int, output_dir: str):
    rgb = camera.data.output["rgb"][0, ..., :3].detach().cpu().numpy()
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"frame_{step:04d}.png")
    Image.fromarray(rgb).save(path)
    print(f"[INFO]: saved {path}, shape={rgb.shape}, mean={rgb.mean():.2f}", flush=True)


def run_simulator(sim: SimulationContext, robot: Articulation, camera: Camera):
    sim_dt = sim.get_physics_dt()
    default_joint_pos = reset_robot(robot)
    target_joint_pos = default_joint_pos.clone()

    camera.set_world_poses_from_view(
        torch.tensor([[2.2, 1.7, 1.35]], device=sim.device),
        torch.tensor([[0.25, 0.0, 0.45]], device=sim.device),
    )

    print("[INFO]: Setup complete.", flush=True)
    print("[INFO]: Joint names:", robot.joint_names, flush=True)

    saved_paths = []
    for count in range(args_cli.num_steps):
        phase = torch.tensor(float(count) * 0.035, device=sim.device)
        target_joint_pos[:] = default_joint_pos
        target_joint_pos[:, 0] = default_joint_pos[:, 0] + 0.30 * torch.sin(phase)
        target_joint_pos[:, 1] = default_joint_pos[:, 1] + 0.18 * torch.sin(phase * 0.8)
        target_joint_pos[:, 3] = default_joint_pos[:, 3] + 0.22 * torch.sin(phase * 0.9)
        target_joint_pos[:, 5] = default_joint_pos[:, 5] + 0.12 * torch.sin(phase * 1.1)

        robot.set_joint_position_target(target_joint_pos)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)
        camera.update(dt=sim_dt)

        if count % args_cli.save_every == 0 or count == args_cli.num_steps - 1:
            save_rgb(camera, count, args_cli.output_dir)
            saved_paths.append(os.path.join(args_cli.output_dir, f"frame_{count:04d}.png"))

    print("[OK]: Franka moved while fixed RGB camera rendered successfully.", flush=True)
    print("[OK]: Saved frames:", saved_paths, flush=True)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.2, 1.7, 1.35], [0.25, 0.0, 0.45])

    robot, camera = design_scene()
    sim.reset()
    run_simulator(sim, robot, camera)


if __name__ == "__main__":
    main()
    simulation_app.close()
