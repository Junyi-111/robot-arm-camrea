"""Minimal IsaacLab Franka load/control smoke test.

Run from the IsaacLab repository with:

    conda activate isaaclab51
    TERM=xterm ./isaaclab.sh -p /home/junyi/robot_aesthetic_rl/aesthetic_rl/scripts/isaac_franka_smoke.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Load Franka in IsaacLab and drive it with joint position targets.")
parser.add_argument("--num_steps", type=int, default=240, help="Number of simulation steps before exiting.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG


def design_scene() -> Articulation:
    """Create a ground plane, light, and one Franka robot."""
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    sim_utils.create_prim("/World/Origin", "Xform", translation=(0.0, 0.0, 0.0))
    robot_cfg = FRANKA_PANDA_CFG.replace(prim_path="/World/Origin/Robot")
    return Articulation(cfg=robot_cfg)


def reset_robot(robot: Articulation):
    """Write the default root and joint state to the simulator."""
    root_state = robot.data.default_root_state.clone()
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()
    return joint_pos


def run_simulator(sim: SimulationContext, robot: Articulation):
    sim_dt = sim.get_physics_dt()
    default_joint_pos = reset_robot(robot)
    target_joint_pos = default_joint_pos.clone()

    print("[INFO]: Setup complete.", flush=True)
    print("[INFO]: Joint names:", robot.joint_names, flush=True)
    print("[INFO]: Default joint positions:", default_joint_pos[0].detach().cpu().numpy().round(4).tolist(), flush=True)

    for count in range(args_cli.num_steps):
        phase = torch.tensor(float(count) * 0.04, device=sim.device)
        target_joint_pos[:] = default_joint_pos
        target_joint_pos[:, 0] = default_joint_pos[:, 0] + 0.35 * torch.sin(phase)
        target_joint_pos[:, 1] = default_joint_pos[:, 1] + 0.20 * torch.sin(phase * 0.7)
        target_joint_pos[:, 3] = default_joint_pos[:, 3] + 0.25 * torch.sin(phase * 0.9)
        target_joint_pos[:, 5] = default_joint_pos[:, 5] + 0.15 * torch.sin(phase * 1.1)

        robot.set_joint_position_target(target_joint_pos)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)

        if count in (0, args_cli.num_steps // 2, args_cli.num_steps - 1):
            current = robot.data.joint_pos[0].detach().cpu().numpy().round(4).tolist()
            target = target_joint_pos[0].detach().cpu().numpy().round(4).tolist()
            print(f"[INFO]: step={count:04d} target={target}", flush=True)
            print(f"[INFO]: step={count:04d} actual={current}", flush=True)

    final_joint_pos = robot.data.joint_pos[0].detach().cpu().numpy().round(4).tolist()
    print("[OK]: Franka loaded and joint position control ran successfully.", flush=True)
    print("[OK]: Final joint positions:", final_joint_pos, flush=True)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.2, 2.0, 1.6], [0.0, 0.0, 0.5])

    robot = design_scene()
    sim.reset()
    run_simulator(sim, robot)


if __name__ == "__main__":
    main()
    simulation_app.close()
