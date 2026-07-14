"""Gymnasium wrapper around an IsaacLab Franka wrist-camera photography scene.

This module intentionally imports IsaacLab objects lazily inside the environment
constructor. Isaac Sim requires ``SimulationApp`` to be created before most
Isaac/Omniverse imports.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class IsaacFrankaWristCameraEnvCfg:
    image_height: int = 128
    image_width: int = 128
    max_episode_steps: int = 50
    control_decimation: int = 4
    action_scale: float = 0.08
    action_mode: str = "joint_delta_7d"
    ee_delta_xyz_scale: float = 0.0125
    ee_workspace_delta_low: tuple[float, float, float] = (-0.10, -0.18, -0.08)
    ee_workspace_delta_high: tuple[float, float, float] = (0.18, 0.18, 0.16)
    ik_damping: float = 0.05
    ik_max_joint_delta: float = 0.02
    ik_grid_settle_steps: int = 48
    ik_grid_position_tolerance: float = 0.004
    device: str = "cuda:0"
    output_dir: str = str(PROJECT_ROOT / "aesthetic_rl/output/isaac_franka_wrist_env")
    reward_mode: str = "dummy"
    artimuse_model_path: str = str(PROJECT_ROOT / "ArtiMuse/checkpoints/ArtiMuse")
    artimuse_max_gpu_memory: str = "5GiB"
    artimuse_max_cpu_memory: str = "32GiB"
    reward_image_name: str = "reward_latest.png"
    scene_variant: str = "residential_living_room"
    target_semantic_type: str = "class"
    target_semantic_label: str = "photography_target"
    residential_asset_scale: tuple[float, float, float] = (0.01, 0.01, 0.01)
    residential_target_mug_pos: tuple[float, float, float] = (1.25, -0.22, 0.365)
    residential_target_color_rgb: tuple[float, float, float] = (0.92, 0.66, 0.08)
    office_scene_usd_path: str = ""
    office_scene_translation: tuple[float, float, float] = (23.523, -14.534, 0.0)
    office_target_mug_usd_path: str = ""
    office_target_mug_pos: tuple[float, float, float] = (1.45, 0.0, 0.46)
    office_target_color_rgb: tuple[float, float, float] = (0.92, 0.66, 0.08)
    use_photo_home_pose: bool = True
    photo_home_joint_pos: tuple[float, float, float, float, float, float, float] = (
        0.0,
        -0.78,
        0.0,
        -2.43,
        0.0,
        2.78,
        0.741,
    )
    camera_base_pos: tuple[float, float, float] = (0.13, 0.0, -0.15)
    camera_base_rot: tuple[float, float, float, float] = (-0.70614, 0.03701, 0.03701, -0.70614)
    camera_initial_pitch_yaw: tuple[float, float] = (0.0, 0.0)
    camera_xyz_scale: float = 0.035
    camera_pitch_yaw_scale: float = 0.18
    enable_wrist_camera_pose_control: bool = False
    camera_pos_low: tuple[float, float, float] = (0.04, -0.22, -0.18)
    camera_pos_high: tuple[float, float, float] = (0.30, 0.22, 0.08)
    target_color_rgb: tuple[float, float, float] = (0.90, 0.22, 0.12)
    min_target_visibility: float = 0.006
    aesthetic_score_scale: float = 1.0
    collision_penalty_weight: float = 1.0
    out_of_view_penalty_weight: float = 0.35
    action_smoothness_penalty_weight: float = 0.02
    joint_limit_penalty_weight: float = 0.35
    enable_overview_camera: bool = False
    overview_image_height: int = 720
    overview_image_width: int = 960
    overview_eye: tuple[float, float, float] = (2.60, -1.55, 1.00)
    overview_target: tuple[float, float, float] = (1.45, 0.0, 0.22)


class IsaacFrankaWristCameraEnv(gym.Env):
    """Minimal Gymnasium env: action controls Franka joints, obs includes wrist RGB."""

    metadata = {"render_modes": []}

    def __init__(self, cfg: IsaacFrankaWristCameraEnvCfg | None = None):
        super().__init__()
        self.cfg = cfg or IsaacFrankaWristCameraEnvCfg()
        self.step_count = 0
        self.reward_provider = None
        self.last_action = None
        self.last_workspace_violation = 0.0
        self.camera_local_pos = np.array(self.cfg.camera_base_pos, dtype=np.float32)
        self.camera_pitch_yaw = np.array(self.cfg.camera_initial_pitch_yaw, dtype=np.float32)
        self._camera_translate_op = None
        self._camera_orient_op = None
        self._warned_missing_target_semantics = False
        self._ee_action_active = False

        import torch

        import isaaclab.sim as sim_utils
        from isaaclab.assets import Articulation
        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
        from isaaclab.sensors.camera import Camera, CameraCfg
        from isaaclab.sim import SimulationCfg, SimulationContext
        from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR
        from isaaclab.utils.math import compute_pose_error, matrix_from_quat, quat_inv, subtract_frame_transforms
        from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

        self.torch = torch
        self.sim_utils = sim_utils
        self.CameraCfg = CameraCfg
        self.DifferentialIKController = DifferentialIKController
        self.DifferentialIKControllerCfg = DifferentialIKControllerCfg
        self.matrix_from_quat = matrix_from_quat
        self.quat_inv = quat_inv
        self.subtract_frame_transforms = subtract_frame_transforms
        self.compute_pose_error = compute_pose_error
        self.isaac_nucleus_dir = ISAAC_NUCLEUS_DIR
        self.nvidia_nucleus_dir = NVIDIA_NUCLEUS_DIR
        self._stage = None
        self.overview_camera = None

        sim_cfg = SimulationCfg(device=self.cfg.device)
        self.sim = SimulationContext(sim_cfg)
        self.sim.set_camera_view([2.2, 1.7, 1.35], [0.25, 0.0, 0.45])

        sim_utils.create_prim("/World/Origin", "Xform", translation=(0.0, 0.0, 0.0))
        self.robot: Articulation = Articulation(
            cfg=FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="/World/Origin/Robot")
        )

        self._spawn_scene()
        self.camera: Camera = self._create_wrist_camera()
        if self.cfg.enable_overview_camera:
            self.overview_camera: Camera = self._create_overview_camera()

        self.sim.reset()
        self.sim_dt = self.sim.get_physics_dt()

        self.default_joint_pos = self._reset_robot()
        self.joint_target = self.default_joint_pos.clone()
        self.arm_joint_ids = self.robot.find_joints("panda_joint.*")[0]
        self.ee_body_id = self.robot.find_bodies("panda_hand")[0][0]
        self.ee_jacobian_id = self.ee_body_id - 1 if self.robot.is_fixed_base else self.ee_body_id
        diff_ik_cfg = self.DifferentialIKControllerCfg(
            command_type="position",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": self.cfg.ik_damping},
        )
        self.diff_ik_controller = self.DifferentialIKController(
            diff_ik_cfg,
            num_envs=1,
            device=self.sim.device,
        )

        # 9 joint positions + 9 joint velocities for Panda including fingers.
        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32),
                "image": spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.cfg.image_height, self.cfg.image_width, 3),
                    dtype=np.uint8,
                ),
                "target_mask": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.cfg.image_height, self.cfg.image_width, 1),
                    dtype=np.uint8,
                ),
            }
        )
        self.action_space = self._make_action_space()

        self._warmup_camera()
        self._reset_ee_target()

    def _make_action_space(self):
        if self.cfg.action_mode in ("camera_xyz", "ee_delta_xyz"):
            action_dim = 3
        elif self.cfg.action_mode == "camera_pose_5d":
            action_dim = 5
        elif self.cfg.action_mode == "joint_delta_7d":
            action_dim = 7
        else:
            raise ValueError(f"Unsupported action_mode: {self.cfg.action_mode}")
        return spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)

    def _material(self, color, roughness=0.65, metallic=0.0):
        return self.sim_utils.PreviewSurfaceCfg(
            diffuse_color=tuple(float(c) for c in color),
            roughness=roughness,
            metallic=metallic,
        )

    def _spawn_box(self, prim_path, size, translation, color, roughness=0.65, semantic_tags=None):
        cfg = self.sim_utils.CuboidCfg(
            size=size,
            visual_material=self._material(color, roughness=roughness),
            semantic_tags=semantic_tags,
        )
        cfg.func(prim_path, cfg, translation=translation)

    def _spawn_cylinder(
        self, prim_path, radius, height, translation, color, roughness=0.55, semantic_tags=None
    ):
        cfg = self.sim_utils.CylinderCfg(
            radius=radius,
            height=height,
            visual_material=self._material(color, roughness=roughness),
            semantic_tags=semantic_tags,
        )
        cfg.func(prim_path, cfg, translation=translation)

    def _spawn_sphere(self, prim_path, radius, translation, color, roughness=0.45, semantic_tags=None):
        cfg = self.sim_utils.SphereCfg(
            radius=radius,
            visual_material=self._material(color, roughness=roughness),
            semantic_tags=semantic_tags,
        )
        cfg.func(prim_path, cfg, translation=translation)

    def _target_semantic_tags(self):
        return [(self.cfg.target_semantic_type, self.cfg.target_semantic_label)]

    def _spawn_scene(self):
        if self.cfg.scene_variant == "office_lounge":
            self._spawn_office_lounge_scene()
            return

        ground_cfg = self.sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

        key_light = self.sim_utils.DomeLightCfg(intensity=1800.0, color=(0.92, 0.90, 0.86))
        key_light.func("/World/KeyDomeLight", key_light)

        fill_light = self.sim_utils.DistantLightCfg(intensity=850.0, color=(0.82, 0.88, 1.0), angle=0.35)
        fill_light.func("/World/FillLight", fill_light)

        if self.cfg.scene_variant == "residential_living_room":
            self._spawn_residential_living_room()
            return
        if self.cfg.scene_variant == "minimal":
            self._spawn_minimal_props()
            return
        if self.cfg.scene_variant != "tabletop_aesthetic":
            raise ValueError(f"Unsupported scene_variant: {self.cfg.scene_variant}")

        self._spawn_tabletop_scene()

    def _spawn_residential_asset(
        self,
        prim_name,
        relative_path,
        translation,
        orientation=(1.0, 0.0, 0.0, 0.0),
        scale=None,
    ):
        asset_cfg = self.sim_utils.UsdFileCfg(
            usd_path=f"{self.nvidia_nucleus_dir}/Assets/ArchVis/Residential/{relative_path}",
            scale=scale or self.cfg.residential_asset_scale,
        )
        asset_cfg.func(
            f"/World/ResidentialLivingRoom/{prim_name}",
            asset_cfg,
            translation=translation,
            orientation=orientation,
        )

    def _spawn_residential_living_room(self):
        self.sim_utils.create_prim("/World/ResidentialLivingRoom", "Xform")
        self._spawn_box(
            "/World/ResidentialLivingRoom/Floor",
            size=(4.20, 4.20, 0.05),
            translation=(1.35, 0.0, -0.025),
            color=(0.42, 0.30, 0.22),
            roughness=0.72,
        )
        self._spawn_box(
            "/World/ResidentialLivingRoom/BackWall",
            size=(0.06, 4.20, 2.60),
            translation=(3.32, 0.0, 1.30),
            color=(0.82, 0.80, 0.75),
            roughness=0.9,
        )
        self._spawn_box(
            "/World/ResidentialLivingRoom/SideWall",
            size=(4.20, 0.06, 2.60),
            translation=(1.35, 2.07, 1.30),
            color=(0.76, 0.79, 0.78),
            roughness=0.9,
        )

        self._spawn_residential_asset(
            "AreaRug",
            "Decor/Rugs/Rug_01.usd",
            translation=(1.55, 0.0, 0.004),
            scale=(0.007, 0.007, 0.007),
        )
        self._spawn_residential_asset(
            "CoffeeTable",
            "Furniture/CoffeeTables/Midtown.usd",
            translation=(1.45, 0.0, 0.01),
        )
        self._spawn_residential_asset(
            "Sofa",
            "Furniture/Sofas/Moline.usd",
            translation=(2.48, 0.0, 0.01),
            orientation=(0.7071068, 0.0, 0.0, 0.7071068),
        )
        self._spawn_residential_asset(
            "CeramicVase",
            "Decor/Vases/Earthenware01.usd",
            translation=(1.62, 0.20, 0.375),
        )
        self._spawn_residential_asset(
            "BookStack",
            "Decor/Books/BookStack_02.usd",
            translation=(1.57, -0.20, 0.375),
        )
        self._spawn_residential_asset(
            "Succulent",
            "Plants/Plant_Succulent_01.usd",
            translation=(1.22, 0.24, 0.375),
        )

        mug_cfg = self.sim_utils.UsdFileCfg(
            usd_path=f"{self.isaac_nucleus_dir}/Props/Mugs/SM_Mug_C1.usd",
            semantic_tags=self._target_semantic_tags(),
        )
        mug_cfg.func(
            "/World/ResidentialLivingRoom/PhotographyTargetMug",
            mug_cfg,
            translation=self.cfg.residential_target_mug_pos,
        )

    def _spawn_office_lounge_scene(self):
        office_usd_path = self.cfg.office_scene_usd_path or (
            f"{self.isaac_nucleus_dir}/Environments/Office/office.usd"
        )
        office_cfg = self.sim_utils.UsdFileCfg(usd_path=office_usd_path)
        office_cfg.func(
            "/World/OfficeLounge",
            office_cfg,
            translation=self.cfg.office_scene_translation,
        )

        mug_usd_path = self.cfg.office_target_mug_usd_path or (
            f"{self.isaac_nucleus_dir}/Props/Mugs/SM_Mug_C1.usd"
        )
        mug_cfg = self.sim_utils.UsdFileCfg(
            usd_path=mug_usd_path,
            semantic_tags=self._target_semantic_tags(),
        )
        mug_cfg.func(
            "/World/PhotographyTargetMug",
            mug_cfg,
            translation=self.cfg.office_target_mug_pos,
        )

    def _spawn_minimal_props(self):
        red_cube_cfg = self.sim_utils.CuboidCfg(
            size=(0.18, 0.18, 0.18),
            visual_material=self.sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.18, 0.10), metallic=0.0),
            semantic_tags=self._target_semantic_tags(),
        )
        red_cube_cfg.func("/World/RedCube", red_cube_cfg, translation=(0.52, -0.18, 0.09))

        blue_cube_cfg = self.sim_utils.CuboidCfg(
            size=(0.14, 0.14, 0.14),
            visual_material=self.sim_utils.PreviewSurfaceCfg(diffuse_color=(0.10, 0.40, 0.95), metallic=0.0),
        )
        blue_cube_cfg.func("/World/BlueCube", blue_cube_cfg, translation=(0.44, 0.18, 0.07))

    def _spawn_tabletop_scene(self):
        self._spawn_box(
            "/World/Table",
            size=(1.60, 1.18, 0.08),
            translation=(1.45, 0.0, 0.04),
            color=(0.72, 0.58, 0.43),
            roughness=0.8,
        )
        self._spawn_box(
            "/World/RightBackdropWall",
            size=(0.08, 1.35, 1.22),
            translation=(2.28, 0.0, 0.61),
            color=(0.78, 0.80, 0.82),
            roughness=0.9,
        )

        target_color = self.cfg.target_color_rgb
        self._spawn_cylinder(
            "/World/PrimaryVase",
            radius=0.075,
            height=0.34,
            translation=(1.42, -0.05, 0.25),
            color=target_color,
            roughness=0.45,
            semantic_tags=self._target_semantic_tags(),
        )
        self._spawn_sphere(
            "/World/VaseTop",
            radius=0.088,
            translation=(1.42, -0.05, 0.44),
            color=(0.96, 0.48, 0.28),
            roughness=0.4,
            semantic_tags=self._target_semantic_tags(),
        )

        self._spawn_box(
            "/World/BookStackA",
            size=(0.22, 0.16, 0.035),
            translation=(1.03, 0.34, 0.117),
            color=(0.18, 0.30, 0.62),
            roughness=0.75,
        )
        self._spawn_box(
            "/World/BookStackB",
            size=(0.20, 0.14, 0.032),
            translation=(1.03, 0.34, 0.152),
            color=(0.92, 0.78, 0.24),
            roughness=0.7,
        )
        self._spawn_cylinder(
            "/World/GlassBottle",
            radius=0.045,
            height=0.24,
            translation=(1.76, 0.30, 0.20),
            color=(0.12, 0.55, 0.48),
            roughness=0.25,
        )
        self._spawn_cylinder(
            "/World/Plate",
            radius=0.14,
            height=0.018,
            translation=(1.78, -0.34, 0.089),
            color=(0.92, 0.90, 0.84),
            roughness=0.52,
        )
        self._spawn_box(
            "/World/SmallBox",
            size=(0.16, 0.10, 0.10),
            translation=(1.03, -0.36, 0.13),
            color=(0.37, 0.22, 0.58),
            roughness=0.68,
        )
        self._spawn_sphere(
            "/World/ToyBall",
            radius=0.055,
            translation=(1.98, -0.02, 0.135),
            color=(0.10, 0.42, 0.86),
            roughness=0.38,
        )

    def _create_wrist_camera(self):
        camera_cfg = self.CameraCfg(
            prim_path="/World/Origin/Robot/panda_hand/wrist_cam",
            update_period=0,
            height=self.cfg.image_height,
            width=self.cfg.image_width,
            data_types=["rgb", "semantic_segmentation"],
            semantic_filter=f"{self.cfg.target_semantic_type}:{self.cfg.target_semantic_label}",
            colorize_semantic_segmentation=False,
            spawn=self.sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.05, 2.5),
            ),
            offset=self.CameraCfg.OffsetCfg(
                pos=self.cfg.camera_base_pos,
                rot=self.cfg.camera_base_rot,
                convention="ros",
            ),
        )
        from isaaclab.sensors.camera import Camera

        return Camera(cfg=camera_cfg)

    def _create_overview_camera(self):
        camera_cfg = self.CameraCfg(
            prim_path="/World/OverviewCamera",
            update_period=0,
            height=self.cfg.overview_image_height,
            width=self.cfg.overview_image_width,
            data_types=["rgb"],
            spawn=self.sim_utils.PinholeCameraCfg(
                focal_length=28.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 100.0),
            ),
        )
        from isaaclab.sensors.camera import Camera

        return Camera(cfg=camera_cfg)

    def _set_overview_camera_pose(self):
        if self.overview_camera is None:
            return
        self.overview_camera.set_world_poses_from_view(
            self.torch.tensor([self.cfg.overview_eye], device=self.sim.device),
            self.torch.tensor([self.cfg.overview_target], device=self.sim.device),
        )

    def _quat_multiply(self, q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=np.float64,
        )

    def _axis_angle_quat(self, axis, angle):
        axis = np.asarray(axis, dtype=np.float64)
        axis = axis / np.linalg.norm(axis)
        half = float(angle) * 0.5
        return np.array([np.cos(half), *(np.sin(half) * axis)], dtype=np.float64)

    def _set_wrist_camera_local_pose(self):
        try:
            import omni.usd
            from pxr import Gf, UsdGeom
        except Exception:
            return

        stage = self._stage
        if stage is None:
            stage = omni.usd.get_context().get_stage()
            self._stage = stage
        if stage is None:
            return

        prim = stage.GetPrimAtPath("/World/Origin/Robot/panda_hand/wrist_cam")
        if not prim.IsValid():
            return

        pitch, yaw = self.camera_pitch_yaw
        base_rot = np.asarray(self.cfg.camera_base_rot, dtype=np.float64)
        pitch_q = self._axis_angle_quat((0.0, 1.0, 0.0), pitch)
        yaw_q = self._axis_angle_quat((0.0, 0.0, 1.0), yaw)
        quat = self._quat_multiply(self._quat_multiply(base_rot, yaw_q), pitch_q)
        quat = quat / np.linalg.norm(quat)

        if self._camera_translate_op is None or self._camera_orient_op is None:
            xform = UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            self._camera_translate_op = xform.AddTranslateOp()
            self._camera_orient_op = xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)

        self._camera_translate_op.Set(Gf.Vec3d(*[float(v) for v in self.camera_local_pos]))
        self._camera_orient_op.Set(Gf.Quatd(float(quat[0]), Gf.Vec3d(float(quat[1]), float(quat[2]), float(quat[3]))))

    def _reset_robot(self):
        root_state = self.robot.data.default_root_state.clone()
        self.robot.write_root_pose_to_sim(root_state[:, :7])
        self.robot.write_root_velocity_to_sim(root_state[:, 7:])

        joint_pos = self.robot.data.default_joint_pos.clone()
        joint_vel = self.robot.data.default_joint_vel.clone()
        if self.cfg.use_photo_home_pose:
            home = self.torch.tensor(
                self.cfg.photo_home_joint_pos,
                device=self.sim.device,
                dtype=joint_pos.dtype,
            )
            joint_pos[:, :7] = home
            joint_vel[:, :7] = 0.0
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel)
        self.robot.reset()
        self.robot.set_joint_position_target(joint_pos)
        self.robot.write_data_to_sim()
        return joint_pos

    def _warmup_camera(self):
        for _ in range(3):
            if self.cfg.enable_wrist_camera_pose_control:
                self._set_wrist_camera_local_pose()
            self._set_overview_camera_pose()
            if hasattr(self, "joint_target"):
                self.robot.set_joint_position_target(self.joint_target)
                self.robot.write_data_to_sim()
            self.sim.step()
            self.robot.update(self.sim_dt)
            self.camera.update(dt=self.sim_dt)
            if self.overview_camera is not None:
                self.overview_camera.update(dt=self.sim_dt)

    def _clamp_arm_targets(self):
        limits = self.robot.data.soft_joint_pos_limits[0, :7]
        lower = limits[:, 0]
        upper = limits[:, 1]
        self.joint_target[:, :7] = self.torch.clamp(self.joint_target[:, :7], lower, upper)

    def _get_rgb(self) -> np.ndarray:
        rgb = self.camera.data.output["rgb"][0, ..., :3].detach().cpu().numpy()
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return rgb

    def _get_target_mask(self) -> np.ndarray:
        segmentation = self.camera.data.output["semantic_segmentation"][0, ..., 0]
        semantic_ids = segmentation.detach().cpu().numpy()
        info = self.camera.data.info[0].get("semantic_segmentation") or {}
        id_to_labels = info.get("idToLabels", {})

        target_ids = []
        for semantic_id, labels in id_to_labels.items():
            if isinstance(labels, dict):
                label_matches = labels.get(self.cfg.target_semantic_type) == self.cfg.target_semantic_label
            else:
                label_matches = self.cfg.target_semantic_label in str(labels)
            if label_matches:
                try:
                    target_ids.append(int(semantic_id))
                except (TypeError, ValueError):
                    continue

        if target_ids:
            mask = np.isin(semantic_ids, target_ids)
        else:
            mask = np.zeros_like(semantic_ids, dtype=bool)
            if not self._warned_missing_target_semantics:
                print(
                    "[WARN]: semantic segmentation did not report the photography_target label; "
                    f"idToLabels={id_to_labels}",
                    flush=True,
                )
                self._warned_missing_target_semantics = True
        return mask[..., None].astype(np.uint8)

    def get_overview_rgb(self) -> np.ndarray:
        if self.overview_camera is None:
            raise RuntimeError("Overview camera is disabled. Set enable_overview_camera=True in the env config.")
        self._set_overview_camera_pose()
        self.overview_camera.update(dt=self.sim_dt)
        rgb = self.overview_camera.data.output["rgb"][0, ..., :3].detach().cpu().numpy()
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return rgb

    def _get_state(self) -> np.ndarray:
        joint_pos = self.robot.data.joint_pos[0].detach().cpu().numpy()
        joint_vel = self.robot.data.joint_vel[0].detach().cpu().numpy()
        return np.concatenate([joint_pos, joint_vel]).astype(np.float32)

    def _get_obs(self) -> dict[str, np.ndarray]:
        return {
            "state": self._get_state(),
            "image": self._get_rgb(),
            "target_mask": self._get_target_mask(),
        }

    def _apply_camera_action(self, action):
        if not self.cfg.enable_wrist_camera_pose_control:
            self.last_workspace_violation = 0.0
            return

        old_pos = self.camera_local_pos.copy()
        old_pitch_yaw = self.camera_pitch_yaw.copy()

        proposed_pos = old_pos + action[:3] * self.cfg.camera_xyz_scale
        low = np.array(self.cfg.camera_pos_low, dtype=np.float32)
        high = np.array(self.cfg.camera_pos_high, dtype=np.float32)
        clipped_pos = np.clip(proposed_pos, low, high)

        pos_violation = float(np.linalg.norm(proposed_pos - clipped_pos))
        self.camera_local_pos = clipped_pos.astype(np.float32)

        if self.cfg.action_mode == "camera_pose_5d":
            proposed_angles = old_pitch_yaw + action[3:5] * self.cfg.camera_pitch_yaw_scale
            clipped_angles = np.clip(proposed_angles, -1.8, 1.8)
            angle_violation = float(np.linalg.norm(proposed_angles - clipped_angles))
            self.camera_pitch_yaw = clipped_angles.astype(np.float32)
        else:
            angle_violation = 0.0

        self.last_workspace_violation = pos_violation + angle_violation
        self._set_wrist_camera_local_pose()

    def _apply_joint_action(self, action):
        delta = self.torch.tensor(action, device=self.sim.device).view(1, 7) * self.cfg.action_scale
        self.joint_target[:, :7] = self.joint_target[:, :7] + delta
        self.joint_target[:, 7:] = self.default_joint_pos[:, 7:]
        before_clamp = self.joint_target[:, :7].clone()
        self._clamp_arm_targets()
        self.last_workspace_violation = float(
            self.torch.norm(before_clamp - self.joint_target[:, :7]).detach().cpu().item()
        )

    def _get_ee_pose_in_base(self):
        ee_pose_w = self.robot.data.body_pose_w[:, self.ee_body_id]
        root_pose_w = self.robot.data.root_pose_w
        return self.subtract_frame_transforms(
            root_pose_w[:, 0:3],
            root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3],
            ee_pose_w[:, 3:7],
        )

    def _reset_ee_target(self):
        ee_pos_b, ee_quat_b = self._get_ee_pose_in_base()
        self.ee_home_pos_b = ee_pos_b.clone()
        self.ee_target_pos_b = ee_pos_b.clone()
        self.ee_target_quat_b = ee_quat_b.clone()
        self._ee_action_active = False
        self.diff_ik_controller.reset()

    def _apply_ee_delta_action(self, action):
        delta = self.torch.tensor(action[:3], device=self.sim.device, dtype=self.ee_target_pos_b.dtype).view(1, 3)
        ee_pos_b, _ = self._get_ee_pose_in_base()
        if float(self.torch.norm(delta).detach().cpu().item()) < 1e-8:
            self.ee_target_pos_b = ee_pos_b.clone()
            self.joint_target[:, self.arm_joint_ids] = self.robot.data.joint_pos[:, self.arm_joint_ids]
            self.last_workspace_violation = 0.0
            self._ee_action_active = False
            return

        self._ee_action_active = True
        proposed = ee_pos_b + delta * self.cfg.ee_delta_xyz_scale
        low = self.ee_home_pos_b + self.torch.tensor(
            self.cfg.ee_workspace_delta_low,
            device=self.sim.device,
            dtype=proposed.dtype,
        ).view(1, 3)
        high = self.ee_home_pos_b + self.torch.tensor(
            self.cfg.ee_workspace_delta_high,
            device=self.sim.device,
            dtype=proposed.dtype,
        ).view(1, 3)
        clipped = self.torch.clamp(proposed, low, high)
        self.last_workspace_violation = float(self.torch.norm(proposed - clipped).detach().cpu().item())
        self.ee_target_pos_b = clipped

    def _set_absolute_ee_workspace_action(self, action):
        action_tensor = self.torch.tensor(
            action[:3], device=self.sim.device, dtype=self.ee_home_pos_b.dtype
        ).view(1, 3)
        action_tensor = self.torch.clamp(action_tensor, -1.0, 1.0)
        low = self.torch.tensor(
            self.cfg.ee_workspace_delta_low, device=self.sim.device, dtype=action_tensor.dtype
        ).view(1, 3)
        high = self.torch.tensor(
            self.cfg.ee_workspace_delta_high, device=self.sim.device, dtype=action_tensor.dtype
        ).view(1, 3)
        offset = self.torch.where(action_tensor >= 0.0, action_tensor * high, -action_tensor * low)
        self.ee_target_pos_b = self.ee_home_pos_b + offset
        self.last_workspace_violation = 0.0
        self._ee_action_active = True

    def _update_ik_joint_target(self):
        jacobian = self.robot.root_physx_view.get_jacobians()[
            :, self.ee_jacobian_id, :, self.arm_joint_ids
        ].clone()
        root_quat_w = self.robot.data.root_pose_w[:, 3:7]
        base_rot = self.matrix_from_quat(self.quat_inv(root_quat_w))
        jacobian[:, :3, :] = self.torch.bmm(base_rot, jacobian[:, :3, :])
        jacobian[:, 3:, :] = self.torch.bmm(base_rot, jacobian[:, 3:, :])

        ee_pos_b, ee_quat_b = self._get_ee_pose_in_base()
        self.diff_ik_controller.set_command(self.ee_target_pos_b, ee_quat=ee_quat_b)
        arm_joint_pos = self.robot.data.joint_pos[:, self.arm_joint_ids]
        raw_arm_target = self.diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, arm_joint_pos)
        joint_delta = self.torch.clamp(
            raw_arm_target - arm_joint_pos,
            min=-self.cfg.ik_max_joint_delta,
            max=self.cfg.ik_max_joint_delta,
        )
        arm_target = arm_joint_pos + joint_delta
        self.joint_target[:, self.arm_joint_ids] = arm_target
        self.joint_target[:, 7:] = self.default_joint_pos[:, 7:]
        self._clamp_arm_targets()

    def _apply_action(self, action):
        if self.cfg.action_mode == "ee_delta_xyz":
            self._apply_ee_delta_action(action)
            return
        if self.cfg.action_mode in ("camera_xyz", "camera_pose_5d"):
            self._apply_camera_action(action)
            return
        if self.cfg.action_mode == "joint_delta_7d":
            self._apply_joint_action(action)
            return
        raise ValueError(f"Unsupported action_mode: {self.cfg.action_mode}")

    def _advance_simulation(self, control_steps: int, stop_at_ee_target: bool = False):
        for control_step in range(control_steps):
            if self.cfg.action_mode == "ee_delta_xyz" and self._ee_action_active:
                self._update_ik_joint_target()
            self.robot.set_joint_position_target(self.joint_target)
            self.robot.write_data_to_sim()
            self.sim.step()
            self.robot.update(self.sim_dt)
            self.camera.update(dt=self.sim_dt)
            if self.overview_camera is not None:
                self._set_overview_camera_pose()
                self.overview_camera.update(dt=self.sim_dt)

            if stop_at_ee_target and control_step >= self.cfg.control_decimation - 1:
                ee_pos_b, _ = self._get_ee_pose_in_base()
                error = self.torch.norm(self.ee_target_pos_b - ee_pos_b, dim=-1)
                if float(error.max().detach().cpu().item()) <= self.cfg.ik_grid_position_tolerance:
                    break

    def step_absolute_ee(self, action, settle_steps: int | None = None):
        """Evaluate one absolute workspace point for an independent grid-search baseline."""
        if self.cfg.action_mode != "ee_delta_xyz":
            raise RuntimeError("Absolute EE grid evaluation requires action_mode='ee_delta_xyz'.")
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self._set_absolute_ee_workspace_action(action)
        self._advance_simulation(
            settle_steps or self.cfg.ik_grid_settle_steps,
            stop_at_ee_target=True,
        )
        self.step_count += 1
        obs = self._get_obs()
        reward, reward_info = self._compute_reward(obs["image"], obs["target_mask"], action)
        self.last_action = action.copy()
        info = {
            "reward": reward,
            "reward_mode": self.cfg.reward_mode,
            "action_mode": self.cfg.action_mode,
            "scene_variant": self.cfg.scene_variant,
            "image_mean": float(obs["image"].mean()),
            "step_count": self.step_count,
            "absolute_grid_action": action.copy(),
        }
        info.update(reward_info)
        return obs, reward, False, False, info

    def _make_reward_provider(self):
        if self.cfg.reward_mode == "dummy":
            return None
        if self.cfg.reward_mode == "artimuse":
            try:
                from aesthetic_rl.reward.artimuse_reward import ArtiMuseReward
            except Exception as exc:
                raise ImportError(
                    "Failed to import ArtiMuseReward. In the IsaacLab conda env, install the ArtiMuse runtime "
                    "dependencies first, especially bitsandbytes and accelerate."
                ) from exc
            return ArtiMuseReward(
                self.cfg.artimuse_model_path,
                device=self.cfg.device,
                max_gpu_memory=self.cfg.artimuse_max_gpu_memory,
                max_cpu_memory=self.cfg.artimuse_max_cpu_memory,
            )
        raise ValueError(f"Unsupported reward_mode: {self.cfg.reward_mode}")

    def _dummy_reward(self, rgb: np.ndarray) -> float:
        return float(rgb.mean() / 255.0)

    def _artimuse_reward(self, rgb: np.ndarray) -> float:
        if self.reward_provider is None:
            self.reward_provider = self._make_reward_provider()
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        image_path = os.path.join(self.cfg.output_dir, self.cfg.reward_image_name)
        from PIL import Image

        Image.fromarray(rgb).save(image_path)
        return float(self.reward_provider.score_image(image_path))

    def _compute_aesthetic_score(self, rgb: np.ndarray) -> float:
        if self.cfg.reward_mode == "dummy":
            return self._dummy_reward(rgb)
        if self.cfg.reward_mode == "artimuse":
            return self._artimuse_reward(rgb)
        raise ValueError(f"Unsupported reward_mode: {self.cfg.reward_mode}")

    def _target_visibility(self, target_mask: np.ndarray) -> tuple[float, float]:
        mask = np.asarray(target_mask, dtype=bool).squeeze(-1)
        visibility = float(mask.mean())
        if not np.any(mask):
            return visibility, 1.0
        ys, xs = np.nonzero(mask)
        centroid = np.array(
            [xs.mean() / max(mask.shape[1] - 1, 1), ys.mean() / max(mask.shape[0] - 1, 1)]
        )
        edge_distance = float(np.min([centroid[0], 1.0 - centroid[0], centroid[1], 1.0 - centroid[1]]))
        center_quality = float(np.clip(edge_distance / 0.35, 0.0, 1.0))
        return visibility, center_quality

    def _joint_limit_penalty(self) -> float:
        limits = self.robot.data.soft_joint_pos_limits[0, :7].detach().cpu().numpy()
        joint_pos = self.robot.data.joint_pos[0, :7].detach().cpu().numpy()
        lower_margin = joint_pos - limits[:, 0]
        upper_margin = limits[:, 1] - joint_pos
        margin = np.minimum(lower_margin, upper_margin)
        return float(np.mean(np.clip(0.08 - margin, 0.0, 0.08) / 0.08))

    def _compute_reward(
        self, rgb: np.ndarray, target_mask: np.ndarray, action: np.ndarray | None
    ) -> tuple[float, dict]:
        raw_score = self._compute_aesthetic_score(rgb)
        normalized_score = raw_score / 100.0 if self.cfg.reward_mode == "artimuse" else raw_score
        normalized_score *= self.cfg.aesthetic_score_scale

        visibility, center_quality = self._target_visibility(target_mask)
        visibility_ratio = float(np.clip(visibility / max(self.cfg.min_target_visibility, 1e-6), 0.0, 1.0))
        task_aesthetic_score = normalized_score * visibility_ratio
        out_of_view_penalty = 0.0
        if visibility < self.cfg.min_target_visibility:
            gap = (self.cfg.min_target_visibility - visibility) / max(self.cfg.min_target_visibility, 1e-6)
            out_of_view_penalty = self.cfg.out_of_view_penalty_weight * float(np.clip(gap, 0.0, 1.0))
        out_of_view_penalty += self.cfg.out_of_view_penalty_weight * 0.25 * (1.0 - center_quality)

        collision_penalty = self.cfg.collision_penalty_weight * float(self.last_workspace_violation > 1e-6)
        joint_limit_penalty = self.cfg.joint_limit_penalty_weight * self._joint_limit_penalty()

        if action is None or self.last_action is None:
            smoothness_penalty = 0.0
        else:
            smoothness_penalty = self.cfg.action_smoothness_penalty_weight * float(
                np.mean(np.square(action - self.last_action))
            )

        total_penalty = collision_penalty + out_of_view_penalty + smoothness_penalty + joint_limit_penalty
        reward = float(task_aesthetic_score - total_penalty)
        info = {
            "aesthetic_score": float(raw_score),
            "normalized_aesthetic_score": float(normalized_score),
            "task_aesthetic_score": float(task_aesthetic_score),
            "collision_penalty": float(collision_penalty),
            "out_of_view_penalty": float(out_of_view_penalty),
            "action_smoothness_penalty": float(smoothness_penalty),
            "joint_limit_penalty": float(joint_limit_penalty),
            "total_penalty": float(total_penalty),
            "target_visibility": float(visibility),
            "target_visibility_ratio": visibility_ratio,
            "target_center_quality": float(center_quality),
            "camera_local_pos": self.camera_local_pos.copy(),
            "camera_pitch_yaw": self.camera_pitch_yaw.copy(),
        }
        if hasattr(self, "ee_target_pos_b"):
            ee_pos_b, ee_quat_b = self._get_ee_pose_in_base()
            pos_error, rot_error = self.compute_pose_error(
                ee_pos_b,
                ee_quat_b,
                self.ee_target_pos_b,
                self.ee_target_quat_b,
                rot_error_type="axis_angle",
            )
            info["ee_pos_b"] = ee_pos_b[0].detach().cpu().numpy().copy()
            info["ee_target_pos_b"] = self.ee_target_pos_b[0].detach().cpu().numpy().copy()
            info["ee_position_error"] = float(self.torch.norm(pos_error, dim=-1)[0].detach().cpu().item())
            info["ee_orientation_error"] = float(self.torch.norm(rot_error, dim=-1)[0].detach().cpu().item())
            joint_error = self.robot.data.joint_pos[:, self.arm_joint_ids] - self.joint_target[:, self.arm_joint_ids]
            info["joint_tracking_error"] = float(
                self.torch.max(self.torch.abs(joint_error)).detach().cpu().item()
            )
        return reward, info

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.step_count = 0
        self.default_joint_pos = self._reset_robot()
        self.joint_target = self.default_joint_pos.clone()
        self.camera_local_pos = np.array(self.cfg.camera_base_pos, dtype=np.float32)
        self.camera_pitch_yaw = np.array(self.cfg.camera_initial_pitch_yaw, dtype=np.float32)
        self.last_action = None
        self.last_workspace_violation = 0.0
        self._warmup_camera()
        self._reset_ee_target()
        obs = self._get_obs()
        if options and options.get("skip_reward", False):
            return obs, {
                "reward": 0.0,
                "reward_mode": self.cfg.reward_mode,
                "action_mode": self.cfg.action_mode,
                "scene_variant": self.cfg.scene_variant,
                "image_mean": float(obs["image"].mean()),
            }
        reward, reward_info = self._compute_reward(obs["image"], obs["target_mask"], action=None)
        info = {
            "reward": reward,
            "reward_mode": self.cfg.reward_mode,
            "action_mode": self.cfg.action_mode,
            "scene_variant": self.cfg.scene_variant,
            "image_mean": float(obs["image"].mean()),
        }
        info.update(reward_info)
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        self._apply_action(action)

        self._advance_simulation(self.cfg.control_decimation)

        self.step_count += 1
        obs = self._get_obs()
        reward, reward_info = self._compute_reward(obs["image"], obs["target_mask"], action)
        self.last_action = action.copy()
        terminated = False
        truncated = self.step_count >= self.cfg.max_episode_steps
        info = {
            "reward": reward,
            "reward_mode": self.cfg.reward_mode,
            "action_mode": self.cfg.action_mode,
            "scene_variant": self.cfg.scene_variant,
            "image_mean": float(obs["image"].mean()),
            "step_count": self.step_count,
        }
        info.update(reward_info)
        return obs, reward, terminated, truncated, info

    def close(self):
        os.makedirs(self.cfg.output_dir, exist_ok=True)


def make_grid_actions(action_space: spaces.Box, levels: int = 3, max_actions: int | None = None):
    """Create a small deterministic action grid for first-stage baseline sweeps."""
    values = np.linspace(float(action_space.low[0]), float(action_space.high[0]), levels, dtype=np.float32)
    actions = [np.asarray(combo, dtype=np.float32) for combo in product(values, repeat=action_space.shape[0])]
    if max_actions is not None and max_actions < len(actions):
        indices = np.linspace(0, len(actions) - 1, max_actions, dtype=int)
        actions = [actions[index] for index in indices]
    return actions
