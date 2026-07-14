# IsaacLab Aesthetic Photography Stage 1

This stage keeps the current IsaacLab + HIL-SERL/SAC direction and only expands
the existing Franka wrist-camera environment.

## Environment

`aesthetic_rl/envs/isaac_franka_wrist_env.py` now supports:

- `scene_variant=residential_living_room` (default): a living room assembled
  from official Isaac Sim 5.1 ArchVis Residential USD assets: Moline sofa,
  Midtown coffee table, rug, ceramic vase, books, succulent, and a yellow mug
  photography target.
- `scene_variant=office_lounge`: the official furnished
  `Environments/Office/office.usd` lounge, retained as a heavier comparison scene.
- `scene_variant=tabletop_aesthetic`: Franka, wrist camera, table, backdrop wall,
  primary vase-like target, books, bottle, plate, box, toy ball, and adjustable
  primitive lighting.
- `action_mode=camera_xyz`: first-stage 3D camera offset control.
- `action_mode=camera_pose_5d`: first-stage x/y/z plus pitch/yaw camera control.
- `action_mode=joint_delta_7d`: previous joint-delta behavior.
- `action_mode=ee_delta_xyz`: recommended task-space control. The action is a
  bounded end-effector position delta; a damped-least-squares Differential IK
  controller converts it to Franka joint targets. Each command is relative to
  the measured end-effector pose, and per-physics-step joint deltas are clipped
  to prevent target runaway and unstable IK jumps. Zero actions bypass IK and
  hold the measured joints exactly. Wrist orientation drift is recorded and
  must pass the smoke-test gate before expanding this position-only controller.
- Franka uses IsaacLab's `FRANKA_PANDA_HIGH_PD_CFG`, which is the upstream
  task-space-control configuration. Reset and camera warmup explicitly write
  the current joint targets so the arm cannot sag under gravity before the
  first environment action.
- `reward_mode=dummy`: fast smoke-test reward.
- `reward_mode=artimuse`: saves the wrist RGB frame and scores it with ArtiMuse.
- The active photography subject is tagged as `class:photography_target`. The
  wrist camera returns an uncolorized semantic segmentation buffer and exposes
  a binary `target_mask` observation, so target visibility does not depend on
  object color.

The observation dict contains `state`, `image`, and `target_mask`. The returned
`info` dict includes `aesthetic_score`, `normalized_aesthetic_score`,
`task_aesthetic_score`, `collision_penalty`, `out_of_view_penalty`,
`action_smoothness_penalty`, `joint_limit_penalty`, `total_penalty`,
`target_visibility`, `target_visibility_ratio`, and camera trajectory fields.
ArtiMuse contributes to the task reward only in proportion to semantic target
visibility.

## Smoke Test

Open the complete scene from the project root:

```bash
conda activate isaaclab51
cd /home/junyi/robot_aesthetic_rl
TERM=xterm PYTHONUNBUFFERED=1 /home/junyi/IsaacLab_232_sim51/isaaclab.sh \
  -p aesthetic_rl/scripts/view_isaac_aesthetic_scene.py \
  --enable_cameras \
  --scene_variant residential_living_room
```

The first launch downloads the referenced Residential materials and textures
from the Isaac Sim 5.1 asset server. Keep the network connected until the scene
is fully loaded.

Render the current raised home pose and nearby alternatives:

```bash
TERM=xterm PYTHONUNBUFFERED=1 /home/junyi/IsaacLab_232_sim51/isaaclab.sh \
  -p aesthetic_rl/scripts/scan_franka_photo_home_pose.py \
  --headless --enable_cameras \
  --scene_variant residential_living_room
```

The scan saves paired overview and wrist images under
`aesthetic_rl/output/franka_photo_home_scan/`.

Run a wrist-camera smoke test from the project root:

```bash
conda activate isaaclab51
cd /home/junyi/robot_aesthetic_rl
TERM=xterm PYTHONUNBUFFERED=1 /home/junyi/IsaacLab_232_sim51/isaaclab.sh \
  -p aesthetic_rl/scripts/test_isaac_franka_wrist_env.py \
  --headless --enable_cameras \
  --scene_variant residential_living_room \
  --action_mode ee_delta_xyz \
  --reward_mode dummy
```

The smoke test also writes `reset_mask.png`, `reset_overlay.png`,
`last_mask.png`, and `last_overlay.png`. The mask must cover only the selected
photography target; the overlay marks those pixels in green.

Switch to ArtiMuse scoring only after the dummy rollout renders correctly:

```bash
TERM=xterm PYTHONUNBUFFERED=1 /home/junyi/IsaacLab_232_sim51/isaaclab.sh \
  -p aesthetic_rl/scripts/test_isaac_franka_wrist_env.py \
  --headless --enable_cameras \
  --scene_variant residential_living_room \
  --action_mode ee_delta_xyz \
  --reward_mode artimuse
```

## Baselines Before RL

Run fixed-view, random-policy, and grid-search baselines before SAC/HIL-SERL:

```bash
TERM=xterm PYTHONUNBUFFERED=1 /home/junyi/IsaacLab_232_sim51/isaaclab.sh \
  -p aesthetic_rl/scripts/run_isaac_aesthetic_baselines.py \
  --headless --enable_cameras \
  --scene_variant residential_living_room \
  --reward_mode artimuse \
  --action_mode ee_delta_xyz \
  --episodes 3 \
  --steps 20
```

For `action_mode=ee_delta_xyz`, grid search evaluates independent absolute
workspace points. Every candidate resets to the same home pose, moves to one
normalized Cartesian grid coordinate, waits for IK convergence, and receives
one reward evaluation. Fixed and random remain sequential trajectory baselines.

Outputs are written to:

- `aesthetic_rl/output/isaac_aesthetic_baselines/summary.json`
- `aesthetic_rl/output/isaac_aesthetic_baselines/metrics.csv`
- `aesthetic_rl/output/isaac_aesthetic_baselines/best_images/`

Each episode saves `*_best.png` by final task reward and a separate
`*_highest_aesthetic.png` diagnostic. A high-scoring image with an empty target
mask is reward hacking and must not become `*_best.png`.

Use these values as the minimum comparison set for RL: average aesthetic score,
best image, final score minus initial score, reward curve, and camera trajectory.

## SAC/HIL-SERL Gate

Only start policy training after:

- the residential living room renders with the target visible,
- `reward_mode=artimuse` produces stable scores,
- the fixed/random/grid baselines have saved metrics,
- the RL objective can be checked against `final_score > initial_score` and
  against the baseline scores.
