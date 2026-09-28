#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/humble/setup.bash
source /home/junyi/handeye/install/setup.bash
set -u

echo "Starting PiPER driver with auto_enable=false. This command does not enable the arm."
exec ros2 launch piper start_single_piper.launch.py \
  can_port:="${PIPER_CAN_PORT:-can0}" \
  auto_enable:=false \
  gripper_exist:=true
