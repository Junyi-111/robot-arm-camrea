#!/usr/bin/env bash
# Read-only fixed-pose camera/model noise calibration. Never commands the arm.
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source /opt/ros/humble/setup.bash
source /home/junyi/handeye/install/setup.bash
source /home/junyi/miniconda3/etc/profile.d/conda.sh
conda activate artimuse
set -u
cd "${ROOT}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6${LD_PRELOAD:+:${LD_PRELOAD}}"
unset PIPER_ARM_WRITES

exec python -m aesthetic_rl.scripts.method_a_calibrate_score_noise "$@"
