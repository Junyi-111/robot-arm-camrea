#!/usr/bin/env bash
# Read-only hardware and vision preflight. Never enables or moves the arm.
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source /opt/ros/humble/setup.bash
source /home/junyi/handeye/install/setup.bash
source /home/junyi/miniconda3/etc/profile.d/conda.sh
conda activate artimuse
cd "${ROOT}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6${LD_PRELOAD:+:${LD_PRELOAD}}"
python -m aesthetic_rl.scripts.method_a_doctor --require-learner-gpu
exec python -m aesthetic_rl.scripts.method_a_actor --preflight --check-robot-feedback "$@"
