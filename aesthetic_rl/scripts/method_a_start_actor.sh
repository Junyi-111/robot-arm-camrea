#!/usr/bin/env bash
set -eo pipefail

if [[ "${PIPER_ARM_WRITES:-}" != "I_ACCEPT_REAL_MOTION" ]]; then
  echo "Real motion remains locked." >&2
  echo "After supporting the arm and checking the configured bounds, run:" >&2
  echo "export PIPER_ARM_WRITES=I_ACCEPT_REAL_MOTION" >&2
  exit 2
fi

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

exec python -m aesthetic_rl.scripts.method_a_actor --arm "$@"
