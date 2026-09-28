#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source /home/junyi/miniconda3/etc/profile.d/conda.sh
conda activate artimuse
set -u
cd "${ROOT}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python -m aesthetic_rl.scripts.method_a_doctor --require-learner-gpu
exec python -m aesthetic_rl.scripts.method_a_learner "$@"
