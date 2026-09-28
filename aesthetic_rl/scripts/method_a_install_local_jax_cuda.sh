#!/usr/bin/env bash
set -euo pipefail

source /home/junyi/miniconda3/etc/profile.d/conda.sh
conda activate artimuse

# Keep the project's tested JAX/JAXLIB 0.4.35 pair and add its plugin-based
# CUDA 12 runtime. NVCC is pinned because its newest namespace-only wheel is
# incompatible with JAX 0.4.35's cuda-path discovery.
python -m pip install \
  'jax-cuda12-plugin[with_cuda]==0.4.35' \
  'jax-cuda12-pjrt==0.4.35' \
  'nvidia-cuda-nvcc-cu12==12.1.105'

XLA_PYTHON_CLIENT_PREALLOCATE=false python - <<'PY'
import jax
devices = jax.devices()
print("JAX", jax.__version__, devices)
if not any(device.platform == "gpu" for device in devices):
    raise SystemExit("JAX CUDA installation completed but no GPU is visible")
PY
