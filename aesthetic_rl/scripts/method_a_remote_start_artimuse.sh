#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-7}"
REMOTE_ROOT="${METHOD_A_REMOTE_ROOT:-/home/hjy/robot_aesthetic_rl}"
MODEL_PATH="${ARTIMUSE_MODEL_PATH:-/datasets/hjy/robot_aesthetic_models/ArtiMuse}"
if [[ -z "${CONDA_SH:-}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
  else
    CONDA_SH="/home/hjy/miniconda3/etc/profile.d/conda.sh"
  fi
fi
# The existing server environment used by the validated ArtiMuse/Isaac runs is
# isaaclab51. The local robot computer uses a different environment named
# artimuse; do not create or assume that local environment on the GPU server.
ARTIMUSE_CONDA_ENV="${ARTIMUSE_CONDA_ENV:-isaaclab51}"

test -f "${CONDA_SH}" || { echo "缺少 conda 初始化脚本: ${CONDA_SH}" >&2; exit 1; }
source "${CONDA_SH}"
if ! conda env list | awk '{print $1}' | grep -qx "${ARTIMUSE_CONDA_ENV}"; then
  echo "服务器缺少 ArtiMuse 环境: ${ARTIMUSE_CONDA_ENV}" >&2
  exit 1
fi
conda activate "${ARTIMUSE_CONDA_ENV}"
cd "${REMOTE_ROOT}"
echo "ArtiMuse Conda 环境: ${ARTIMUSE_CONDA_ENV}"
echo "ArtiMuse 使用物理 GPU ${GPU_ID}；进程内设备名为 cuda:0"
CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONPATH="${REMOTE_ROOT}" \
  python -m aesthetic_rl.method_a.remote.artimuse_server \
  --project "${REMOTE_ROOT}" \
  --model "${MODEL_PATH}" \
  --device cuda:0 \
  --host 127.0.0.1 \
  --port 8000
