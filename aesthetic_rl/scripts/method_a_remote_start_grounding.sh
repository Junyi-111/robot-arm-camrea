#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-6}"
GROUNDING_ROOT="${GROUNDING_ROOT:-/home/hjy/robot_aesthetic_rl/third_party/GroundingDINO}"
CHECKPOINT="${GROUNDING_CHECKPOINT:-/datasets/hjy/robot_aesthetic_models/GroundingDINO/groundingdino_swint_ogc.pth}"
GROUNDING_HF_HOME="${GROUNDING_HF_HOME:-/datasets/hjy/robot_aesthetic_models/GroundingDINO/huggingface}"
CONFIG="${GROUNDING_CONFIG:-${GROUNDING_ROOT}/groundingdino/config/GroundingDINO_SwinT_OGC.py}"
REMOTE_ROOT="${METHOD_A_REMOTE_ROOT:-/home/hjy/robot_aesthetic_rl}"
if [[ -z "${CONDA_SH:-}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
  else
    CONDA_SH="/home/hjy/miniconda3/etc/profile.d/conda.sh"
  fi
fi
GROUNDING_CONDA_ENV="${GROUNDING_CONDA_ENV:-groundingdino}"

test -f "${CHECKPOINT}" || { echo "缺少 Grounding DINO 权重: ${CHECKPOINT}" >&2; exit 1; }
test -f "${CONFIG}" || { echo "缺少 Grounding DINO 配置: ${CONFIG}" >&2; exit 1; }
test -d "${GROUNDING_HF_HOME}/hub" || { echo "缺少 BERT 缓存: ${GROUNDING_HF_HOME}" >&2; echo "请重新运行 Grounding DINO 安装脚本" >&2; exit 1; }
test -f "${CONDA_SH}" || { echo "缺少 conda 初始化脚本: ${CONDA_SH}" >&2; exit 1; }
source "${CONDA_SH}"
conda activate "${GROUNDING_CONDA_ENV}"

REQUIRED_GLIBCXX="GLIBCXX_3.4.29"
CONDA_LIBSTDCXX="${CONDA_PREFIX}/lib/libstdc++.so.6"
SYSTEM_LIBSTDCXX="${METHOD_A_SYSTEM_LIBSTDCXX:-/usr/lib/x86_64-linux-gnu/libstdc++.so.6}"

# Do not use `grep -q` here: with `set -o pipefail`, grep can close the pipe
# after the first match and make `strings` exit via SIGPIPE, causing a false
# "symbol not found" result even though the symbol is present.
has_required_glibcxx() {
  local library="$1"
  [[ -f "${library}" ]] && strings "${library}" | grep -Fx "${REQUIRED_GLIBCXX}" >/dev/null
}

if has_required_glibcxx "${CONDA_LIBSTDCXX}"; then
  :
elif has_required_glibcxx "${SYSTEM_LIBSTDCXX}"; then
  export LD_PRELOAD="${SYSTEM_LIBSTDCXX}${LD_PRELOAD:+:${LD_PRELOAD}}"
  echo "使用系统 libstdc++: ${SYSTEM_LIBSTDCXX}"
else
  echo "找不到包含 ${REQUIRED_GLIBCXX} 的 libstdc++.so.6" >&2
  echo "请先重新运行：bash aesthetic_rl/scripts/method_a_remote_install_grounding.sh" >&2
  exit 1
fi
if ! python - <<'PY'
import transformers
from transformers import BertModel

assert transformers.__version__ == "4.41.2", transformers.__version__
assert hasattr(BertModel, "get_head_mask")
PY
then
  echo "Transformers 版本与 Grounding DINO 不兼容。" >&2
  echo "请先重新运行：bash aesthetic_rl/scripts/method_a_remote_install_grounding.sh" >&2
  exit 1
fi
cd "${REMOTE_ROOT}"
echo "Grounding DINO 使用物理 GPU ${GPU_ID}；进程内设备名为 cuda:0"
CUDA_VISIBLE_DEVICES="${GPU_ID}" \
HF_HOME="${GROUNDING_HF_HOME}" \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHONPATH="${REMOTE_ROOT}:${GROUNDING_ROOT}" \
  python -m aesthetic_rl.method_a.remote.grounding_server \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --device cuda:0 \
  --prompt flower \
  --host 127.0.0.1 \
  --port 8001
