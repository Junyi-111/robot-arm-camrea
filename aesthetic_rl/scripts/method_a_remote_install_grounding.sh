#!/usr/bin/env bash
set -euo pipefail

# Run this on 10.120.17.111. It installs the official Grounding DINO repository
# in a dedicated conda environment and downloads its official Swin-T checkpoint.
PROJECT_ROOT="${PROJECT_ROOT:-/home/hjy/robot_aesthetic_rl}"
REPO_DIR="${GROUNDING_REPO_DIR:-${PROJECT_ROOT}/third_party/GroundingDINO}"
WEIGHT_DIR="${GROUNDING_WEIGHT_DIR:-/datasets/hjy/robot_aesthetic_models/GroundingDINO}"
GROUNDING_HF_HOME="${GROUNDING_HF_HOME:-${WEIGHT_DIR}/huggingface}"
METHOD_A_HF_ENDPOINT="${METHOD_A_HF_ENDPOINT:-https://hf-mirror.com}"
ENV_NAME="${GROUNDING_CONDA_ENV:-groundingdino}"
if [[ -z "${CONDA_SH:-}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
  else
    CONDA_SH="/home/hjy/miniconda3/etc/profile.d/conda.sh"
  fi
fi

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "Cannot find conda initialization script: ${CONDA_SH}" >&2
  exit 1
fi
source "${CONDA_SH}"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" python=3.10 pip
fi
conda activate "${ENV_NAME}"

# The server's Conda libstdc++ can be older than the system compiler used for
# the Grounding DINO extension. Prefer the Conda library when it has the needed
# symbol; otherwise preload a compatible system library. If neither copy is
# recent enough, upgrade only the C++ runtime inside this dedicated environment.
REQUIRED_GLIBCXX="GLIBCXX_3.4.29"
CONDA_LIBSTDCXX="${CONDA_PREFIX}/lib/libstdc++.so.6"
SYSTEM_LIBSTDCXX="${METHOD_A_SYSTEM_LIBSTDCXX:-/usr/lib/x86_64-linux-gnu/libstdc++.so.6}"

has_required_glibcxx() {
  local library="$1"
  [[ -f "${library}" ]] && strings "${library}" | grep -Fx "${REQUIRED_GLIBCXX}" >/dev/null
}

if ! has_required_glibcxx "${CONDA_LIBSTDCXX}" && ! has_required_glibcxx "${SYSTEM_LIBSTDCXX}"; then
  echo "Neither Conda nor system libstdc++ provides ${REQUIRED_GLIBCXX}."
  echo "Upgrading the C++ runtime only in Conda environment ${ENV_NAME} ..."
  conda install -y -n "${ENV_NAME}" -c conda-forge \
    'libstdcxx-ng>=12' 'libgcc-ng>=12'
  conda activate "${ENV_NAME}"
  CONDA_LIBSTDCXX="${CONDA_PREFIX}/lib/libstdc++.so.6"
fi

if has_required_glibcxx "${CONDA_LIBSTDCXX}"; then
  echo "Conda libstdc++ provides ${REQUIRED_GLIBCXX}"
elif has_required_glibcxx "${SYSTEM_LIBSTDCXX}"; then
  export LD_PRELOAD="${SYSTEM_LIBSTDCXX}${LD_PRELOAD:+:${LD_PRELOAD}}"
  echo "Using system libstdc++ for ${REQUIRED_GLIBCXX}: ${SYSTEM_LIBSTDCXX}"
else
  echo "The Conda C++ runtime upgrade did not provide ${REQUIRED_GLIBCXX}." >&2
  echo "Conda library checked: ${CONDA_LIBSTDCXX}" >&2
  echo "System library checked: ${SYSTEM_LIBSTDCXX}" >&2
  exit 1
fi

if ! python -c 'import torch, torchvision' >/dev/null 2>&1; then
  python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
    'torch==2.5.1+cu121' 'torchvision==0.20.1+cu121'
fi
if ! python -c 'import torch; assert torch.cuda.is_available()'; then
  echo "PyTorch cannot see CUDA. Stop here and repair the CUDA/PyTorch environment." >&2
  exit 1
fi

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc is required to build Grounding DINO CUDA operators." >&2
  exit 1
fi
export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"

# Grounding DINO predates NumPy 2 and its setup.py imports torch while
# determining build requirements. Keep a compatible NumPy and expose the
# already-installed torch by disabling PEP 517's temporary isolated env.
# Its requirement file does not constrain Transformers, but its BertModelWarper
# relies on the legacy BertModel.get_head_mask API. Pin a version that provides
# that API before installing the editable package so pip cannot select a newer,
# incompatible major/minor release.
TRANSFORMERS_VERSION="4.41.2"
python -m pip install --upgrade pip setuptools wheel \
  'numpy==1.26.4' \
  'opencv-python==4.10.0.84' \
  "transformers==${TRANSFORMERS_VERSION}"

if [[ ! -d "${REPO_DIR}/.git" ]]; then
  mkdir -p "$(dirname "${REPO_DIR}")"
  git clone https://github.com/IDEA-Research/GroundingDINO.git "${REPO_DIR}"
fi
git -C "${REPO_DIR}" fetch --tags origin
python -m pip install --no-build-isolation -e "${REPO_DIR}"
python - <<'PY'
import transformers
from transformers import BertModel

assert transformers.__version__ == "4.41.2", transformers.__version__
assert hasattr(BertModel, "get_head_mask"), "BertModel.get_head_mask is unavailable"
print("Transformers:", transformers.__version__)
print("BertModel.get_head_mask: available")
PY

mkdir -p "${WEIGHT_DIR}"
WEIGHT="${WEIGHT_DIR}/groundingdino_swint_ogc.pth"
LEGACY_WEIGHT="${REPO_DIR}/weights/groundingdino_swint_ogc.pth"
EXPECTED_SHA256="3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799"

verify_weight() {
  local candidate="$1"
  local actual
  actual="$(sha256sum "${candidate}" | awk '{print $1}')"
  if [[ "${actual}" != "${EXPECTED_SHA256}" ]]; then
    echo "Checkpoint checksum mismatch: ${candidate}" >&2
    echo "expected=${EXPECTED_SHA256}" >&2
    echo "actual=${actual}" >&2
    return 1
  fi
}

# Migrate a verified checkpoint produced by the older script instead of
# downloading 694 MB again. If both copies exist and are identical, remove
# only the verified legacy duplicate.
if [[ -f "${LEGACY_WEIGHT}" ]]; then
  verify_weight "${LEGACY_WEIGHT}"
  if [[ ! -f "${WEIGHT}" ]]; then
    mv "${LEGACY_WEIGHT}" "${WEIGHT}"
    echo "Migrated legacy checkpoint to ${WEIGHT}"
  else
    verify_weight "${WEIGHT}"
    rm -f "${LEGACY_WEIGHT}"
    echo "Removed verified duplicate legacy checkpoint: ${LEGACY_WEIGHT}"
  fi
fi

if [[ ! -f "${WEIGHT}" ]]; then
  wget -c -O "${WEIGHT}.part" \
    https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
  verify_weight "${WEIGHT}.part"
  mv "${WEIGHT}.part" "${WEIGHT}"
fi
verify_weight "${WEIGHT}"

# Grounding DINO's official Swin-T config constructs a bert-base-uncased text
# backbone before loading the detector checkpoint. Keep that dependency in the
# datasets tree as well, and pin/check the safe tensor instead of relying on a
# per-user Hugging Face cache or network access at service startup.
BERT_REPO="bert-base-uncased"
BERT_EXPECTED_SHA256="68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3"
mkdir -p "${GROUNDING_HF_HOME}"
BERT_MODEL_FILE="$(
  find -L "${GROUNDING_HF_HOME}/hub" \
    -path '*/snapshots/*/model.safetensors' -type f -print -quit 2>/dev/null || true
)"
if [[ -z "${BERT_MODEL_FILE}" ]]; then
  echo "Downloading ${BERT_REPO} into ${GROUNDING_HF_HOME}"
  echo "Hugging Face endpoint: ${METHOD_A_HF_ENDPOINT}"
  HF_HOME="${GROUNDING_HF_HOME}" \
  HF_ENDPOINT="${METHOD_A_HF_ENDPOINT}" \
  HF_HUB_DISABLE_XET=1 \
    python - <<'PY'
from transformers import AutoTokenizer, BertModel

repo = "bert-base-uncased"
AutoTokenizer.from_pretrained(repo)
BertModel.from_pretrained(repo, use_safetensors=True)
PY
  BERT_MODEL_FILE="$(
    find -L "${GROUNDING_HF_HOME}/hub" \
      -path '*/snapshots/*/model.safetensors' -type f -print -quit 2>/dev/null || true
  )"
fi
if [[ -z "${BERT_MODEL_FILE}" ]]; then
  echo "Downloaded BERT cache does not contain model.safetensors" >&2
  exit 1
fi
BERT_ACTUAL_SHA256="$(sha256sum "${BERT_MODEL_FILE}" | awk '{print $1}')"
if [[ "${BERT_ACTUAL_SHA256}" != "${BERT_EXPECTED_SHA256}" ]]; then
  echo "BERT safetensors checksum mismatch: ${BERT_MODEL_FILE}" >&2
  echo "expected=${BERT_EXPECTED_SHA256}" >&2
  echo "actual=${BERT_ACTUAL_SHA256}" >&2
  exit 1
fi

# Prove that both pieces can be loaded with networking disabled. The service
# exports the same cache/offline variables on every startup.
HF_HOME="${GROUNDING_HF_HOME}" \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
  python - <<'PY'
from transformers import AutoTokenizer, BertModel

repo = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(repo, local_files_only=True)
model = BertModel.from_pretrained(repo, local_files_only=True, use_safetensors=True)
print("BERT tokenizer:", type(tokenizer).__name__)
print("BERT hidden size:", model.config.hidden_size)
PY

python - <<'PY'
import torch
import groundingdino
from groundingdino import _C
print("Grounding DINO import:", groundingdino.__file__)
print("Grounding DINO CUDA extension:", _C.__file__)
print("CUDA device:", torch.cuda.get_device_name(0))
print("PyTorch CUDA runtime:", torch.version.cuda)
PY
git -C "${REPO_DIR}" rev-parse HEAD
echo "Installed repository: ${REPO_DIR}"
echo "Checkpoint: ${WEIGHT}"
echo "BERT cache: ${GROUNDING_HF_HOME}"
