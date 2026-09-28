#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REMOTE_HOST="${METHOD_A_REMOTE_HOST:-hjy@10.120.17.111}"
REMOTE_ROOT="${METHOD_A_REMOTE_ROOT:-/home/hjy/robot_aesthetic_rl}"

echo "将方法A服务代码上传到 ${REMOTE_HOST}:${REMOTE_ROOT}"
echo "不会上传 demo、checkpoint、日志或本地运行数据。"
rsync -av --exclude='__pycache__' \
  "${PROJECT_ROOT}/aesthetic_rl/method_a/" \
  "${REMOTE_HOST}:${REMOTE_ROOT}/aesthetic_rl/method_a/"
rsync -av \
  "${PROJECT_ROOT}/aesthetic_rl/configs/method_a.yaml" \
  "${REMOTE_HOST}:${REMOTE_ROOT}/aesthetic_rl/configs/method_a.yaml"
rsync -av \
  "${PROJECT_ROOT}/aesthetic_rl/scripts/method_a_remote_install_grounding.sh" \
  "${PROJECT_ROOT}/aesthetic_rl/scripts/method_a_remote_start_artimuse.sh" \
  "${PROJECT_ROOT}/aesthetic_rl/scripts/method_a_remote_start_grounding.sh" \
  "${REMOTE_HOST}:${REMOTE_ROOT}/aesthetic_rl/scripts/"

echo "UPLOAD_OK"
