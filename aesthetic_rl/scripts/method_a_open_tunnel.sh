#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${METHOD_A_REMOTE_HOST:-hjy@10.120.17.111}"
echo "映射 ArtiMuse 18080→8000，Grounding DINO 18081→8001"
echo "保持此终端运行；Ctrl+C 只关闭隧道，不控制机械臂。"
exec ssh -N \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=15 \
  -o ServerAliveCountMax=3 \
  -L 18080:127.0.0.1:8000 \
  -L 18081:127.0.0.1:8001 \
  "${REMOTE_HOST}"
