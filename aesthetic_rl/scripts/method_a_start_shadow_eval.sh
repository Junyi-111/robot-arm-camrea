#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "Starting frozen checkpoint-1731 evaluation with passive SHADOW STOP v2."
echo "WOULD_RETURN_TO_BEST is logged only; no return motion or episode stop occurs."
exec "${ROOT}/aesthetic_rl/scripts/method_a_start_actor.sh" \
  --eval-only --shadow-stop "$@"
