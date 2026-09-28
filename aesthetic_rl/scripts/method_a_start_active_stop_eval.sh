#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "Starting frozen checkpoint-1731 evaluation with ACTIVE STOP v3."
echo "The arm may return slowly to a recorded pose; support/observe it continuously."
echo "A confirmed STOP ends the episode and HOLDS; it never disables the arm."
exec "${ROOT}/aesthetic_rl/scripts/method_a_start_actor.sh" \
  --eval-only --active-stop "$@"
