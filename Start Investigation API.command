#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNTIME_DIR="$ROOT_DIR/innova-ai-investigation-runtime"

cd "$RUNTIME_DIR"

if [[ ! -d ".venv" ]]; then
  bash scripts/bootstrap_local_env.sh
fi

if [[ -f ".env" ]]; then
  set -a
  source ".env"
  set +a
fi

exec "$RUNTIME_DIR/.venv/bin/python" -m innova_investigation
