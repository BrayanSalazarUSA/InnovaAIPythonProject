#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$ROOT_DIR"

if [[ ! -d ".venv" ]]; then
  bash scripts/bootstrap_local_env.sh
fi

if [[ -f ".env" ]]; then
  set -a
  source ".env"
  set +a
fi

exec "$ROOT_DIR/.venv/bin/python" -m innova_investigation
