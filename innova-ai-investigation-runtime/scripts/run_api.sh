#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f ".venv/bin/activate" ]]; then
  echo "Virtual environment not found. Bootstrapping local environment..."
  bash scripts/bootstrap_local_env.sh
fi

source .venv/bin/activate

ENV_FILE="${INNOVA_ENV_FILE:-.env}"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  source "$ENV_FILE"
  set +a
fi

python -m innova_investigation
