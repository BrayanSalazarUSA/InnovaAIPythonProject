#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .

mkdir -p output/api_jobs resources/keys

if [[ ! -f .env ]]; then
  cp .env.example .env
fi

echo "Local environment ready at $ROOT_DIR"
