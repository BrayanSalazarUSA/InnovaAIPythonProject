#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNTIME_DIR="$ROOT_DIR/innova-ai-investigation-runtime"

cd "$RUNTIME_DIR"
exec bash scripts/bootstrap_local_env.sh
