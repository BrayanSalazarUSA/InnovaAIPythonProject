#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PROFILE="${1:-auto}"
UNAME="$(uname -s)"

resolve_profile() {
  case "$PROFILE" in
    auto)
      if [[ "$UNAME" == "Darwin" ]]; then
        echo ".env.local-macos"
      else
        echo ".env.ubuntu"
      fi
      ;;
    macos|local|local-macos)
      echo ".env.local-macos"
      ;;
    ubuntu|server|linux)
      echo ".env.ubuntu"
      ;;
    *)
      echo "$PROFILE"
      ;;
  esac
}

ENV_FILE="$(resolve_profile)"

if [[ ! -f "$ENV_FILE" && -f "${ENV_FILE}.example" ]]; then
  cp "${ENV_FILE}.example" "$ENV_FILE"
  echo "Created $ENV_FILE from ${ENV_FILE}.example. Adjust paths before retrying."
fi

echo "Starting runtime with env file: $ENV_FILE"
INNOVA_ENV_FILE="$ENV_FILE" bash scripts/run_api.sh
