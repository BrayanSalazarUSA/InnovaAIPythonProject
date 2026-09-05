#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ENV_FILE="${INNOVA_ENV_FILE:-.env.local-macos}"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  source "$ENV_FILE"
  set +a
elif [[ -f ".env" ]]; then
  set -a
  source ".env"
  set +a
fi

echo "== Local runtime config =="
PYTHON_BIN="python3"
if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
fi

"$PYTHON_BIN" - <<'PY'
from src.innova_investigation import config

checks = {
    "bridge_host": config.REMOTE_BRIDGE_HOST,
    "bridge_user": config.REMOTE_BRIDGE_USER,
    "ssh_key": str(config.SSH_KEY_PATH),
    "ssh_key_exists": config.SSH_KEY_PATH.exists(),
    "hikvision_remote_sdk": config.HIKVISION_REMOTE_SDK_DIR,
    "dahua_remote_sdk": config.DAHUA_REMOTE_SDK_DIR,
    "nvr_profiles": str(config.NVR_PROFILES_PATH),
    "nvr_profiles_exists": config.NVR_PROFILES_PATH.exists(),
}
for key, value in checks.items():
    print(f"{key}: {value}")
PY

if [[ -z "${INNOVA_REMOTE_BRIDGE_HOST:-}" || -z "${INNOVA_REMOTE_BRIDGE_USER:-}" ]]; then
  echo "ERROR: faltan INNOVA_REMOTE_BRIDGE_HOST o INNOVA_REMOTE_BRIDGE_USER."
  exit 2
fi

if [[ -z "${INNOVA_SSH_KEY_PATH:-}" || ! -f "$INNOVA_SSH_KEY_PATH" ]]; then
  echo "ERROR: no existe la llave SSH: ${INNOVA_SSH_KEY_PATH:-<empty>}"
  exit 2
fi

echo
echo "== SSH bridge check =="
ssh \
  -o BatchMode=yes \
  -o StrictHostKeyChecking=accept-new \
  -o ConnectTimeout=15 \
  -i "$INNOVA_SSH_KEY_PATH" \
  "${INNOVA_REMOTE_BRIDGE_USER}@${INNOVA_REMOTE_BRIDGE_HOST}" \
  'set -e; hostname; python3 --version; echo "hikvision_sdk=$([ -d /opt/innova/hikvision/current/lib ] && echo ok || echo missing)"; echo "dahua_sdk=$([ -d /opt/innova/dahua ] && echo ok || echo missing)"'

echo
echo "Bridge reachable. Si esto pasa, SSH y las rutas base del SDK estan OK."
