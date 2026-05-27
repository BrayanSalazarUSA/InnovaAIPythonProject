#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p /opt/innova/hikvision /opt/innova/dahua /opt/innova/keys /opt/innova/models

HIK_ARCHIVE="$ROOT_DIR/vendor/sdk_archives/EN-HCNetSDKV6.1.9.4_build20220412_linux64.zip"
DAHUA_ARCHIVE="$ROOT_DIR/vendor/sdk_archives/General_NetSDK_Eng_Linux64_IS_V3.060.0000003.0.R.251127.tar.gz"

if [[ -f "$HIK_ARCHIVE" ]]; then
  unzip -o "$HIK_ARCHIVE" -d /opt/innova/hikvision >/dev/null
fi

if [[ -f "$DAHUA_ARCHIVE" ]]; then
  tar -xzf "$DAHUA_ARCHIVE" -C /opt/innova/dahua
fi

HIK_LIB_DIR="$(find /opt/innova/hikvision -type d -path '*/lib' | head -n 1 || true)"
if [[ -n "${HIK_LIB_DIR}" ]]; then
  HIK_ROOT_DIR="$(dirname "${HIK_LIB_DIR}")"
  ln -sfn "${HIK_ROOT_DIR}" /opt/innova/hikvision/current
fi

mkdir -p "$ROOT_DIR/resources/keys" "$ROOT_DIR/output/api_jobs"

echo "Linux SDK layout prepared."
