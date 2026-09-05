#!/usr/bin/env python3
from __future__ import annotations

import json
import sys

from innova_investigation.bridges.hikvision import REMOTE_HIKVISION_SCRIPT


REQUIRED_ENV = (
    "HIK_HOST",
    "HIK_SDK_PORT",
    "HIK_USER",
    "HIK_PASSWORD",
    "HIK_CHANNEL",
    "HIK_START_ISO",
    "HIK_END_ISO",
    "HIK_OUTPUT_NAME",
)


def main() -> int:
    import os

    missing = [key for key in REQUIRED_ENV if not os.environ.get(key)]
    if missing:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "MISSING_ENV",
                    "missing": missing,
                    "usage": (
                        "Set HIK_HOST, HIK_SDK_PORT, HIK_USER, HIK_PASSWORD, "
                        "HIK_CHANNEL, HIK_START_ISO, HIK_END_ISO, HIK_OUTPUT_NAME "
                        "and optionally HIK_SDK_ROOT/HIK_EVIDENCE_ROOT."
                    ),
                }
            )
        )
        return 2

    if not sys.platform.startswith("linux"):
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "UNSUPPORTED_PLATFORM",
                    "detail": (
                        "The local HCNetSDK archive in this project is linux64 and "
                        "loads libhcnetsdk.so. Run this command on Ubuntu/Linux, "
                        "or inside a Linux container/VM."
                    ),
                }
            )
        )
        return 2

    exec_globals = {"__name__": "__main__"}
    exec(REMOTE_HIKVISION_SCRIPT, exec_globals)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
