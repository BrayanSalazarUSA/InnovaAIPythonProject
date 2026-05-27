from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import quote

import cv2
import numpy as np
import requests
from requests.auth import HTTPBasicAuth, HTTPDigestAuth


def build_rtsp_url(
    *,
    host: str,
    rtsp_port: int,
    username: str,
    password: str,
    logical_channel: int,
    stream_variant: str = "sub",
) -> str:
    stream = 1 if str(stream_variant or "").lower() == "sub" else 0
    encoded_user = quote(username, safe="")
    encoded_password = quote(password, safe="")
    return (
        f"rtsp://{encoded_user}:{encoded_password}@{host}:{rtsp_port}"
        f"/unicast/c{logical_channel}/s{stream}/live"
    )


def _validate_snapshot_payload(payload: bytes, *, source: str) -> None:
    if not payload or len(payload) < 2048:
        raise RuntimeError(f"{source}: snapshot demasiado pequeño ({len(payload or b'')} bytes).")
    header = payload[:128].lstrip().lower()
    if header.startswith(b"<") or b"<html" in header or b"<?xml" in header:
        raise RuntimeError(f"{source}: el NVR devolvió HTML/XML en vez de una imagen.")

    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise RuntimeError(f"{source}: no se pudo decodificar la imagen.")
    height, width = image.shape[:2]
    if width < 80 or height < 60:
        raise RuntimeError(f"{source}: imagen inválida ({width}x{height}).")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if float(gray.std()) < 3.5:
        raise RuntimeError(f"{source}: imagen sin detalle útil.")
    if float((gray < 12).mean()) > 0.92:
        raise RuntimeError(f"{source}: imagen casi totalmente negra.")


def _get_with_auth_fallback(url: str, username: str, password: str, timeout_seconds: int) -> requests.Response:
    response = requests.get(
        url,
        auth=HTTPDigestAuth(username, password),
        timeout=timeout_seconds,
    )
    if response.status_code == 401:
        response = requests.get(
            url,
            auth=HTTPBasicAuth(username, password),
            timeout=timeout_seconds,
        )
    response.raise_for_status()
    return response


def fetch_snapshot_bytes_via_lapi(
    *,
    host: str,
    http_port: int,
    username: str,
    password: str,
    logical_channel: int,
    stream_variant: str = "sub",
    timeout_seconds: int = 12,
) -> bytes:
    stream = 1 if str(stream_variant or "").lower() == "sub" else 0
    candidates = [
        f"http://{host}:{http_port}/LAPI/V1.0/Channels/{logical_channel}/Media/Video/Streams/{stream}/Snapshot",
        f"http://{host}:{http_port}/LAPI/V1.0/Channels/{logical_channel}/Media/Video/Streams/{stream}/Snapshot?format=JPEG",
        f"http://{host}:{http_port}/LAPI/V1.0/Channels/{logical_channel}/Media/Video/Streams/{stream}/Snapshot/",
    ]

    errors: list[str] = []
    for url in candidates:
        try:
            response = _get_with_auth_fallback(url, username, password, timeout_seconds)
            payload = response.content
            _validate_snapshot_payload(payload, source=f"Uniview LAPI c{logical_channel}s{stream}")
            return payload
        except Exception as exc:
            errors.append(f"{url}: {exc}")

    raise RuntimeError(" | ".join(errors))


def fetch_snapshot_bytes_via_rtsp(
    *,
    host: str,
    rtsp_port: int,
    username: str,
    password: str,
    logical_channel: int,
    stream_variant: str = "sub",
    transport: str = "tcp",
    timeout_seconds: int = 18,
) -> bytes:
    ffmpeg_bin = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    if not Path(ffmpeg_bin).exists():
        raise FileNotFoundError("No encontré ffmpeg para capturar miniaturas Uniview.")

    rtsp_url = build_rtsp_url(
        host=host,
        rtsp_port=rtsp_port,
        username=username,
        password=password,
        logical_channel=logical_channel,
        stream_variant=stream_variant,
    )
    with tempfile.TemporaryDirectory(prefix="innova-unv-rtsp-") as temp_dir:
        output_path = Path(temp_dir) / f"channel_{logical_channel}_{stream_variant}.jpg"
        command = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-rtsp_transport",
            "udp" if str(transport).lower() == "udp" else "tcp",
            "-timeout",
            str(max(3, int(timeout_seconds)) * 1_000_000),
            "-analyzeduration",
            "8000000",
            "-probesize",
            "8000000",
            "-fflags",
            "+genpts+discardcorrupt",
            "-i",
            rtsp_url,
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(output_path),
        ]
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            timeout=max(10, timeout_seconds + 5),
        )
        if result.returncode != 0 or not output_path.exists():
            raise RuntimeError(result.stderr.strip() or "No se pudo capturar snapshot RTSP Uniview.")
        payload = output_path.read_bytes()
        _validate_snapshot_payload(payload, source=f"Uniview RTSP {logical_channel} {stream_variant}")
        return payload


def list_channels(
    *,
    host: str,
    http_port: int,
    rtsp_port: int,
    username: str,
    password: str,
    timeout_seconds: int = 5,
    max_channels: int | None = None,
) -> list[dict[str, Any]]:
    channel_limit = max_channels or int(os.getenv("UNIVIEW_MAX_CHANNELS", "32"))
    channels: list[dict[str, Any]] = []
    consecutive_misses = 0

    for channel in range(1, max(1, channel_limit) + 1):
        online = False
        detail = "-"
        try:
            fetch_snapshot_bytes_via_lapi(
                host=host,
                http_port=http_port,
                username=username,
                password=password,
                logical_channel=channel,
                stream_variant="sub",
                timeout_seconds=timeout_seconds,
            )
            online = True
            detail = "LAPI snapshot"
            consecutive_misses = 0
        except Exception as exc:
            detail = str(exc).split(" | ", 1)[0]
            consecutive_misses += 1

        if online:
            channels.append(
                {
                    "id": channel,
                    "sdk_channel": channel,
                    "name": f"Canal {channel}",
                    "vendor": "Uniview",
                    "online": True,
                    "status_label": "Online",
                    "detect_result": detail,
                    "ip_address": "-",
                    "password_status": "-",
                }
            )

        if channel >= 16 and consecutive_misses >= 8:
            break

    channels.sort(key=lambda item: int(item["id"]) if isinstance(item["id"], int) else 9999)
    return channels
