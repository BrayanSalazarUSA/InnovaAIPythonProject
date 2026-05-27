from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import date, time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests
from requests.auth import HTTPDigestAuth

from .. import config

DEFAULT_CLASSES_DIR = Path("/tmp/dahua-sdk-test/classes")

CHANNEL_COUNT_HINT_PATTERN = re.compile(
    r"^(?P<key>videoInputChannels|videoInChannel|channelNum|nChannel|VideoInChannel)\s*=\s*(?P<count>\d+)\s*$",
    re.IGNORECASE,
)

RECORD_LINE_PATTERN = re.compile(
    r"Record\s+(?P<index>\d+)\s+->\s+channel=(?P<channel>\d+)\s+start=(?P<start>[\d:\-\s]+)\s+end=(?P<end>[\d:\-\s]+)\s+size=(?P<size>\d+)"
)
CHANNEL_COUNT_PATTERN = re.compile(r"Channel count from device info:\s*(?P<count>\d+)")
RECORD_COUNT_PATTERN = re.compile(r"Record count:\s*(?P<count>\d+)")
CHANNEL_TITLE_PATTERN = re.compile(r"table\.ChannelTitle\[(?P<index>\d+)\]\.Name=(?P<name>.*)")
DOWNLOAD_OK_PATTERN = re.compile(
    r"DOWNLOAD_OK\s+path=(?P<path>.+?)\s+size=(?P<size>\d+)\s+format=(?P<format>\w+)",
)


def find_dahua_sdk_root() -> Path:
    configured = os.getenv("INNOVA_DAHUA_JAVA_SDK_ROOT", "").strip()
    if configured:
        configured_path = Path(configured).expanduser()
        if configured_path.exists():
            return configured_path

    candidates = sorted(config.PROJECT_ROOT.glob("General_NetSDK*JAVA_*"))
    if not candidates:
        candidates = sorted(config.VENDOR_DIR.glob("General_NetSDK*JAVA_*"))
    if not candidates:
        raise FileNotFoundError("No encontré el SDK Dahua Java. Configura INNOVA_DAHUA_JAVA_SDK_ROOT o coloca el SDK en vendor/.")
    return candidates[0]


@dataclass(slots=True)
class DahuaBridgeSettings:
    sdk_root: Path | None = None
    java_bin: str = "java"
    javac_bin: str = "javac"
    classes_dir: Path = DEFAULT_CLASSES_DIR


def _dahua_get(
    *,
    host: str,
    port: int,
    username: str,
    password: str,
    path: str,
    timeout_seconds: int = 12,
) -> requests.Response:
    url = f"http://{host}:{port}{path}"
    response = requests.get(
        url,
        auth=HTTPDigestAuth(username, password),
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    return response


def get_channel_count_via_http(
    *,
    host: str,
    http_port: int,
    username: str,
    password: str,
    timeout_seconds: int = 8,
) -> int:
    """
    Best-effort: algunos NVRs exponen el total de canales en endpoints HTTP.
    Si este equipo no lo expone, esta funcion falla y hacemos fallback a SDK (remoto).
    """
    for path in [
        "/cgi-bin/magicBox.cgi?action=getSystemInfo",
        "/cgi-bin/magicBox.cgi?action=getSystemInfoEx",
    ]:
        try:
            response = _dahua_get(
                host=host,
                port=http_port,
                username=username,
                password=password,
                path=path,
                timeout_seconds=timeout_seconds,
            )
        except Exception:
            continue
        for line in response.text.splitlines():
            match = CHANNEL_COUNT_HINT_PATTERN.match(line.strip())
            if not match:
                continue
            count = int(match.group("count"))
            if count > 0:
                return count

    raise RuntimeError("No pude determinar el número total de canales vía HTTP.")


def get_channel_titles_via_http(
    *,
    host: str,
    http_port: int,
    username: str,
    password: str,
    timeout_seconds: int = 12,
) -> dict[int, str]:
    response = _dahua_get(
        host=host,
        port=http_port,
        username=username,
        password=password,
        path="/cgi-bin/configManager.cgi?action=getConfig&name=ChannelTitle",
        timeout_seconds=timeout_seconds,
    )
    titles: dict[int, str] = {}
    for line in response.text.splitlines():
        match = CHANNEL_TITLE_PATTERN.match(line.strip())
        if not match:
            continue
        index = int(match.group("index")) + 1
        name = match.group("name").strip()
        titles[index] = name or f"Canal {index}"
    return titles


def fetch_snapshot_bytes_via_http(
    *,
    host: str,
    http_port: int,
    username: str,
    password: str,
    channel: int,
    timeout_seconds: int = 12,
) -> bytes:
    response = _dahua_get(
        host=host,
        port=http_port,
        username=username,
        password=password,
        path=f"/cgi-bin/snapshot.cgi?channel={channel}",
        timeout_seconds=timeout_seconds,
    )
    return response.content


def list_channels_via_http(
    *,
    host: str,
    http_port: int,
    username: str,
    password: str,
    channel_count: int,
    timeout_seconds: int = 10,
) -> list[dict[str, Any]]:
    """
    Devuelve una lista completa (1..N) usando:
    - N via systemInfo (rápido).
    - nombres via ChannelTitle si están disponibles.
    """
    titles: dict[int, str] = {}
    try:
        titles = get_channel_titles_via_http(
            host=host,
            http_port=http_port,
            username=username,
            password=password,
            timeout_seconds=timeout_seconds,
        )
    except Exception:
        titles = {}

    channels: list[dict[str, Any]] = []
    for channel_id in range(1, channel_count + 1):
        title = titles.get(channel_id, "").strip()
        channels.append(
            {
                "id": channel_id,
                "sdk_channel": channel_id - 1,
                "name": title or f"Canal {channel_id}",
                "status_label": "Disponible",
                "online": True,
                "ip_address": "-",
                "vendor": "Dahua",
            }
        )
    return channels


def _copy_native_libs(settings: DahuaBridgeSettings) -> None:
    libs_dir = settings.sdk_root / "libs" / "mac64"
    for library in libs_dir.glob("*.dylib"):
        shutil.copy2(library, Path("/tmp") / library.name)


def _ensure_compiled(settings: DahuaBridgeSettings) -> None:
    smoke_source = settings.sdk_root / "src" / "main" / "java" / "com" / "netsdk" / "demo" / "custom" / "DahuaInvestigationSmokeTest.java"
    download_source = settings.sdk_root / "src" / "main" / "java" / "com" / "netsdk" / "demo" / "custom" / "DahuaDownloadByTimeHelper.java"
    download_module_source = settings.sdk_root / "src" / "main" / "java" / "com" / "netsdk" / "demo" / "module" / "DownLoadRecordModule.java"
    login_module_source = settings.sdk_root / "src" / "main" / "java" / "com" / "netsdk" / "demo" / "module" / "LoginModule.java"
    smoke_class = settings.classes_dir / "com" / "netsdk" / "demo" / "custom" / "DahuaInvestigationSmokeTest.class"
    download_class = settings.classes_dir / "com" / "netsdk" / "demo" / "custom" / "DahuaDownloadByTimeHelper.class"
    latest_source_mtime = max(
        smoke_source.stat().st_mtime,
        download_source.stat().st_mtime,
        download_module_source.stat().st_mtime,
        login_module_source.stat().st_mtime,
    )
    if (
        smoke_class.exists()
        and download_class.exists()
        and smoke_class.stat().st_mtime >= latest_source_mtime
        and download_class.stat().st_mtime >= latest_source_mtime
    ):
        return

    settings.classes_dir.mkdir(parents=True, exist_ok=True)
    sources = [str(path) for path in (settings.sdk_root / "src" / "main" / "java").rglob("*.java")]
    compile_cmd = [
        settings.javac_bin,
        "-encoding",
        "UTF-8",
        "-d",
        str(settings.classes_dir),
        "-cp",
        str(settings.sdk_root / "libs" / "jna.jar"),
        *sources,
    ]
    result = subprocess.run(compile_cmd, cwd=settings.sdk_root, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "No se pudo compilar el helper Dahua.")


def _parse_probe_output(output: str) -> dict[str, Any]:
    channel_count = 0
    record_count = 0
    records: list[dict[str, Any]] = []

    for line in output.splitlines():
        channel_match = CHANNEL_COUNT_PATTERN.search(line)
        if channel_match:
            channel_count = int(channel_match.group("count"))
            continue

        record_count_match = RECORD_COUNT_PATTERN.search(line)
        if record_count_match:
            record_count = int(record_count_match.group("count"))
            continue

        record_match = RECORD_LINE_PATTERN.search(line)
        if record_match:
            records.append(
                {
                    "index": int(record_match.group("index")),
                    "channel": int(record_match.group("channel")),
                    "start": record_match.group("start").strip(),
                    "end": record_match.group("end").strip(),
                    "size": int(record_match.group("size")),
                }
            )

    return {
        "channel_count": channel_count,
        "record_count": record_count,
        "records": records,
    }


def query_recordings_via_sdk(
    *,
    settings: DahuaBridgeSettings,
    host: str,
    sdk_port: int,
    username: str,
    password: str,
    channel: int,
    start_dt: datetime,
    end_dt: datetime,
) -> dict[str, Any]:
    _copy_native_libs(settings)
    _ensure_compiled(settings)

    cmd = [
        settings.java_bin,
        "-Djava.library.path=/tmp:libs/mac64",
        "-cp",
        f"{settings.classes_dir}:{settings.sdk_root / 'libs' / 'jna.jar'}:{settings.sdk_root / 'res'}",
        "com.netsdk.demo.custom.DahuaInvestigationSmokeTest",
        host,
        str(sdk_port),
        username,
        password,
        str(channel),
        start_dt.strftime("%Y-%m-%d %H:%M:%S"),
        end_dt.strftime("%Y-%m-%d %H:%M:%S"),
    ]
    result = subprocess.run(
        cmd,
        cwd=settings.sdk_root,
        text=True,
        capture_output=True,
        check=False,
    )

    combined_output = "\n".join(part for part in [result.stdout.strip(), result.stderr.strip()] if part).strip()
    parsed = _parse_probe_output(result.stdout)

    return {
        "ok": result.returncode == 0 and parsed["record_count"] >= 0,
        "returncode": result.returncode,
        "output": combined_output,
        "channel_count": parsed["channel_count"],
        "record_count": parsed["record_count"],
        "records": parsed["records"],
    }


def download_clip_via_sdk(
    *,
    settings: DahuaBridgeSettings,
    host: str,
    sdk_port: int,
    username: str,
    password: str,
    channel: int,
    start_dt: datetime,
    end_dt: datetime,
    local_target_dir: Path,
) -> dict[str, Any]:
    _copy_native_libs(settings)
    _ensure_compiled(settings)

    clip_stem = f"channel_{channel}_{start_dt:%Y%m%d_%H%M%S}_{end_dt:%H%M%S}"
    output_base = local_target_dir / clip_stem
    output_base.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        settings.java_bin,
        "-Djava.library.path=/tmp:libs/mac64",
        "-cp",
        f"{settings.classes_dir}:{settings.sdk_root / 'libs' / 'jna.jar'}:{settings.sdk_root / 'res'}",
        "com.netsdk.demo.custom.DahuaDownloadByTimeHelper",
        host,
        str(sdk_port),
        username,
        password,
        str(channel),
        start_dt.strftime("%Y-%m-%d %H:%M:%S"),
        end_dt.strftime("%Y-%m-%d %H:%M:%S"),
        str(output_base),
    ]
    result = subprocess.run(
        cmd,
        cwd=settings.sdk_root,
        text=True,
        capture_output=True,
        check=False,
    )

    combined_output = "\n".join(part for part in [result.stdout.strip(), result.stderr.strip()] if part).strip()
    match = DOWNLOAD_OK_PATTERN.search(combined_output)
    if result.returncode != 0 or not match:
        raise RuntimeError(combined_output or "No se pudo descargar el clip Dahua por SDK.")

    final_path = Path(match.group("path")).expanduser()
    if not final_path.exists():
        raise FileNotFoundError(f"El helper Dahua reportó un archivo que no existe: {final_path}")

    return {
        "ok": True,
        "vendor": "Dahua",
        "final_local_path": str(final_path),
        "raw_local_path": str(final_path),
        "size_bytes": int(match.group("size")),
        "format": match.group("format"),
        "output": combined_output,
    }


def list_channels_via_sdk(
    *,
    settings: DahuaBridgeSettings,
    host: str,
    sdk_port: int,
    username: str,
    password: str,
) -> list[dict[str, Any]]:
    if settings.sdk_root is None:
        raise RuntimeError("No hay SDK Dahua local disponible para discovery por SDK.")
    today = datetime.combine(date.today(), time(0, 0, 0))
    probe = query_recordings_via_sdk(
        settings=settings,
        host=host,
        sdk_port=sdk_port,
        username=username,
        password=password,
        channel=1,
        start_dt=today,
        end_dt=today.replace(hour=23, minute=59, second=59),
    )
    channel_count = int(probe.get("channel_count", 0))
    if channel_count <= 0:
        raise RuntimeError(probe.get("output", "No se pudo determinar la cantidad de canales Dahua."))

    return [
        {
            "id": channel,
            "sdk_channel": channel - 1,
            "name": f"Canal {channel}",
            "status_label": "Disponible",
            "online": True,
            "ip_address": "-",
            "vendor": "Dahua",
        }
        for channel in range(1, channel_count + 1)
    ]


def list_channels(
    *,
    settings: DahuaBridgeSettings | None,
    host: str,
    sdk_port: int,
    http_port: int,
    username: str,
    password: str,
    bridge: DahuaRemoteBridgeSettings | None = None,
) -> list[dict[str, Any]]:
    titles: dict[int, str] = {}
    try:
        titles = get_channel_titles_via_http(
            host=host,
            http_port=http_port,
            username=username,
            password=password,
        )
    except Exception:
        titles = {}

    channel_count = 0
    if bridge is not None:
        try:
            channel_count = get_channel_count_via_bridge(
                bridge=bridge,
                host=host,
                sdk_port=sdk_port,
                username=username,
                password=password,
            )
        except Exception:
            channel_count = 0

    # If we at least got some titles, we can make a minimal list from them.
    if channel_count <= 0 and titles:
        channel_count = max(titles.keys())

    if channel_count > 0:
        channels = []
        for channel_id in range(1, channel_count + 1):
            name = titles.get(channel_id, "").strip()
            channels.append(
                {
                    "id": channel_id,
                    "sdk_channel": channel_id - 1,
                    "name": name or f"Canal {channel_id}",
                    "status_label": "Disponible",
                    "online": True,
                    "ip_address": "-",
                    "vendor": "Dahua",
                }
            )
        return channels

    if settings is not None:
        channels = list_channels_via_sdk(
            settings=settings,
            host=host,
            sdk_port=sdk_port,
            username=username,
            password=password,
        )
        return channels

    raise RuntimeError("No pude descubrir canales Dahua por HTTP/bridge y no hay SDK local disponible.")


def build_rtsp_url(
    *,
    host: str,
    rtsp_port: int,
    username: str,
    password: str,
    channel: int,
    subtype: int = 0,
) -> str:
    encoded_user = quote(username, safe="")
    encoded_password = quote(password, safe="")
    return (
        f"rtsp://{encoded_user}:{encoded_password}@{host}:{rtsp_port}"
        f"/cam/realmonitor?channel={channel}&subtype={subtype}"
    )


def fetch_snapshot_via_rtsp(
    *,
    host: str,
    rtsp_port: int,
    username: str,
    password: str,
    channel: int,
    output_path: Path,
) -> Path:
    ffmpeg_bin = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    if not Path(ffmpeg_bin).exists():
        raise FileNotFoundError("No encontré ffmpeg para capturar miniaturas RTSP.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rtsp_url = build_rtsp_url(
        host=host,
        rtsp_port=rtsp_port,
        username=username,
        password=password,
        channel=channel,
    )
    command = [
        ffmpeg_bin,
        "-y",
        "-rtsp_transport",
        "tcp",
        "-i",
        rtsp_url,
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(output_path),
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0 or not output_path.exists():
        raise RuntimeError(result.stderr.strip() or "No se pudo capturar snapshot RTSP.")
    return output_path


def fetch_snapshot_bytes_via_rtsp(
    *,
    host: str,
    rtsp_port: int,
    username: str,
    password: str,
    channel: int,
    timeout_seconds: int = 18,
) -> bytes:
    with tempfile.TemporaryDirectory(prefix="innova-dahua-rtsp-") as temp_dir:
        output_path = Path(temp_dir) / f"channel_{channel}.jpg"
        fetch_snapshot_via_rtsp(
            host=host,
            rtsp_port=rtsp_port,
            username=username,
            password=password,
            channel=channel,
            output_path=output_path,
        )
        payload = output_path.read_bytes()
        if len(payload) < 512:
            raise RuntimeError(f"Snapshot RTSP Dahua demasiado pequeño ({len(payload)} bytes).")
        return payload


def fetch_snapshot_via_http(
    *,
    host: str,
    http_port: int,
    username: str,
    password: str,
    channel: int,
    output_path: Path,
    timeout_seconds: int = 12,
) -> Path:
    snapshot = fetch_snapshot_bytes_via_http(
        host=host,
        http_port=http_port,
        username=username,
        password=password,
        channel=channel,
        timeout_seconds=timeout_seconds,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(snapshot)
    return output_path


@dataclass(slots=True)
class DahuaRemoteBridgeSettings:
    ssh_host: str = config.REMOTE_BRIDGE_HOST
    ssh_user: str = config.REMOTE_BRIDGE_USER
    ssh_key_path: Path = config.SSH_KEY_PATH
    remote_python: str = config.REMOTE_BRIDGE_PYTHON
    remote_sdk_dir: str = config.DAHUA_REMOTE_SDK_DIR


REMOTE_DAHUA_SCRIPT = r"""
import ctypes
import json
import os
import sys
import time
from ctypes import byref, c_bool, c_char_p, c_int, c_longlong, c_ushort, c_void_p, c_uint32, POINTER, Structure
from datetime import datetime
from pathlib import Path


class NET_TIME(Structure):
    _fields_ = [
        ("dwYear", c_uint32),
        ("dwMonth", c_uint32),
        ("dwDay", c_uint32),
        ("dwHour", c_uint32),
        ("dwMinute", c_uint32),
        ("dwSecond", c_uint32),
    ]


class NET_DEVICEINFO_Ex(Structure):
    _fields_ = [
        ("sSerialNumber", ctypes.c_ubyte * 48),
        ("nAlarmInPortNum", c_int),
        ("nAlarmOutPortNum", c_int),
        ("nDiskNum", c_int),
        ("nDVRType", c_int),
        ("nChanNum", c_int),
        ("byLimitLoginTime", ctypes.c_ubyte),
        ("byLeftLogTimes", ctypes.c_ubyte),
        ("bReserved", ctypes.c_ubyte * 2),
        ("nLockLeftTime", c_int),
        ("Reserved", ctypes.c_char * 4),
        ("nNTlsPort", c_int),
        ("nKeyFrameEncrypt", c_int),
        ("emAlgorithm", c_int),
        ("Reserved2", ctypes.c_char * 8),
    ]


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _net_time(dt: datetime) -> NET_TIME:
    return NET_TIME(
        dwYear=dt.year,
        dwMonth=dt.month,
        dwDay=dt.day,
        dwHour=dt.hour,
        dwMinute=dt.minute,
        dwSecond=dt.second,
    )


def main() -> None:
    host = _env("DAHUA_HOST").strip()
    port = int(_env("DAHUA_SDK_PORT", "37777"))
    username = _env("DAHUA_USER").strip()
    password = _env("DAHUA_PASSWORD")
    channel = int(_env("DAHUA_CHANNEL", "0"))
    start_iso = _env("DAHUA_START_ISO").strip()
    end_iso = _env("DAHUA_END_ISO").strip()
    output_name = _env("DAHUA_OUTPUT_NAME", "dahua_clip.mp4").strip()
    sdk_dir = _env("DAHUA_SDK_DIR", "/opt/innova/dahua").strip()

    try:
        start_dt = datetime.fromisoformat(start_iso)
        end_dt = datetime.fromisoformat(end_iso)
    except Exception:
        raise RuntimeError("Invalid DAHUA_START_ISO/DAHUA_END_ISO format. Use ISO 'YYYY-MM-DD HH:MM:SS'.")

    lib_candidates = [
        Path(sdk_dir) / "libdhnetsdk.so",
        Path(sdk_dir) / "Bin" / "libdhnetsdk.so",
        Path("/opt/innova/dahua/libdhnetsdk.so"),
        Path("/opt/innova/dahua/Bin/libdhnetsdk.so"),
    ]
    lib_path = next((p for p in lib_candidates if p.exists()), None)
    if lib_path is None:
        raise FileNotFoundError(f"libdhnetsdk.so not found. Tried: {[str(p) for p in lib_candidates]}")

    # Ensure dependencies are resolvable (libssl/libcrypto included with SDK).
    os.environ["LD_LIBRARY_PATH"] = f"{lib_path.parent}:{Path(sdk_dir)}:{Path(sdk_dir)/'Bin'}:" + os.environ.get("LD_LIBRARY_PATH", "")

    sdk = ctypes.CDLL(str(lib_path))

    sdk.CLIENT_Init.argtypes = [c_void_p, c_void_p]
    sdk.CLIENT_Init.restype = c_bool
    sdk.CLIENT_Cleanup.argtypes = []
    sdk.CLIENT_Cleanup.restype = None
    sdk.CLIENT_SetConnectTime.argtypes = [c_int, c_int]
    sdk.CLIENT_SetConnectTime.restype = None
    sdk.CLIENT_SetAutoReconnect.argtypes = [c_void_p, c_void_p]
    sdk.CLIENT_SetAutoReconnect.restype = None
    sdk.CLIENT_GetLastError.argtypes = []
    sdk.CLIENT_GetLastError.restype = c_uint32

    sdk.CLIENT_LoginEx2.argtypes = [
        c_char_p,
        c_ushort,
        c_char_p,
        c_char_p,
        c_int,
        c_void_p,
        POINTER(NET_DEVICEINFO_Ex),
        POINTER(c_int),
    ]
    sdk.CLIENT_LoginEx2.restype = c_longlong
    sdk.CLIENT_Logout.argtypes = [c_longlong]
    sdk.CLIENT_Logout.restype = c_bool

    sdk.CLIENT_DownloadByTimeEx.argtypes = [
        c_longlong,
        c_int,
        c_int,
        POINTER(NET_TIME),
        POINTER(NET_TIME),
        c_char_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    sdk.CLIENT_DownloadByTimeEx.restype = c_longlong

    sdk.CLIENT_GetDownloadPos.argtypes = [c_longlong, POINTER(c_int), POINTER(c_int)]
    sdk.CLIENT_GetDownloadPos.restype = c_bool
    sdk.CLIENT_StopDownload.argtypes = [c_longlong]
    sdk.CLIENT_StopDownload.restype = c_bool

    if not sdk.CLIENT_Init(None, None):
        raise RuntimeError(f"CLIENT_Init failed, err={sdk.CLIENT_GetLastError()}")
    sdk.CLIENT_SetAutoReconnect(None, None)
    sdk.CLIENT_SetConnectTime(5000, 3)

    device_info = NET_DEVICEINFO_Ex()
    error = c_int(0)
    login_id = sdk.CLIENT_LoginEx2(
        host.encode("utf-8"),
        port,
        username.encode("utf-8"),
        password.encode("utf-8"),
        0,  # EM_LOGIN_SPEC_CAP_TCP
        None,
        byref(device_info),
        byref(error),
    )
    if login_id == 0:
        err = int(sdk.CLIENT_GetLastError())
        raise RuntimeError(f"Login failed, sdk_err={err}, login_err={error.value}")

    try:
        out_dir = Path("/tmp/innova_dahua_downloads")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / output_name

        start_time = _net_time(start_dt)
        end_time = _net_time(end_dt)
        handle = sdk.CLIENT_DownloadByTimeEx(
            login_id,
            int(channel),
            0,  # EM_RECORD_TYPE_ALL
            byref(start_time),
            byref(end_time),
            str(out_path).encode("utf-8"),
            None,
            None,
            None,
            None,
            None,
        )
        if handle == 0:
            err = int(sdk.CLIENT_GetLastError())
            raise RuntimeError(
                f"DownloadByTimeEx failed, err={err} (0x{err:08x}). "
                f"Check channel index and that recordings exist in the requested time range."
            )

        total = c_int(0)
        done = c_int(0)
        deadline = time.time() + 60 * 60  # 1h safety timeout
        last_progress = time.time()
        while True:
            if time.time() > deadline:
                raise RuntimeError("Download timeout exceeded.")
            ok = sdk.CLIENT_GetDownloadPos(handle, byref(total), byref(done))
            if not ok:
                raise RuntimeError(f"GetDownloadPos failed, err={sdk.CLIENT_GetLastError()}")
            if total.value > 0 and done.value >= total.value:
                break
            if time.time() - last_progress > 15:
                last_progress = time.time()
            time.sleep(0.35)

        sdk.CLIENT_StopDownload(handle)
        if not out_path.exists() or out_path.stat().st_size <= 0:
            raise RuntimeError("Download completed but output file is missing/empty.")

        payload = {"ok": True, "remote_path": str(out_path), "size_bytes": int(out_path.stat().st_size)}
        print(json.dumps(payload))
    finally:
        sdk.CLIENT_Logout(login_id)
        sdk.CLIENT_Cleanup()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        sys.exit(2)
"""


REMOTE_DAHUA_COUNT_SCRIPT = r"""
import ctypes
import json
import os
import sys
from ctypes import byref, c_int, c_longlong, c_ushort, c_char_p, c_bool, POINTER, Structure, c_uint32

class NET_DEVICEINFO_Ex(Structure):
    _fields_ = [
        ("sSerialNumber", ctypes.c_ubyte * 48),
        ("nAlarmInPortNum", c_int),
        ("nAlarmOutPortNum", c_int),
        ("nDiskNum", c_int),
        ("nDVRType", c_int),
        ("nChanNum", c_int),
        ("byLimitLoginTime", ctypes.c_ubyte),
        ("byLeftLogTimes", ctypes.c_ubyte),
        ("bReserved", ctypes.c_ubyte * 2),
        ("nLockLeftTime", c_int),
        ("Reserved", ctypes.c_char * 4),
        ("nNTlsPort", c_int),
        ("nKeyFrameEncrypt", c_int),
        ("emAlgorithm", c_int),
        ("Reserved2", ctypes.c_char * 8),
    ]

def main() -> None:
    host = os.environ.get("DAHUA_HOST", "").strip()
    port = int(os.environ.get("DAHUA_SDK_PORT", "37777"))
    username = os.environ.get("DAHUA_USER", "").strip()
    password = os.environ.get("DAHUA_PASSWORD", "")
    sdk_dir = os.environ.get("DAHUA_SDK_DIR", "/opt/innova/dahua").strip()

    lib_candidates = [
        os.path.join(sdk_dir, "libdhnetsdk.so"),
        os.path.join(sdk_dir, "Bin", "libdhnetsdk.so"),
        "/opt/innova/dahua/libdhnetsdk.so",
        "/opt/innova/dahua/Bin/libdhnetsdk.so",
    ]
    lib_path = next((p for p in lib_candidates if os.path.exists(p)), None)
    if lib_path is None:
        raise FileNotFoundError(f"libdhnetsdk.so not found: {lib_candidates}")

    os.environ["LD_LIBRARY_PATH"] = f"{os.path.dirname(lib_path)}:{sdk_dir}:{os.path.join(sdk_dir,'Bin')}:" + os.environ.get("LD_LIBRARY_PATH", "")
    sdk = ctypes.CDLL(lib_path)

    sdk.CLIENT_Init.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    sdk.CLIENT_Init.restype = c_bool
    sdk.CLIENT_Cleanup.argtypes = []
    sdk.CLIENT_Cleanup.restype = None
    sdk.CLIENT_SetConnectTime.argtypes = [c_int, c_int]
    sdk.CLIENT_SetConnectTime.restype = None
    sdk.CLIENT_GetLastError.argtypes = []
    sdk.CLIENT_GetLastError.restype = c_uint32
    sdk.CLIENT_LoginEx2.argtypes = [c_char_p, c_ushort, c_char_p, c_char_p, c_int, ctypes.c_void_p, POINTER(NET_DEVICEINFO_Ex), POINTER(c_int)]
    sdk.CLIENT_LoginEx2.restype = c_longlong
    sdk.CLIENT_Logout.argtypes = [c_longlong]
    sdk.CLIENT_Logout.restype = c_bool

    if not sdk.CLIENT_Init(None, None):
        raise RuntimeError(f"CLIENT_Init failed, err={sdk.CLIENT_GetLastError()}")
    sdk.CLIENT_SetConnectTime(5000, 3)

    info = NET_DEVICEINFO_Ex()
    err = c_int(0)
    login = sdk.CLIENT_LoginEx2(host.encode(), port, username.encode(), password.encode(), 0, None, byref(info), byref(err))
    if login == 0:
        raise RuntimeError(f"Login failed, sdk_err={sdk.CLIENT_GetLastError()}, login_err={err.value}")

    sdk.CLIENT_Logout(login)
    sdk.CLIENT_Cleanup()
    print(json.dumps({"ok": True, "channel_count": int(info.nChanNum)}))

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        sys.exit(2)
"""


def get_channel_count_via_bridge(
    *,
    bridge: DahuaRemoteBridgeSettings,
    host: str,
    sdk_port: int,
    username: str,
    password: str,
) -> int:
    import shlex

    env_prefix = " ".join(
        [
            f"DAHUA_HOST={shlex.quote(host)}",
            f"DAHUA_SDK_PORT={sdk_port}",
            f"DAHUA_USER={shlex.quote(username)}",
            f"DAHUA_PASSWORD={shlex.quote(password)}",
            f"DAHUA_SDK_DIR={shlex.quote(bridge.remote_sdk_dir)}",
        ]
    )
    remote_command = f"{env_prefix} {bridge.remote_python} - <<'PY'\n{REMOTE_DAHUA_COUNT_SCRIPT}\nPY"

    result = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            "-i",
            str(bridge.ssh_key_path),
            f"{bridge.ssh_user}@{bridge.ssh_host}",
            remote_command,
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "Fallo SSH al consultar canales Dahua.")

    payload = json.loads(result.stdout.strip().splitlines()[-1])
    if not payload.get("ok"):
        raise RuntimeError(payload.get("error", "No se pudo obtener channel_count Dahua por puente."))
    return int(payload.get("channel_count") or 0)


def download_clip_via_bridge_dahua(
    *,
    bridge: DahuaRemoteBridgeSettings,
    host: str,
    sdk_port: int,
    username: str,
    password: str,
    sdk_channel: int,
    start_dt: datetime,
    end_dt: datetime,
    local_target_dir: Path,
    progress_callback=None,
) -> dict[str, Any]:
    """
    Descarga un clip Dahua desde un servidor Ubuntu (igual que Hikvision), pero usando Dahua NetSDK (Linux).
    `sdk_channel` es 0-based (como lo usa la SDK de Dahua).
    """
    import shlex
    import subprocess

    local_target_dir.mkdir(parents=True, exist_ok=True)
    output_name = f"dahua_ch{sdk_channel:02d}_{start_dt:%Y%m%d_%H%M%S}_{end_dt:%H%M%S}.dav"

    def _emit(stage: str, detail: str) -> None:
        if progress_callback is not None:
            progress_callback(stage, detail)

    _emit("prepare", f"Preparando descarga Dahua (canal SDK {sdk_channel}) {start_dt:%H:%M:%S}-{end_dt:%H:%M:%S}.")

    env_prefix = " ".join(
        [
            f"DAHUA_HOST={shlex.quote(host)}",
            f"DAHUA_SDK_PORT={sdk_port}",
            f"DAHUA_USER={shlex.quote(username)}",
            f"DAHUA_PASSWORD={shlex.quote(password)}",
            f"DAHUA_CHANNEL={sdk_channel}",
            f"DAHUA_START_ISO={shlex.quote(start_dt.isoformat(sep=' '))}",
            f"DAHUA_END_ISO={shlex.quote(end_dt.isoformat(sep=' '))}",
            f"DAHUA_OUTPUT_NAME={shlex.quote(output_name)}",
            f"DAHUA_SDK_DIR={shlex.quote(bridge.remote_sdk_dir)}",
        ]
    )
    remote_command = f"{env_prefix} {bridge.remote_python} - <<'PY'\n{REMOTE_DAHUA_SCRIPT}\nPY"

    ssh_cmd = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "-i",
        str(bridge.ssh_key_path),
        f"{bridge.ssh_user}@{bridge.ssh_host}",
        remote_command,
    ]
    _emit("server_download", "Descargando clip en el servidor Ubuntu con Dahua NetSDK.")
    result = subprocess.run(ssh_cmd, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "Fallo SSH en descarga Dahua.")

    payload = json.loads(result.stdout.strip().splitlines()[-1])
    if not payload.get("ok"):
        raise RuntimeError(payload.get("error", "El servidor Dahua no pudo descargar el clip."))

    local_path = local_target_dir / output_name
    _emit("transfer", "Copiando clip Dahua del servidor al Mac.")
    scp_cmd = [
        "scp",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "-i",
        str(bridge.ssh_key_path),
        f"{bridge.ssh_user}@{bridge.ssh_host}:{payload['remote_path']}",
        str(local_path),
    ]
    scp_res = subprocess.run(scp_cmd, text=True, capture_output=True, check=False)
    if scp_res.returncode != 0:
        raise RuntimeError(scp_res.stderr.strip() or "No se pudo copiar el clip Dahua al Mac.")

    return {
        "ok": True,
        "vendor": "Dahua",
        "raw_local_path": str(local_path),
        "final_local_path": str(local_path),
        "size_bytes": int(payload.get("size_bytes", local_path.stat().st_size if local_path.exists() else 0)),
    }
