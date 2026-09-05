from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote
from xml.etree import ElementTree

import cv2
import numpy as np
import requests
from requests.auth import HTTPDigestAuth

from .. import config


def _tag_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _find_text(element: ElementTree.Element, target_name: str) -> str:
    for child in element.iter():
        if _tag_name(child.tag) == target_name and child.text:
            return child.text.strip()
    return ""


def list_channels_via_isapi(
    *,
    host: str,
    port: int,
    username: str,
    password: str,
    timeout_seconds: int = 12,
) -> list[dict[str, Any]]:
    url = f"http://{host}:{port}/ISAPI/ContentMgmt/InputProxy/channels"
    response = requests.get(
        url,
        auth=HTTPDigestAuth(username, password),
        timeout=timeout_seconds,
    )
    response.raise_for_status()

    root = ElementTree.fromstring(response.content)
    channels: list[dict[str, Any]] = []
    for channel in root.iter():
        if _tag_name(channel.tag) != "InputProxyChannel":
            continue

        logical_id = _find_text(channel, "id")
        name = _find_text(channel, "name") or f"Canal {logical_id or '?'}"
        source_input_port = _find_text(channel, "sourceInputPortDescriptor")
        channel_info = {
            "id": int(logical_id) if logical_id.isdigit() else logical_id,
            "name": name,
            "source_input_port": source_input_port,
        }
        channels.append(channel_info)

    channels.sort(key=lambda item: int(item["id"]) if isinstance(item["id"], int) else 9999)
    return channels


def list_channels_via_streaming_isapi(
    *,
    host: str,
    port: int,
    username: str,
    password: str,
    timeout_seconds: int = 12,
) -> list[dict[str, Any]]:
    url = f"http://{host}:{port}/ISAPI/Streaming/channels"
    response = requests.get(
        url,
        auth=HTTPDigestAuth(username, password),
        timeout=timeout_seconds,
    )
    response.raise_for_status()

    root = ElementTree.fromstring(response.content)
    channels_by_id: dict[int, dict[str, Any]] = {}
    for stream in root.iter():
        if _tag_name(stream.tag) != "StreamingChannel":
            continue

        stream_id_text = _find_text(stream, "id")
        video_input_text = _find_text(stream, "videoInputChannelID")
        if video_input_text.isdigit():
            logical_id = int(video_input_text)
        elif stream_id_text.isdigit():
            logical_id = max(1, int(stream_id_text) // 100)
        else:
            continue

        enabled_text = _find_text(stream, "enabled").lower()
        if enabled_text and enabled_text != "true":
            continue

        existing = channels_by_id.get(logical_id, {})
        stream_ids = existing.get("stream_ids", [])
        if stream_id_text and stream_id_text not in stream_ids:
            stream_ids.append(stream_id_text)

        channels_by_id[logical_id] = {
            "id": logical_id,
            "name": existing.get("name") or f"Canal {logical_id}",
            "source_input_port": existing.get("source_input_port") or f"Streaming channels {', '.join(stream_ids)}",
            "stream_ids": stream_ids,
            "online": True,
            "status_label": "Online",
            "detect_result": "Streaming ISAPI",
            "ip_address": "-",
            "password_status": "-",
        }

    channels = list(channels_by_id.values())
    channels.sort(key=lambda item: int(item["id"]) if isinstance(item["id"], int) else 9999)
    return channels


def list_channel_status_via_isapi(
    *,
    host: str,
    port: int,
    username: str,
    password: str,
    timeout_seconds: int = 12,
) -> dict[int, dict[str, Any]]:
    url = f"http://{host}:{port}/ISAPI/ContentMgmt/InputProxy/channels/status"
    response = requests.get(
        url,
        auth=HTTPDigestAuth(username, password),
        timeout=timeout_seconds,
    )
    response.raise_for_status()

    root = ElementTree.fromstring(response.content)
    status_map: dict[int, dict[str, Any]] = {}
    for channel in root.iter():
        if _tag_name(channel.tag) != "InputProxyChannelStatus":
            continue

        logical_id = _find_text(channel, "id")
        if not logical_id.isdigit():
            continue

        online_text = _find_text(channel, "online").lower()
        detect_result = _find_text(channel, "chanDetectResult")
        status_map[int(logical_id)] = {
            "online": online_text == "true",
            "status_label": "Online" if online_text == "true" else "Offline",
            "detect_result": detect_result or "-",
            "ip_address": _find_text(channel, "ipAddress") or "-",
            "password_status": _find_text(channel, "PasswordStatus") or "-",
        }

    return status_map


def list_channels_with_status_via_isapi(
    *,
    host: str,
    port: int,
    username: str,
    password: str,
    timeout_seconds: int = 12,
) -> list[dict[str, Any]]:
    channels = list_channels_via_isapi(
        host=host,
        port=port,
        username=username,
        password=password,
        timeout_seconds=timeout_seconds,
    )
    if not channels:
        return list_channels_via_streaming_isapi(
            host=host,
            port=port,
            username=username,
            password=password,
            timeout_seconds=timeout_seconds,
        )

    try:
        status_map = list_channel_status_via_isapi(
            host=host,
            port=port,
            username=username,
            password=password,
            timeout_seconds=timeout_seconds,
        )
    except Exception:
        status_map = {}

    for channel in channels:
        channel_id = int(channel["id"]) if isinstance(channel["id"], int) else None
        channel_status = status_map.get(channel_id or -1, {})
        channel["online"] = channel_status.get("online", False)
        channel["status_label"] = channel_status.get("status_label", "Sin dato")
        channel["detect_result"] = channel_status.get("detect_result", "-")
        channel["ip_address"] = channel_status.get("ip_address", "-")
        channel["password_status"] = channel_status.get("password_status", "-")

    return channels


def fetch_snapshot_bytes_via_isapi(
    *,
    host: str,
    port: int,
    username: str,
    password: str,
    logical_channel: int,
    stream_variant: str = "sub",
    timeout_seconds: int = 12,
) -> bytes:
    stream_suffix = 2 if stream_variant == "sub" else 1
    stream_channel_id = (logical_channel * 100) + stream_suffix
    url = f"http://{host}:{port}/ISAPI/Streaming/channels/{stream_channel_id}/picture"
    response = requests.get(
        url,
        auth=HTTPDigestAuth(username, password),
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    payload = response.content
    _validate_snapshot_payload(payload, source=f"Hikvision ISAPI {stream_channel_id}")
    return payload


def _validate_snapshot_payload(payload: bytes, *, source: str) -> None:
    if not payload or len(payload) < 2048:
        raise RuntimeError(f"{source}: snapshot demasiado pequeño ({len(payload or b'')} bytes).")

    header = payload[:128].lstrip().lower()
    if header.startswith(b"<") or b"<html" in header or b"<?xml" in header:
        raise RuntimeError(f"{source}: el NVR devolvió HTML/XML en vez de una imagen.")

    data = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise RuntimeError(f"{source}: no se pudo decodificar la imagen.")

    height, width = image.shape[:2]
    if width < 80 or height < 60:
        raise RuntimeError(f"{source}: imagen inválida ({width}x{height}).")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean = float(gray.mean())
    stddev = float(gray.std())
    dark_ratio = float((gray < 12).mean())
    bright_ratio = float((gray > 245).mean())

    if stddev < 3.5:
        raise RuntimeError(f"{source}: imagen sin detalle útil.")
    if dark_ratio > 0.92 and bright_ratio < 0.01:
        raise RuntimeError(f"{source}: imagen casi totalmente negra.")

    center = image[height // 4 : (height * 3) // 4, width // 4 : (width * 3) // 4]
    if center.size:
        blue = center[:, :, 0].astype(np.int16)
        green = center[:, :, 1].astype(np.int16)
        red = center[:, :, 2].astype(np.int16)
        blue_notice_ratio = float(((blue > 120) & (blue > green + 35) & (blue > red + 35)).mean())
        if dark_ratio > 0.55 and blue_notice_ratio > 0.05:
            raise RuntimeError(f"{source}: parece una pantalla de error del NVR, no video real.")


def build_rtsp_url(
    *,
    host: str,
    rtsp_port: int,
    username: str,
    password: str,
    logical_channel: int,
    stream_variant: str = "sub",
) -> str:
    stream_suffix = 2 if stream_variant == "sub" else 1
    stream_channel_id = (logical_channel * 100) + stream_suffix
    encoded_user = quote(username, safe="")
    encoded_password = quote(password, safe="")
    return (
        f"rtsp://{encoded_user}:{encoded_password}@{host}:{rtsp_port}"
        f"/Streaming/Channels/{stream_channel_id}"
    )


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
        raise FileNotFoundError("No encontré ffmpeg para capturar miniaturas RTSP Hikvision.")

    rtsp_url = build_rtsp_url(
        host=host,
        rtsp_port=rtsp_port,
        username=username,
        password=password,
        logical_channel=logical_channel,
        stream_variant=stream_variant,
    )
    with tempfile.TemporaryDirectory(prefix="innova-hik-rtsp-") as temp_dir:
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
            "10000000",
            "-probesize",
            "10000000",
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
            raise RuntimeError(result.stderr.strip() or "No se pudo capturar snapshot RTSP Hikvision.")
        payload = output_path.read_bytes()
        _validate_snapshot_payload(payload, source=f"Hikvision RTSP {logical_channel} {stream_variant}")
        return payload


REMOTE_HIKVISION_SCRIPT = textwrap.dedent(
    """
    import ctypes
    import json
    import os
    import sys
    import time
    from ctypes import byref, c_char, c_int, c_uint, c_void_p, c_ubyte, c_ushort, c_char_p, POINTER
    from pathlib import Path
    from datetime import datetime

    SERIALNO_LEN = 48
    NET_DVR_DEV_ADDRESS_MAX_LEN = 129
    NET_DVR_LOGIN_USERNAME_MAX_LEN = 64
    NET_DVR_LOGIN_PASSWD_MAX_LEN = 64
    NET_SDK_INIT_CFG_SDK_PATH = 2
    NET_SDK_INIT_CFG_LIBEAY_PATH = 3
    NET_SDK_INIT_CFG_SSLEAY_PATH = 4
    NET_DVR_PLAYSTART = 1

    class NET_DVR_DEVICEINFO_V30(ctypes.Structure):
        _fields_ = [
            ('sSerialNumber', c_ubyte * SERIALNO_LEN),
            ('byAlarmInPortNum', c_ubyte),
            ('byAlarmOutPortNum', c_ubyte),
            ('byDiskNum', c_ubyte),
            ('byDVRType', c_ubyte),
            ('byChanNum', c_ubyte),
            ('byStartChan', c_ubyte),
            ('byAudioChanNum', c_ubyte),
            ('byIPChanNum', c_ubyte),
            ('byZeroChanNum', c_ubyte),
            ('byMainProto', c_ubyte),
            ('bySubProto', c_ubyte),
            ('bySupport', c_ubyte),
            ('bySupport1', c_ubyte),
            ('bySupport2', c_ubyte),
            ('wDevType', c_ushort),
            ('bySupport3', c_ubyte),
            ('byMultiStreamProto', c_ubyte),
            ('byStartDChan', c_ubyte),
            ('byStartDTalkChan', c_ubyte),
            ('byHighDChanNum', c_ubyte),
            ('bySupport4', c_ubyte),
            ('byLanguageType', c_ubyte),
            ('byVoiceInChanNum', c_ubyte),
            ('byStartVoiceInChanNo', c_ubyte),
            ('bySupport5', c_ubyte),
            ('bySupport6', c_ubyte),
            ('byMirrorChanNum', c_ubyte),
            ('wStartMirrorChanNo', c_ushort),
            ('bySupport7', c_ubyte),
            ('byRes2', c_ubyte),
        ]

    class NET_DVR_DEVICEINFO_V40(ctypes.Structure):
        _fields_ = [
            ('struDeviceV30', NET_DVR_DEVICEINFO_V30),
            ('bySupportLock', c_ubyte),
            ('byRetryLoginTime', c_ubyte),
            ('byPasswordLevel', c_ubyte),
            ('byProxyType', c_ubyte),
            ('dwSurplusLockTime', c_uint),
            ('byCharEncodeType', c_ubyte),
            ('bySupportDev5', c_ubyte),
            ('bySupport', c_ubyte),
            ('byLoginMode', c_ubyte),
            ('dwOEMCode', c_uint),
            ('iResidualValidity', c_int),
            ('byResidualValidity', c_ubyte),
            ('bySingleStartDTalkChan', c_ubyte),
            ('bySingleDTalkChanNums', c_ubyte),
            ('byPassWordResetLevel', c_ubyte),
            ('bySupportStreamEncrypt', c_ubyte),
            ('byMarketType', c_ubyte),
            ('byRes2', c_ubyte * 238),
        ]

    class NET_DVR_USER_LOGIN_INFO(ctypes.Structure):
        _fields_ = [
            ('sDeviceAddress', c_char * NET_DVR_DEV_ADDRESS_MAX_LEN),
            ('byUseTransport', c_ubyte),
            ('wPort', c_ushort),
            ('sUserName', c_char * NET_DVR_LOGIN_USERNAME_MAX_LEN),
            ('sPassword', c_char * NET_DVR_LOGIN_PASSWD_MAX_LEN),
            ('cbLoginResult', c_void_p),
            ('pUser', c_void_p),
            ('bUseAsynLogin', c_int),
            ('byProxyType', c_ubyte),
            ('byUseUTCTime', c_ubyte),
            ('byLoginMode', c_ubyte),
            ('byHttps', c_ubyte),
            ('iProxyID', c_int),
            ('byVerifyMode', c_ubyte),
            ('byRes3', c_ubyte * 119),
        ]

    class NET_DVR_LOCAL_SDK_PATH(ctypes.Structure):
        _fields_ = [
            ('sPath', c_char * 256),
            ('byRes', c_ubyte * 128),
        ]

    class NET_DVR_TIME(ctypes.Structure):
        _fields_ = [
            ('dwYear', c_uint),
            ('dwMonth', c_uint),
            ('dwDay', c_uint),
            ('dwHour', c_uint),
            ('dwMinute', c_uint),
            ('dwSecond', c_uint),
        ]

    class NET_DVR_PLAYCOND(ctypes.Structure):
        _fields_ = [
            ('dwChannel', c_uint),
            ('struStartTime', NET_DVR_TIME),
            ('struStopTime', NET_DVR_TIME),
            ('byDrawFrame', c_ubyte),
            ('byStreamType', c_ubyte),
            ('byStreamID', c_ubyte * 32),
            ('byCourseFile', c_ubyte),
            ('byDownload', c_ubyte),
            ('byOptimalStreamType', c_ubyte),
            ('byVODFileType', c_ubyte),
            ('byRes', c_ubyte * 26),
        ]

    def make_time(value: str):
        dt = datetime.fromisoformat(value)
        t = NET_DVR_TIME()
        t.dwYear = dt.year
        t.dwMonth = dt.month
        t.dwDay = dt.day
        t.dwHour = dt.hour
        t.dwMinute = dt.minute
        t.dwSecond = dt.second
        return t

    host = os.environ['HIK_HOST']
    sdk_port = int(os.environ['HIK_SDK_PORT'])
    username = os.environ['HIK_USER']
    password = os.environ['HIK_PASSWORD']
    logical_channel = int(os.environ['HIK_CHANNEL'])
    start_iso = os.environ['HIK_START_ISO']
    end_iso = os.environ['HIK_END_ISO']
    output_name = os.environ['HIK_OUTPUT_NAME']
    sdk_candidates = []
    env_sdk_root = os.environ.get('HIK_SDK_ROOT', '').strip()
    if env_sdk_root:
        sdk_candidates.append(env_sdk_root)
    sdk_candidates.extend([
        '/opt/innova/hikvision/EN-HCNetSDKV6.1.9.4_build20220412_linux64/EN-HCNetSDKV6.1.9.4_build20220412_linux64/lib',
        '/home/ubuntu/hikvision/EN-HCNetSDKV6.1.9.4_build20220412_linux64/EN-HCNetSDKV6.1.9.4_build20220412_linux64/lib',
    ])
    sdk_root = ''
    for candidate in sdk_candidates:
        if candidate and Path(candidate).exists():
            sdk_root = candidate
            break
    if not sdk_root:
        raise FileNotFoundError(
            'No encontré el SDK Hikvision en el servidor remoto. '
            f'Candidatos revisados: {sdk_candidates}'
        )
    evidence_root = os.environ.get('HIK_EVIDENCE_ROOT', '/tmp/innova_hikvision')

    libcrypto = sdk_root + '/libcrypto.so.1.1'
    libssl = sdk_root + '/libssl.so.1.1'
    libhcnetsdk = sdk_root + '/libhcnetsdk.so'
    os.chdir(sdk_root)

    ctypes.CDLL(libcrypto, mode=ctypes.RTLD_GLOBAL)
    ctypes.CDLL(libssl, mode=ctypes.RTLD_GLOBAL)
    sdk = ctypes.CDLL(libhcnetsdk, mode=ctypes.RTLD_GLOBAL)

    sdk.NET_DVR_SetSDKInitCfg.argtypes = [c_int, c_void_p]
    sdk.NET_DVR_SetSDKInitCfg.restype = c_int
    sdk.NET_DVR_Init.restype = c_int
    sdk.NET_DVR_GetLastError.restype = c_uint
    sdk.NET_DVR_Login_V40.argtypes = [POINTER(NET_DVR_USER_LOGIN_INFO), POINTER(NET_DVR_DEVICEINFO_V40)]
    sdk.NET_DVR_Login_V40.restype = c_int
    sdk.NET_DVR_Logout_V30.argtypes = [c_int]
    sdk.NET_DVR_Logout_V30.restype = c_int
    sdk.NET_DVR_Cleanup.restype = c_int
    sdk.NET_DVR_GetFileByTime_V40.argtypes = [c_int, c_char_p, POINTER(NET_DVR_PLAYCOND)]
    sdk.NET_DVR_GetFileByTime_V40.restype = c_int
    sdk.NET_DVR_PlayBackControl.argtypes = [c_int, c_uint, c_uint, POINTER(c_uint)]
    sdk.NET_DVR_PlayBackControl.restype = c_int
    sdk.NET_DVR_GetDownloadPos.argtypes = [c_int]
    sdk.NET_DVR_GetDownloadPos.restype = c_int
    sdk.NET_DVR_StopGetFile.argtypes = [c_int]
    sdk.NET_DVR_StopGetFile.restype = c_int

    sdk_path = NET_DVR_LOCAL_SDK_PATH()
    sdk_path.sPath = sdk_root.encode()
    crypto_buf = ctypes.create_string_buffer(libcrypto.encode())
    ssl_buf = ctypes.create_string_buffer(libssl.encode())
    sdk.NET_DVR_SetSDKInitCfg(NET_SDK_INIT_CFG_SDK_PATH, byref(sdk_path))
    sdk.NET_DVR_SetSDKInitCfg(NET_SDK_INIT_CFG_LIBEAY_PATH, crypto_buf)
    sdk.NET_DVR_SetSDKInitCfg(NET_SDK_INIT_CFG_SSLEAY_PATH, ssl_buf)
    sdk.NET_DVR_Init()

    login = NET_DVR_USER_LOGIN_INFO()
    device = NET_DVR_DEVICEINFO_V40()
    login.sDeviceAddress = host.encode()
    login.wPort = sdk_port
    login.sUserName = username.encode()
    login.sPassword = password.encode()
    login.bUseAsynLogin = 0
    login.byLoginMode = 0
    login.byHttps = 0

    user_id = sdk.NET_DVR_Login_V40(byref(login), byref(device))
    if user_id < 0:
        print(json.dumps({'ok': False, 'error': f'LOGIN_FAIL:{sdk.NET_DVR_GetLastError()}'}))
        raise SystemExit(1)

    start_dchan = device.struDeviceV30.byStartDChan
    sdk_channel = start_dchan + logical_channel - 1

    output_dir = Path(evidence_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    remote_path = output_dir / output_name
    if remote_path.exists():
        remote_path.unlink()

    cond = NET_DVR_PLAYCOND()
    cond.dwChannel = sdk_channel
    cond.struStartTime = make_time(start_iso)
    cond.struStopTime = make_time(end_iso)
    cond.byDrawFrame = 0
    cond.byStreamType = 0
    cond.byCourseFile = 0
    cond.byDownload = 1
    cond.byOptimalStreamType = 0
    cond.byVODFileType = 0

    handle = sdk.NET_DVR_GetFileByTime_V40(user_id, str(remote_path).encode(), byref(cond))
    if handle < 0:
        print(json.dumps({'ok': False, 'error': f'DOWNLOAD_FAIL:{sdk.NET_DVR_GetLastError()}'}))
        sdk.NET_DVR_Logout_V30(user_id)
        sdk.NET_DVR_Cleanup()
        raise SystemExit(1)

    sdk.NET_DVR_PlayBackControl(handle, NET_DVR_PLAYSTART, 0, None)
    final_pos = -1
    for _ in range(180):
        final_pos = sdk.NET_DVR_GetDownloadPos(handle)
        if final_pos >= 100 or final_pos < 0:
            break
        time.sleep(1)

    sdk.NET_DVR_StopGetFile(handle)
    sdk.NET_DVR_Logout_V30(user_id)
    sdk.NET_DVR_Cleanup()

    exists = remote_path.exists()
    size_bytes = remote_path.stat().st_size if exists else 0
    print(
        json.dumps(
            {
                'ok': exists and size_bytes > 0,
                'remote_path': str(remote_path),
                'logical_channel': logical_channel,
                'sdk_channel': sdk_channel,
                'download_pos': final_pos,
                'size_bytes': size_bytes,
            }
        )
    )
    """
).strip()


@dataclass(slots=True)
class HikvisionBridgeSettings:
    ssh_host: str = config.REMOTE_BRIDGE_HOST
    ssh_user: str = config.REMOTE_BRIDGE_USER
    ssh_key_path: Path = config.SSH_KEY_PATH
    remote_python: str = config.REMOTE_BRIDGE_PYTHON
    remote_sdk_root: str = config.HIKVISION_REMOTE_SDK_DIR
    remote_evidence_root: str = "/tmp/innova_hikvision"
    local_download_dir: Path = config.DEFAULT_OUTPUT_DIR / "hikvision_downloads"


def _run_subprocess(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
    )


def _emit_progress(
    progress_callback,
    stage: str,
    detail: str,
) -> None:
    if progress_callback is not None:
        progress_callback(stage, detail)


def _normalize_with_opencv(raw_path: Path, normalized_path: Path) -> tuple[bool, str]:
    capture = cv2.VideoCapture(str(raw_path))
    if not capture.isOpened():
        return False, "OpenCV no pudo abrir el clip crudo."

    fps = capture.get(cv2.CAP_PROP_FPS) or 20.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        return False, "OpenCV no obtuvo tamaño válido del video."

    writer = cv2.VideoWriter(
        str(normalized_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        return False, "OpenCV no pudo crear el MP4 estándar."

    written_frames = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        writer.write(frame)
        written_frames += 1

    capture.release()
    writer.release()
    if written_frames == 0 or not normalized_path.exists():
        return False, "OpenCV no escribió frames al MP4 estándar."
    return True, f"Normalizado con OpenCV ({written_frames} frames)."


def download_clip_via_bridge(
    *,
    bridge: HikvisionBridgeSettings,
    host: str,
    sdk_port: int,
    username: str,
    password: str,
    logical_channel: int,
    start_dt: datetime,
    end_dt: datetime,
    local_target_dir: Path,
    normalize_for_review: bool = True,
    progress_callback=None,
) -> dict[str, Any]:
    local_target_dir.mkdir(parents=True, exist_ok=True)
    timestamp_label = start_dt.strftime("%Y%m%d_%H%M%S")
    output_name = f"channel_{logical_channel:02d}_{timestamp_label}_{end_dt.strftime('%H%M%S')}.mp4"
    _emit_progress(
        progress_callback,
        "prepare",
        f"Preparando solicitud para canal {logical_channel} desde {start_dt:%H:%M:%S} hasta {end_dt:%H:%M:%S}.",
    )

    env_prefix = " ".join(
        [
            f"HIK_HOST={shlex.quote(host)}",
            f"HIK_SDK_PORT={sdk_port}",
            f"HIK_USER={shlex.quote(username)}",
            f"HIK_PASSWORD={shlex.quote(password)}",
            f"HIK_CHANNEL={logical_channel}",
            f"HIK_START_ISO={shlex.quote(start_dt.isoformat(sep=' '))}",
            f"HIK_END_ISO={shlex.quote(end_dt.isoformat(sep=' '))}",
            f"HIK_OUTPUT_NAME={shlex.quote(output_name)}",
            f"HIK_SDK_ROOT={shlex.quote(bridge.remote_sdk_root)}",
            f"HIK_EVIDENCE_ROOT={shlex.quote(bridge.remote_evidence_root)}",
        ]
    )
    remote_command = (
        f"{env_prefix} {bridge.remote_python} - <<'PY'\n"
        f"{REMOTE_HIKVISION_SCRIPT}\n"
        "PY"
    )

    ssh_command = [
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
    _emit_progress(
        progress_callback,
        "server_download",
        "Descargando video en el servidor Ubuntu con HCNetSDK.",
    )
    result = _run_subprocess(ssh_command)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "Fallo SSH al descargar clip.")

    payload = json.loads(result.stdout.strip().splitlines()[-1])
    if not payload.get("ok"):
        raise RuntimeError(payload.get("error", "El servidor Hikvision no pudo descargar el clip."))

    local_raw_path = local_target_dir / output_name
    _emit_progress(
        progress_callback,
        "transfer",
        "Copiando clip del servidor al Mac.",
    )
    scp_command = [
        "scp",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "-i",
        str(bridge.ssh_key_path),
        f"{bridge.ssh_user}@{bridge.ssh_host}:{payload['remote_path']}",
        str(local_raw_path),
    ]
    scp_result = _run_subprocess(scp_command)
    if scp_result.returncode != 0:
        raise RuntimeError(scp_result.stderr.strip() or "No se pudo copiar el clip al Mac.")

    final_path = local_raw_path
    normalized_path = local_target_dir / f"{local_raw_path.stem}_standard.mp4"
    ffmpeg_result = subprocess.CompletedProcess(args=[], returncode=127, stdout="", stderr="ffmpeg no disponible")
    normalization_note = "Se conserva el clip crudo descargado desde Hikvision para analisis."
    if normalize_for_review:
        ffmpeg_binary = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
        normalization_note = "Se conserva el clip crudo descargado desde Hikvision."
        if Path(ffmpeg_binary).exists():
            _emit_progress(
                progress_callback,
                "normalize",
                "Convirtiendo video a MP4 estándar con ffmpeg.",
            )
            ffmpeg_command = [
                ffmpeg_binary,
                "-y",
                "-i",
                str(local_raw_path),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                str(normalized_path),
            ]
            ffmpeg_result = _run_subprocess(ffmpeg_command)
            if ffmpeg_result.returncode == 0 and normalized_path.exists():
                final_path = normalized_path
                normalization_note = "Normalizado con ffmpeg."
        elif local_raw_path.exists():
            _emit_progress(
                progress_callback,
                "normalize",
                "Convirtiendo video a MP4 estándar con OpenCV.",
            )
            ok, note = _normalize_with_opencv(local_raw_path, normalized_path)
            normalization_note = note
            if ok:
                final_path = normalized_path

    _emit_progress(
        progress_callback,
        "done",
        "Clip listo para reproducir y descargar.",
    )

    return {
        "remote_path": payload["remote_path"],
        "raw_local_path": str(local_raw_path),
        "final_local_path": str(final_path),
        "logical_channel": payload["logical_channel"],
        "sdk_channel": payload["sdk_channel"],
        "size_bytes": payload["size_bytes"],
        "download_pos": payload["download_pos"],
        "normalize_for_review": normalize_for_review,
        "converted_to_standard_mp4": final_path == normalized_path,
        "normalization_note": normalization_note,
        "ffmpeg_stderr": ffmpeg_result.stderr.strip(),
    }


def download_clip_via_local_sdk(
    *,
    host: str,
    sdk_port: int,
    username: str,
    password: str,
    logical_channel: int,
    start_dt: datetime,
    end_dt: datetime,
    local_target_dir: Path,
    sdk_root: str | None = None,
    normalize_for_review: bool = True,
    progress_callback=None,
) -> dict[str, Any]:
    local_target_dir.mkdir(parents=True, exist_ok=True)
    timestamp_label = start_dt.strftime("%Y%m%d_%H%M%S")
    output_name = f"channel_{logical_channel:02d}_{timestamp_label}_{end_dt.strftime('%H%M%S')}.mp4"
    local_raw_path = local_target_dir / output_name
    if local_raw_path.exists():
        local_raw_path.unlink()

    _emit_progress(
        progress_callback,
        "local_hikvision_sdk",
        f"Descargando Hikvision localmente con HCNetSDK canal {logical_channel}.",
    )

    env = os.environ.copy()
    env.update(
        {
            "HIK_HOST": host,
            "HIK_SDK_PORT": str(int(sdk_port)),
            "HIK_USER": username,
            "HIK_PASSWORD": password,
            "HIK_CHANNEL": str(int(logical_channel)),
            "HIK_START_ISO": start_dt.isoformat(sep=" "),
            "HIK_END_ISO": end_dt.isoformat(sep=" "),
            "HIK_OUTPUT_NAME": output_name,
            "HIK_EVIDENCE_ROOT": str(local_target_dir),
        }
    )
    resolved_sdk_root = (sdk_root or config.HIKVISION_LOCAL_SDK_DIR or "").strip()
    if resolved_sdk_root:
        env["HIK_SDK_ROOT"] = resolved_sdk_root

    result = subprocess.run(
        [sys.executable, "-m", "innova_investigation.tools.hikvision_sdk_download"],
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "Fallo descarga local Hikvision.")

    payload = json.loads(result.stdout.strip().splitlines()[-1])
    if not payload.get("ok"):
        raise RuntimeError(payload.get("error", "HCNetSDK local no pudo descargar el clip."))

    sdk_output_path = Path(payload["remote_path"])
    if sdk_output_path != local_raw_path and sdk_output_path.exists():
        shutil.copy2(sdk_output_path, local_raw_path)
    if not local_raw_path.exists() or local_raw_path.stat().st_size <= 0:
        raise RuntimeError("HCNetSDK local reporto OK, pero el archivo local no existe o esta vacio.")

    final_path = local_raw_path
    normalized_path = local_target_dir / f"{local_raw_path.stem}_standard.mp4"
    ffmpeg_result = subprocess.CompletedProcess(args=[], returncode=127, stdout="", stderr="ffmpeg no disponible")
    normalization_note = "Se conserva el clip crudo descargado desde Hikvision para analisis."
    if normalize_for_review:
        ffmpeg_binary = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
        normalization_note = "Se conserva el clip crudo descargado desde Hikvision."
        if Path(ffmpeg_binary).exists():
            _emit_progress(
                progress_callback,
                "normalize",
                "Convirtiendo video a MP4 estandar con ffmpeg.",
            )
            ffmpeg_command = [
                ffmpeg_binary,
                "-y",
                "-i",
                str(local_raw_path),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                str(normalized_path),
            ]
            ffmpeg_result = _run_subprocess(ffmpeg_command)
            if ffmpeg_result.returncode == 0 and normalized_path.exists():
                final_path = normalized_path
                normalization_note = "Normalizado con ffmpeg."
        elif local_raw_path.exists():
            _emit_progress(
                progress_callback,
                "normalize",
                "Convirtiendo video a MP4 estandar con OpenCV.",
            )
            ok, note = _normalize_with_opencv(local_raw_path, normalized_path)
            normalization_note = note
            if ok:
                final_path = normalized_path

    _emit_progress(
        progress_callback,
        "done",
        "Clip Hikvision listo para analizar.",
    )

    return {
        "remote_path": str(sdk_output_path),
        "raw_local_path": str(local_raw_path),
        "final_local_path": str(final_path),
        "logical_channel": payload["logical_channel"],
        "sdk_channel": payload["sdk_channel"],
        "size_bytes": payload["size_bytes"],
        "download_pos": payload["download_pos"],
        "normalize_for_review": normalize_for_review,
        "converted_to_standard_mp4": final_path == normalized_path,
        "normalization_note": normalization_note,
        "ffmpeg_stderr": ffmpeg_result.stderr.strip(),
    }
