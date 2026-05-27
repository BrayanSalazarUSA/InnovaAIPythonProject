from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from typing import Optional
import json
import os
import platform
import shutil
import subprocess
import threading
import time
import urllib.request

APP_ROOT = Path('/opt/innova/live-streaming')
HLS_ROOT = Path('/var/www/innova-live-view-runtime')
LOG_ROOT = APP_ROOT / 'logs'
TMP_ROOT = APP_ROOT / 'tmp'
PUBLIC_BASE_URL = os.getenv('LIVEVIEW_PUBLIC_BASE_URL', 'http://18.234.252.123/live-view-runtime').rstrip('/')
FFMPEG_PATH = os.getenv('FFMPEG_PATH', 'ffmpeg')
HLS_TIME = os.getenv('LIVEVIEW_HLS_TIME_SECONDS', '1')
HLS_LIST_SIZE = os.getenv('LIVEVIEW_HLS_LIST_SIZE', '6')
STARTUP_TIMEOUT = float(os.getenv('LIVEVIEW_STARTUP_TIMEOUT_SECONDS', '20'))
WEBRTC_STARTUP_TIMEOUT = float(os.getenv('LIVEVIEW_WEBRTC_STARTUP_TIMEOUT_SECONDS', '12'))
MEDIAMTX_RTSP_BASE = os.getenv('MEDIAMTX_RTSP_URL', 'rtsp://127.0.0.1:8554').rstrip('/')
MEDIAMTX_WHEP_BASE = os.getenv('MEDIAMTX_WHEP_URL', 'http://18.234.252.123/live-view-webrtc').rstrip('/')
MEDIAMTX_API_BASE = os.getenv('MEDIAMTX_API_URL', 'http://127.0.0.1:9997').rstrip('/')
REGISTRY_PATH = APP_ROOT / 'runtime' / 'registry.json'
APP_ROOT.mkdir(parents=True, exist_ok=True)
HLS_ROOT.mkdir(parents=True, exist_ok=True)
LOG_ROOT.mkdir(parents=True, exist_ok=True)
TMP_ROOT.mkdir(parents=True, exist_ok=True)
REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)

app = FastAPI(title='Innova Live View Stream Service')
_lock = threading.Lock()
_processes: dict[int, dict] = {}
_cpu_prev = {'total': None, 'idle': None}


def _slot_public_url(slot: int) -> str:
    return f'{PUBLIC_BASE_URL}/slot-{slot}/index.m3u8'


def _webrtc_path_name(slot: int) -> str:
    return f'innova-live-{slot}'


def _slot_webrtc_url(slot: int) -> str:
    return f'{MEDIAMTX_WHEP_BASE}/{_webrtc_path_name(slot)}/whep'


def _normalize_text(value: Optional[str]) -> str:
    return (value or '').strip()


def _save_registry():
    data = {}
    for slot, info in _processes.items():
        proc = info.get('process')
        data[str(slot)] = {
            'slot': slot,
            'source': info.get('source'),
            'playlist': info.get('playlist'),
            'log_file': info.get('log_file'),
            'slot_dir': info.get('slot_dir'),
            'public_url': info.get('public_url') or _slot_public_url(slot),
            'pid': proc.pid if proc and proc.poll() is None else None,
            'mode': info.get('mode', 'ffmpeg'),
            'started_at': info.get('started_at'),
            'alive': bool(proc and proc.poll() is None),
            'web_rtc_url': info.get('web_rtc_url'),
        }
    REGISTRY_PATH.write_text(json.dumps(data, indent=2), encoding='utf-8')


def _cleanup_slot_dir(slot_dir: Path):
    if not slot_dir.exists():
        return
    for item in slot_dir.iterdir():
        try:
            if item.is_file() or item.is_symlink():
                item.unlink()
        except Exception:
            pass


def _playlist_ready(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    return any(path.parent.glob('*.ts'))


def _wait_ready(playlist: Path, timeout: float = STARTUP_TIMEOUT):
    started = time.time()
    while time.time() - started < timeout:
        if _playlist_ready(playlist):
            return True
        time.sleep(0.4)
    return False


def _read_log_tail(log_file: Path, max_chars: int = 4000) -> str:
    try:
        return log_file.read_text(encoding='utf-8', errors='ignore')[-max_chars:]
    except Exception:
        return ''


def _mediamtx_paths():
    try:
        req = urllib.request.Request(
            f'{MEDIAMTX_API_BASE}/v3/paths/list',
            headers={'Accept': 'application/json'},
        )
        with urllib.request.urlopen(req, timeout=3) as response:
            payload = json.loads(response.read().decode('utf-8'))
        return payload.get('items') or []
    except Exception:
        return []


def _webrtc_ready(path_name: str) -> bool:
    for item in _mediamtx_paths():
        if _normalize_text(item.get('name')) != path_name:
            continue
        ready = item.get('ready')
        source_ready = item.get('sourceReady')
        tracks = item.get('tracks') or []
        return bool(ready or source_ready or tracks)
    return False


def _wait_webrtc_ready(path_name: str, timeout: float = WEBRTC_STARTUP_TIMEOUT):
    started = time.time()
    while time.time() - started < timeout:
        if _webrtc_ready(path_name):
            return True
        time.sleep(0.35)
    return False


def _stop_process(slot: int):
    info = _processes.pop(slot, None)
    if not info:
        _save_registry()
        return
    proc = info.get('process')
    if proc and proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    _cleanup_slot_dir(Path(info['slot_dir']))
    _save_registry()


def _linux_meminfo():
    info = {}
    try:
        for line in Path('/proc/meminfo').read_text(encoding='utf-8').splitlines():
            if ':' not in line:
                continue
            key, raw = line.split(':', 1)
            parts = raw.strip().split()
            if not parts:
                continue
            info[key] = int(parts[0]) * 1024
    except Exception:
        return {}
    return info


def _cpu_times():
    try:
        first = Path('/proc/stat').read_text(encoding='utf-8').splitlines()[0]
        parts = first.split()
        values = [int(value) for value in parts[1:8]]
        idle = values[3] + values[4]
        total = sum(values)
        return total, idle
    except Exception:
        return None, None


def _sample_cpu_percent():
    total, idle = _cpu_times()
    if total is None:
        return -1
    prev_total = _cpu_prev['total']
    prev_idle = _cpu_prev['idle']
    if prev_total is None or prev_idle is None:
        time.sleep(0.12)
        total, idle = _cpu_times()
        if total is None:
            return -1
        prev_total = _cpu_prev['total'] or 0
        prev_idle = _cpu_prev['idle'] or 0
    _cpu_prev['total'] = total
    _cpu_prev['idle'] = idle
    delta_total = total - prev_total
    delta_idle = idle - prev_idle
    if delta_total <= 0:
        return -1
    return round((1 - (delta_idle / delta_total)) * 100, 2)


def _host_os():
    try:
        for line in Path('/etc/os-release').read_text(encoding='utf-8').splitlines():
            if line.startswith('PRETTY_NAME='):
                return line.split('=', 1)[1].strip().strip('"')
    except Exception:
        pass
    return platform.system()


def _system_health():
    meminfo = _linux_meminfo()
    mem_total = int(meminfo.get('MemTotal', 0))
    mem_available = int(meminfo.get('MemAvailable', 0) or meminfo.get('MemFree', 0))
    mem_used = max(0, mem_total - mem_available)
    disk = shutil.disk_usage('/')
    process = psutil_process_memory()
    process_cpu = -1
    system_cpu = _sample_cpu_percent()
    return {
        'availableProcessors': os.cpu_count() or 0,
        'processCpuPercent': process_cpu,
        'systemCpuPercent': system_cpu,
        'jvmMemoryUsedBytes': process,
        'jvmMemoryMaxBytes': mem_total,
        'jvmMemoryPercent': round((process * 100.0 / mem_total), 2) if mem_total > 0 else -1,
        'systemMemoryTotalBytes': mem_total,
        'systemMemoryFreeBytes': mem_available,
        'systemMemoryPercent': round((mem_used * 100.0 / mem_total), 2) if mem_total > 0 else -1,
        'diskTotalBytes': disk.total,
        'diskFreeBytes': disk.free,
        'diskPercent': round(((disk.total - disk.free) * 100.0 / disk.total), 2) if disk.total > 0 else -1,
        'hostOs': _host_os(),
    }


def psutil_process_memory():
    try:
        statm = Path('/proc/self/statm').read_text(encoding='utf-8').strip().split()
        resident_pages = int(statm[1])
        page_size = os.sysconf('SC_PAGE_SIZE')
        return resident_pages * page_size
    except Exception:
        return 0


def _summary(slots_payload: dict) -> dict:
    active = [slot for slot in slots_payload.values() if slot.get('alive')]
    return {
        'activeStreams': len(active),
        'ffmpegProcesses': len(active),
        'orphanProcesses': 0,
        'uniqueUsers': 0,
        'uniqueCameras': 0,
        'slotsInUse': len(active),
    }


class StartRequest(BaseModel):
    slot: int
    rtspUrl: str = ''
    hlsUrl: str = ''
    streamProfile: str = 'sub'


class StopRequest(BaseModel):
    slot: int


@app.get('/health')
def health():
    return {'ok': True, 'service': 'live-view-stream-service', 'system': _system_health()}


@app.get('/status')
def status():
    with _lock:
        _save_registry()
        slots = json.loads(REGISTRY_PATH.read_text(encoding='utf-8')) if REGISTRY_PATH.exists() else {}
    return {
        'ok': True,
        'generatedAt': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'system': _system_health(),
        'summary': _summary(slots),
        'slots': slots,
    }


@app.post('/start')
def start(req: StartRequest):
    if req.slot < 0 or req.slot > 63:
        raise HTTPException(status_code=400, detail='Invalid slot')

    hls_url = _normalize_text(req.hlsUrl)
    if hls_url.lower().endswith('.m3u8'):
        with _lock:
            _stop_process(req.slot)
        return {'ok': True, 'url': hls_url, 'mode': 'hls-external'}

    rtsp_url = _normalize_text(req.rtspUrl)
    if not rtsp_url.lower().startswith('rtsp://'):
        raise HTTPException(status_code=400, detail='Missing rtspUrl')

    with _lock:
        existing = _processes.get(req.slot)
        if existing:
            existing_source = _normalize_text(existing.get('source'))
            existing_proc = existing.get('process')
            existing_playlist = Path(existing.get('playlist') or '')
            if existing_source == rtsp_url and existing_proc and existing_proc.poll() is None and _playlist_ready(existing_playlist):
                _save_registry()
                return {
                    'ok': True,
                    'url': existing.get('public_url') or _slot_public_url(req.slot),
                    'mode': existing.get('mode', 'ffmpeg-hls'),
                    'reused': True,
                }

        _stop_process(req.slot)
        slot_dir = HLS_ROOT / f'slot-{req.slot}'
        slot_dir.mkdir(parents=True, exist_ok=True)
        playlist = slot_dir / 'index.m3u8'
        log_file = LOG_ROOT / f'slot-{req.slot}-ffmpeg.log'
        cmd = [
            FFMPEG_PATH,
            '-rtsp_transport', 'tcp',
            '-i', rtsp_url,
            '-an',
            '-vf', 'scale=640:-2',
            '-r', '10',
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-pix_fmt', 'yuv420p',
            '-b:v', '650k',
            '-maxrate', '650k',
            '-bufsize', '900k',
            '-g', '20',
            '-keyint_min', '20',
            '-sc_threshold', '0',
            '-f', 'hls',
            '-hls_time', str(HLS_TIME),
            '-hls_list_size', str(HLS_LIST_SIZE),
            '-hls_flags', 'delete_segments+append_list+omit_endlist+independent_segments',
            '-hls_segment_filename', str(slot_dir / 'seg_%06d.ts'),
            str(playlist),
        ]
        with open(log_file, 'ab') as fh:
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
        _processes[req.slot] = {
            'process': proc,
            'source': rtsp_url,
            'playlist': str(playlist),
            'log_file': str(log_file),
            'slot_dir': str(slot_dir),
            'public_url': _slot_public_url(req.slot),
            'mode': 'ffmpeg-hls',
            'started_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        }
        _save_registry()

    if not _wait_ready(playlist, timeout=STARTUP_TIMEOUT):
        with _lock:
            log_tail = _read_log_tail(log_file)
            _stop_process(req.slot)
        raise HTTPException(status_code=502, detail=f'HLS playlist not ready. ffmpeg log tail: {log_tail}')

    return {'ok': True, 'url': _slot_public_url(req.slot), 'mode': 'hls-transcode', 'reused': False}


@app.post('/webrtc/start')
def start_webrtc(req: StartRequest):
    if req.slot < 0 or req.slot > 63:
        raise HTTPException(status_code=400, detail='Invalid slot')

    rtsp_url = _normalize_text(req.rtspUrl)
    if not rtsp_url.lower().startswith('rtsp://'):
        raise HTTPException(status_code=400, detail='Missing rtspUrl')

    path_name = _webrtc_path_name(req.slot)
    publish_url = f'{MEDIAMTX_RTSP_BASE}/{path_name}'

    with _lock:
        existing = _processes.get(req.slot)
        if existing:
            existing_source = _normalize_text(existing.get('source'))
            existing_proc = existing.get('process')
            existing_mode = _normalize_text(existing.get('mode'))
            if (
                existing_source == rtsp_url
                and existing_mode == 'ffmpeg-webrtc'
                and existing_proc
                and existing_proc.poll() is None
                and _webrtc_ready(path_name)
            ):
                _save_registry()
                return {
                    'ok': True,
                    'url': existing.get('web_rtc_url') or _slot_webrtc_url(req.slot),
                    'webRtcUrl': existing.get('web_rtc_url') or _slot_webrtc_url(req.slot),
                    'mode': existing_mode,
                    'reused': True,
                }

        _stop_process(req.slot)
        slot_dir = TMP_ROOT / f'webrtc-slot-{req.slot}'
        slot_dir.mkdir(parents=True, exist_ok=True)
        log_file = LOG_ROOT / f'slot-{req.slot}-webrtc.log'
        cmd = [
            FFMPEG_PATH,
            '-rtsp_transport', 'tcp',
            '-i', rtsp_url,
            '-an',
            '-vf', 'scale=640:-2',
            '-r', '10',
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-pix_fmt', 'yuv420p',
            '-b:v', '650k',
            '-maxrate', '650k',
            '-bufsize', '900k',
            '-g', '20',
            '-keyint_min', '20',
            '-sc_threshold', '0',
            '-f', 'rtsp',
            '-rtsp_transport', 'tcp',
            publish_url,
        ]
        with open(log_file, 'ab') as fh:
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
        _processes[req.slot] = {
            'process': proc,
            'source': rtsp_url,
            'playlist': '',
            'log_file': str(log_file),
            'slot_dir': str(slot_dir),
            'public_url': '',
            'web_rtc_url': _slot_webrtc_url(req.slot),
            'mode': 'ffmpeg-webrtc',
            'started_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        }
        _save_registry()

    if not _wait_webrtc_ready(path_name, timeout=WEBRTC_STARTUP_TIMEOUT):
        with _lock:
            log_tail = _read_log_tail(log_file)
            _stop_process(req.slot)
        raise HTTPException(status_code=502, detail=f'WebRTC publish not ready. ffmpeg log tail: {log_tail}')

    return {
        'ok': True,
        'url': _slot_webrtc_url(req.slot),
        'webRtcUrl': _slot_webrtc_url(req.slot),
        'mode': 'webrtc-publish',
        'reused': False,
    }


@app.post('/stop')
def stop(req: StopRequest):
    with _lock:
        _stop_process(req.slot)
    return {'ok': True}
