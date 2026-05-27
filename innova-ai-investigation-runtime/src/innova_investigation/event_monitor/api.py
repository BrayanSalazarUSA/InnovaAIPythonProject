from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .. import config as runtime_config
from ..bridges.dahua import build_rtsp_url as build_dahua_rtsp_url
from ..bridges.hikvision import build_rtsp_url as build_hikvision_rtsp_url
from ..bridges.uniview import build_rtsp_url as build_uniview_rtsp_url
from .models import MonitorConfig, resolve_path
from .monitor import EventMonitor, load_monitor_config
from .storage import load_recent_events, load_recent_objects


LOGGER = logging.getLogger(__name__)


@dataclass
class MonitorJob:
    job_id: str
    status: str = "running"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    progress: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: str = ""
    source_label: str = "api"
    auto_restart: bool = False
    stop_requested: bool = False
    monitor: EventMonitor | None = None
    thread: threading.Thread | None = None
    config_summary: dict[str, Any] = field(default_factory=dict)


JOBS: dict[str, MonitorJob] = {}
JOBS_LOCK = threading.Lock()

app = FastAPI(title="Innova Event Monitor MVP")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def register_event_monitor(target_app: FastAPI) -> FastAPI:
    target_app.include_router(app.router)
    return target_app


def _touch(job: MonitorJob) -> None:
    job.updated_at = datetime.now(timezone.utc).isoformat()


def _normalize_vendor(value: str) -> str:
    raw = str(value or "").strip().lower()
    if raw.startswith("hik"):
        return "hikvision"
    if raw.startswith("dah"):
        return "dahua"
    if raw.startswith("unv") or raw.startswith("uni") or "uniview" in raw or "uniarch" in raw:
        return "uniview"
    return raw


def _load_backend_nvr_profile(nvr_id: str | int | None) -> dict[str, Any] | None:
    raw_id = str(nvr_id or "").strip()
    if not raw_id:
        return None
    try:
        response = requests.get(
            f"{runtime_config.BACKEND_API_BASE_URL}/nvrs/{raw_id}/investigation-profile",
            timeout=5,
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _resolve_monitor_source(config: MonitorConfig) -> MonitorConfig:
    source = str(config.source or "").strip()
    vendor = _normalize_vendor(config.vendor)
    host = str(config.host or "").strip()
    rtsp_port = int(config.rtsp_port or 554)
    logical_channel = max(1, int(config.logical_channel or 1))
    username = str(config.username or "").strip()
    password = str(config.password or "").strip()
    nvr_name = str(config.nvr_name or "").strip()

    backend_profile = _load_backend_nvr_profile(config.nvr_id)
    if backend_profile:
        vendor = _normalize_vendor(str(backend_profile.get("brand") or backend_profile.get("vendor") or vendor))
        host = str(backend_profile.get("host") or host or "").strip()
        rtsp_port = int(backend_profile.get("rtspPort") or rtsp_port or 554)
        username = str(backend_profile.get("username") or username or "").strip()
        password = str(backend_profile.get("password") or password or "").strip()
        nvr_name = str(backend_profile.get("name") or nvr_name or "").strip()

    needs_source_resolution = (not source) or ("USER:PASSWORD@" in source)
    if not needs_source_resolution:
        return config

    if not vendor or not host:
        raise RuntimeError("No tengo suficiente contexto del NVR para construir el stream RTSP.")
    if not username or not password:
        raise RuntimeError("No pude resolver las credenciales del NVR para abrir el stream.")

    if vendor == "hikvision":
        source = build_hikvision_rtsp_url(
            host=host,
            rtsp_port=rtsp_port,
            username=username,
            password=password,
            logical_channel=logical_channel,
            stream_variant="sub" if str(config.stream_variant or "").lower() == "sub" else "main",
        )
    elif vendor == "dahua":
        source = build_dahua_rtsp_url(
            host=host,
            rtsp_port=rtsp_port,
            username=username,
            password=password,
            channel=logical_channel,
            subtype=1 if str(config.stream_variant or "").lower() == "sub" else 0,
        )
    elif vendor == "uniview":
        source = build_uniview_rtsp_url(
            host=host,
            rtsp_port=rtsp_port,
            username=username,
            password=password,
            logical_channel=logical_channel,
            stream_variant="sub" if str(config.stream_variant or "").lower() == "sub" else "main",
        )
    else:
        raise RuntimeError(f"Vendor no soportado para modo vigilancia: {vendor}")

    return config.model_copy(
        update={
            "source": source,
            "vendor": vendor,
            "host": host,
            "rtsp_port": rtsp_port,
            "logical_channel": logical_channel,
            "username": username,
            "password": password,
            "nvr_name": nvr_name,
        }
    )


def _serialize_job(job: MonitorJob) -> dict[str, object]:
    return {
        "jobId": job.job_id,
        "status": job.status,
        "createdAt": job.created_at,
        "updatedAt": job.updated_at,
        "sourceLabel": job.source_label,
        "autoRestart": job.auto_restart,
        "progress": job.progress,
        "result": job.result,
        "error": job.error,
        "config": job.config_summary,
        "cameraId": job.config_summary.get("cameraId"),
        "cameraName": job.config_summary.get("cameraName"),
        "outputDir": job.config_summary.get("outputDir"),
        "ruleNames": job.config_summary.get("ruleNames", []),
    }


def _start_job_thread(
    config: MonitorConfig,
    *,
    source_label: str = "api",
    auto_restart: bool = False,
) -> MonitorJob:
    config = _resolve_monitor_source(config)
    job_id = uuid.uuid4().hex
    job = MonitorJob(
        job_id=job_id,
        source_label=source_label,
        auto_restart=auto_restart,
        config_summary={
            "cameraId": config.camera_id,
            "cameraName": config.camera_name,
            "source": config.source,
            "outputDir": config.output_dir,
            "sampleEverySeconds": config.sample_every_seconds,
            "maxRuntimeSeconds": config.max_runtime_seconds,
            "ruleNames": [rule.name for rule in config.rules],
            "ruleTypes": [rule.type for rule in config.rules],
        },
    )

    def on_progress(update: dict[str, object]) -> None:
        with JOBS_LOCK:
            current = JOBS.get(job_id)
            if current:
                current.progress = dict(update)
                _touch(current)

    def runner() -> None:
        restart_delay_seconds = 5
        while True:
            monitor = EventMonitor(config, on_progress=on_progress)
            with JOBS_LOCK:
                current = JOBS.get(job_id)
                if not current or current.stop_requested:
                    return
                current.status = "running"
                current.monitor = monitor
                current.error = ""
                _touch(current)

            try:
                result = monitor.run()
                with JOBS_LOCK:
                    current = JOBS.get(job_id)
                    if not current:
                        return
                    current.result = result
                    _touch(current)
                    should_restart = current.auto_restart and not current.stop_requested
                    if not should_restart:
                        current.status = "done" if not current.stop_requested else "stopped"
                        return
                    current.progress = {
                        "stage": "restart",
                        "detail": f"Stream finalizado; reiniciando en {restart_delay_seconds}s.",
                        "last_result": result,
                    }
            except Exception as exc:
                with JOBS_LOCK:
                    current = JOBS.get(job_id)
                    if not current:
                        return
                    current.error = str(exc)
                    _touch(current)
                    should_restart = current.auto_restart and not current.stop_requested
                    if not should_restart:
                        current.status = "error"
                        return
                    current.progress = {
                        "stage": "restart",
                        "detail": f"Error del monitor; reiniciando en {restart_delay_seconds}s: {exc}",
                    }

            time.sleep(restart_delay_seconds)

    job.thread = threading.Thread(target=runner, name=f"event-monitor-{job_id[:8]}", daemon=True)
    with JOBS_LOCK:
        JOBS[job_id] = job
    job.thread.start()
    return job


def _configured_autostart_paths() -> list[str]:
    raw = os.getenv("INNOVA_EVENT_MONITOR_AUTOSTART_CONFIGS") or os.getenv("INNOVA_EVENT_MONITOR_AUTOSTART_CONFIG") or ""
    return [item.strip() for item in raw.split(",") if item.strip()]


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


@app.on_event("startup")
def autostart_configured_jobs() -> None:
    for config_path in _configured_autostart_paths():
        try:
            config = load_monitor_config(config_path)
            _start_job_thread(
                config,
                source_label=f"autostart:{config_path}",
                auto_restart=_env_flag("INNOVA_EVENT_MONITOR_RESTART_ON_ERROR", default=True),
            )
            LOGGER.info("Autostarted event monitor config: %s", config_path)
        except Exception:
            LOGGER.exception("Could not autostart event monitor config: %s", config_path)


@app.get("/api/event-monitor/health")
def health() -> dict[str, object]:
    return {"ok": True, "service": "event-monitor", "jobs": len(JOBS)}


@app.get("/api/event-monitor/jobs")
def list_monitor_jobs() -> dict[str, object]:
    with JOBS_LOCK:
        return {"jobs": [_serialize_job(job) for job in JOBS.values()]}


@app.post("/api/event-monitor/jobs")
def start_monitor_job(config: MonitorConfig, background: BackgroundTasks) -> dict[str, str]:
    _ = background
    job = _start_job_thread(config)
    return {"jobId": job.job_id}


@app.get("/api/event-monitor/jobs/{job_id}")
def get_monitor_job(job_id: str) -> dict[str, object]:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return _serialize_job(job)


@app.post("/api/event-monitor/jobs/{job_id}/stop")
def stop_monitor_job(job_id: str) -> dict[str, object]:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        job.stop_requested = True
        if job.monitor:
            job.monitor.stop()
        job.status = "stopping"
        _touch(job)
    return {"ok": True, "jobId": job_id}


@app.get("/api/event-monitor/events")
def list_events(outputDir: str = "output/event_monitor", limit: int = 100) -> dict[str, object]:
    output_dir = resolve_path(outputDir, base_dir=runtime_config.PROJECT_ROOT)
    return {"events": load_recent_events(output_dir, limit=max(1, min(limit, 500)))}


@app.get("/api/event-monitor/objects")
def list_objects(outputDir: str = "output/event_monitor", limit: int = 100) -> dict[str, object]:
    output_dir = resolve_path(outputDir, base_dir=runtime_config.PROJECT_ROOT)
    return {"objects": load_recent_objects(output_dir, limit=max(1, min(limit, 500)))}


@app.get("/api/event-monitor/artifact")
def get_artifact(path: str) -> FileResponse:
    candidate = Path(path).expanduser().resolve()
    project_root = runtime_config.PROJECT_ROOT.resolve()
    # Keep file serving scoped to the project output tree.
    if project_root not in candidate.parents and candidate != project_root:
        raise HTTPException(status_code=400, detail="Invalid artifact path")
    if not candidate.exists():
        raise HTTPException(status_code=404, detail="Artifact not found")
    media_type = "image/jpeg" if candidate.suffix.lower() in {".jpg", ".jpeg"} else None
    return FileResponse(candidate, media_type=media_type)


def main() -> None:
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Innova Event Monitor API")
    parser.add_argument("--host", default=os.getenv("INNOVA_EVENT_MONITOR_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("INNOVA_EVENT_MONITOR_PORT", "8522")))
    parser.add_argument(
        "--autostart-config",
        action="append",
        default=[],
        help="JSON config to start automatically. Can be passed more than once.",
    )
    parser.add_argument(
        "--restart-on-error",
        action="store_true",
        default=_env_flag("INNOVA_EVENT_MONITOR_RESTART_ON_ERROR", default=False),
        help="Restart autostart monitors if the stream ends or errors.",
    )
    args = parser.parse_args()

    if args.autostart_config:
        os.environ["INNOVA_EVENT_MONITOR_AUTOSTART_CONFIGS"] = ",".join(args.autostart_config)
    os.environ["INNOVA_EVENT_MONITOR_RESTART_ON_ERROR"] = "true" if args.restart_on_error else "false"

    host = args.host
    port = args.port
    uvicorn.run("innova_investigation.event_monitor.api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
