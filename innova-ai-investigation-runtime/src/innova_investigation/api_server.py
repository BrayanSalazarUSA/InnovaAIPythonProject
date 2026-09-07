from __future__ import annotations

import json
import threading
import time
import uuid
import base64
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import os
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import cv2
import numpy as np
import requests
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict

from . import config as runtime_config
from .bridges.dahua import (
    DahuaBridgeSettings,
    DahuaRemoteBridgeSettings,
    download_clip_via_bridge_dahua,
    download_clip_via_sdk,
    fetch_snapshot_bytes_via_http,
    fetch_snapshot_bytes_via_rtsp as fetch_dahua_snapshot_bytes_via_rtsp,
    find_dahua_sdk_root,
    list_channels as list_dahua_channels,
)
from .bridges.hikvision import (
    HikvisionBridgeSettings,
    download_clip_via_bridge,
    download_clip_via_local_sdk,
    fetch_snapshot_bytes_via_isapi,
    fetch_snapshot_bytes_via_rtsp as fetch_hikvision_snapshot_bytes_via_rtsp,
    list_channels_with_status_via_isapi,
)
from .bridges.uniview import (
    fetch_snapshot_bytes_via_lapi as fetch_uniview_snapshot_bytes_via_lapi,
    fetch_snapshot_bytes_via_rtsp as fetch_uniview_snapshot_bytes_via_rtsp,
    list_channels as list_uniview_channels,
)
from .investigation_api_engine import (
    _prefer_native_clip_variant,
    ensure_analysis_clip,
    ensure_openable_clip,
    extract_video_segment,
    format_seconds,
    probe_static_object_clip,
    quick_scan_clip,
    run_deep_analysis,
)
from .event_monitor.api import register_event_monitor
from .similarity_search import SimilaritySearcher

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional runtime dependency
    YOLO = None


OUTPUT_ROOT = runtime_config.OUTPUT_DIR / "api_jobs"
NVR_PROFILES_PATH = runtime_config.NVR_PROFILES_PATH
DAHUA_PLAYBACK_CHANNEL_MAP_PATH = runtime_config.DAHUA_PLAYBACK_CHANNEL_MAP_PATH
BACKEND_API_BASE_URL = runtime_config.BACKEND_API_BASE_URL
VIDEO_ARTIFACT_EXTENSIONS = {".mp4", ".dav", ".avi", ".mov", ".mkv", ".264", ".h264"}
STATIC_DISCOVERY_DEFAULT_MICROCLIP_SECONDS = 8.0
STATIC_DISCOVERY_MAX_MICROCLIP_SECONDS = 30.0
STATIC_DISCOVERY_MIN_TARGET_WINDOW_SECONDS = 30.0
STATIC_DISCOVERY_MAX_ITERATIONS = 24
STATIC_DISCOVERY_MAX_SAFE_CANDIDATE_WINDOW_SECONDS = max(
    STATIC_DISCOVERY_MIN_TARGET_WINDOW_SECONDS,
    float(os.getenv("STATIC_DISCOVERY_MAX_SAFE_CANDIDATE_WINDOW_SECONDS", "600") or 600),
)
STATIC_DISCOVERY_MIN_ROI_AREA = 0.0025
STATIC_DISCOVERY_MAX_ROI_AREA = 0.80
FIRST_APPEARANCE_MAX_INITIAL_CLIP_SECONDS = max(
    60.0,
    float(os.getenv("FIRST_APPEARANCE_MAX_INITIAL_CLIP_SECONDS", "300") or 300),
)
BROWSER_TRANSCODE_MAX_BYTES = max(
    1,
    int(float(os.getenv("INNOVA_BROWSER_TRANSCODE_MAX_MB", "80") or 80) * 1024 * 1024),
)
DEEP_PERSON_DETECTION_FRAME_STEP = max(
    1,
    int(float(os.getenv("INNOVA_DEEP_PERSON_DETECTION_FRAME_STEP", "3") or 3)),
)
DEEP_SAVE_ANNOTATED_VIDEO = str(os.getenv("INNOVA_DEEP_SAVE_ANNOTATED_VIDEO", "0")).lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _parse_iso_dt(value: str) -> datetime:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("Fecha vacía.")
    raw = raw.replace("Z", "+00:00")
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is not None:
        dt = dt.astimezone().replace(tzinfo=None)
    return dt


def _normalize_vendor(value: str) -> str:
    raw = str(value or "").strip().lower()
    if raw.startswith("hik"):
        return "hikvision"
    if raw.startswith("dah"):
        return "dahua"
    if raw.startswith("unv") or raw.startswith("uni") or "uniview" in raw or "uniarch" in raw:
        return "uniview"
    return raw or "hikvision"


def _normalize_lookup_key(value: Any) -> str:
    return str(value or "").strip().lower()


def _load_dahua_playback_channel_map() -> list[dict[str, Any]]:
    try:
        if not DAHUA_PLAYBACK_CHANNEL_MAP_PATH.exists():
            return []
        payload = json.loads(DAHUA_PLAYBACK_CHANNEL_MAP_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []
    entries = payload.get("mappings") if isinstance(payload, dict) else payload
    return [entry for entry in entries or [] if isinstance(entry, dict)]


def _resolve_dahua_playback_sdk_channel(
    *,
    host: str,
    nvr_id: str | int | None,
    nvr_name: str,
    camera_id: str | int | None,
    camera_name: str,
    logical_channel: int,
) -> tuple[int, dict[str, Any]]:
    logical = max(1, int(logical_channel or 1))
    default_sdk_channel = max(0, logical - 1)
    target_host = _normalize_lookup_key(host)
    target_nvr_id = _normalize_lookup_key(nvr_id)
    target_nvr_name = _normalize_lookup_key(nvr_name)
    target_camera_id = _normalize_lookup_key(camera_id)
    target_camera_name = _normalize_lookup_key(camera_name)

    for entry in _load_dahua_playback_channel_map():
        if entry.get("enabled") is False:
            continue
        entry_host = _normalize_lookup_key(entry.get("host"))
        entry_nvr_id = _normalize_lookup_key(entry.get("nvrId") or entry.get("nvr_id"))
        entry_nvr_name = _normalize_lookup_key(entry.get("nvrName") or entry.get("nvr_name"))
        entry_camera_id = _normalize_lookup_key(entry.get("cameraId") or entry.get("camera_id"))
        entry_camera_name = _normalize_lookup_key(entry.get("cameraName") or entry.get("camera_name"))
        entry_logical = int(entry.get("logicalChannel") or entry.get("logical_channel") or 0)

        nvr_matches = (
            bool(target_nvr_id and entry_nvr_id == target_nvr_id)
            or bool(target_host and entry_host == target_host)
            or bool(target_nvr_name and entry_nvr_name == target_nvr_name)
        )
        camera_matches = (
            bool(target_camera_id and entry_camera_id == target_camera_id)
            or bool(target_camera_name and entry_camera_name == target_camera_name)
            or bool(entry_logical and entry_logical == logical)
        )
        if not (nvr_matches and camera_matches):
            continue

        raw_sdk_channel = (
            entry.get("sdkPlaybackChannel")
            if entry.get("sdkPlaybackChannel") is not None
            else entry.get("sdk_channel")
            if entry.get("sdk_channel") is not None
            else entry.get("playbackChannel")
        )
        if raw_sdk_channel is None:
            continue
        sdk_channel = max(0, int(raw_sdk_channel))
        return sdk_channel, {
            "source": "dahua_playback_channel_map",
            "mapPath": str(DAHUA_PLAYBACK_CHANNEL_MAP_PATH),
            "logicalChannel": logical,
            "sdkPlaybackChannel": sdk_channel,
            "cameraId": str(camera_id or ""),
            "cameraName": str(camera_name or ""),
            "nvrId": str(nvr_id or ""),
            "nvrName": str(nvr_name or ""),
        }

    return default_sdk_channel, {
        "source": "default_zero_based",
        "logicalChannel": logical,
        "sdkPlaybackChannel": default_sdk_channel,
        "cameraId": str(camera_id or ""),
        "cameraName": str(camera_name or ""),
        "nvrId": str(nvr_id or ""),
        "nvrName": str(nvr_name or ""),
    }


def _load_nvr_profiles() -> list[dict[str, Any]]:
    try:
        if not NVR_PROFILES_PATH.exists():
            return []
        import json

        payload = json.loads(NVR_PROFILES_PATH.read_text(encoding="utf-8"))
        profiles = payload.get("profiles", [])
        if isinstance(profiles, list):
            return [p for p in profiles if isinstance(p, dict)]
        return []
    except Exception:
        return []


def _pick_profile(
    *,
    profiles: list[dict[str, Any]],
    nvr_name: str,
    host: str,
    vendor: str,
    http_port: int,
    sdk_port: int,
) -> dict[str, Any] | None:
    def norm(s: Any) -> str:
        return str(s or "").strip().lower()

    target_vendor = norm(vendor)
    target_name = str(nvr_name or "").strip()
    target_host = norm(host)

    # 1) By exact name (best)
    if target_name:
        for profile in profiles:
            if str(profile.get("name", "")).strip() == target_name:
                return profile

    # 2) By host+vendor+ports
    for profile in profiles:
        if target_vendor and norm(profile.get("vendor")) != target_vendor:
            continue
        if target_host and norm(profile.get("host")) != target_host:
            continue
        if http_port and int(profile.get("http_port") or 0) != int(http_port):
            continue
        if sdk_port and int(profile.get("sdk_port") or 0) != int(sdk_port):
            continue
        return profile

    # 3) By host+vendor only
    for profile in profiles:
        if target_vendor and norm(profile.get("vendor")) != target_vendor:
            continue
        if target_host and norm(profile.get("host")) != target_host:
            continue
        return profile

    return None


def _load_backend_nvr_profile(nvr_id: str | int | None) -> dict[str, Any] | None:
    raw_id = str(nvr_id or "").strip()
    if not raw_id:
        return None

    try:
        response = requests.get(
            f"{BACKEND_API_BASE_URL}/nvrs/{raw_id}/investigation-profile",
            timeout=5,
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _load_backend_nvr_profile_by_hint(
    *,
    nvr_name: str,
    property_id: str | int | None,
    property_name: str,
    host: str,
) -> dict[str, Any] | None:
    def norm(value: Any) -> str:
        return str(value or "").strip().lower()

    target_name = norm(nvr_name)
    target_property_id = str(property_id or "").strip()
    target_property_name = norm(property_name)
    target_host = norm(host)

    if not any((target_name, target_property_id, target_property_name, target_host)):
        return None

    try:
        response = requests.get(f"{BACKEND_API_BASE_URL}/nvrs", timeout=5)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return None

    nvrs = payload if isinstance(payload, list) else []
    best_id: Any = None
    for item in nvrs:
        if not isinstance(item, dict):
            continue
        item_id = item.get("id")
        item_property = item.get("property") if isinstance(item.get("property"), dict) else {}
        item_property_id = str(item.get("propertyId") or item_property.get("id") or "").strip()
        item_property_name = norm(item.get("propertyName") or item_property.get("name"))
        item_name = norm(item.get("name"))
        item_host = norm(item.get("host"))

        if target_host and item_host == target_host:
            best_id = item_id
            break
        if target_name and item_name == target_name:
            best_id = item_id
            break
        if target_property_id and item_property_id == target_property_id and (target_name in item_name or item_name in target_property_name):
            best_id = item_id
            break
        if target_property_name and item_property_name == target_property_name and (target_name in item_name or item_name in target_property_name):
            best_id = item_id
            break

    return _load_backend_nvr_profile(best_id)


@dataclass(slots=True)
class JobState:
    job_id: str
    created_at: str
    updated_at: str
    status: str = "running"  # running|done|error
    stage: str = "prepare"
    detail: str = "Iniciando..."
    progress: float = 0.0
    error: str = ""
    result: dict[str, Any] | None = None
    partial_result: dict[str, Any] | None = None
    cancel_requested: bool = False
    base_url: str = ""
    job_dir: Path = field(default_factory=Path)
    performance: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "jobId": self.job_id,
            "status": self.status,
            "stage": self.stage,
            "detail": self.detail,
            "progress": round(float(self.progress), 4),
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
        }
        if self.status == "error":
            payload["error"] = self.error or "Job failed"
        if self.status == "cancelled":
            payload["cancelled"] = True
        if self.status == "running" and self.partial_result is not None:
            payload["partialResult"] = self.partial_result
        if self.status == "done" and self.result is not None:
            payload["result"] = self.result
        if self.performance:
            payload["performance"] = self.performance
        return payload


JOBS: dict[str, JobState] = {}
JOBS_LOCK = threading.Lock()
ENGINE_LOCK = threading.Lock()
PERSON_DETECTOR: Any | None = None


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _persist_job_state(job: JobState) -> None:
    if not job.job_dir:
        return
    try:
        job.job_dir.mkdir(parents=True, exist_ok=True)
        with (job.job_dir / "job_state.json").open("w", encoding="utf-8") as file:
            json.dump(job.to_payload(), file, indent=2, ensure_ascii=False)
        if job.performance:
            with (job.job_dir / "performance.json").open("w", encoding="utf-8") as file:
                json.dump(job.performance, file, indent=2, ensure_ascii=False)
    except Exception:
        pass


def _record_job_metric(job_id: str, phase: str, elapsed_seconds: float, **metadata: Any) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        phases = job.performance.setdefault("phases", {})
        current = phases.setdefault(
            phase,
            {
                "count": 0,
                "totalSeconds": 0.0,
                "lastSeconds": 0.0,
                "maxSeconds": 0.0,
                "samples": [],
            },
        )
        elapsed = round(max(0.0, float(elapsed_seconds)), 3)
        current["count"] = int(current.get("count", 0) or 0) + 1
        current["totalSeconds"] = round(float(current.get("totalSeconds", 0.0) or 0.0) + elapsed, 3)
        current["lastSeconds"] = elapsed
        current["maxSeconds"] = round(max(float(current.get("maxSeconds", 0.0) or 0.0), elapsed), 3)
        if metadata:
            samples = current.setdefault("samples", [])
            samples.append({"seconds": elapsed, **metadata})
            current["samples"] = samples[-30:]
        job.performance["updatedAt"] = _now_iso()
        job.updated_at = job.performance["updatedAt"]
        _persist_job_state(job)


@contextmanager
def _measure_job_phase(job_id: str, phase: str, **metadata: Any):
    started = time.perf_counter()
    try:
        yield
    finally:
        _record_job_metric(job_id, phase, time.perf_counter() - started, **metadata)


def _update_job(job_id: str, *, stage: str | None = None, detail: str | None = None, progress: float | None = None) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        if stage is not None:
            job.stage = stage
        if detail is not None:
            job.detail = detail
        if progress is not None:
            job.progress = max(0.0, min(1.0, float(progress)))
        job.updated_at = _now_iso()
        _persist_job_state(job)


def _update_job_partial(job_id: str, partial_result: dict[str, Any]) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        job.partial_result = partial_result
        job.updated_at = _now_iso()
        _persist_job_state(job)


def _make_live_preview_callback(output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    last_write = {"at": 0.0}

    def write_preview(frame: Any) -> None:
        now = time.monotonic()
        if now - last_write["at"] < 0.75:
            return
        try:
            cv2.imwrite(str(output_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            last_write["at"] = now
        except Exception:
            pass

    return write_preview


def _job_cancel_requested(job_id: str) -> bool:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        return bool(job and job.cancel_requested)


def _artifact_url(job: JobState, relative_path: Path) -> str:
    rel = relative_path.as_posix().lstrip("/")
    return f"{job.base_url}/investigation/artifacts/{job.job_id}/{rel}"


def _resolve_public_api_base(request: Request) -> str:
    configured = str(getattr(runtime_config, "PUBLIC_API_BASE_URL", "") or "").strip().rstrip("/")
    if configured:
        return configured

    base = str(request.base_url).rstrip("/")
    forwarded_prefix = str(request.headers.get("x-forwarded-prefix", "") or "").strip().rstrip("/")
    if forwarded_prefix:
        return f"{base}{forwarded_prefix}"
    return f"{base}/api"


def _artifact_url_for_path(job: JobState, path: str | Path | None) -> str:
    if not path:
        return ""
    try:
        p = Path(path)
        if p.exists():
            return _artifact_url(job, p.relative_to(job.job_dir))
    except Exception:
        return ""
    return ""


def _artifact_url_for_openable_video_path(job: JobState, path: str | Path | None) -> str:
    if not path:
        return ""
    try:
        source = Path(path)
        if not source.exists():
            return ""
        if source.suffix.lower() not in VIDEO_ARTIFACT_EXTENSIONS:
            return _artifact_url(job, source.relative_to(job.job_dir))
        if source.stat().st_size > BROWSER_TRANSCODE_MAX_BYTES:
            return _artifact_url(job, source.relative_to(job.job_dir))
        browser_candidate = source.parent / f"{source.stem}_browser.mp4"
        if browser_candidate.exists() and browser_candidate.resolve() != source.resolve():
            openable = ensure_openable_clip(browser_candidate, output_dir=browser_candidate.parent)
            return _artifact_url(job, openable.relative_to(job.job_dir))
        openable = ensure_openable_clip(source, output_dir=source.parent)
        return _artifact_url(job, openable.relative_to(job.job_dir))
    except Exception:
        return ""


def _artifact_url_for_media_path(job: JobState, path: str | Path | None) -> str:
    try:
        p = Path(path) if path else None
    except Exception:
        return ""
    if p is not None and p.suffix.lower() in VIDEO_ARTIFACT_EXTENSIONS:
        return _artifact_url_for_openable_video_path(job, p)
    return _artifact_url_for_path(job, p)


def _media_url_from_payload(job: JobState, payload: dict[str, Any], path_key: str = "path", url_key: str = "url") -> str:
    path_value = payload.get(path_key)
    try:
        path = Path(path_value) if path_value else None
    except Exception:
        path = None
    if path is not None and path.suffix.lower() in VIDEO_ARTIFACT_EXTENSIONS:
        video_url = _artifact_url_for_openable_video_path(job, path)
        if video_url:
            return video_url
    existing_url = str(payload.get(url_key) or "").strip()
    if existing_url:
        return existing_url
    return _artifact_url_for_media_path(job, path_value)


def _append_unique_url(values: list[str], url: str) -> None:
    normalized = str(url or "").strip()
    if normalized and normalized not in values:
        values.append(normalized)


def _enrich_path_url(job: JobState, payload: dict[str, Any], path_key: str, url_key: str | None = None) -> dict[str, Any]:
    updated = dict(payload)
    url = _artifact_url_for_media_path(job, updated.get(path_key))
    if url:
        updated[url_key or path_key.replace("_path", "_url")] = url
    return updated


def _build_report_artifacts(job: JobState, result: dict[str, Any]) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}

    clip = result.get("clip")
    if isinstance(clip, dict):
        url = _media_url_from_payload(job, clip)
        if url:
            artifacts["object_source_clip_url"] = url
            artifacts["object_clip_url"] = url

    moment_clip = result.get("object_moment_clip")
    if isinstance(moment_clip, dict):
        url = _media_url_from_payload(job, moment_clip)
        if url:
            artifacts["object_moment_clip_url"] = url

    deep = result.get("deep")
    if isinstance(deep, dict):
        analysis_video_urls: list[str] = []
        for url_key in ["annotated_video_url"]:
            _append_unique_url(analysis_video_urls, str(deep.get(url_key) or ""))
        for path_key in ["annotated_video_path"]:
            _append_unique_url(analysis_video_urls, _artifact_url_for_openable_video_path(job, deep.get(path_key)))
        if analysis_video_urls:
            artifacts["object_analysis_video_url"] = analysis_video_urls[0]
            artifacts["object_analysis_video_urls"] = analysis_video_urls

        matches = deep.get("matches") or []
        if isinstance(matches, list) and not artifacts.get("object_moment_clip_url"):
            clip_urls: list[str] = []
            for match in matches:
                if not isinstance(match, dict):
                    continue
                _append_unique_url(clip_urls, str(match.get("clip_url") or ""))
                _append_unique_url(clip_urls, _artifact_url_for_openable_video_path(job, match.get("clip_path")))
            if clip_urls:
                artifacts.setdefault("object_moment_clip_url", clip_urls[0])
                artifacts["object_moment_clip_urls"] = clip_urls

    person_track = result.get("person_track")
    if isinstance(person_track, dict):
        track_clip = person_track.get("clip")
        if isinstance(track_clip, dict):
            url = _media_url_from_payload(job, track_clip)
            if url:
                artifacts["person_tracking_source_clip_url"] = url
                artifacts["person_tracking_clip_url"] = url

        tracking_clip = person_track.get("tracking_clip")
        if isinstance(tracking_clip, dict):
            url = _media_url_from_payload(job, tracking_clip)
            if url:
                artifacts["person_tracking_moment_clip_url"] = url

        track_deep = person_track.get("deep")
        if isinstance(track_deep, dict):
            matches = track_deep.get("matches") or track_deep.get("top_hits") or []
            if isinstance(matches, list):
                tracking_clip_urls: list[str] = []
                frame_urls = [
                    str(match.get("annotated_frame_url"))
                    for match in matches
                    if isinstance(match, dict) and str(match.get("annotated_frame_url") or "").strip()
                ]
                crop_urls = [
                    str(match.get("crop_url"))
                    for match in matches
                    if isinstance(match, dict) and str(match.get("crop_url") or "").strip()
                ]
                for match in matches:
                    if not isinstance(match, dict):
                        continue
                    _append_unique_url(tracking_clip_urls, str(match.get("clip_url") or ""))
                    _append_unique_url(
                        tracking_clip_urls,
                        _artifact_url_for_openable_video_path(job, match.get("clip_path")),
                    )
                if tracking_clip_urls:
                    artifacts.setdefault("person_tracking_moment_clip_url", tracking_clip_urls[0])
                    artifacts["person_tracking_moment_clip_urls"] = tracking_clip_urls
                if frame_urls:
                    artifacts["person_tracking_frame_url"] = frame_urls[0]
                    artifacts["person_tracking_frame_urls"] = frame_urls
                if crop_urls:
                    artifacts["person_tracking_crop_url"] = crop_urls[0]
                    artifacts["person_tracking_crop_urls"] = crop_urls

    return artifacts


def _rewrite_paths_to_urls(job: JobState, payload: dict[str, Any]) -> dict[str, Any]:
    def convert_hit(hit: dict[str, Any], *, include_clip: bool = True) -> dict[str, Any]:
        updated = dict(hit)
        for key in ["crop_path", "annotated_frame_path", "clip_path"]:
            if key == "clip_path" and not include_clip:
                continue
            path = str(updated.get(key, "") or "").strip()
            if not path:
                continue
            try:
                p = Path(path)
                if p.exists():
                    if p.suffix.lower() in VIDEO_ARTIFACT_EXTENSIONS:
                        url = _artifact_url_for_openable_video_path(job, p)
                    else:
                        url = _artifact_url(job, p.relative_to(job.job_dir))
                    if url:
                        updated[key.replace("_path", "_url")] = url
            except Exception:
                continue
        return updated

    def convert_people(match: dict[str, Any]) -> dict[str, Any]:
        updated = dict(match)
        people = updated.get("associated_people") or []
        if isinstance(people, list):
            converted = []
            for person in people:
                if not isinstance(person, dict):
                    continue
                p2 = dict(person)
                crop_path = str(p2.get("crop_path", "") or "").strip()
                if crop_path:
                    try:
                        pth = Path(crop_path)
                        if pth.exists():
                            p2["crop_url"] = _artifact_url(job, pth.relative_to(job.job_dir))
                    except Exception:
                        pass
                converted.append(p2)
            updated["associated_people"] = converted
        return updated

    result = dict(payload)
    clip = result.get("clip")
    if isinstance(clip, dict):
        result["clip"] = _enrich_path_url(job, clip, "path", "url")
    object_moment_clip = result.get("object_moment_clip")
    if isinstance(object_moment_clip, dict):
        result["object_moment_clip"] = _enrich_path_url(job, object_moment_clip, "path", "url")

    for report_key in ["coarse_report", "refine_report"]:
        report = result.get(report_key)
        if isinstance(report, dict):
            report2 = dict(report)
            top = report2.get("top_hits") or []
            if isinstance(top, list):
                report2["top_hits"] = [convert_hit(item) for item in top if isinstance(item, dict)]
            earliest = report2.get("earliest_hit")
            if isinstance(earliest, dict):
                report2["earliest_hit"] = convert_hit(earliest)
            result[report_key] = report2

    static_discovery = result.get("static_discovery")
    if isinstance(static_discovery, dict):
        discovery2 = dict(static_discovery)
        checks = discovery2.get("checks") or []
        if isinstance(checks, list):
            converted_checks = []
            for check in checks:
                if not isinstance(check, dict):
                    continue
                check2 = dict(check)
                clip = check2.get("clip")
                if isinstance(clip, dict):
                    check2["clip"] = _enrich_path_url(job, clip, "path", "url")
                probe = check2.get("probe")
                if isinstance(probe, dict):
                    probe2 = dict(probe)
                    for path_key in ["roi_reference_path", "roi_sample_path", "roi_annotated_frame_path"]:
                        path = str(probe2.get(path_key, "") or "").strip()
                        if not path:
                            continue
                        try:
                            p = Path(path)
                            if p.exists():
                                probe2[path_key.replace("_path", "_url")] = _artifact_url(job, p.relative_to(job.job_dir))
                        except Exception:
                            pass
                    best_hit = probe2.get("best_hit")
                    if isinstance(best_hit, dict):
                        probe2["best_hit"] = convert_hit(best_hit)
                    check2["probe"] = probe2
                converted_checks.append(check2)
            discovery2["checks"] = converted_checks
            by_index = {
                int(check.get("index")): check
                for check in converted_checks
                if isinstance(check, dict) and check.get("index") is not None
            }
            for window_key in ["candidateWindow"]:
                window = discovery2.get(window_key)
                if not isinstance(window, dict):
                    continue
                window2 = dict(window)
                for check_key in ["preAppearanceCheck", "postAppearanceCheck"]:
                    nested = window2.get(check_key)
                    if isinstance(nested, dict):
                        idx = nested.get("index")
                        try:
                            window2[check_key] = by_index.get(int(idx), nested)
                        except Exception:
                            pass
                discovery2[window_key] = window2
        result["static_discovery"] = discovery2
        if isinstance(result.get("checks"), list) and isinstance(discovery2.get("checks"), list):
            result["checks"] = discovery2["checks"]
        if isinstance(result.get("candidateWindow"), dict) and isinstance(discovery2.get("candidateWindow"), dict):
            result["candidateWindow"] = discovery2["candidateWindow"]

    first_match = result.get("first_match")
    if isinstance(first_match, dict):
        result["first_match"] = convert_people(convert_hit(first_match))

    confirmed_object_hit = result.get("confirmed_object_hit")
    if isinstance(confirmed_object_hit, dict):
        result["confirmed_object_hit"] = convert_people(convert_hit(confirmed_object_hit))

    deep = result.get("deep")
    if isinstance(deep, dict):
        deep2 = dict(deep)
        deep2.pop("video_path", None)
        matches = deep2.get("matches") or []
        if isinstance(matches, list):
            deep2["matches"] = [
                convert_people(convert_hit(item, include_clip=False)) for item in matches if isinstance(item, dict)
            ]
        deep_first_match = deep2.get("first_match")
        if isinstance(deep_first_match, dict):
            deep2["first_match"] = convert_people(convert_hit(deep_first_match, include_clip=False))
        for key in ["annotated_video_path"]:
            path = str(deep2.get(key, "") or "").strip()
            if not path:
                continue
            try:
                p = Path(path)
                if p.exists():
                    url = _artifact_url_for_openable_video_path(job, p)
                    if url:
                        deep2[key.replace("_path", "_url")] = url
            except Exception:
                continue
        result["deep"] = deep2

    person_track = result.get("person_track")
    if isinstance(person_track, dict):
        track2 = dict(person_track)
        for report_key in ["coarse_report", "refine_report"]:
            report = track2.get(report_key)
            if isinstance(report, dict):
                track2[report_key] = _convert_report_paths(job, report)
        deep_track = track2.get("deep")
        if isinstance(deep_track, dict):
            deep_track2 = _convert_report_paths(job, deep_track)
            matches = deep_track2.get("matches") or []
            if isinstance(matches, list):
                deep_track2["matches"] = [convert_people(convert_hit(item)) for item in matches if isinstance(item, dict)]
            for key in ["annotated_video_path", "video_path"]:
                path = str(deep_track2.get(key, "") or "").strip()
                if not path:
                    continue
                try:
                    p = Path(path)
                    if p.exists():
                        url = _artifact_url_for_openable_video_path(job, p)
                        if url:
                            deep_track2[key.replace("_path", "_url")] = url
                except Exception:
                    continue
            track2["deep"] = deep_track2
        first_match_track = track2.get("first_match")
        if isinstance(first_match_track, dict):
            track2["first_match"] = convert_people(convert_hit(first_match_track))
        clip = track2.get("clip")
        if isinstance(clip, dict):
            track2["clip"] = _enrich_path_url(job, clip, "path", "url")
        tracking_clip = track2.get("tracking_clip")
        if isinstance(tracking_clip, dict):
            track2["tracking_clip"] = _enrich_path_url(job, tracking_clip, "path", "url")
        refs = track2.get("references") or []
        if isinstance(refs, list):
            refs2 = []
            for ref in refs:
                if not isinstance(ref, dict):
                    continue
                ref2 = dict(ref)
                path = str(ref2.get("path", "") or "").strip()
                if path:
                    try:
                        p = Path(path)
                        if p.exists():
                            ref2["url"] = _artifact_url(job, p.relative_to(job.job_dir))
                    except Exception:
                        pass
                refs2.append(ref2)
            track2["references"] = refs2
        result["person_track"] = track2

    report_artifacts = _build_report_artifacts(job, result)
    if report_artifacts:
        result["report_artifacts"] = report_artifacts

    return result


def _enrich_deep_matches(deep_report: dict[str, Any], *, deep_start: float) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    matches = deep_report.get("matches") or []
    if not isinstance(matches, list) or not matches:
        return [], None

    enriched: list[dict[str, Any]] = []
    for match in matches:
        if not isinstance(match, dict):
            continue
        absolute_seconds = deep_start + float(match.get("timestamp_seconds", 0.0) or 0.0)
        next_match = dict(match)
        next_match["absolute_seconds"] = round(absolute_seconds, 2)
        next_match["absolute_label"] = format_seconds(absolute_seconds)
        enriched.append(next_match)

    if not enriched:
        return [], None

    def match_time(item: dict[str, Any]) -> float:
        return float(item.get("absolute_seconds", 0.0) or 0.0)

    def match_score(item: dict[str, Any]) -> float:
        return float(item.get("score", 0.0) or 0.0)

    def nearby_people(item: dict[str, Any]) -> int:
        return int(item.get("nearby_person_count", item.get("nearbyPersonCount", 0)) or 0)

    def visible_people(item: dict[str, Any]) -> int:
        return int(item.get("person_count", item.get("personCount", 0)) or 0)

    nearby_matches = [item for item in enriched if nearby_people(item) > 0]
    visible_matches = [item for item in enriched if visible_people(item) > 0]
    preferred_pool = nearby_matches or visible_matches or enriched
    first_match = min(preferred_pool, key=lambda item: (match_time(item), -match_score(item)))
    return enriched, first_match


def _build_object_moment_clip_payload(
    *,
    deep_segment_path: Path,
    window_start_seconds: float,
    window_end_seconds: float,
    first_match: dict[str, Any] | None,
) -> dict[str, Any]:
    if isinstance(first_match, dict):
        clip_path = str(first_match.get("clip_path") or "").strip()
        if clip_path and Path(clip_path).exists():
            absolute_seconds = float(
                first_match.get("absolute_seconds", first_match.get("timestamp_seconds", window_start_seconds)) or 0.0
            )
            return {
                "path": clip_path,
                "start_seconds": round(max(0.0, absolute_seconds - runtime_config.EVIDENCE_CLIP_SECONDS_BEFORE), 2),
                "end_seconds": round(absolute_seconds + runtime_config.EVIDENCE_CLIP_SECONDS_AFTER, 2),
                "source": "best_match_clip",
            }

    return {
        "path": str(deep_segment_path),
        "start_seconds": round(float(window_start_seconds), 2),
        "end_seconds": round(float(window_end_seconds), 2),
        "source": "deep_window",
    }


def _pick_hit_from_result(
    result_payload: dict[str, Any],
    *,
    selected_hit_index: int | None = None,
    selected_absolute_seconds: float | None = None,
) -> dict[str, Any] | None:
    refine_report = result_payload.get("refine_report") or {}
    refine_hits = refine_report.get("top_hits") or []
    if isinstance(refine_hits, list) and refine_hits:
        if selected_hit_index is not None and 0 <= int(selected_hit_index) < len(refine_hits):
            selected = refine_hits[int(selected_hit_index)]
            if isinstance(selected, dict):
                return selected
        if selected_absolute_seconds is not None:
            candidates = [item for item in refine_hits if isinstance(item, dict)]
            if candidates:
                return min(
                    candidates,
                    key=lambda item: abs(
                        float(item.get("absolute_seconds", item.get("timestamp_seconds", 0.0)) or 0.0)
                        - float(selected_absolute_seconds)
                    ),
                )
    refined_first = result_payload.get("refined_first")
    if isinstance(refined_first, dict):
        return refined_first
    earliest = refine_report.get("earliest_hit")
    if isinstance(earliest, dict):
        return earliest
    static_hits = _collect_static_discovery_hits(result_payload)
    if static_hits:
        if selected_hit_index is not None and 0 <= int(selected_hit_index) < len(static_hits):
            return static_hits[int(selected_hit_index)]
        if selected_absolute_seconds is not None:
            return min(
                static_hits,
                key=lambda item: abs(float(item.get("absolute_seconds", 0.0) or 0.0) - float(selected_absolute_seconds)),
            )
        transition_hit = _build_static_transition_hit_from_result(result_payload)
        if transition_hit:
            return transition_hit
        return static_hits[0]
    return None


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except Exception:
        return default


def _parse_optional_dt(value: Any) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return _parse_iso_dt(raw)
    except Exception:
        return None


def _static_candidate_window_from_result(result_payload: dict[str, Any]) -> dict[str, Any] | None:
    static_discovery = result_payload.get("static_discovery")
    if isinstance(static_discovery, dict) and isinstance(static_discovery.get("candidateWindow"), dict):
        return dict(static_discovery.get("candidateWindow") or {})
    if isinstance(result_payload.get("candidateWindow"), dict):
        return dict(result_payload.get("candidateWindow") or {})
    return None


def _build_static_transition_hit_from_result(result_payload: dict[str, Any]) -> dict[str, Any] | None:
    candidate_window = _static_candidate_window_from_result(result_payload)
    if not isinstance(candidate_window, dict):
        return None

    post_check = candidate_window.get("postAppearanceCheck")
    transition_hit = _build_static_hit_from_check(post_check) if isinstance(post_check, dict) else None
    start_offset = _safe_float(candidate_window.get("startOffsetSeconds"))
    end_offset = _safe_float(candidate_window.get("endOffsetSeconds"))
    if transition_hit is None:
        transition_hit = {}
    if start_offset is not None and end_offset is not None and end_offset >= start_offset:
        transition_hit["absolute_seconds"] = round((start_offset + end_offset) / 2.0, 2)
        transition_hit["transition_start_seconds"] = round(start_offset, 2)
        transition_hit["transition_end_seconds"] = round(end_offset, 2)
    elif "absolute_seconds" not in transition_hit:
        return None
    transition_hit["static_candidate_window"] = candidate_window
    transition_hit["static_check_label"] = transition_hit.get("static_check_label") or "candidate_window"
    return transition_hit


def _resolve_job_artifact_path(job: JobState, value: Any) -> Path | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        direct = Path(raw)
        if direct.exists():
            return direct
    except Exception:
        pass

    marker = f"/api/investigation/artifacts/{job.job_id}/"
    marker_index = raw.find(marker)
    if marker_index < 0:
        return None
    relative_part = raw[marker_index + len(marker) :].split("?", 1)[0].split("#", 1)[0]
    relative_part = unquote(relative_part).strip("/")
    if not relative_part:
        return None
    candidate = (job.job_dir / relative_part).resolve()
    try:
        job_root = job.job_dir.resolve()
        if candidate != job_root and job_root not in candidate.parents:
            return None
    except Exception:
        return None
    return candidate if candidate.exists() else None


def _build_static_hit_from_check(check: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(check, dict):
        return None
    probe = check.get("probe") if isinstance(check.get("probe"), dict) else {}
    best_hit = probe.get("best_hit") if isinstance(probe.get("best_hit"), dict) else {}
    clip = check.get("clip") if isinstance(check.get("clip"), dict) else {}
    probe_window = check.get("probeWindow") if isinstance(check.get("probeWindow"), dict) else {}

    timestamp_seconds = _safe_float(best_hit.get("timestamp_seconds"), 0.0) or 0.0
    clip_start_offset = _safe_float(probe_window.get("startOffsetSeconds"))
    if clip_start_offset is None:
        check_offset = _safe_float(check.get("offsetSeconds"))
        clip_duration = _safe_float(clip.get("duration_seconds"), 0.0) or 0.0
        if check_offset is not None:
            clip_start_offset = max(0.0, check_offset - (clip_duration / 2.0))

    absolute_seconds = _safe_float(best_hit.get("absolute_seconds"))
    if absolute_seconds is None:
        absolute_seconds = _safe_float(check.get("absoluteSeconds"))
    if absolute_seconds is None and clip_start_offset is not None:
        absolute_seconds = clip_start_offset + timestamp_seconds
    if absolute_seconds is None:
        absolute_seconds = _safe_float(check.get("offsetSeconds"), 0.0) or 0.0

    return {
        **best_hit,
        "score": _safe_float(best_hit.get("score"), _safe_float(check.get("score"), 0.0)) or 0.0,
        "absolute_seconds": round(float(absolute_seconds), 2),
        "timestamp_seconds": round(float(timestamp_seconds), 2),
        "timestamp_label": best_hit.get("timestamp_label") or check.get("timestamp") or "",
        "static_check_index": check.get("index"),
        "static_check_label": check.get("label"),
        "clip": clip,
        "probeWindow": probe_window,
    }


def _collect_static_discovery_hits(result_payload: dict[str, Any]) -> list[dict[str, Any]]:
    static_discovery = result_payload.get("static_discovery")
    if not isinstance(static_discovery, dict):
        return []
    checks = static_discovery.get("checks")
    if not isinstance(checks, list):
        return []
    hits: list[dict[str, Any]] = []
    for check in checks:
        hit = _build_static_hit_from_check(check)
        if isinstance(hit, dict):
            hits.append(hit)
    hits.sort(
        key=lambda item: (
            -float(item.get("score", 0.0) or 0.0),
            float(item.get("absolute_seconds", 0.0) or 0.0),
        )
    )
    return hits


def _pick_static_check_from_result(
    result_payload: dict[str, Any],
    *,
    selected_check_index: int | None = None,
    selected_hit_index: int | None = None,
    selected_absolute_seconds: float | None = None,
) -> dict[str, Any] | None:
    static_discovery = result_payload.get("static_discovery")
    if not isinstance(static_discovery, dict):
        return None
    checks = [check for check in static_discovery.get("checks", []) if isinstance(check, dict)]
    if not checks:
        return None

    if selected_check_index is not None:
        for check in checks:
            try:
                if int(check.get("index")) == int(selected_check_index):
                    return check
            except Exception:
                continue

    if selected_hit_index is not None:
        static_hits = _collect_static_discovery_hits(result_payload)
        if 0 <= int(selected_hit_index) < len(static_hits):
            target_index = static_hits[int(selected_hit_index)].get("static_check_index")
            if target_index is not None:
                for check in checks:
                    try:
                        if int(check.get("index")) == int(target_index):
                            return check
                    except Exception:
                        continue

    if selected_absolute_seconds is not None:
        with_offsets = []
        for check in checks:
            hit = _build_static_hit_from_check(check)
            if isinstance(hit, dict):
                with_offsets.append((check, abs(float(hit.get("absolute_seconds", 0.0) or 0.0) - float(selected_absolute_seconds))))
        if with_offsets:
            with_offsets.sort(key=lambda item: item[1])
            return with_offsets[0][0]

    ranked = sorted(
        checks,
        key=lambda check: (
            not bool(check.get("present")),
            -float(check.get("combinedScore", check.get("score", 0.0)) or 0.0),
        ),
    )
    return ranked[0] if ranked else None


def _get_static_source_context(result_payload: dict[str, Any]) -> dict[str, Any] | None:
    direct = result_payload.get("investigation_context")
    if isinstance(direct, dict):
        return direct
    static_discovery = result_payload.get("static_discovery")
    if isinstance(static_discovery, dict):
        nested = static_discovery.get("source_context")
        if isinstance(nested, dict):
            return nested
    return None


def _resolve_nvr_credentials(
    *,
    nvr_id: str | None,
    nvr_name: str,
    vendor: str,
    host: str,
    http_port: int,
    sdk_port: int,
    username: str,
    password: str | None,
) -> tuple[str, str]:
    resolved_user = (username or "").strip()
    resolved_pass = str(password if password is not None else "")

    backend_profile = _load_backend_nvr_profile(nvr_id)
    if backend_profile:
        resolved_user = str(backend_profile.get("username") or resolved_user or "").strip()
        resolved_pass = str(
            backend_profile.get("password") if backend_profile.get("password") is not None else resolved_pass
        )

    if resolved_user and resolved_pass:
        return resolved_user, resolved_pass

    profiles = _load_nvr_profiles()
    profile = _pick_profile(
        profiles=profiles,
        nvr_name=nvr_name,
        host=host,
        vendor=vendor,
        http_port=http_port,
        sdk_port=sdk_port,
    )
    if profile:
        resolved_user = str(profile.get("username", "")).strip() or resolved_user
        if not resolved_pass:
            resolved_pass = str(profile.get("password", "") if profile.get("password", "") is not None else "")

    return resolved_user, resolved_pass


def _dt_to_payload(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def _offset_seconds(base_dt: datetime, value_dt: datetime) -> float:
    return round(max(0.0, (value_dt - base_dt).total_seconds()), 2)


def _clamp_float(value: Any, *, default: float, min_value: float, max_value: float) -> float:
    try:
        numeric = float(value)
    except Exception:
        numeric = default
    if numeric != numeric:
        numeric = default
    return max(min_value, min(max_value, numeric))


def _parse_static_discovery_roi(raw_roi: str | None) -> dict[str, float] | None:
    raw = str(raw_roi or "").strip()
    if not raw:
        return None

    def read_number(payload: dict[str, Any], *keys: str) -> float | None:
        for key in keys:
            value = payload.get(key)
            if value is not None:
                return float(value)
        return None

    try:
        parsed: Any
        if raw.startswith("{") or raw.startswith("["):
            parsed = json.loads(raw)
        else:
            parsed = [float(part.strip()) for part in raw.replace(";", ",").split(",") if part.strip()]

        if isinstance(parsed, list):
            if len(parsed) != 4:
                raise ValueError("La ROI debe tener 4 valores: x,y,width,height.")
            x, y, width, height = [float(value) for value in parsed]
        elif isinstance(parsed, dict):
            if isinstance(parsed.get("roi"), dict):
                parsed = parsed["roi"]
            x = read_number(parsed, "x", "left")
            y = read_number(parsed, "y", "top")
            width = read_number(parsed, "width", "w")
            height = read_number(parsed, "height", "h")
            if width is None and read_number(parsed, "right") is not None and x is not None:
                width = float(read_number(parsed, "right") or 0.0) - x
            if height is None and read_number(parsed, "bottom") is not None and y is not None:
                height = float(read_number(parsed, "bottom") or 0.0) - y
            if x is None or y is None or width is None or height is None:
                raise ValueError("La ROI JSON debe incluir x,y,width,height o left,top,right,bottom.")
        else:
            raise ValueError("Formato de ROI no soportado.")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"ROI inválida: {exc}") from exc

    values = [x, y, width, height]
    if any(value != value for value in values):
        raise ValueError("La ROI contiene valores NaN.")
    if x < 0.0 or y < 0.0 or width <= 0.0 or height <= 0.0:
        raise ValueError("La ROI debe usar coordenadas normalizadas positivas.")
    if x >= 1.0 or y >= 1.0 or x + width > 1.0 or y + height > 1.0:
        raise ValueError("La ROI debe quedar dentro del frame normalizado [0,1].")
    area = width * height
    if area < STATIC_DISCOVERY_MIN_ROI_AREA:
        raise ValueError(
            f"La ROI es demasiado pequeña ({area:.4f}); mínimo {STATIC_DISCOVERY_MIN_ROI_AREA:.4f} del frame."
        )
    if area > STATIC_DISCOVERY_MAX_ROI_AREA:
        raise ValueError(
            f"La ROI es demasiado grande ({area:.4f}); máximo {STATIC_DISCOVERY_MAX_ROI_AREA:.2f} del frame."
        )
    return {
        "x": round(float(x), 6),
        "y": round(float(y), 6),
        "width": round(float(width), 6),
        "height": round(float(height), 6),
        "area": round(float(area), 6),
        "coordinateSpace": "normalized",
    }


def _lighten_static_check(check: dict[str, Any]) -> dict[str, Any]:
    light = {
        "index": check.get("index"),
        "label": check.get("label"),
        "timestamp": check.get("timestamp"),
        "offsetSeconds": check.get("offsetSeconds"),
        "present": bool(check.get("present")),
        "candidate": bool(check.get("candidate")),
        "candidateConfidence": check.get("candidateConfidence"),
        "candidateReason": check.get("candidateReason"),
        "score": check.get("score"),
        "roi": check.get("roi"),
        "changeScore": check.get("changeScore"),
        "similarityScore": check.get("similarityScore"),
        "baselineSimilarityScore": check.get("baselineSimilarityScore"),
        "similarityDelta": check.get("similarityDelta"),
        "darkObjectScore": check.get("darkObjectScore"),
        "baselineDarkObjectScore": check.get("baselineDarkObjectScore"),
        "darkObjectDelta": check.get("darkObjectDelta"),
        "darkAreaFraction": check.get("darkAreaFraction"),
        "largestDarkComponent": check.get("largestDarkComponent"),
        "darkStructurePenalty": check.get("darkStructurePenalty"),
        "largestDarkComponentWidth": check.get("largestDarkComponentWidth"),
        "largestDarkComponentHeight": check.get("largestDarkComponentHeight"),
        "largestDarkComponentAspect": check.get("largestDarkComponentAspect"),
        "darkComponentEdgeContact": check.get("darkComponentEdgeContact"),
        "darkComponentTouchesBorder": check.get("darkComponentTouchesBorder"),
        "darkAreaDelta": check.get("darkAreaDelta"),
        "darkComponentDelta": check.get("darkComponentDelta"),
        "darkPersistence": check.get("darkPersistence"),
        "persistentDarkFrames": check.get("persistentDarkFrames"),
        "roiObjectScore": check.get("roiObjectScore"),
        "combinedScore": check.get("combinedScore"),
        "decision": check.get("decision"),
        "reason": check.get("reason"),
        "probeWindow": check.get("probeWindow"),
    }
    clip = check.get("clip")
    if isinstance(clip, dict):
        light["clip"] = {
            "url": clip.get("url") or "",
            "start_dt": clip.get("start_dt"),
            "end_dt": clip.get("end_dt"),
            "duration_seconds": clip.get("duration_seconds"),
        }
    probe = check.get("probe")
    if isinstance(probe, dict):
        best_hit = probe.get("best_hit")
        best_hit_light: dict[str, Any] | None = None
        if isinstance(best_hit, dict):
            best_hit_light = {
                "timestamp_seconds": best_hit.get("timestamp_seconds"),
                "timestamp_label": best_hit.get("timestamp_label"),
                "score": best_hit.get("score"),
                "similarityScore": best_hit.get("similarityScore"),
                "changeScore": best_hit.get("changeScore"),
                "darkObjectScore": best_hit.get("darkObjectScore"),
                "darkAreaFraction": best_hit.get("darkAreaFraction"),
                "largestDarkComponent": best_hit.get("largestDarkComponent"),
                "darkStructurePenalty": best_hit.get("darkStructurePenalty"),
                "darkAreaDelta": best_hit.get("darkAreaDelta"),
                "darkComponentDelta": best_hit.get("darkComponentDelta"),
                "rawRoiObjectScore": best_hit.get("rawRoiObjectScore"),
                "roiObjectScore": best_hit.get("roiObjectScore"),
                "combinedScore": best_hit.get("combinedScore"),
                "bbox": best_hit.get("bbox"),
                "zone_id": best_hit.get("zone_id"),
                "crop_url": best_hit.get("crop_url") or "",
                "annotated_frame_url": best_hit.get("annotated_frame_url") or "",
            }
        light["probe"] = {
            "present": bool(probe.get("present")),
            "candidate": bool(probe.get("candidate")),
            "candidateConfidence": probe.get("candidateConfidence"),
            "candidateReason": probe.get("candidateReason"),
            "duration_seconds": probe.get("duration_seconds"),
            "frames_reviewed": probe.get("frames_reviewed"),
            "similarity_threshold": probe.get("similarity_threshold"),
            "effective_similarity_threshold": probe.get("effective_similarity_threshold"),
            "changeThreshold": probe.get("changeThreshold"),
            "combinedThreshold": probe.get("combinedThreshold"),
            "visualSupportThreshold": probe.get("visualSupportThreshold"),
            "changeScore": probe.get("changeScore"),
            "similarityScore": probe.get("similarityScore"),
            "baselineSimilarityScore": probe.get("baselineSimilarityScore"),
            "similarityDelta": probe.get("similarityDelta"),
            "darkObjectScore": probe.get("darkObjectScore"),
            "baselineDarkObjectScore": probe.get("baselineDarkObjectScore"),
            "darkObjectDelta": probe.get("darkObjectDelta"),
            "darkAreaFraction": probe.get("darkAreaFraction"),
            "largestDarkComponent": probe.get("largestDarkComponent"),
            "darkStructurePenalty": probe.get("darkStructurePenalty"),
            "largestDarkComponentWidth": probe.get("largestDarkComponentWidth"),
            "largestDarkComponentHeight": probe.get("largestDarkComponentHeight"),
            "largestDarkComponentAspect": probe.get("largestDarkComponentAspect"),
            "darkComponentEdgeContact": probe.get("darkComponentEdgeContact"),
            "darkComponentTouchesBorder": probe.get("darkComponentTouchesBorder"),
            "darkAreaDelta": probe.get("darkAreaDelta"),
            "darkComponentDelta": probe.get("darkComponentDelta"),
            "darkPersistence": probe.get("darkPersistence"),
            "persistentDarkFrames": probe.get("persistentDarkFrames"),
            "roiObjectScore": probe.get("roiObjectScore"),
            "combinedScore": probe.get("combinedScore"),
            "decision": probe.get("decision"),
            "reason": probe.get("reason"),
            "roi": probe.get("roi"),
            "roi_reference_url": probe.get("roi_reference_url") or "",
            "roi_sample_url": probe.get("roi_sample_url") or "",
            "roi_annotated_frame_url": probe.get("roi_annotated_frame_url") or "",
            "best_hit": best_hit_light,
        }
    return light


def _lighten_static_candidate_window(window: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(window, dict):
        return None
    light = dict(window)
    for key in ("preAppearanceCheck", "postAppearanceCheck"):
        nested = light.get(key)
        if isinstance(nested, dict):
            light[key] = {
                "index": nested.get("index"),
                "label": nested.get("label"),
                "timestamp": nested.get("timestamp"),
                "offsetSeconds": nested.get("offsetSeconds"),
                "present": bool(nested.get("present")),
                "candidate": bool(nested.get("candidate")),
                "candidateConfidence": nested.get("candidateConfidence"),
                "candidateReason": nested.get("candidateReason"),
                "score": nested.get("score"),
                "changeScore": nested.get("changeScore"),
                "similarityScore": nested.get("similarityScore"),
                "darkStructurePenalty": nested.get("darkStructurePenalty"),
                "combinedScore": nested.get("combinedScore"),
                "decision": nested.get("decision"),
                "reason": nested.get("reason"),
            }
    return light


def _build_investigation_bridge(vendor: str, downloads_dir: Path) -> HikvisionBridgeSettings | DahuaRemoteBridgeSettings:
    key_path = runtime_config.SSH_KEY_PATH
    if vendor == "hikvision":
        return HikvisionBridgeSettings(
            ssh_host=runtime_config.REMOTE_BRIDGE_HOST,
            ssh_user=runtime_config.REMOTE_BRIDGE_USER,
            ssh_key_path=key_path,
            local_download_dir=downloads_dir,
        )
    return DahuaRemoteBridgeSettings(
        ssh_host=runtime_config.REMOTE_BRIDGE_HOST,
        ssh_user=runtime_config.REMOTE_BRIDGE_USER,
        ssh_key_path=key_path,
    )


def _should_use_local_dahua_sdk() -> bool:
    return runtime_config.DAHUA_DOWNLOAD_MODE in {"auto", "local", "mac", "macos"}


def _should_use_remote_dahua_bridge() -> bool:
    return runtime_config.DAHUA_DOWNLOAD_MODE in {"auto", "remote", "bridge", "ssh"}


def _should_use_local_hikvision_sdk() -> bool:
    mode = runtime_config.HIKVISION_DOWNLOAD_MODE
    if mode in {"local", "direct", "linux"}:
        return True
    if mode == "auto":
        return sys.platform.startswith("linux")
    return False


def _should_use_remote_hikvision_bridge() -> bool:
    mode = runtime_config.HIKVISION_DOWNLOAD_MODE
    if mode in {"auto", "remote", "bridge", "ssh"}:
        return True
    return False


def _download_dahua_clip_local(
    *,
    host: str,
    sdk_port: int,
    username: str,
    password: str,
    logical_channel: int,
    start_dt: datetime,
    end_dt: datetime,
    local_target_dir: Path,
    playback_sdk_channel: int | None = None,
    channel_resolution: dict[str, Any] | None = None,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    settings = DahuaBridgeSettings(sdk_root=find_dahua_sdk_root())
    primary_channel = max(0, int(playback_sdk_channel)) if playback_sdk_channel is not None else max(0, int(logical_channel) - 1)
    try:
        if progress_callback is not None:
            resolution_source = str((channel_resolution or {}).get("source") or "default_zero_based")
            progress_callback(
                "local_dahua_sdk",
                f"Descargando clip Dahua localmente con NetSDK Java Mac canal SDK {primary_channel} ({resolution_source}).",
            )
        result = download_clip_via_sdk(
            settings=settings,
            host=host,
            sdk_port=int(sdk_port),
            username=username,
            password=password,
            channel=primary_channel,
            start_dt=start_dt,
            end_dt=end_dt,
            local_target_dir=local_target_dir,
        )
        result["channel_resolution"] = channel_resolution or {}
        return result
    except Exception as exc:
        msg = str(exc)
        if "2147483650" not in msg and "0x80000002" not in msg:
            raise
        fallback_channel = int(logical_channel)
        if progress_callback is not None:
            progress_callback(
                "local_dahua_sdk",
                f"Reintentando Dahua local con canal alterno {fallback_channel} (mapeo 1-based).",
            )
        return download_clip_via_sdk(
            settings=settings,
            host=host,
            sdk_port=int(sdk_port),
            username=username,
            password=password,
            channel=fallback_channel,
            start_dt=start_dt,
            end_dt=end_dt,
            local_target_dir=local_target_dir,
        )


def _download_dahua_clip_remote(
    *,
    bridge: DahuaRemoteBridgeSettings,
    host: str,
    sdk_port: int,
    username: str,
    password: str,
    logical_channel: int,
    start_dt: datetime,
    end_dt: datetime,
    local_target_dir: Path,
    playback_sdk_channel: int | None = None,
    channel_resolution: dict[str, Any] | None = None,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    primary_channel = max(0, int(playback_sdk_channel)) if playback_sdk_channel is not None else max(0, int(logical_channel) - 1)
    try:
        result = download_clip_via_bridge_dahua(
            bridge=bridge,
            host=host,
            sdk_port=int(sdk_port),
            username=username,
            password=password,
            sdk_channel=primary_channel,
            start_dt=start_dt,
            end_dt=end_dt,
            local_target_dir=local_target_dir,
            progress_callback=progress_callback,
        )
        result["channel_resolution"] = channel_resolution or {}
        return result
    except Exception as exc:
        msg = str(exc)
        if "2147483650" not in msg and "0x80000002" not in msg:
            raise
        fallback_channel = int(logical_channel)
        if progress_callback is not None:
            progress_callback(
                "server_download",
                f"Reintentando Dahua NetSDK con canal alterno {fallback_channel} (mapeo 1-based).",
            )
        return download_clip_via_bridge_dahua(
            bridge=bridge,
            host=host,
            sdk_port=int(sdk_port),
            username=username,
            password=password,
            sdk_channel=fallback_channel,
            start_dt=start_dt,
            end_dt=end_dt,
            local_target_dir=local_target_dir,
            progress_callback=progress_callback,
        )


def _download_dahua_investigation_clip(
    *,
    bridge: DahuaRemoteBridgeSettings | None,
    host: str,
    sdk_port: int,
    username: str,
    password: str,
    logical_channel: int,
    start_dt: datetime,
    end_dt: datetime,
    local_target_dir: Path,
    nvr_id: str | int | None = None,
    nvr_name: str = "",
    camera_id: str | int | None = None,
    camera_name: str = "",
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    playback_sdk_channel, channel_resolution = _resolve_dahua_playback_sdk_channel(
        host=host,
        nvr_id=nvr_id,
        nvr_name=nvr_name,
        camera_id=camera_id,
        camera_name=camera_name,
        logical_channel=logical_channel,
    )
    local_error: Exception | None = None
    if _should_use_local_dahua_sdk():
        try:
            return _download_dahua_clip_local(
                host=host,
                sdk_port=sdk_port,
                username=username,
                password=password,
                logical_channel=logical_channel,
                playback_sdk_channel=playback_sdk_channel,
                channel_resolution=channel_resolution,
                start_dt=start_dt,
                end_dt=end_dt,
                local_target_dir=local_target_dir,
                progress_callback=progress_callback,
            )
        except Exception as exc:
            local_error = exc
            if runtime_config.DAHUA_DOWNLOAD_MODE in {"local", "mac", "macos"}:
                raise
            if progress_callback is not None:
                progress_callback(
                    "local_dahua_sdk",
                    f"Dahua local no disponible ({exc}). Intentando bridge remoto.",
                )

    if _should_use_remote_dahua_bridge() and bridge is not None:
        try:
            return _download_dahua_clip_remote(
                bridge=bridge,
                host=host,
                sdk_port=sdk_port,
                username=username,
                password=password,
                logical_channel=logical_channel,
                playback_sdk_channel=playback_sdk_channel,
                channel_resolution=channel_resolution,
                start_dt=start_dt,
                end_dt=end_dt,
                local_target_dir=local_target_dir,
                progress_callback=progress_callback,
            )
        except Exception:
            if local_error is not None:
                raise RuntimeError(
                    f"Dahua local falló ({local_error}) y el bridge remoto también falló."
                ) from local_error
            raise

    if local_error is not None:
        raise RuntimeError(f"Dahua local falló y el bridge remoto está deshabilitado: {local_error}") from local_error
    raise RuntimeError(
        "Modo Dahua inválido. Usa INNOVA_DAHUA_DOWNLOAD_MODE=auto, local o remote."
    )


def _download_investigation_clip(
    *,
    vendor: str,
    bridge: HikvisionBridgeSettings | DahuaRemoteBridgeSettings,
    host: str,
    sdk_port: int,
    username: str,
    password: str,
    logical_channel: int,
    start_dt: datetime,
    end_dt: datetime,
    local_target_dir: Path,
    nvr_id: str | int | None = None,
    nvr_name: str = "",
    camera_id: str | int | None = None,
    camera_name: str = "",
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    if vendor == "hikvision":
        local_error: Exception | None = None
        if _should_use_local_hikvision_sdk():
            try:
                return download_clip_via_local_sdk(
                    host=host,
                    sdk_port=int(sdk_port),
                    username=username,
                    password=password,
                    logical_channel=int(logical_channel),
                    start_dt=start_dt,
                    end_dt=end_dt,
                    local_target_dir=local_target_dir,
                    sdk_root=runtime_config.HIKVISION_LOCAL_SDK_DIR,
                    normalize_for_review=False,
                    progress_callback=progress_callback,
                )
            except Exception as exc:
                local_error = exc
                if runtime_config.HIKVISION_DOWNLOAD_MODE in {"local", "direct", "linux"}:
                    raise
                if progress_callback is not None:
                    progress_callback(
                        "local_hikvision_sdk",
                        f"Hikvision local no disponible ({exc}). Intentando bridge remoto.",
                    )

        if _should_use_remote_hikvision_bridge():
            try:
                return download_clip_via_bridge(
                    bridge=bridge,  # type: ignore[arg-type]
                    host=host,
                    sdk_port=int(sdk_port),
                    username=username,
                    password=password,
                    logical_channel=int(logical_channel),
                    start_dt=start_dt,
                    end_dt=end_dt,
                    local_target_dir=local_target_dir,
                    normalize_for_review=False,
                    progress_callback=progress_callback,
                )
            except Exception:
                if local_error is not None:
                    raise RuntimeError(
                        f"Hikvision local fallo ({local_error}) y el bridge remoto tambien fallo."
                    ) from local_error
                raise

        if local_error is not None:
            raise RuntimeError(f"Hikvision local fallo y el bridge remoto esta deshabilitado: {local_error}") from local_error
        raise RuntimeError(
            "Modo Hikvision invalido. Usa INNOVA_HIKVISION_DOWNLOAD_MODE=auto, local o remote."
        )

    return _download_dahua_investigation_clip(
        bridge=bridge if isinstance(bridge, DahuaRemoteBridgeSettings) else None,
        host=host,
        sdk_port=sdk_port,
        username=username,
        password=password,
        logical_channel=logical_channel,
        start_dt=start_dt,
        end_dt=end_dt,
        local_target_dir=local_target_dir,
        nvr_id=nvr_id,
        nvr_name=nvr_name,
        camera_id=camera_id,
        camera_name=camera_name,
        progress_callback=progress_callback,
    )


def _collect_person_reference_paths(
    result_payload: dict[str, Any],
    *,
    selected_person_index: int | None = None,
    max_references: int = 12,
) -> list[Path]:
    first_match = result_payload.get("first_match") or {}
    people = first_match.get("associated_people") or []

    candidates: list[dict[str, Any]] = []
    if isinstance(people, list):
        for person in people:
            if not isinstance(person, dict):
                continue
            crop_path = str(person.get("crop_path") or "").strip()
            if not crop_path:
                continue
            path = Path(crop_path)
            if not path.exists():
                continue
            candidates.append(person)

    def append_path(path: Path, *, confidence: float = 0.0, near: bool = False) -> None:
        if path.exists():
            candidates.append(
                {
                    "crop_path": str(path),
                    "confidence": confidence,
                    "is_near_object": near,
                    "distance_to_object": 0.0 if near else 999999.0,
                }
            )

    deep = result_payload.get("deep") or {}
    if isinstance(deep, dict):
        persons_dir = str(deep.get("persons_dir") or "").strip()
        if persons_dir:
            for path in sorted(Path(persons_dir).glob("*.jpg")):
                append_path(path, confidence=0.0, near=("near" in path.name or "scene" in path.name))

        matches = deep.get("matches") or []
        if isinstance(matches, list):
            for match in matches:
                if not isinstance(match, dict):
                    continue
                for person in match.get("associated_people") or []:
                    if not isinstance(person, dict):
                        continue
                    crop_path = str(person.get("crop_path") or "").strip()
                    if crop_path:
                        append_path(
                            Path(crop_path),
                            confidence=float(person.get("confidence", 0.0) or 0.0),
                            near=bool(person.get("is_near_object")),
                        )

    def sort_key(person: dict[str, Any]) -> tuple[int, float, float]:
        return (
            0 if person.get("is_near_object") else 1,
            float(person.get("distance_to_object", 999999.0) or 999999.0),
            -float(person.get("confidence", 0.0) or 0.0),
        )

    ordered = sorted(candidates, key=sort_key)
    if selected_person_index is not None and 0 <= int(selected_person_index) < len(candidates):
        picked = candidates[int(selected_person_index)]
        ordered = [picked] + [person for person in ordered if person is not picked]

    unique: list[Path] = []
    seen: set[str] = set()
    for person in ordered:
        path = Path(str(person.get("crop_path"))).resolve()
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
        if len(unique) >= max_references:
            break
    return unique


def _collect_reference_paths_from_urls(
    *,
    job: JobState,
    urls: list[str] | None,
    max_references: int = 12,
) -> list[Path]:
    if not isinstance(urls, list):
        return []
    unique: list[Path] = []
    seen: set[str] = set()
    for raw in urls:
        path = _resolve_job_artifact_path(job, raw)
        if path is None or not path.exists():
            continue
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
        if len(unique) >= max_references:
            break
    return unique


def _get_person_detector() -> Any:
    global PERSON_DETECTOR
    if PERSON_DETECTOR is not None:
        return PERSON_DETECTOR
    if YOLO is None:
        raise RuntimeError("YOLO no está disponible para rastrear personas.")
    PERSON_DETECTOR = YOLO(str(runtime_config.YOLO_MODEL_PATH))
    return PERSON_DETECTOR


def _detect_people(frame: np.ndarray) -> list[dict[str, Any]]:
    if frame.size == 0:
        return []
    detector = _get_person_detector()
    results = detector.predict(
        frame,
        classes=[0],
        conf=float(getattr(runtime_config, "PERSON_CONFIDENCE_THRESHOLD", 0.35)),
        device=str(getattr(runtime_config, "PERSON_DETECTION_DEVICE", "cpu")),
        verbose=False,
    )
    detections: list[dict[str, Any]] = []
    height, width = frame.shape[:2]
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        for box in boxes:
            xyxy = box.xyxy[0].detach().cpu().numpy().astype(int).tolist()
            x1, y1, x2, y2 = xyxy[:4]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            conf = float(box.conf[0].detach().cpu().item()) if getattr(box, "conf", None) is not None else 0.0
            detections.append({"bbox": [x1, y1, x2, y2], "confidence": conf})
    return detections


INTERACTION_CLASS_IDS = [0, 2, 3, 5, 7]
INTERACTION_VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "motorbike"}


def _detect_interaction_objects(frame: np.ndarray, *, min_confidence: float = 0.28) -> list[dict[str, Any]]:
    if frame.size == 0:
        return []
    detector = _get_person_detector()
    results = detector.predict(
        frame,
        classes=INTERACTION_CLASS_IDS,
        conf=float(min_confidence),
        device=str(getattr(runtime_config, "PERSON_DETECTION_DEVICE", "cpu")),
        verbose=False,
    )
    detections: list[dict[str, Any]] = []
    height, width = frame.shape[:2]
    names = getattr(detector, "names", {}) or {}
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        for box in boxes:
            class_id = int(box.cls[0].detach().cpu().item()) if getattr(box, "cls", None) is not None else -1
            class_name = str(names.get(class_id, class_id))
            xyxy = box.xyxy[0].detach().cpu().numpy().astype(int).tolist()
            x1, y1, x2, y2 = xyxy[:4]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            conf = float(box.conf[0].detach().cpu().item()) if getattr(box, "conf", None) is not None else 0.0
            detections.append(
                {
                    "classId": class_id,
                    "className": class_name,
                    "confidence": round(conf, 4),
                    "bbox": [x1, y1, x2, y2],
                }
            )
    return detections


def _normalized_roi_to_pixels(frame_shape: tuple[int, int], roi: dict[str, Any]) -> tuple[int, int, int, int]:
    height, width = frame_shape[:2]
    x = float(roi.get("x", 0.0) or 0.0)
    y = float(roi.get("y", 0.0) or 0.0)
    roi_width = float(roi.get("width", 0.0) or 0.0)
    roi_height = float(roi.get("height", 0.0) or 0.0)
    x1 = max(0, min(width - 1, int(round(x * width))))
    y1 = max(0, min(height - 1, int(round(y * height))))
    x2 = max(x1 + 1, min(width, int(round((x + roi_width) * width))))
    y2 = max(y1 + 1, min(height, int(round((y + roi_height) * height))))
    return x1, y1, x2, y2


def _expand_box(box: tuple[int, int, int, int], frame_shape: tuple[int, int], *, scale: float = 0.55) -> tuple[int, int, int, int]:
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = box
    pad_x = int(round((x2 - x1) * scale))
    pad_y = int(round((y2 - y1) * scale))
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x),
        min(height, y2 + pad_y),
    )


def _box_intersection_ratio(a: list[int] | tuple[int, ...], b: list[int] | tuple[int, ...]) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in a[:4]]
    bx1, by1, bx2, by2 = [float(value) for value in b[:4]]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    return float(inter / area_a)


def _annotate_interaction_frame(
    frame: np.ndarray,
    *,
    roi_box: tuple[int, int, int, int],
    expanded_box: tuple[int, int, int, int],
    detections: list[dict[str, Any]],
    label: str,
) -> np.ndarray:
    annotated = frame.copy()
    ex1, ey1, ex2, ey2 = expanded_box
    rx1, ry1, rx2, ry2 = roi_box
    cv2.rectangle(annotated, (ex1, ey1), (ex2, ey2), (90, 120, 255), 2)
    cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), runtime_config.BRAND_GOLD, 3)
    for detection in detections:
        x1, y1, x2, y2 = [int(value) for value in detection.get("bbox", [0, 0, 0, 0])]
        class_name = str(detection.get("className") or "object")
        confidence = float(detection.get("confidence", 0.0) or 0.0)
        color = (0, 220, 255) if class_name in INTERACTION_VEHICLE_CLASSES else (80, 220, 120)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            f"{class_name} {confidence:.2f}",
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            color,
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        annotated,
        label,
        (max(10, rx1), max(28, ry1 - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        runtime_config.BRAND_GOLD,
        2,
        cv2.LINE_AA,
    )
    return annotated


def _scan_clip_for_roi_interactions(
    *,
    video_path: Path,
    output_dir: Path,
    roi: dict[str, Any],
    chunk_start_dt: datetime,
    range_start_dt: datetime,
    sample_every_seconds: float,
    pre_post_seconds: float,
    max_events: int = 8,
    on_progress: Any | None = None,
    should_cancel: Any | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    clips_dir = output_dir / "clips"
    frames_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"No pude abrir el clip para ROI interaction search: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 1.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_seconds = total_frames / fps if fps > 0 else 0.0
    frame_step = max(1, int(round(max(float(sample_every_seconds), 0.4) * fps)))

    events: list[dict[str, Any]] = []
    prev_gray: np.ndarray | None = None
    last_event_second = -9999.0
    sampled = 0
    frame_index = 0
    roi_box: tuple[int, int, int, int] | None = None
    expanded_box: tuple[int, int, int, int] | None = None

    try:
        while frame_index < max(total_frames, 1):
            if should_cancel and should_cancel():
                break
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                frame_index += frame_step
                continue
            sampled += 1
            timestamp_seconds = frame_index / fps if fps else 0.0
            if roi_box is None or expanded_box is None:
                roi_box = _normalized_roi_to_pixels(frame.shape[:2], roi)
                expanded_box = _expand_box(roi_box, frame.shape[:2], scale=0.65)

            ex1, ey1, ex2, ey2 = expanded_box
            roi_frame = frame[ey1:ey2, ex1:ex2]
            if roi_frame.size == 0:
                frame_index += frame_step
                continue
            gray = cv2.cvtColor(cv2.resize(roi_frame, (220, 160), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            motion_score = 0.0
            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                motion_score = float(np.mean(diff) / 255.0)
            prev_gray = gray

            should_probe = motion_score >= 0.030
            if should_probe and (timestamp_seconds - last_event_second) >= max(6.0, pre_post_seconds * 0.65):
                detections = _detect_interaction_objects(frame, min_confidence=0.25)
                relevant: list[dict[str, Any]] = []
                proximity_score = 0.0
                vehicle_near = False
                person_near = False
                assert roi_box is not None and expanded_box is not None
                for detection in detections:
                    bbox = detection.get("bbox") or []
                    if len(bbox) < 4:
                        continue
                    expanded_overlap = _box_intersection_ratio(bbox, expanded_box)
                    roi_overlap = _box_intersection_ratio(bbox, roi_box)
                    if expanded_overlap <= 0.02 and roi_overlap <= 0.0:
                        continue
                    item = dict(detection)
                    item["expandedRoiOverlap"] = round(expanded_overlap, 4)
                    item["targetRoiOverlap"] = round(roi_overlap, 4)
                    relevant.append(item)
                    proximity_score = max(proximity_score, min(1.0, (expanded_overlap * 0.65) + (roi_overlap * 1.2)))
                    class_name = str(item.get("className") or "").lower()
                    vehicle_near = vehicle_near or class_name in INTERACTION_VEHICLE_CLASSES
                    person_near = person_near or class_name == "person"

                if relevant or motion_score >= 0.055:
                    confidence = min(
                        0.98,
                        (motion_score * 9.0)
                        + (0.24 if vehicle_near else 0.0)
                        + (0.10 if person_near else 0.0)
                        + (0.30 * proximity_score),
                    )
                    reasons: list[str] = []
                    if motion_score >= 0.055:
                        reasons.append("strong ROI motion/change")
                    else:
                        reasons.append("ROI motion/change")
                    if vehicle_near:
                        reasons.append("vehicle near selected target")
                    if person_near:
                        reasons.append("person near selected target")
                    if proximity_score > 0.12:
                        reasons.append("object overlaps expanded target area")

                    event_number = len(events) + 1
                    absolute_dt = chunk_start_dt + timedelta(seconds=timestamp_seconds)
                    absolute_seconds = (absolute_dt - range_start_dt).total_seconds()
                    frame_path = frames_dir / f"event_{event_number:03d}_{absolute_dt.strftime('%Y%m%d_%H%M%S')}.jpg"
                    annotated = _annotate_interaction_frame(
                        frame,
                        roi_box=roi_box,
                        expanded_box=expanded_box,
                        detections=relevant,
                        label=f"Interaction candidate {confidence:.2f}",
                    )
                    cv2.imwrite(str(frame_path), annotated)
                    clip_path = clips_dir / f"event_{event_number:03d}_{absolute_dt.strftime('%Y%m%d_%H%M%S')}.mp4"
                    try:
                        clip_path = extract_video_segment(
                            source_video=video_path,
                            output_video=clip_path,
                            start_seconds=max(0.0, timestamp_seconds - pre_post_seconds),
                            end_seconds=min(duration_seconds, timestamp_seconds + pre_post_seconds),
                        )
                    except Exception:
                        clip_path = Path("")

                    events.append(
                        {
                            "eventId": f"roi-event-{event_number:03d}",
                            "timestamp": _dt_to_payload(absolute_dt),
                            "timestampSeconds": round(timestamp_seconds, 2),
                            "absoluteSeconds": round(absolute_seconds, 2),
                            "confidence": round(float(confidence), 4),
                            "reason": ", ".join(reasons),
                            "objectsDetected": relevant,
                            "motionScore": round(float(motion_score), 4),
                            "proximityScore": round(float(proximity_score), 4),
                            "frame_path": str(frame_path),
                            "clip_path": str(clip_path) if clip_path else "",
                            "roi": roi,
                        }
                    )
                    last_event_second = timestamp_seconds
                    if len(events) >= max_events:
                        break

            if on_progress and sampled % 6 == 0:
                on_progress(frame_index / max(total_frames, 1))
            frame_index += frame_step
    finally:
        capture.release()

    return {
        "ok": True,
        "durationSeconds": round(duration_seconds, 2),
        "framesReviewed": sampled,
        "events": events,
    }


def _score_person_crop(
    searcher: SimilaritySearcher,
    query_signatures: list[dict[str, Any]],
    crop: np.ndarray,
) -> dict[str, Any]:
    best: dict[str, Any] = {"score": 0.0, "reference_index": 0}
    if crop.size == 0:
        return best
    for ref in query_signatures:
        query = ref["signature"]
        color_score = searcher._compare_histograms(query, crop)
        feature_score = searcher._compare_features(query, crop)
        structure_score = searcher._compare_structure(query, crop)
        aspect_ratio = crop.shape[1] / max(crop.shape[0], 1)
        aspect_score = 1.0 - min(abs(aspect_ratio - query.aspect_ratio) / max(query.aspect_ratio, 1e-6), 1.0)
        score = float(
            (0.34 * color_score)
            + (0.24 * feature_score)
            + (0.26 * structure_score)
            + (0.16 * aspect_score)
        )
        if score > float(best.get("score", 0.0) or 0.0):
            best = {
                "score": round(score, 4),
                "reference_index": int(ref["index"]),
                "reference_path": str(ref["path"]),
                "color_score": round(float(color_score), 4),
                "feature_score": round(float(feature_score), 4),
                "structure_score": round(float(structure_score), 4),
                "aspect_score": round(float(aspect_score), 4),
            }
    return best


def _draw_person_hit(frame: np.ndarray, hit: dict[str, Any]) -> np.ndarray:
    annotated = frame.copy()
    x1, y1, x2, y2 = [int(v) for v in hit.get("bbox", [0, 0, 0, 0])]
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (72, 173, 255), 3)
    label = f"Persona {hit.get('score', 0):.3f}"
    cv2.putText(
        annotated,
        label,
        (x1, max(24, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (18, 28, 40),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        label,
        (x1, max(24, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 230, 156),
        2,
        cv2.LINE_AA,
    )
    return annotated


def _bbox_center(bbox: list[int] | tuple[int, ...]) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _bbox_iou(a: list[int] | tuple[int, ...], b: list[int] | tuple[int, ...]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in b[:4]]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    return float(inter / max(1.0, area_a + area_b - inter))


def _group_person_hits_into_tracks(
    hits: list[dict[str, Any]],
    *,
    max_time_gap_seconds: float = 3.0,
    min_iou: float = 0.08,
) -> list[dict[str, Any]]:
    """Collapse repeated frame detections into review-friendly person tracks."""
    tracks: list[dict[str, Any]] = []
    ordered_hits = sorted(
        [item for item in hits if isinstance(item, dict) and item.get("bbox")],
        key=lambda item: float(item.get("absolute_seconds", item.get("timestamp_seconds", 0.0)) or 0.0),
    )
    for hit in ordered_hits:
        bbox = hit.get("bbox") or [0, 0, 0, 0]
        hit_time = float(hit.get("absolute_seconds", hit.get("timestamp_seconds", 0.0)) or 0.0)
        hit_center = _bbox_center(bbox)
        hit_w = max(1.0, float(bbox[2] - bbox[0]))
        hit_h = max(1.0, float(bbox[3] - bbox[1]))
        max_center_shift = max(90.0, ((hit_w * hit_w + hit_h * hit_h) ** 0.5) * 0.85)

        best_track: dict[str, Any] | None = None
        best_score = -1.0
        for track in tracks:
            last_hit = track.get("last_hit") or {}
            last_bbox = last_hit.get("bbox") or []
            if len(last_bbox) < 4:
                continue
            last_time = float(last_hit.get("absolute_seconds", last_hit.get("timestamp_seconds", 0.0)) or 0.0)
            time_gap = abs(hit_time - last_time)
            if time_gap > max_time_gap_seconds:
                continue
            last_center = _bbox_center(last_bbox)
            center_distance = ((hit_center[0] - last_center[0]) ** 2 + (hit_center[1] - last_center[1]) ** 2) ** 0.5
            iou = _bbox_iou(bbox, last_bbox)
            if iou < min_iou and center_distance > max_center_shift:
                continue
            continuity_score = (iou * 3.0) + max(0.0, 1.0 - (center_distance / max_center_shift))
            if continuity_score > best_score:
                best_track = track
                best_score = continuity_score

        if best_track is None:
            track_id = len(tracks) + 1
            best_track = {
                "track_id": track_id,
                "hits": [],
                "last_hit": None,
            }
            tracks.append(best_track)

        hit["track_id"] = int(best_track["track_id"])
        best_track["hits"].append(hit)
        best_track["last_hit"] = hit

    summaries: list[dict[str, Any]] = []
    for track in tracks:
        track_hits = [item for item in track.get("hits", []) if isinstance(item, dict)]
        if not track_hits:
            continue
        best_hit = max(
            track_hits,
            key=lambda item: (
                float(item.get("score", 0.0) or 0.0),
                -float(item.get("absolute_seconds", item.get("timestamp_seconds", 0.0)) or 0.0),
            ),
        )
        first_hit = min(track_hits, key=lambda item: float(item.get("absolute_seconds", 0.0) or 0.0))
        last_hit = max(track_hits, key=lambda item: float(item.get("absolute_seconds", 0.0) or 0.0))
        summaries.append(
            {
                "track_id": int(track["track_id"]),
                "hit_count": len(track_hits),
                "first_seen_seconds": first_hit.get("absolute_seconds"),
                "last_seen_seconds": last_hit.get("absolute_seconds"),
                "first_seen_label": first_hit.get("absolute_label") or first_hit.get("timestamp_label") or "",
                "last_seen_label": last_hit.get("absolute_label") or last_hit.get("timestamp_label") or "",
                "best_score": best_hit.get("score", 0.0),
                "best_hit": best_hit,
                "sample_hits": sorted(
                    track_hits,
                    key=lambda item: (-float(item.get("score", 0.0) or 0.0), float(item.get("absolute_seconds", 0.0) or 0.0)),
                )[:6],
            }
        )
    summaries.sort(
        key=lambda item: (
            -float(item.get("best_score", 0.0) or 0.0),
            -int(item.get("hit_count", 0) or 0),
            float(item.get("first_seen_seconds", 0.0) or 0.0),
        )
    )
    return summaries


def _scan_clip_for_person_references(
    *,
    reference_paths: list[Path],
    video_path: Path,
    output_dir: Path,
    sample_every_seconds: float,
    similarity_threshold: float,
    stage_label: str,
    keep_top: int,
    time_offset_seconds: float = 0.0,
    on_progress: Any | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = output_dir / "person_crops"
    frames_dir = output_dir / "person_frames"
    crops_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    searcher = SimilaritySearcher()
    queries: list[dict[str, Any]] = []
    for index, path in enumerate(reference_paths):
        image = cv2.imread(str(path))
        if image is None or image.size == 0:
            continue
        queries.append({"index": index, "path": path, "signature": searcher.build_query_signature(image)})
    if not queries:
        raise RuntimeError("No pude leer las referencias de persona confirmada.")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"No pude abrir el clip para rastrear persona: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 1.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / fps if fps > 0 else 0.0
    frame_step = max(1, int(round(max(float(sample_every_seconds), 0.2) * fps)))

    hits: list[dict[str, Any]] = []
    frame_index = 0
    sampled = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % frame_step != 0:
                frame_index += 1
                continue

            timestamp = frame_index / fps if fps else 0.0
            people = _detect_people(frame)
            sampled += 1
            if on_progress and sampled % 2 == 0:
                ratio = frame_index / max(total_frames, 1)
                on_progress(stage_label, f"Buscando persona frame {frame_index}/{total_frames}", ratio)

            for person_index, person in enumerate(people, start=1):
                x1, y1, x2, y2 = [int(v) for v in person["bbox"]]
                crop = frame[y1:y2, x1:x2]
                score_payload = _score_person_crop(searcher, queries, crop)
                score = float(score_payload.get("score", 0.0) or 0.0)
                detection_conf = float(person.get("confidence", 0.0) or 0.0)
                final_score = round((0.88 * score) + (0.12 * min(detection_conf, 1.0)), 4)
                if final_score < float(similarity_threshold):
                    continue

                rank_seed = len(hits) + 1
                crop_path = crops_dir / f"hit_{rank_seed:03d}_t_{timestamp:08.2f}s_person_{person_index:02d}_score_{final_score:.3f}.jpg"
                frame_path = frames_dir / f"hit_{rank_seed:03d}_t_{timestamp:08.2f}s_person_{person_index:02d}_score_{final_score:.3f}.jpg"
                cv2.imwrite(str(crop_path), crop)
                cv2.imwrite(str(frame_path), _draw_person_hit(frame, {**person, **score_payload, "score": final_score}))
                hit = {
                    "frame_index": int(frame_index),
                    "timestamp_seconds": round(timestamp, 2),
                    "absolute_seconds": round(time_offset_seconds + timestamp, 2),
                    "timestamp_label": format_seconds(timestamp),
                    "absolute_label": format_seconds(time_offset_seconds + timestamp),
                    "score": final_score,
                    "base_score": round(score, 4),
                    "person_confidence": round(detection_conf, 4),
                    "bbox": [x1, y1, x2, y2],
                    "reference_index": int(score_payload.get("reference_index", 0) or 0),
                    "reference_path": str(score_payload.get("reference_path", "")),
                    "color_score": score_payload.get("color_score", 0.0),
                    "feature_score": score_payload.get("feature_score", 0.0),
                    "structure_score": score_payload.get("structure_score", 0.0),
                    "aspect_score": score_payload.get("aspect_score", 0.0),
                    "crop_path": str(crop_path),
                    "annotated_frame_path": str(frame_path),
                    "person_count": len(people),
                }
                hits.append(hit)
            frame_index += 1
    finally:
        capture.release()

    person_tracks = _group_person_hits_into_tracks(hits)
    track_representatives = [
        track.get("best_hit")
        for track in person_tracks
        if isinstance(track.get("best_hit"), dict)
    ]
    track_representatives.sort(
        key=lambda item: (-float(item.get("score", 0.0)), float(item.get("absolute_seconds", 0.0)))
    )
    hits.sort(key=lambda item: (-float(item.get("score", 0.0)), float(item.get("absolute_seconds", 0.0))))
    top_hits = (track_representatives or hits)[:keep_top]
    earliest = min(top_hits, key=lambda item: float(item.get("absolute_seconds", 0.0))) if top_hits else None
    return {
        "ok": True,
        "stage": stage_label,
        "duration_seconds": round(duration, 2),
        "sample_every_seconds": float(sample_every_seconds),
        "frames_reviewed": sampled,
        "references_used": len(queries),
        "matches_found": len(hits),
        "tracks_found": len(person_tracks),
        "person_tracks": person_tracks,
        "earliest_hit": earliest,
        "top_hits": top_hits,
    }


def _scan_clip_for_person_candidates(
    *,
    video_path: Path,
    output_dir: Path,
    sample_every_seconds: float,
    stage_label: str,
    keep_top: int,
    time_offset_seconds: float = 0.0,
    on_progress: Any | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = output_dir / "person_crops"
    frames_dir = output_dir / "person_frames"
    crops_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"No pude abrir el clip para descubrir personas: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 1.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / fps if fps > 0 else 0.0
    frame_step = max(1, int(round(max(float(sample_every_seconds), 0.2) * fps)))

    hits: list[dict[str, Any]] = []
    frame_index = 0
    sampled = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % frame_step != 0:
                frame_index += 1
                continue

            timestamp = frame_index / fps if fps else 0.0
            people = _detect_people(frame)
            sampled += 1
            if on_progress and sampled % 2 == 0:
                ratio = frame_index / max(total_frames, 1)
                on_progress(stage_label, f"Detectando personas frame {frame_index}/{total_frames}", ratio)

            frame_area = float(max(1, frame.shape[0] * frame.shape[1]))
            for person_index, person in enumerate(people, start=1):
                x1, y1, x2, y2 = [int(v) for v in person["bbox"]]
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                detection_conf = float(person.get("confidence", 0.0) or 0.0)
                area_ratio = float(((x2 - x1) * (y2 - y1)) / frame_area)
                score = round((0.78 * detection_conf) + (0.22 * min(1.0, area_ratio * 9.0)), 4)

                rank_seed = len(hits) + 1
                crop_path = crops_dir / f"candidate_{rank_seed:03d}_t_{timestamp:08.2f}s_person_{person_index:02d}_score_{score:.3f}.jpg"
                frame_path = frames_dir / f"candidate_{rank_seed:03d}_t_{timestamp:08.2f}s_person_{person_index:02d}_score_{score:.3f}.jpg"
                cv2.imwrite(str(crop_path), crop)
                cv2.imwrite(str(frame_path), _draw_person_hit(frame, {**person, "score": score}))
                hits.append(
                    {
                        "frame_index": int(frame_index),
                        "timestamp_seconds": round(timestamp, 2),
                        "absolute_seconds": round(time_offset_seconds + timestamp, 2),
                        "timestamp_label": format_seconds(timestamp),
                        "absolute_label": format_seconds(time_offset_seconds + timestamp),
                        "score": score,
                        "person_confidence": round(detection_conf, 4),
                        "bbox": [x1, y1, x2, y2],
                        "crop_path": str(crop_path),
                        "annotated_frame_path": str(frame_path),
                        "person_count": len(people),
                        "mode": "person_discovery_candidate",
                    }
                )
            frame_index += 1
    finally:
        capture.release()

    person_tracks = _group_person_hits_into_tracks(hits)
    track_representatives = [
        track.get("best_hit")
        for track in person_tracks
        if isinstance(track.get("best_hit"), dict)
    ]
    track_representatives.sort(
        key=lambda item: (
            -float(item.get("score", 0.0)),
            float(item.get("absolute_seconds", 0.0)),
        )
    )
    hits.sort(
        key=lambda item: (
            -float(item.get("score", 0.0)),
            float(item.get("absolute_seconds", 0.0)),
        )
    )
    top_hits = (track_representatives or hits)[:keep_top]
    earliest = min(top_hits, key=lambda item: float(item.get("absolute_seconds", 0.0))) if top_hits else None
    return {
        "ok": True,
        "stage": stage_label,
        "duration_seconds": round(duration, 2),
        "sample_every_seconds": float(sample_every_seconds),
        "frames_reviewed": sampled,
        "references_used": 0,
        "matches_found": len(hits),
        "tracks_found": len(person_tracks),
        "person_tracks": person_tracks,
        "earliest_hit": earliest,
        "top_hits": top_hits,
    }


def _convert_report_paths(job: JobState, report: dict[str, Any]) -> dict[str, Any]:
    def convert_hit(hit: dict[str, Any]) -> dict[str, Any]:
        updated = dict(hit)
        for key in ["crop_path", "annotated_frame_path", "clip_path"]:
            path = str(updated.get(key, "") or "").strip()
            if not path:
                continue
            try:
                p = Path(path)
                if p.exists():
                    if p.suffix.lower() in VIDEO_ARTIFACT_EXTENSIONS:
                        url = _artifact_url_for_openable_video_path(job, p)
                    else:
                        url = _artifact_url(job, p.relative_to(job.job_dir))
                    if url:
                        updated[key.replace("_path", "_url")] = url
            except Exception:
                continue

        people = updated.get("associated_people") or []
        if isinstance(people, list):
            converted_people = []
            for person in people:
                if not isinstance(person, dict):
                    continue
                p2 = dict(person)
                crop_path = str(p2.get("crop_path", "") or "").strip()
                if crop_path:
                    try:
                        pth = Path(crop_path)
                        if pth.exists():
                            p2["crop_url"] = _artifact_url(job, pth.relative_to(job.job_dir))
                    except Exception:
                        pass
                converted_people.append(p2)
            updated["associated_people"] = converted_people
        return updated

    updated_report = dict(report)
    top = updated_report.get("top_hits") or []
    if isinstance(top, list):
        updated_report["top_hits"] = [convert_hit(item) for item in top if isinstance(item, dict)]
    earliest = updated_report.get("earliest_hit")
    if isinstance(earliest, dict):
        updated_report["earliest_hit"] = convert_hit(earliest)
    tracks = updated_report.get("person_tracks") or []
    if isinstance(tracks, list):
        converted_tracks: list[dict[str, Any]] = []
        for track in tracks:
            if not isinstance(track, dict):
                continue
            updated_track = dict(track)
            best_hit = updated_track.get("best_hit")
            if isinstance(best_hit, dict):
                updated_track["best_hit"] = convert_hit(best_hit)
            sample_hits = updated_track.get("sample_hits") or []
            if isinstance(sample_hits, list):
                updated_track["sample_hits"] = [
                    convert_hit(item) for item in sample_hits if isinstance(item, dict)
                ]
            updated_track.pop("hits", None)
            updated_track.pop("last_hit", None)
            converted_tracks.append(updated_track)
        updated_report["person_tracks"] = converted_tracks
    return updated_report


class DeepSearchRequest(BaseModel):
    selectedHitIndex: int | None = None
    selectedCheckIndex: int | None = None
    selectedAbsoluteSeconds: float | None = None
    investigationRadius: float = 25.0
    similarityThreshold: float = 0.58
    preferWindow: bool = False


def _person_context_window_seconds(radius: float) -> tuple[float, float]:
    safe_radius = max(10.0, float(radius or 25.0))
    before_seconds = min(75.0, max(30.0, safe_radius * 1.5))
    after_seconds = min(75.0, max(30.0, safe_radius * 1.5))
    return before_seconds, after_seconds


class TrackPersonRequest(BaseModel):
    vendor: str
    host: str
    httpPort: int = 0
    sdkPort: int
    nvrName: str = ""
    nvrId: str = ""
    username: str = ""
    password: str | None = None
    logicalChannel: int
    cameraId: str = ""
    cameraName: str = ""
    startDt: str
    endDt: str
    similarityThreshold: float = 0.45
    coarseStepSeconds: float = 2.0
    refineStepSeconds: float = 1.0
    investigationRadius: float = 20.0
    selectedPersonIndex: int | None = None
    referenceUrls: list[str] | None = None
    discoveryOnly: bool = False


class BridgeNvrRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    vendor: str = ""
    host: str = ""
    httpPort: int | None = 80
    sdkPort: int | None = 8000
    rtspPort: int | None = 554
    nvrName: str = ""
    nvrId: str = ""
    username: str = ""
    password: str | None = None
    bridgeHost: str = ""
    bridgeUser: str = ""
    bridgeSdkPath: str = ""


class BridgeSnapshotRequest(BridgeNvrRequest):
    logicalChannel: int | None = None
    streamVariant: str = "sub"
    timeoutSeconds: int | None = 12
    deep: bool = False


def _resolved_runtime_nvr_payload(payload: BridgeNvrRequest) -> dict[str, Any]:
    resolved_vendor = str(payload.vendor or "").strip()
    resolved_host = str(payload.host or "").strip()
    resolved_http_port = int(payload.httpPort or 0) or 80
    resolved_sdk_port = int(payload.sdkPort or 0) or 8000
    resolved_rtsp_port = int(payload.rtspPort or 0) or 554
    resolved_nvr_name = str(payload.nvrName or "").strip()
    resolved_username = str(payload.username or "").strip()
    resolved_password = payload.password or ""
    resolved_bridge_host = str(payload.bridgeHost or "").strip()
    resolved_bridge_user = str(payload.bridgeUser or "").strip()
    resolved_bridge_sdk_path = str(payload.bridgeSdkPath or "").strip()

    backend_profile = _load_backend_nvr_profile(payload.nvrId)
    if backend_profile:
        resolved_vendor = str(backend_profile.get("brand") or resolved_vendor or "").strip()
        resolved_host = str(backend_profile.get("host") or resolved_host or "").strip()
        resolved_http_port = int(backend_profile.get("httpPort") or resolved_http_port or 80)
        resolved_sdk_port = int(backend_profile.get("sdkPort") or resolved_sdk_port or 8000)
        resolved_rtsp_port = int(backend_profile.get("rtspPort") or resolved_rtsp_port or 554)
        resolved_nvr_name = str(backend_profile.get("name") or resolved_nvr_name or "").strip()
        resolved_username = str(backend_profile.get("username") or resolved_username or "").strip()
        resolved_password = str(backend_profile.get("password") or resolved_password or "")
        resolved_bridge_host = str(backend_profile.get("bridgeHost") or resolved_bridge_host or "").strip()
        resolved_bridge_user = str(backend_profile.get("bridgeSshUser") or resolved_bridge_user or "").strip()
        resolved_bridge_sdk_path = str(backend_profile.get("bridgeSdkPath") or resolved_bridge_sdk_path or "").strip()

    if not resolved_password:
        profiles = _load_nvr_profiles()
        profile = _pick_profile(
            profiles=profiles,
            nvr_name=resolved_nvr_name,
            host=resolved_host,
            vendor=resolved_vendor,
            http_port=resolved_http_port,
            sdk_port=resolved_sdk_port,
        )
        if profile:
            resolved_password = str(profile.get("password") or resolved_password or "")
            resolved_username = str(profile.get("username") or resolved_username or "").strip()

    if not resolved_host:
        raise HTTPException(status_code=400, detail="El bridge necesita host del NVR.")
    if not resolved_username or not resolved_password:
        raise HTTPException(status_code=400, detail="El bridge necesita usuario y contraseña válidos.")

    return {
        "vendor": _normalize_vendor(resolved_vendor),
        "host": resolved_host,
        "httpPort": resolved_http_port,
        "sdkPort": resolved_sdk_port,
        "rtspPort": resolved_rtsp_port,
        "nvrName": resolved_nvr_name,
        "username": resolved_username,
        "password": resolved_password,
        "bridgeHost": resolved_bridge_host or runtime_config.REMOTE_BRIDGE_HOST,
        "bridgeUser": resolved_bridge_user or runtime_config.REMOTE_BRIDGE_USER,
        "bridgeSdkPath": resolved_bridge_sdk_path,
    }


def _bridge_camera_payload(channel: dict[str, Any]) -> dict[str, Any]:
    channel_number = int(channel.get("id") or 0)
    return {
        "channelNumber": channel_number,
        "channelCode": str(channel.get("sdk_channel") if channel.get("sdk_channel") is not None else channel.get("id") or ""),
        "name": str(channel.get("name") or f"Canal {channel_number}"),
        "vendor": str(channel.get("vendor") or ""),
        "online": bool(channel.get("online")),
        "statusLabel": str(channel.get("status_label") or "Sin dato"),
        "ipAddress": str(channel.get("ip_address") or "-"),
        "streamMainPath": f"/stream/main/{channel_number}" if channel_number > 0 else "",
        "streamSubPath": f"/stream/sub/{channel_number}" if channel_number > 0 else "",
        "snapshotPath": f"/snapshot/{channel_number}" if channel_number > 0 else "",
    }


def _bridge_fetch_snapshot_once(
    *,
    vendor: str,
    host: str,
    http_port: int,
    rtsp_port: int,
    username: str,
    password: str,
    logical_channel: int,
    stream_variant: str,
    timeout_seconds: int,
    strategy_type: str,
) -> bytes:
    normalized_vendor = _normalize_vendor(vendor)
    if normalized_vendor == "hikvision":
        if strategy_type == "rtsp":
            return fetch_hikvision_snapshot_bytes_via_rtsp(
                host=host,
                rtsp_port=rtsp_port,
                username=username,
                password=password,
                logical_channel=logical_channel,
                stream_variant=stream_variant,
                timeout_seconds=timeout_seconds,
            )
        if strategy_type == "rtsp-udp":
            return fetch_hikvision_snapshot_bytes_via_rtsp(
                host=host,
                rtsp_port=rtsp_port,
                username=username,
                password=password,
                logical_channel=logical_channel,
                stream_variant=stream_variant,
                transport="udp",
                timeout_seconds=timeout_seconds,
            )
        return fetch_snapshot_bytes_via_isapi(
            host=host,
            port=http_port,
            username=username,
            password=password,
            logical_channel=logical_channel,
            stream_variant=stream_variant,
            timeout_seconds=timeout_seconds,
        )

    if normalized_vendor == "uniview":
        if strategy_type == "rtsp":
            return fetch_uniview_snapshot_bytes_via_rtsp(
                host=host,
                rtsp_port=rtsp_port,
                username=username,
                password=password,
                logical_channel=logical_channel,
                stream_variant=stream_variant,
                timeout_seconds=timeout_seconds,
            )
        if strategy_type == "rtsp-udp":
            return fetch_uniview_snapshot_bytes_via_rtsp(
                host=host,
                rtsp_port=rtsp_port,
                username=username,
                password=password,
                logical_channel=logical_channel,
                stream_variant=stream_variant,
                transport="udp",
                timeout_seconds=timeout_seconds,
            )
        return fetch_uniview_snapshot_bytes_via_lapi(
            host=host,
            http_port=http_port,
            username=username,
            password=password,
            logical_channel=logical_channel,
            stream_variant=stream_variant,
            timeout_seconds=timeout_seconds,
        )

    if strategy_type == "rtsp":
        return fetch_dahua_snapshot_bytes_via_rtsp(
            host=host,
            rtsp_port=rtsp_port,
            username=username,
            password=password,
            channel=logical_channel,
            timeout_seconds=timeout_seconds,
        )
    return fetch_snapshot_bytes_via_http(
        host=host,
        http_port=http_port,
        username=username,
        password=password,
        channel=logical_channel,
        timeout_seconds=timeout_seconds,
    )


def _bridge_snapshot_strategies(vendor: str, deep: bool, stream_variant: str, timeout_seconds: int) -> list[dict[str, Any]]:
    normalized_vendor = _normalize_vendor(vendor)
    if not deep:
        return [{
            "label": f"{normalized_vendor} direct {stream_variant}",
            "streamVariant": stream_variant,
            "timeoutSeconds": min(max(4, timeout_seconds), 8) if normalized_vendor == "hikvision" else timeout_seconds,
            "strategyType": "isapi" if normalized_vendor == "hikvision" else "lapi" if normalized_vendor == "uniview" else "http",
        }]

    if normalized_vendor == "hikvision":
        return [
            {"label": "ISAPI substream", "streamVariant": "sub", "timeoutSeconds": 5, "strategyType": "isapi"},
            {"label": "ISAPI mainstream", "streamVariant": "main", "timeoutSeconds": 7, "strategyType": "isapi"},
            {"label": "ISAPI mainstream retry", "streamVariant": "main", "timeoutSeconds": 10, "strategyType": "isapi"},
            {"label": "RTSP substream", "streamVariant": "sub", "timeoutSeconds": 18, "strategyType": "rtsp"},
            {"label": "RTSP mainstream", "streamVariant": "main", "timeoutSeconds": 25, "strategyType": "rtsp"},
            {"label": "RTSP UDP substream", "streamVariant": "sub", "timeoutSeconds": 18, "strategyType": "rtsp-udp"},
            {"label": "RTSP UDP mainstream", "streamVariant": "main", "timeoutSeconds": 25, "strategyType": "rtsp-udp"},
            {"label": "RTSP mainstream final", "streamVariant": "main", "timeoutSeconds": 35, "strategyType": "rtsp"},
        ]

    if normalized_vendor == "uniview":
        return [
            {"label": "LAPI substream snapshot", "streamVariant": "sub", "timeoutSeconds": 8, "strategyType": "lapi"},
            {"label": "LAPI mainstream snapshot", "streamVariant": "main", "timeoutSeconds": 10, "strategyType": "lapi"},
            {"label": "RTSP substream snapshot", "streamVariant": "sub", "timeoutSeconds": 18, "strategyType": "rtsp"},
            {"label": "RTSP mainstream snapshot", "streamVariant": "main", "timeoutSeconds": 25, "strategyType": "rtsp"},
            {"label": "RTSP UDP substream snapshot", "streamVariant": "sub", "timeoutSeconds": 18, "strategyType": "rtsp-udp"},
            {"label": "RTSP UDP mainstream snapshot", "streamVariant": "main", "timeoutSeconds": 25, "strategyType": "rtsp-udp"},
        ]

    return [
        {"label": "HTTP snapshot", "streamVariant": stream_variant, "timeoutSeconds": 12, "strategyType": "http"},
        {"label": "RTSP snapshot", "streamVariant": stream_variant, "timeoutSeconds": 18, "strategyType": "rtsp"},
        {"label": "HTTP snapshot extended", "streamVariant": stream_variant, "timeoutSeconds": 28, "strategyType": "http"},
        {"label": "RTSP snapshot final", "streamVariant": stream_variant, "timeoutSeconds": 35, "strategyType": "rtsp"},
    ]


def _bridge_is_terminal_snapshot_error(message: str) -> bool:
    normalized = str(message or "").lower()
    terminal_fragments = (
        "401",
        "unauthorized",
        "403",
        "forbidden",
        "404",
        "not found",
        "405",
        "method not allowed",
        "400",
        "bad request",
    )
    return any(fragment in normalized for fragment in terminal_fragments)


app = FastAPI(title="Innova AI Investigation Runtime API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_event_monitor(app)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "service": "investigation-mvp", "time": _now_iso()}


@app.post("/api/bridge/discover-cameras")
def bridge_discover_cameras(payload: BridgeNvrRequest) -> dict[str, Any]:
    resolved = _resolved_runtime_nvr_payload(payload)
    vendor = resolved["vendor"]

    if vendor == "hikvision":
        channels = list_channels_with_status_via_isapi(
            host=resolved["host"],
            port=int(resolved["httpPort"]),
            username=resolved["username"],
            password=resolved["password"],
            timeout_seconds=12,
        )
    elif vendor == "uniview":
        channels = list_uniview_channels(
            host=resolved["host"],
            http_port=int(resolved["httpPort"]),
            rtsp_port=int(resolved["rtspPort"]),
            username=resolved["username"],
            password=resolved["password"],
            timeout_seconds=5,
        )
    elif vendor == "dahua":
        bridge = DahuaRemoteBridgeSettings(
            ssh_host=resolved["bridgeHost"],
            ssh_user=resolved["bridgeUser"],
            remote_sdk_dir=resolved["bridgeSdkPath"] or runtime_config.DAHUA_REMOTE_SDK_DIR,
        )
        dahua_settings: DahuaBridgeSettings | None = None
        try:
            dahua_settings = DahuaBridgeSettings(sdk_root=find_dahua_sdk_root())
        except Exception:
            dahua_settings = None
        channels = list_dahua_channels(
            settings=dahua_settings,
            host=resolved["host"],
            sdk_port=int(resolved["sdkPort"]),
            http_port=int(resolved["httpPort"]),
            username=resolved["username"],
            password=resolved["password"],
            bridge=bridge,
        )
    else:
        raise HTTPException(status_code=400, detail=f"Vendor no soportado: {vendor}")

    return {
        "ok": True,
        "vendor": vendor,
        "count": len(channels),
        "cameras": [_bridge_camera_payload(channel) for channel in channels],
    }


@app.post("/api/bridge/fetch-snapshot")
def bridge_fetch_snapshot(payload: BridgeSnapshotRequest) -> dict[str, Any]:
    if int(payload.logicalChannel or 0) <= 0:
        raise HTTPException(status_code=400, detail="El canal lógico es inválido.")

    resolved = _resolved_runtime_nvr_payload(payload)
    strategies = _bridge_snapshot_strategies(
        resolved["vendor"],
        bool(payload.deep),
        str(payload.streamVariant or "sub").strip().lower() or "sub",
        int(payload.timeoutSeconds or 12),
    )

    attempts: list[str] = []
    for strategy in strategies:
        label = str(strategy["label"])
        try:
            snapshot_bytes = _bridge_fetch_snapshot_once(
                vendor=resolved["vendor"],
                host=resolved["host"],
                http_port=int(resolved["httpPort"]),
                rtsp_port=int(resolved["rtspPort"]),
                username=resolved["username"],
                password=resolved["password"],
                logical_channel=int(payload.logicalChannel),
                stream_variant=str(strategy["streamVariant"]),
                timeout_seconds=int(strategy["timeoutSeconds"]),
                strategy_type=str(strategy["strategyType"]),
            )
            attempts.append(f"{label}: ok ({len(snapshot_bytes)} bytes)")
            return {
                "ok": True,
                "vendor": resolved["vendor"],
                "strategy": label,
                "attempts": " | ".join(attempts),
                "message": f"Snapshot actualizado usando {label}.",
                "imageBase64": base64.b64encode(snapshot_bytes).decode("ascii"),
                "sizeBytes": len(snapshot_bytes),
            }
        except Exception as exc:
            error_text = str(exc)
            attempts.append(f"{label}: {error_text}")
            if _normalize_vendor(resolved["vendor"]) == "hikvision":
                is_rtsp_strategy = str(strategy.get("strategyType") or "").lower().startswith("rtsp")
                if _bridge_is_terminal_snapshot_error(error_text) and is_rtsp_strategy:
                    break

    raise HTTPException(
        status_code=502,
        detail={
            "message": "No pudimos refrescar la imagen de la cámara después de múltiples intentos.",
            "error": " | ".join(attempts),
        },
    )


@app.post("/api/investigation/roi-interaction-search")
async def start_roi_interaction_search(
    request: Request,
    background: BackgroundTasks,
    propertyId: str = Form(""),
    propertyName: str = Form(""),
    vendor: str = Form(...),
    host: str = Form(""),
    httpPort: int = Form(0),
    sdkPort: int = Form(0),
    nvrName: str = Form(""),
    nvrId: str = Form(""),
    username: str = Form(""),
    password: str | None = Form(None),
    cameraId: str = Form(""),
    cameraName: str = Form(""),
    logicalChannel: int = Form(...),
    startDt: str = Form(...),
    endDt: str = Form(...),
    caseName: str = Form("roi-interaction-search"),
    roi: str = Form(...),
    interactionType: str = Form("possible_vehicle_impact"),
    chunkMinutes: float = Form(10.0),
    sampleEverySeconds: float = Form(1.5),
    prePostSeconds: float = Form(20.0),
) -> dict[str, Any]:
    resolved_vendor = str(vendor or "").strip()
    resolved_host = str(host or "").strip()
    resolved_http_port = int(httpPort or 0)
    resolved_sdk_port = int(sdkPort or 0)
    resolved_nvr_name = str(nvrName or "").strip()

    backend_profile = _load_backend_nvr_profile(nvrId)
    if not backend_profile:
        backend_profile = _load_backend_nvr_profile_by_hint(
            nvr_name=nvrName,
            property_id=propertyId,
            property_name=propertyName,
            host=host,
        )
    if backend_profile:
        resolved_vendor = str(backend_profile.get("brand") or backend_profile.get("vendor") or resolved_vendor or "").strip()
        resolved_host = str(backend_profile.get("host") or resolved_host or "").strip()
        resolved_http_port = int(backend_profile.get("httpPort") or resolved_http_port or 0)
        resolved_sdk_port = int(backend_profile.get("sdkPort") or resolved_sdk_port or 0)
        resolved_nvr_name = str(backend_profile.get("name") or resolved_nvr_name or "").strip()

    normalized_vendor = _normalize_vendor(resolved_vendor)
    if normalized_vendor not in ("hikvision", "dahua"):
        raise HTTPException(status_code=400, detail=f"Vendor no soportado para ROI interaction search: {resolved_vendor or vendor}")

    resolved_user, resolved_pass = _resolve_nvr_credentials(
        nvr_id=nvrId,
        nvr_name=resolved_nvr_name,
        vendor=normalized_vendor,
        host=resolved_host,
        http_port=resolved_http_port,
        sdk_port=resolved_sdk_port,
        username=username,
        password=password,
    )
    if not resolved_host or not resolved_sdk_port:
        raise HTTPException(
            status_code=400,
            detail=(
                "Falta host o sdkPort del NVR. "
                f"Recibido nvrId={nvrId or '-'}, nvrName={nvrName or '-'}, "
                f"propertyId={propertyId or '-'}, host={resolved_host or '-'}, "
                f"sdkPort={resolved_sdk_port or 0}, backendProfile={'yes' if backend_profile else 'no'}."
            ),
        )
    if not resolved_user or not resolved_pass:
        raise HTTPException(
            status_code=400,
            detail=(
                "Faltan credenciales del NVR para ROI interaction search. "
                f"Recibido nvrId={nvrId or '-'}, nvrName={resolved_nvr_name or nvrName or '-'}, "
                f"propertyId={propertyId or '-'}, username={'yes' if resolved_user else 'no'}, "
                f"password={'yes' if resolved_pass else 'no'}, backendProfile={'yes' if backend_profile else 'no'}."
            ),
        )

    try:
        start_dt = _parse_iso_dt(startDt)
        end_dt = _parse_iso_dt(endDt)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Fechas inválidas: {exc}") from exc
    if end_dt <= start_dt:
        raise HTTPException(status_code=400, detail="endDt debe ser mayor que startDt.")
    if start_dt > datetime.now() or end_dt > datetime.now():
        raise HTTPException(status_code=400, detail="Selecciona una fecha y hora anteriores a la actual.")

    try:
        normalized_roi = _parse_static_discovery_roi(roi)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    chunk_minutes = _clamp_float(chunkMinutes, default=10.0, min_value=2.0, max_value=30.0)
    sample_seconds = _clamp_float(sampleEverySeconds, default=1.5, min_value=0.5, max_value=8.0)
    context_seconds = _clamp_float(prePostSeconds, default=20.0, min_value=5.0, max_value=60.0)

    job_id = uuid.uuid4().hex
    job_dir = OUTPUT_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    job = JobState(
        job_id=job_id,
        created_at=_now_iso(),
        updated_at=_now_iso(),
        status="running",
        stage="prepare",
        detail="Job creado. Preparando ROI interaction search...",
        progress=0.02,
        base_url=_resolve_public_api_base(request),
        job_dir=job_dir,
    )
    with JOBS_LOCK:
        JOBS[job_id] = job
        _persist_job_state(job)

    def runner() -> None:
        events: list[dict[str, Any]] = []
        chunks_processed = 0
        chunks_failed = 0
        limitations: list[str] = [
            "AI candidates require human review; this workflow does not make a legal fault determination.",
            "Low light, occlusion, glare, or low camera angle can hide physical contact.",
        ]
        try:
            with ENGINE_LOCK:
                downloads_dir = job_dir / "roi_interaction" / "downloads"
                analysis_dir = job_dir / "roi_interaction" / "analysis"
                downloads_dir.mkdir(parents=True, exist_ok=True)
                analysis_dir.mkdir(parents=True, exist_ok=True)
                bridge = _build_investigation_bridge(normalized_vendor, downloads_dir)
                total_seconds = max(1.0, (end_dt - start_dt).total_seconds())
                cursor = start_dt
                chunk_index = 0
                consecutive_failures = 0
                chunk_delta = timedelta(minutes=chunk_minutes)

                while cursor < end_dt:
                    if _job_cancel_requested(job_id):
                        break
                    chunk_index += 1
                    chunk_start = cursor
                    chunk_end = min(end_dt, cursor + chunk_delta)
                    ratio_start = max(0.0, (chunk_start - start_dt).total_seconds() / total_seconds)
                    ratio_end = max(0.0, (chunk_end - start_dt).total_seconds() / total_seconds)
                    _update_job(
                        job_id,
                        stage="download",
                        detail=f"Chunk {chunk_index}: descargando {_dt_to_payload(chunk_start)} a {_dt_to_payload(chunk_end)}. Eventos: {len(events)}",
                        progress=0.04 + (0.88 * ratio_start),
                    )
                    chunk_dir = downloads_dir / f"chunk_{chunk_index:04d}"
                    chunk_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        clip_result = _download_investigation_clip(
                            vendor=normalized_vendor,
                            bridge=bridge,
                            host=resolved_host,
                            sdk_port=resolved_sdk_port,
                            username=resolved_user,
                            password=resolved_pass,
                            logical_channel=int(logicalChannel),
                            start_dt=chunk_start,
                            end_dt=chunk_end,
                            local_target_dir=chunk_dir,
                            progress_callback=lambda s, d: _update_job(job_id, stage=s, detail=d),
                        )
                        clip_path = ensure_openable_clip(Path(clip_result["final_local_path"]), output_dir=chunk_dir)
                        scan = _scan_clip_for_roi_interactions(
                            video_path=clip_path,
                            output_dir=analysis_dir / f"chunk_{chunk_index:04d}",
                            roi=normalized_roi,
                            chunk_start_dt=chunk_start,
                            range_start_dt=start_dt,
                            sample_every_seconds=sample_seconds,
                            pre_post_seconds=context_seconds,
                            max_events=6,
                            should_cancel=lambda: _job_cancel_requested(job_id),
                            on_progress=lambda ratio: _update_job(
                                job_id,
                                stage="analysis",
                                detail=f"Chunk {chunk_index}: analizando ROI {int(max(0.0, min(1.0, ratio)) * 100)}%. Eventos: {len(events)}",
                                progress=0.04 + (0.88 * (ratio_start + ((ratio_end - ratio_start) * max(0.0, min(1.0, ratio))))),
                            ),
                        )
                        chunks_processed += 1
                        consecutive_failures = 0
                        chunk_events = scan.get("events") if isinstance(scan, dict) else []
                        if isinstance(chunk_events, list):
                            for event in chunk_events:
                                if not isinstance(event, dict):
                                    continue
                                event2 = dict(event)
                                event2["eventId"] = f"roi-event-{len(events) + 1:03d}"
                                event2["chunkIndex"] = chunk_index
                                event2["chunkStartDt"] = _dt_to_payload(chunk_start)
                                event2["chunkEndDt"] = _dt_to_payload(chunk_end)
                                events.append(event2)
                            events.sort(key=lambda item: (-float(item.get("confidence", 0.0) or 0.0), str(item.get("timestamp") or "")))
                        if not chunk_events:
                            for child in chunk_dir.glob("*"):
                                try:
                                    if child.is_file():
                                        child.unlink()
                                except Exception:
                                    pass

                        partial_events: list[dict[str, Any]] = []
                        for event in events[:20]:
                            event_light = dict(event)
                            event_light["frameUrl"] = _artifact_url_for_path(job, event_light.get("frame_path"))
                            event_light["clipUrl"] = _artifact_url_for_openable_video_path(job, event_light.get("clip_path"))
                            partial_events.append(event_light)
                        _update_job_partial(
                            job_id,
                            {
                                "mode": "roi_interaction_search",
                                "events": partial_events,
                                "summary": {
                                    "chunksProcessed": chunks_processed,
                                    "chunksFailed": chunks_failed,
                                    "eventsFound": len(events),
                                    "currentChunk": chunk_index,
                                },
                            },
                        )
                    except Exception as chunk_exc:
                        chunks_failed += 1
                        consecutive_failures += 1
                        failure_detail = (
                            f"Chunk {chunk_index} falló "
                            f"({_dt_to_payload(chunk_start)} a {_dt_to_payload(chunk_end)}): {chunk_exc}"
                        )
                        limitations.append(failure_detail)
                        _update_job(
                            job_id,
                            stage="download_warning",
                            detail=failure_detail,
                            progress=0.04 + (0.88 * ratio_end),
                        )
                        _update_job_partial(
                            job_id,
                            {
                                "mode": "roi_interaction_search",
                                "events": [],
                                "summary": {
                                    "chunksProcessed": chunks_processed,
                                    "chunksFailed": chunks_failed,
                                    "consecutiveFailures": consecutive_failures,
                                    "eventsFound": len(events),
                                    "currentChunk": chunk_index,
                                    "lastError": failure_detail,
                                },
                            },
                        )
                    cursor = chunk_end

                final_events: list[dict[str, Any]] = []
                for event in events[:30]:
                    event_final = dict(event)
                    event_final["frameUrl"] = _artifact_url_for_path(job, event_final.get("frame_path"))
                    event_final["clipUrl"] = _artifact_url_for_openable_video_path(job, event_final.get("clip_path"))
                    final_events.append(event_final)

                result_payload = {
                    "job_id": job_id,
                    "mode": "roi_interaction_search",
                    "case_name": caseName,
                    "interactionType": interactionType,
                    "investigation_context": {
                        "vendor": normalized_vendor,
                        "host": resolved_host,
                        "httpPort": resolved_http_port,
                        "sdkPort": resolved_sdk_port,
                        "nvrName": resolved_nvr_name,
                        "nvrId": str(nvrId or "").strip(),
                        "cameraId": str(cameraId or "").strip(),
                        "cameraName": str(cameraName or "").strip(),
                        "logicalChannel": int(logicalChannel),
                        "username": resolved_user,
                    },
                    "events": final_events,
                    "summary": {
                        "rangeStartDt": _dt_to_payload(start_dt),
                        "rangeEndDt": _dt_to_payload(end_dt),
                        "durationSeconds": round(total_seconds, 2),
                        "chunksProcessed": chunks_processed,
                        "chunksFailed": chunks_failed,
                        "eventsFound": len(events),
                        "eventsReturned": len(final_events),
                        "roi": normalized_roi,
                        "settings": {
                            "chunkMinutes": chunk_minutes,
                            "sampleEverySeconds": sample_seconds,
                            "prePostSeconds": context_seconds,
                        },
                        "limitations": limitations,
                    },
                }
                with JOBS_LOCK:
                    job2 = JOBS.get(job_id)
                    if not job2:
                        return
                    if job2.cancel_requested:
                        job2.status = "cancelled"
                        job2.stage = "cancelled"
                        job2.detail = "ROI interaction search cancelado por el usuario."
                    else:
                        job2.status = "done"
                        job2.stage = "done"
                        job2.detail = "ROI interaction search completado."
                    job2.progress = 1.0 if not job2.cancel_requested else min(0.99, float(job2.progress or 0.0))
                    job2.result = result_payload
                    job2.partial_result = None
                    job2.updated_at = _now_iso()
                    _persist_job_state(job2)
        except Exception as exc:
            with JOBS_LOCK:
                job2 = JOBS.get(job_id)
                if job2:
                    job2.status = "error"
                    job2.stage = "error"
                    job2.detail = "ROI interaction search falló."
                    job2.error = str(exc)
                    job2.updated_at = _now_iso()
                    _persist_job_state(job2)

    background.add_task(runner)
    return {"ok": True, "jobId": job_id, "status": "running"}


@app.post("/api/investigation/static-object-discovery")
async def start_static_object_discovery(
    request: Request,
    background: BackgroundTasks,
    vendor: str = Form(...),
    host: str = Form(...),
    httpPort: int = Form(0),
    sdkPort: int = Form(...),
    nvrName: str = Form(""),
    nvrId: str = Form(""),
    username: str = Form(""),
    password: str | None = Form(None),
    cameraId: str = Form(""),
    cameraName: str = Form(""),
    logicalChannel: int = Form(...),
    startDt: str = Form(...),
    endDt: str = Form(...),
    caseName: str = Form("static-object-discovery"),
    similarityThreshold: float = Form(0.58),
    microclipSeconds: float = Form(8.0),
    targetWindowSeconds: float = Form(30.0),
    maxIterations: int = Form(12),
    representativeFrames: int = Form(5),
    roi: str = Form(""),
    queryImage: UploadFile = File(...),
) -> dict[str, Any]:
    resolved_vendor = str(vendor or "").strip()
    resolved_host = str(host or "").strip()
    resolved_http_port = int(httpPort or 0)
    resolved_sdk_port = int(sdkPort or 0)
    resolved_nvr_name = str(nvrName or "").strip()

    backend_profile = _load_backend_nvr_profile(nvrId)
    if backend_profile:
        resolved_vendor = str(backend_profile.get("brand") or resolved_vendor or "").strip()
        resolved_host = str(backend_profile.get("host") or resolved_host or "").strip()
        resolved_http_port = int(backend_profile.get("httpPort") or resolved_http_port or 0)
        resolved_sdk_port = int(backend_profile.get("sdkPort") or resolved_sdk_port or 0)
        resolved_nvr_name = str(backend_profile.get("name") or resolved_nvr_name or "").strip()

    normalized_vendor = _normalize_vendor(resolved_vendor)
    if normalized_vendor not in ("hikvision", "dahua"):
        raise HTTPException(status_code=400, detail=f"Vendor no soportado: {resolved_vendor or vendor}")

    resolved_user, resolved_pass = _resolve_nvr_credentials(
        nvr_id=nvrId,
        nvr_name=resolved_nvr_name,
        vendor=normalized_vendor,
        host=resolved_host,
        http_port=resolved_http_port,
        sdk_port=resolved_sdk_port,
        username=username,
        password=password,
    )
    if not resolved_user or not resolved_pass:
        raise HTTPException(status_code=400, detail="Faltan credenciales del NVR para discovery estático.")

    try:
        start_dt = _parse_iso_dt(startDt)
        end_dt = _parse_iso_dt(endDt)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Fechas inválidas: {exc}") from exc
    if end_dt <= start_dt:
        raise HTTPException(status_code=400, detail="endDt debe ser mayor que startDt.")
    now_local = datetime.now()
    if start_dt > now_local or end_dt > now_local:
        raise HTTPException(
            status_code=400,
            detail="Selecciona una fecha y hora anteriores a la actual. No podemos buscar grabaciones futuras.",
        )
    requested_microclip_seconds = float(microclipSeconds or STATIC_DISCOVERY_DEFAULT_MICROCLIP_SECONDS)
    if requested_microclip_seconds > STATIC_DISCOVERY_MAX_MICROCLIP_SECONDS:
        raise HTTPException(
            status_code=400,
            detail=(
                "microclipSeconds excede el máximo permitido. "
                f"Máximo: {STATIC_DISCOVERY_MAX_MICROCLIP_SECONDS:.0f}s."
            ),
        )
    clip_seconds = _clamp_float(
        microclipSeconds,
        default=STATIC_DISCOVERY_DEFAULT_MICROCLIP_SECONDS,
        min_value=2.0,
        max_value=STATIC_DISCOVERY_MAX_MICROCLIP_SECONDS,
    )
    target_seconds = max(STATIC_DISCOVERY_MIN_TARGET_WINDOW_SECONDS, float(targetWindowSeconds or 30.0))
    iterations_limit = max(1, min(int(maxIterations or 12), STATIC_DISCOVERY_MAX_ITERATIONS))
    representative_frame_count = max(1, min(int(representativeFrames or 5), 12))
    try:
        normalized_roi = _parse_static_discovery_roi(roi)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if normalized_roi and representative_frame_count < 7:
        representative_frame_count = 7

    job_id = uuid.uuid4().hex
    job_dir = OUTPUT_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir = job_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    query_path = inputs_dir / "query.jpg"
    content = await queryImage.read()
    if not content:
        raise HTTPException(status_code=400, detail="queryImage vacío.")
    query_path.write_bytes(content)

    base_url = _resolve_public_api_base(request)
    job = JobState(
        job_id=job_id,
        created_at=_now_iso(),
        updated_at=_now_iso(),
        status="running",
        stage="prepare",
        detail="Job creado. Iniciando discovery temporal estático...",
        progress=0.02,
        base_url=base_url,
        job_dir=job_dir,
    )
    with JOBS_LOCK:
        JOBS[job_id] = job
        _persist_job_state(job)

    def runner() -> None:
        runner_started = time.perf_counter()
        try:
            with ENGINE_LOCK:
                discovery_dir = job_dir / "static_discovery"
                downloads_dir = discovery_dir / "downloads"
                probes_dir = discovery_dir / "probes"
                downloads_dir.mkdir(parents=True, exist_ok=True)
                probes_dir.mkdir(parents=True, exist_ok=True)

                total_range_seconds = max(1.0, (end_dt - start_dt).total_seconds())
                half_clip = timedelta(seconds=clip_seconds / 2.0)
                checks: list[dict[str, Any]] = []
                check_index = 0
                bridge = _build_investigation_bridge(normalized_vendor, downloads_dir)
                baseline_roi_path: Path | None = None
                baseline_similarity_score: float | None = None

                def on_progress(stage: str, detail: str, ratio: float) -> None:
                    progress = 0.04 + (0.94 * max(0.0, min(1.0, float(ratio))))
                    _update_job(job_id, stage=stage, detail=detail, progress=progress)

                def clamp_probe_window(center_dt: datetime) -> tuple[datetime, datetime]:
                    desired = timedelta(seconds=clip_seconds)
                    probe_start = center_dt - half_clip
                    probe_end = center_dt + half_clip
                    if probe_start < start_dt:
                        probe_start = start_dt
                        probe_end = min(end_dt, probe_start + desired)
                    if probe_end > end_dt:
                        probe_end = end_dt
                        probe_start = max(start_dt, probe_end - desired)
                    if probe_end <= probe_start:
                        probe_start = max(start_dt, min(center_dt, end_dt) - desired)
                        probe_end = min(end_dt, probe_start + desired)
                    return probe_start, probe_end

                def run_check(center_dt: datetime, label: str, ratio: float) -> dict[str, Any]:
                    nonlocal check_index, baseline_roi_path, baseline_similarity_score
                    check_index += 1
                    safe_label = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in label.lower())
                    check_dir = probes_dir / f"{check_index:02d}_{safe_label}"
                    check_downloads_dir = check_dir / "downloads"
                    check_dir.mkdir(parents=True, exist_ok=True)
                    check_downloads_dir.mkdir(parents=True, exist_ok=True)
                    probe_start, probe_end = clamp_probe_window(center_dt)
                    probe_seconds = (probe_end - probe_start).total_seconds()
                    if probe_seconds > STATIC_DISCOVERY_MAX_MICROCLIP_SECONDS + 0.001:
                        raise RuntimeError(
                            f"Static discovery intentó descargar un microclip demasiado largo ({probe_seconds:.1f}s)."
                        )
                    on_progress(
                        "static_probe",
                        f"Check {check_index}: {label} en {_dt_to_payload(center_dt)}",
                        ratio,
                    )
                    with _measure_job_phase(
                        job_id,
                        "static_probe_download",
                        label=label,
                        channel=int(logicalChannel),
                        requestedSeconds=round(probe_seconds, 2),
                    ):
                        clip_result = _download_investigation_clip(
                            vendor=normalized_vendor,
                            bridge=bridge,
                            host=resolved_host,
                            sdk_port=resolved_sdk_port,
                            username=resolved_user,
                            password=resolved_pass,
                            logical_channel=int(logicalChannel),
                            start_dt=probe_start,
                            end_dt=probe_end,
                            local_target_dir=check_downloads_dir,
                            nvr_id=str(nvrId or "").strip(),
                            nvr_name=resolved_nvr_name,
                            camera_id=str(cameraId or "").strip(),
                            camera_name=str(cameraName or "").strip(),
                            progress_callback=lambda s, d: _update_job(job_id, stage=s, detail=d),
                        )
                    clip_path = ensure_openable_clip(
                        Path(clip_result["final_local_path"]),
                        output_dir=check_downloads_dir,
                    )
                    with _measure_job_phase(
                        job_id,
                        "static_probe_analysis",
                        label=label,
                        representativeFrames=representative_frame_count,
                        hasRoi=bool(normalized_roi),
                    ):
                        probe = probe_static_object_clip(
                            query_path=query_path,
                            video_path=clip_path,
                            output_dir=check_dir,
                            similarity_threshold=float(similarityThreshold or 0.58),
                            stage_label="static_probe",
                            sample_count=representative_frame_count,
                            roi=normalized_roi,
                            baseline_roi_path=baseline_roi_path,
                            baseline_similarity_score=baseline_similarity_score,
                            change_threshold=0.14,
                            combined_threshold=0.52,
                            require_change=bool(normalized_roi and baseline_roi_path is not None),
                        )
                    if normalized_roi and baseline_roi_path is None:
                        roi_reference = str(probe.get("roi_reference_path") or "").strip()
                        if roi_reference:
                            candidate_baseline = Path(roi_reference)
                            if candidate_baseline.exists():
                                baseline_roi_path = candidate_baseline
                        baseline_similarity_score = float(probe.get("similarityScore", 0.0) or 0.0)
                    best_hit = probe.get("best_hit") or {}
                    check = {
                        "index": check_index,
                        "label": label,
                        "timestamp": _dt_to_payload(center_dt),
                        "offsetSeconds": _offset_seconds(start_dt, center_dt),
                        "present": bool(probe.get("present")),
                        "candidate": bool(probe.get("candidate")),
                        "candidateConfidence": probe.get("candidateConfidence"),
                        "candidateReason": probe.get("candidateReason"),
                        "score": float(best_hit.get("score", 0.0) or 0.0) if isinstance(best_hit, dict) else 0.0,
                        "roi": normalized_roi,
                        "changeScore": float(probe.get("changeScore", 0.0) or 0.0),
                        "similarityScore": float(probe.get("similarityScore", 0.0) or 0.0),
                        "baselineSimilarityScore": probe.get("baselineSimilarityScore"),
                        "similarityDelta": float(probe.get("similarityDelta", 0.0) or 0.0),
                        "darkObjectScore": float(probe.get("darkObjectScore", 0.0) or 0.0),
                        "baselineDarkObjectScore": probe.get("baselineDarkObjectScore"),
                        "darkObjectDelta": float(probe.get("darkObjectDelta", 0.0) or 0.0),
                        "darkAreaFraction": float(probe.get("darkAreaFraction", 0.0) or 0.0),
                        "largestDarkComponent": float(probe.get("largestDarkComponent", 0.0) or 0.0),
                        "darkStructurePenalty": float(probe.get("darkStructurePenalty", 0.0) or 0.0),
                        "largestDarkComponentWidth": float(probe.get("largestDarkComponentWidth", 0.0) or 0.0),
                        "largestDarkComponentHeight": float(probe.get("largestDarkComponentHeight", 0.0) or 0.0),
                        "largestDarkComponentAspect": float(probe.get("largestDarkComponentAspect", 0.0) or 0.0),
                        "darkComponentEdgeContact": float(probe.get("darkComponentEdgeContact", 0.0) or 0.0),
                        "darkComponentTouchesBorder": float(probe.get("darkComponentTouchesBorder", 0.0) or 0.0),
                        "darkAreaDelta": float(probe.get("darkAreaDelta", 0.0) or 0.0),
                        "darkComponentDelta": float(probe.get("darkComponentDelta", 0.0) or 0.0),
                        "darkPersistence": float(probe.get("darkPersistence", 0.0) or 0.0),
                        "persistentDarkFrames": int(probe.get("persistentDarkFrames", 0) or 0),
                        "persistentVisualFrames": int(probe.get("persistentVisualFrames", 0) or 0),
                        "persistentVisualRatio": float(probe.get("persistentVisualRatio", 0.0) or 0.0),
                        "roiObjectScore": float(probe.get("roiObjectScore", 0.0) or 0.0),
                        "combinedScore": float(probe.get("combinedScore", 0.0) or 0.0),
                        "decision": probe.get("decision"),
                        "reason": probe.get("reason"),
                        "probeWindow": {
                            "startDt": _dt_to_payload(probe_start),
                            "endDt": _dt_to_payload(probe_end),
                            "startOffsetSeconds": _offset_seconds(start_dt, probe_start),
                            "endOffsetSeconds": _offset_seconds(start_dt, probe_end),
                        },
                        "clip": {
                            "path": str(clip_path),
                            "start_dt": _dt_to_payload(probe_start),
                            "end_dt": _dt_to_payload(probe_end),
                            "duration_seconds": round((probe_end - probe_start).total_seconds(), 2),
                        },
                        "probe": probe,
                    }
                    checks.append(check)
                    return check

                def candidate_supports_presence(check: dict[str, Any]) -> bool:
                    if bool(check.get("present")):
                        return True
                    if str(check.get("candidateConfidence") or "").strip().lower() != "candidate":
                        return False
                    probe = check.get("probe") or {}
                    persistent_visual_frames = int(
                        probe.get("persistentVisualFrames", check.get("persistentVisualFrames", 0)) or 0
                    )
                    similarity_score = float(
                        probe.get("similarityScore", check.get("similarityScore", 0.0)) or 0.0
                    )
                    change_score = float(probe.get("changeScore", check.get("changeScore", 0.0)) or 0.0)
                    similarity_delta = float(
                        probe.get("similarityDelta", check.get("similarityDelta", 0.0)) or 0.0
                    )
                    roi_object_score = float(
                        probe.get("roiObjectScore", check.get("roiObjectScore", 0.0)) or 0.0
                    )
                    return (
                        persistent_visual_frames >= 2
                        and similarity_score >= max(0.30, float(similarityThreshold or 0.58) - 0.14)
                        and (
                            change_score >= 0.08
                            or similarity_delta >= 0.06
                            or roi_object_score >= 0.18
                        )
                    )

                def refine_candidate_anchor(seed_check: dict[str, Any], label: str, ratio: float) -> dict[str, Any] | None:
                    probe_window = seed_check.get("probeWindow") or {}
                    best_hit = (seed_check.get("probe") or {}).get("best_hit") or {}
                    best_offset = float(best_hit.get("timestamp_seconds", 0.0) or 0.0)
                    start_raw = str(probe_window.get("startDt") or "").strip()
                    if not start_raw:
                        return None
                    try:
                        anchor_start = _parse_iso_dt(start_raw)
                    except Exception:
                        return None
                    anchor_dt = anchor_start + timedelta(seconds=max(0.0, best_offset))
                    return run_check(anchor_dt, label, ratio)

                def check_timestamp_dt(check: dict[str, Any]) -> datetime | None:
                    raw = str(check.get("timestamp") or "").strip()
                    if not raw:
                        return None
                    try:
                        return _parse_iso_dt(raw)
                    except Exception:
                        return None

                start_check = run_check(start_dt, "range_start", 0.08)
                end_check = run_check(end_dt, "range_end", 0.18)
                object_present_at_start = bool(start_check.get("present"))
                object_present_at_end = bool(end_check.get("present"))
                lower_check = start_check
                upper_check = end_check
                lower_dt_seed = start_dt
                upper_dt_seed = end_dt
                if normalized_roi and not object_present_at_end and candidate_supports_presence(end_check):
                    refined_end = refine_candidate_anchor(end_check, "range_end_refine", 0.21)
                    if refined_end is not None:
                        end_check = refined_end
                        object_present_at_end = bool(end_check.get("present")) or candidate_supports_presence(end_check)
                if normalized_roi and not object_present_at_end:
                    sweep_ratios = (0.20, 0.35, 0.50, 0.65, 0.80)
                    last_absent_check = start_check
                    for idx, sweep_ratio in enumerate(sweep_ratios, start=1):
                        sweep_dt = start_dt + ((end_dt - start_dt) * sweep_ratio)
                        sweep_check = run_check(
                            sweep_dt,
                            f"sweep_{idx:02d}_{int(round(sweep_ratio * 100)):02d}",
                            0.20 + (0.12 * idx),
                        )
                        if bool(sweep_check.get("present")) or candidate_supports_presence(sweep_check):
                            object_present_at_end = True
                            upper_check = sweep_check
                            upper_dt_seed = check_timestamp_dt(sweep_check) or sweep_dt
                            lower_check = last_absent_check
                            lower_dt_seed = check_timestamp_dt(last_absent_check) or start_dt
                            break
                        last_absent_check = sweep_check

                status = "unknown"
                estimated_window: dict[str, Any] | None = None

                if object_present_at_start:
                    status = "present_at_range_start"
                    estimated_window = {
                        "startDt": None,
                        "endDt": _dt_to_payload(start_dt),
                        "startOffsetSeconds": None,
                        "endOffsetSeconds": 0.0,
                        "confidence": "bounded_after",
                        "note": "El objeto ya aparece al inicio del rango; la aparición probable ocurrió antes de startDt.",
                    }
                elif not object_present_at_end:
                    status = "not_present_at_range_end"
                    estimated_window = None
                else:
                    lower_dt = lower_dt_seed
                    upper_dt = upper_dt_seed
                    for iteration in range(iterations_limit):
                        window_seconds = (upper_dt - lower_dt).total_seconds()
                        if window_seconds <= target_seconds:
                            break
                        mid_dt = lower_dt + ((upper_dt - lower_dt) / 2)
                        ratio = 0.22 + (0.70 * ((iteration + 1) / iterations_limit))
                        mid_check = run_check(mid_dt, f"binary_{iteration + 1:02d}", ratio)
                        if bool(mid_check.get("present")) or (normalized_roi and candidate_supports_presence(mid_check)):
                            upper_dt = mid_dt
                            upper_check = mid_check
                        else:
                            lower_dt = mid_dt
                            lower_check = mid_check

                    status = "bounded"
                    estimated_window = {
                        "startDt": _dt_to_payload(lower_dt),
                        "endDt": _dt_to_payload(upper_dt),
                        "startOffsetSeconds": _offset_seconds(start_dt, lower_dt),
                        "endOffsetSeconds": _offset_seconds(start_dt, upper_dt),
                        "durationSeconds": round((upper_dt - lower_dt).total_seconds(), 2),
                        "confidence": "bounded_by_binary_search",
                    }

                candidate_window: dict[str, Any] | None = None
                if estimated_window and estimated_window.get("startDt") and estimated_window.get("endDt"):
                    candidate_start_dt = _parse_iso_dt(str(estimated_window["startDt"]))
                    candidate_end_dt = _parse_iso_dt(str(estimated_window["endDt"]))
                    binary_candidate_start_dt = candidate_start_dt
                    binary_candidate_end_dt = candidate_end_dt
                    lower_probe_window = lower_check.get("probeWindow") if isinstance(lower_check.get("probeWindow"), dict) else {}
                    upper_probe_window = upper_check.get("probeWindow") if isinstance(upper_check.get("probeWindow"), dict) else {}
                    try:
                        lower_probe_start_dt = _parse_iso_dt(str(lower_probe_window.get("startDt") or ""))
                    except Exception:
                        lower_probe_start_dt = None
                    try:
                        upper_probe_end_dt = _parse_iso_dt(str(upper_probe_window.get("endDt") or ""))
                    except Exception:
                        upper_probe_end_dt = None
                    if lower_probe_start_dt and lower_probe_start_dt < candidate_start_dt:
                        candidate_start_dt = lower_probe_start_dt
                    if upper_probe_end_dt and upper_probe_end_dt > candidate_end_dt:
                        candidate_end_dt = upper_probe_end_dt
                    original_candidate_seconds = max(0.0, (candidate_end_dt - candidate_start_dt).total_seconds())
                    candidate_warnings: list[str] = []
                    original_estimated_window: dict[str, Any] | None = None
                    if original_candidate_seconds > STATIC_DISCOVERY_MAX_SAFE_CANDIDATE_WINDOW_SECONDS:
                        original_estimated_window = dict(estimated_window)
                        candidate_warnings.append(
                            "La ventana estimada excede el rango seguro configurado; "
                            "se devuelve una ventana candidata acotada sin descargar el rango completo."
                        )
                        candidate_start_dt = max(
                            candidate_start_dt,
                            candidate_end_dt - timedelta(seconds=STATIC_DISCOVERY_MAX_SAFE_CANDIDATE_WINDOW_SECONDS),
                        )
                    candidate_seconds = max(0.0, (candidate_end_dt - candidate_start_dt).total_seconds())
                    candidate_window = {
                        "startDt": _dt_to_payload(candidate_start_dt),
                        "endDt": _dt_to_payload(candidate_end_dt),
                        "startOffsetSeconds": _offset_seconds(start_dt, candidate_start_dt),
                        "endOffsetSeconds": _offset_seconds(start_dt, candidate_end_dt),
                        "durationSeconds": round(candidate_seconds, 2),
                        "preAppearanceCheck": lower_check,
                        "postAppearanceCheck": upper_check,
                        "binaryCenterWindow": {
                            "startDt": _dt_to_payload(binary_candidate_start_dt),
                            "endDt": _dt_to_payload(binary_candidate_end_dt),
                            "startOffsetSeconds": _offset_seconds(start_dt, binary_candidate_start_dt),
                            "endOffsetSeconds": _offset_seconds(start_dt, binary_candidate_end_dt),
                            "durationSeconds": round(
                                max(0.0, (binary_candidate_end_dt - binary_candidate_start_dt).total_seconds()),
                                2,
                            ),
                        },
                        "edgeProbeWindow": {
                            "startDt": _dt_to_payload(candidate_start_dt),
                            "endDt": _dt_to_payload(candidate_end_dt),
                            "startOffsetSeconds": _offset_seconds(start_dt, candidate_start_dt),
                            "endOffsetSeconds": _offset_seconds(start_dt, candidate_end_dt),
                            "durationSeconds": round(candidate_seconds, 2),
                        },
                        "nextStep": "run_deep_search_in_candidate_window",
                        "safeWindowPolicy": {
                            "maxCandidateWindowSeconds": STATIC_DISCOVERY_MAX_SAFE_CANDIDATE_WINDOW_SECONDS,
                            "fullRangeDownloadAllowed": False,
                        },
                    }
                    if original_estimated_window:
                        candidate_window["originalEstimatedWindow"] = original_estimated_window
                    if candidate_warnings:
                        candidate_window["warnings"] = candidate_warnings

                suggested_action: dict[str, Any]
                if status == "present_at_range_start":
                    suggested_action = {
                        "type": "expand_backward",
                        "label": "El objeto ya estaba al inicio; buscar hacia atrás",
                        "recommendedMinutes": 30,
                        "reason": "Para encontrar quién lo dejó se necesita un inicio sin objeto y un final con objeto.",
                    }
                elif status == "not_present_at_range_end":
                    suggested_action = {
                        "type": "review_motion_candidates",
                        "label": "Revisar movimientos candidatos o ampliar rango",
                        "recommendedMinutes": 30,
                        "reason": "No se confirmó objeto al final; se devuelven candidatos visuales para revisión manual.",
                    }
                elif candidate_window:
                    suggested_action = {
                        "type": "confirm_candidate_window",
                        "label": "Confirmar ventana encontrada",
                        "reason": "Se encontró una transición temporal usable para analizar el clip corto.",
                    }
                else:
                    suggested_action = {
                        "type": "adjust_roi_or_range",
                        "label": "Ajustar ROI o rango",
                        "reason": "El sistema no obtuvo una transición confiable con los datos actuales.",
                    }

                result_payload: dict[str, Any] = {
                    "job_id": job_id,
                    "vendor": normalized_vendor,
                    "case_name": caseName,
                    "mode": "static_object_discovery",
                    "investigation_context": {
                        "vendor": normalized_vendor,
                        "host": resolved_host,
                        "httpPort": resolved_http_port,
                        "sdkPort": resolved_sdk_port,
                        "nvrName": resolved_nvr_name,
                        "nvrId": str(nvrId or "").strip(),
                        "logicalChannel": int(logicalChannel),
                        "cameraId": str(cameraId or "").strip(),
                        "cameraName": str(cameraName or "").strip(),
                        "username": resolved_user,
                    },
                    "next_step": suggested_action["type"],
                    "suggestedAction": suggested_action,
                    "static_discovery": {
                        "strategy": "roi_temporal_change_plus_visual_similarity_microclips"
                        if normalized_roi
                        else "temporal_binary_search_microclips_similarity_fallback",
                        "acquisition": "microclip_probe",
                        "range": {
                            "startDt": _dt_to_payload(start_dt),
                            "endDt": _dt_to_payload(end_dt),
                            "durationSeconds": round(total_range_seconds, 2),
                        },
                        "settings": {
                            "similarityThreshold": float(similarityThreshold or 0.58),
                            "microclipSeconds": clip_seconds,
                            "targetWindowSeconds": target_seconds,
                            "maxIterations": iterations_limit,
                            "representativeFrames": representative_frame_count,
                            "roi": normalized_roi,
                            "roiPolicy": "user_marked_normalized_roi" if normalized_roi else "no_roi_similarity_fallback",
                            "changeThreshold": 0.14,
                            "combinedThreshold": 0.52,
                            "maxSafeCandidateWindowSeconds": STATIC_DISCOVERY_MAX_SAFE_CANDIDATE_WINDOW_SECONDS,
                            "longRangePolicy": "microclips_only_never_download_full_range",
                        },
                        "checks": checks,
                        "objectPresentAtStart": object_present_at_start,
                        "objectPresentAtEnd": object_present_at_end,
                        "estimatedAppearanceWindow": estimated_window,
                        "candidateWindow": candidate_window,
                        "suggestedAction": suggested_action,
                        "status": status,
                        "source_context": {
                            "vendor": normalized_vendor,
                            "host": resolved_host,
                            "httpPort": resolved_http_port,
                            "sdkPort": resolved_sdk_port,
                            "nvrName": resolved_nvr_name,
                            "nvrId": str(nvrId or "").strip(),
                            "logicalChannel": int(logicalChannel),
                            "cameraId": str(cameraId or "").strip(),
                            "cameraName": str(cameraName or "").strip(),
                            "username": resolved_user,
                        },
                    },
                    "checks": checks,
                    "objectPresentAtStart": object_present_at_start,
                    "objectPresentAtEnd": object_present_at_end,
                    "estimatedAppearanceWindow": estimated_window,
                    "candidateWindow": candidate_window,
                    "suggestedAction": suggested_action,
                }

                result_payload = _rewrite_paths_to_urls(job, result_payload)
                try:
                    with (discovery_dir / "result_full.json").open("w", encoding="utf-8") as file:
                        json.dump(result_payload, file, indent=2, ensure_ascii=False)
                except Exception:
                    pass
                static_discovery = result_payload.get("static_discovery")
                if isinstance(static_discovery, dict):
                    light_checks = [
                        _lighten_static_check(check)
                        for check in static_discovery.get("checks", [])
                        if isinstance(check, dict)
                    ]
                    static_discovery["checks"] = light_checks
                    static_discovery["candidateWindow"] = _lighten_static_candidate_window(
                        static_discovery.get("candidateWindow")
                    )
                    result_payload["static_discovery"] = static_discovery
                    result_payload["checks"] = light_checks
                    result_payload["candidateWindow"] = static_discovery.get("candidateWindow")
                try:
                    with (discovery_dir / "result.json").open("w", encoding="utf-8") as file:
                        json.dump(result_payload, file, indent=2, ensure_ascii=False)
                except Exception:
                    pass
                _record_job_metric(
                    job_id,
                    "static_discovery_total",
                    time.perf_counter() - runner_started,
                    status="done",
                    checks=len(checks),
                )
                with JOBS_LOCK:
                    job2 = JOBS.get(job_id)
                    if not job2:
                        return
                    job2.status = "done"
                    job2.stage = "done"
                    job2.detail = "Discovery temporal estático completado."
                    job2.progress = 1.0
                    job2.result = result_payload
                    job2.updated_at = _now_iso()
                    _persist_job_state(job2)
        except Exception as exc:
            _record_job_metric(
                job_id,
                "static_discovery_total",
                time.perf_counter() - runner_started,
                status="error",
            )
            with JOBS_LOCK:
                job2 = JOBS.get(job_id)
                if job2:
                    job2.status = "error"
                    job2.stage = "error"
                    job2.detail = "El discovery temporal estático falló."
                    job2.error = str(exc)
                    job2.progress = min(float(job2.progress or 0.0), 0.99)
                    job2.updated_at = _now_iso()
                    _persist_job_state(job2)

    background.add_task(runner)
    return {"jobId": job_id}


@app.post("/api/investigation/first-appearance")
async def start_first_appearance(
    request: Request,
    background: BackgroundTasks,
    vendor: str = Form(...),
    host: str = Form(...),
    httpPort: int = Form(0),
    sdkPort: int = Form(...),
    nvrName: str = Form(""),
    nvrId: str = Form(""),
    # Optional: el frontend (React) puede NO enviar username/password.
    # Con default "" FastAPI no lo marca como "required" (evita HTTP 422)
    # y nosotros lo resolvemos desde `nvr_profiles.local.json`.
    username: str = Form(""),
    password: str | None = Form(None),
    cameraId: str = Form(""),
    cameraName: str = Form(""),
    logicalChannel: int = Form(...),
    startDt: str = Form(...),
    endDt: str = Form(...),
    caseName: str = Form("first-appearance"),
    coarseStepSeconds: float = Form(8),
    refineStepSeconds: float = Form(1),
    investigationRadius: float = Form(25),
    similarityThreshold: float = Form(0.58),
    deferDeepSearch: bool = Form(False),
    allowLongClip: bool = Form(False),
    maxInitialClipSeconds: float = Form(300),
    queryImage: UploadFile = File(...),
) -> dict[str, Any]:
    resolved_vendor = str(vendor or "").strip()
    resolved_host = str(host or "").strip()
    resolved_http_port = int(httpPort or 0)
    resolved_sdk_port = int(sdkPort or 0)
    resolved_nvr_name = str(nvrName or "").strip()

    backend_profile = _load_backend_nvr_profile(nvrId)
    if backend_profile:
        resolved_vendor = str(backend_profile.get("brand") or resolved_vendor or "").strip()
        resolved_host = str(backend_profile.get("host") or resolved_host or "").strip()
        resolved_http_port = int(backend_profile.get("httpPort") or resolved_http_port or 0)
        resolved_sdk_port = int(backend_profile.get("sdkPort") or resolved_sdk_port or 0)
        resolved_nvr_name = str(backend_profile.get("name") or resolved_nvr_name or "").strip()

    normalized_vendor = _normalize_vendor(resolved_vendor)
    if normalized_vendor not in ("hikvision", "dahua"):
        raise HTTPException(status_code=400, detail=f"Vendor no soportado: {resolved_vendor or vendor}")

    # Credentials: React normalmente NO envía username/password (por seguridad).
    # Intentamos resolverlos desde el archivo local de perfiles.
    resolved_user, resolved_pass = _resolve_nvr_credentials(
        nvr_id=nvrId,
        nvr_name=resolved_nvr_name,
        vendor=normalized_vendor,
        host=resolved_host,
        http_port=resolved_http_port,
        sdk_port=resolved_sdk_port,
        username=username,
        password=password,
    )

    if not resolved_user or not resolved_pass:
        raise HTTPException(
            status_code=400,
            detail=(
                "Faltan credenciales del NVR (username/password). "
                "Para el MVP, agrega el NVR a nvr_profiles.local.json o envía username/password en el request."
            ),
        )

    try:
        start_dt = _parse_iso_dt(startDt)
        end_dt = _parse_iso_dt(endDt)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Fechas inválidas: {exc}") from exc
    if end_dt <= start_dt:
        raise HTTPException(status_code=400, detail="endDt debe ser mayor que startDt.")
    now_local = datetime.now()
    if start_dt > now_local or end_dt > now_local:
        raise HTTPException(
            status_code=400,
            detail="Selecciona una fecha y hora anteriores a la actual. No podemos buscar grabaciones futuras.",
        )
    requested_seconds = (end_dt - start_dt).total_seconds()
    requested_max_initial = max(60.0, float(maxInitialClipSeconds or FIRST_APPEARANCE_MAX_INITIAL_CLIP_SECONDS))
    max_clip_seconds = min(requested_max_initial, FIRST_APPEARANCE_MAX_INITIAL_CLIP_SECONDS)
    if requested_seconds > max_clip_seconds:
        raise HTTPException(
            status_code=400,
            detail=(
                "El barrido inicial no puede descargar rangos largos. "
                f"Rango solicitado: {requested_seconds / 60:.1f} min; límite: {max_clip_seconds / 60:.1f} min. "
                "Usa el modo Objeto abandonado/static-object-discovery para rangos largos."
            ),
        )

    job_id = uuid.uuid4().hex
    job_dir = OUTPUT_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir = job_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    # Save query image
    query_path = inputs_dir / "query.jpg"
    content = await queryImage.read()
    if not content:
        raise HTTPException(status_code=400, detail="queryImage vacío.")
    query_path.write_bytes(content)

    base_url = _resolve_public_api_base(request)

    job = JobState(
        job_id=job_id,
        created_at=_now_iso(),
        updated_at=_now_iso(),
        status="running",
        stage="prepare",
        detail="Job creado. Iniciando...",
        progress=0.02,
        base_url=base_url,
        job_dir=job_dir,
    )
    with JOBS_LOCK:
        JOBS[job_id] = job
        _persist_job_state(job)

    def runner() -> None:
        # The runner needs the query path inside the job directory; we pass it through the job_dir.
        try:
            with JOBS_LOCK:
                job_ref = JOBS.get(job_id)
            if job_ref is None:
                return

            # Local re-implementation of the pipeline (now that query exists).
            with ENGINE_LOCK:
                downloads_dir = job_dir / "downloads"
                scans_dir = job_dir / "scans"
                deep_dir = job_dir / "deep"
                downloads_dir.mkdir(parents=True, exist_ok=True)
                scans_dir.mkdir(parents=True, exist_ok=True)
                deep_dir.mkdir(parents=True, exist_ok=True)

                def on_progress(stage: str, detail: str, ratio: float) -> None:
                    base = 0.0
                    span = 1.0
                    if stage in ("download", "server_download", "transfer", "normalize"):
                        base, span = 0.05, 0.40
                    elif stage in ("barrido inicial", "coarse"):
                        base, span = 0.45, 0.18
                    elif stage in ("refinamiento", "refine"):
                        base, span = 0.63, 0.12
                    elif stage in ("deep", "deep_analysis"):
                        base, span = 0.78, 0.20
                    progress = base + (span * max(0.0, min(1.0, float(ratio))))
                    _update_job(job_id, stage=stage, detail=detail, progress=progress)

                _update_job(job_id, stage="download", detail="Descargando clip desde el NVR (via bridge)...", progress=0.06)
                if normalized_vendor == "hikvision":
                    bridge = HikvisionBridgeSettings(
                        ssh_host=runtime_config.REMOTE_BRIDGE_HOST,
                        ssh_user=runtime_config.REMOTE_BRIDGE_USER,
                        ssh_key_path=runtime_config.SSH_KEY_PATH,
                        remote_python=runtime_config.REMOTE_BRIDGE_PYTHON,
                        local_download_dir=downloads_dir,
                    )
                    clip_result = download_clip_via_bridge(
                        bridge=bridge,
                        host=resolved_host,
                        sdk_port=int(resolved_sdk_port),
                        username=resolved_user,
                        password=resolved_pass,
                        logical_channel=int(logicalChannel),
                        start_dt=start_dt,
                        end_dt=end_dt,
                        local_target_dir=downloads_dir,
                        normalize_for_review=False,
                        progress_callback=lambda s, d: on_progress(s, d, 0.5),
                    )
                else:
                    bridge = DahuaRemoteBridgeSettings(
                        ssh_host=runtime_config.REMOTE_BRIDGE_HOST,
                        ssh_user=runtime_config.REMOTE_BRIDGE_USER,
                        ssh_key_path=runtime_config.SSH_KEY_PATH,
                        remote_python=runtime_config.REMOTE_BRIDGE_PYTHON,
                        remote_sdk_dir=runtime_config.DAHUA_REMOTE_SDK_DIR,
                    )
                    clip_result = _download_dahua_investigation_clip(
                        bridge=bridge,
                        host=resolved_host,
                        sdk_port=int(resolved_sdk_port),
                        username=resolved_user,
                        password=resolved_pass,
                        logical_channel=int(logicalChannel),
                        start_dt=start_dt,
                        end_dt=end_dt,
                        local_target_dir=downloads_dir,
                        nvr_id=str(nvrId or "").strip(),
                        nvr_name=resolved_nvr_name,
                        camera_id=str(cameraId or "").strip(),
                        camera_name=str(cameraName or "").strip(),
                        progress_callback=lambda s, d: on_progress(s, d, 0.5),
                    )

                clip_path = Path(clip_result["final_local_path"])
                on_progress("normalize", "Preparando clip para análisis local...", 0.15)
                clip_path = ensure_analysis_clip(clip_path, output_dir=downloads_dir)
                on_progress("normalize", "Clip listo para análisis.", 1.0)

                coarse_report = quick_scan_clip(
                    query_path=query_path,
                    video_path=clip_path,
                    output_dir=scans_dir / "coarse_scan",
                    sample_every_seconds=float(coarseStepSeconds),
                    similarity_threshold=float(similarityThreshold),
                    stage_label="barrido inicial",
                    keep_top=12,
                    time_offset_seconds=0.0,
                    on_progress=on_progress,
                )
                if not coarse_report.get("earliest_hit"):
                    raise RuntimeError("No encontré el objeto en el rango elegido.")

                coarse_first = coarse_report["earliest_hit"]
                duration_seconds = float(coarse_report.get("duration_seconds", 0.0) or 0.0)
                coarse_start = max(0.0, float(coarse_first["absolute_seconds"]) - (float(coarseStepSeconds) * 2))
                coarse_end = min(duration_seconds, float(coarse_first["absolute_seconds"]) + float(coarseStepSeconds))
                refine_segment_path = extract_video_segment(
                    source_video=clip_path,
                    output_video=scans_dir / "refine_segment.mp4",
                    start_seconds=coarse_start,
                    end_seconds=coarse_end,
                )

                refine_report = quick_scan_clip(
                    query_path=query_path,
                    video_path=refine_segment_path,
                    output_dir=scans_dir / "refine_scan",
                    sample_every_seconds=float(refineStepSeconds),
                    similarity_threshold=float(similarityThreshold),
                    stage_label="refinamiento",
                    keep_top=8,
                    time_offset_seconds=float(coarse_start),
                    on_progress=on_progress,
                )
                refined_first = refine_report.get("earliest_hit") or coarse_first

                result_payload: dict[str, Any] = {
                    "job_id": job_id,
                    "vendor": normalized_vendor,
                    "case_name": caseName,
                    "mode": "initial_scan" if deferDeepSearch else "full_pipeline",
                    "clip": {
                        "path": str(clip_path),
                        "duration_seconds": duration_seconds,
                    },
                    "coarse_report": coarse_report,
                    "refine_report": refine_report,
                    "refined_first": refined_first,
                    "investigation_context": {
                        "vendor": normalized_vendor,
                        "host": resolved_host,
                        "httpPort": resolved_http_port,
                        "sdkPort": resolved_sdk_port,
                        "nvrName": resolved_nvr_name,
                        "nvrId": str(nvrId or "").strip(),
                        "logicalChannel": int(logicalChannel),
                        "cameraId": str(cameraId or "").strip(),
                        "cameraName": str(cameraName or "").strip(),
                        "username": resolved_user,
                    },
                }
                if deferDeepSearch:
                    result_payload["next_step"] = "confirm_object"
                else:
                    radius = float(investigationRadius)
                    deep_start = max(0.0, float(refined_first["absolute_seconds"]) - radius)
                    deep_end = min(duration_seconds, float(refined_first["absolute_seconds"]) + radius)
                    deep_segment_path = extract_video_segment(
                        source_video=clip_path,
                        output_video=deep_dir / "deep_segment.mp4",
                        start_seconds=deep_start,
                        end_seconds=deep_end,
                    )
                    deep_report = run_deep_analysis(
                        query_path=query_path,
                        video_path=deep_segment_path,
                        output_dir=deep_dir / "analysis",
                        similarity_threshold=float(similarityThreshold),
                        frame_step=2,
                        person_trigger_mode="always",
                        person_detection_frame_step=DEEP_PERSON_DETECTION_FRAME_STEP,
                        preview_callback_sample_interval=3,
                        max_results=6,
                        save_annotated_video=DEEP_SAVE_ANNOTATED_VIDEO,
                        early_stop_on_person_match=True,
                        on_progress=on_progress,
                    )
                    deep_matches, first_match = _enrich_deep_matches(deep_report, deep_start=deep_start)
                    if deep_matches:
                        deep_report["matches"] = deep_matches
                    if first_match:
                        deep_report["first_match"] = first_match
                    result_payload["deep"] = deep_report
                    result_payload["first_match"] = first_match or {}
                    result_payload["object_moment_clip"] = _build_object_moment_clip_payload(
                        deep_segment_path=deep_segment_path,
                        window_start_seconds=deep_start,
                        window_end_seconds=deep_end,
                        first_match=first_match,
                    )

                result_payload = _rewrite_paths_to_urls(job, result_payload)

                with JOBS_LOCK:
                    job2 = JOBS.get(job_id)
                    if not job2:
                        return
                    job2.status = "done"
                    job2.stage = "done"
                    job2.detail = "Listo."
                    job2.progress = 1.0
                    job2.result = result_payload
                    job2.updated_at = _now_iso()
                    _persist_job_state(job2)
        except Exception as exc:
            with JOBS_LOCK:
                job2 = JOBS.get(job_id)
                if job2:
                    job2.status = "error"
                    job2.stage = "error"
                    job2.detail = "El job falló."
                    job2.error = str(exc)
                    job2.progress = min(float(job2.progress or 0.0), 0.99)
                    job2.updated_at = _now_iso()
                    _persist_job_state(job2)

    background.add_task(runner)
    return {"jobId": job_id}


@app.get("/api/investigation/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return job.to_payload()


@app.post("/api/investigation/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict[str, Any]:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if job.status != "running":
            return {"ok": True, "jobId": job_id, "status": job.status}
        job.cancel_requested = True
        job.stage = "cancelling"
        job.detail = "Cancelación solicitada. Cerrando al terminar el microclip actual..."
        job.updated_at = _now_iso()
        return {"ok": True, "jobId": job_id, "status": "cancelling"}


@app.post("/api/investigation/jobs/{job_id}/deep-search")
def start_deep_search(
    job_id: str,
    payload: DeepSearchRequest,
    background: BackgroundTasks,
) -> dict[str, Any]:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if job.status == "running":
            raise HTTPException(status_code=409, detail="El job todavía está ejecutándose.")
        result_payload = dict(job.result or {})
        job_dir = job.job_dir

    static_source_context = _get_static_source_context(result_payload)
    selected_hit = _pick_hit_from_result(
        result_payload,
        selected_hit_index=payload.selectedHitIndex,
        selected_absolute_seconds=payload.selectedAbsoluteSeconds,
    )
    static_check = _pick_static_check_from_result(
        result_payload,
        selected_check_index=payload.selectedCheckIndex,
        selected_hit_index=payload.selectedHitIndex,
        selected_absolute_seconds=payload.selectedAbsoluteSeconds,
    )
    clip = result_payload.get("clip") or {}
    clip_path = Path(str(clip.get("path") or "").strip()) if str(clip.get("path") or "").strip() else None

    static_candidate_window = _static_candidate_window_from_result(result_payload)
    prefer_static_window = bool(payload.preferWindow and static_candidate_window and static_source_context)
    if prefer_static_window:
        selected_hit = _build_static_transition_hit_from_result(result_payload) or selected_hit

    if (not clip_path or not clip_path.exists()) and not static_source_context:
        raise HTTPException(
            status_code=400,
            detail="No encontré un clip base válido para profundizar. Ejecuta nuevamente el barrido inicial o el discovery estático.",
        )
    if not selected_hit and not (payload.preferWindow and static_candidate_window):
        raise HTTPException(status_code=400, detail="No encontré un hallazgo útil para lanzar la búsqueda profunda.")

    query_path = job_dir / "inputs" / "query.jpg"
    if not query_path.exists():
        raise HTTPException(status_code=400, detail="No encontré la imagen query asociada al job.")

    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        job.status = "running"
        job.stage = "deep_prepare"
        job.detail = "Objeto confirmado. Preparando búsqueda profunda..."
        job.progress = 0.78
        job.updated_at = _now_iso()

    def runner() -> None:
        runner_started = time.perf_counter()
        try:
            with ENGINE_LOCK:
                deep_dir = job_dir / "deep"
                deep_dir.mkdir(parents=True, exist_ok=True)
                live_preview_path = deep_dir / "analysis" / "live_preview.jpg"
                live_preview_callback = _make_live_preview_callback(live_preview_path)

                def on_progress(stage: str, detail: str, ratio: float) -> None:
                    base, span = 0.78, 0.20
                    progress = base + (span * max(0.0, min(1.0, float(ratio))))
                    _update_job(job_id, stage=stage, detail=detail, progress=progress)
                    live_payload = dict(result_payload)
                    live_payload["deepLive"] = {
                        "stage": stage,
                        "detail": detail,
                        "progress": round(progress, 4),
                        "previewPath": str(live_preview_path),
                        "previewUrl": _artifact_url(job, live_preview_path.relative_to(job.job_dir)),
                        "updatedAt": _now_iso(),
                    }
                    _update_job_partial(job_id, _rewrite_paths_to_urls(job, live_payload))

                source_clip_path = clip_path if clip_path and clip_path.exists() else None
                source_clip_start_offset = 0.0
                source_clip_end_offset = float(clip.get("duration_seconds", 0.0) or 0.0)
                selected_relative_seconds = float(
                    selected_hit.get("absolute_seconds", selected_hit.get("timestamp_seconds", 0.0)) or 0.0
                ) if selected_hit else 0.0
                source_analysis_mode = "initial_clip"

                if source_clip_path is None and static_source_context:
                    source_analysis_mode = "static_discovery_clip"
                    downloads_dir = deep_dir / "source_window"
                    downloads_dir.mkdir(parents=True, exist_ok=True)

                    window_start_dt = None
                    window_end_dt = None
                    window_start_offset = 0.0
                    window_end_offset = 0.0
                    fallback_clip_url = ""
                    fallback_clip_start_dt = None
                    fallback_clip_end_dt = None
                    range_start_dt = None
                    range_end_dt = None
                    static_range = (
                        result_payload.get("static_discovery", {}).get("range")
                        if isinstance(result_payload.get("static_discovery"), dict)
                        else {}
                    )
                    if isinstance(static_range, dict):
                        range_start_dt = _parse_optional_dt(static_range.get("startDt"))
                        range_end_dt = _parse_optional_dt(static_range.get("endDt"))

                    selected_absolute_seconds = None
                    if selected_hit:
                        try:
                            selected_absolute_seconds = float(
                                selected_hit.get("absolute_seconds", selected_hit.get("timestamp_seconds", 0.0)) or 0.0
                            )
                        except Exception:
                            selected_absolute_seconds = None

                    if prefer_static_window and static_candidate_window:
                        candidate_start_dt = _parse_optional_dt(static_candidate_window.get("startDt"))
                        candidate_end_dt = _parse_optional_dt(static_candidate_window.get("endDt"))
                        if candidate_start_dt and candidate_end_dt and candidate_end_dt > candidate_start_dt:
                            before_seconds, after_seconds = _person_context_window_seconds(payload.investigationRadius)
                            window_start_dt = candidate_start_dt - timedelta(seconds=before_seconds)
                            window_end_dt = candidate_end_dt + timedelta(seconds=after_seconds)
                            if range_start_dt:
                                window_start_dt = max(range_start_dt, window_start_dt)
                            if range_end_dt:
                                window_end_dt = min(range_end_dt, window_end_dt)
                            if range_start_dt:
                                window_start_offset = max(0.0, (window_start_dt - range_start_dt).total_seconds())
                                window_end_offset = max(window_start_offset, (window_end_dt - range_start_dt).total_seconds())
                            else:
                                window_start_offset = float(static_candidate_window.get("startOffsetSeconds", 0.0) or 0.0)
                                window_end_offset = float(static_candidate_window.get("endOffsetSeconds", 0.0) or 0.0)

                    if not prefer_static_window and range_start_dt and selected_absolute_seconds is not None:
                        before_seconds, after_seconds = _person_context_window_seconds(payload.investigationRadius)
                        anchor_dt = range_start_dt + timedelta(seconds=max(0.0, selected_absolute_seconds))
                        window_start_dt = anchor_dt - timedelta(seconds=before_seconds)
                        window_end_dt = anchor_dt + timedelta(seconds=after_seconds)
                        if range_start_dt:
                            window_start_dt = max(range_start_dt, window_start_dt)
                        if range_end_dt:
                            window_end_dt = min(range_end_dt, window_end_dt)
                        window_start_offset = max(0.0, (window_start_dt - range_start_dt).total_seconds())
                        window_end_offset = max(window_start_offset, (window_end_dt - range_start_dt).total_seconds())

                    if (not window_start_dt or not window_end_dt or window_end_dt <= window_start_dt) and static_candidate_window:
                        window_start_dt = _parse_optional_dt(static_candidate_window.get("startDt"))
                        window_end_dt = _parse_optional_dt(static_candidate_window.get("endDt"))
                        window_start_offset = float(static_candidate_window.get("startOffsetSeconds", 0.0) or 0.0)
                        window_end_offset = float(static_candidate_window.get("endOffsetSeconds", 0.0) or 0.0)
                        if (window_start_offset <= 0.0 and window_end_offset <= 0.0) and isinstance(
                            static_source_context, dict
                        ):
                            if range_start_dt and window_start_dt and window_end_dt:
                                window_start_offset = max(0.0, (window_start_dt - range_start_dt).total_seconds())
                                window_end_offset = max(0.0, (window_end_dt - range_start_dt).total_seconds())

                    if not window_start_dt or not window_end_dt or window_end_dt <= window_start_dt:
                        if isinstance(static_check, dict):
                            clip_info = static_check.get("clip") if isinstance(static_check.get("clip"), dict) else {}
                            probe_window = (
                                static_check.get("probeWindow") if isinstance(static_check.get("probeWindow"), dict) else {}
                            )
                            fallback_clip_url = str(clip_info.get("url") or "")
                            fallback_clip_start_dt = _parse_optional_dt(
                                clip_info.get("start_dt") or probe_window.get("startDt")
                            )
                            fallback_clip_end_dt = _parse_optional_dt(
                                clip_info.get("end_dt") or probe_window.get("endDt")
                            )
                            window_start_dt = fallback_clip_start_dt
                            window_end_dt = fallback_clip_end_dt
                            window_start_offset = float(probe_window.get("startOffsetSeconds", 0.0) or 0.0)
                            window_end_offset = float(probe_window.get("endOffsetSeconds", 0.0) or 0.0)

                    if not window_start_dt or not window_end_dt or window_end_dt <= window_start_dt:
                        raise RuntimeError("No encontré una ventana válida para profundizar sobre el discovery estático.")

                    fallback_artifact = _resolve_job_artifact_path(job, fallback_clip_url) if fallback_clip_url else None
                    can_reuse_fallback = (
                        fallback_artifact
                        and fallback_artifact.exists()
                        and fallback_clip_start_dt
                        and fallback_clip_end_dt
                        and fallback_clip_start_dt <= window_start_dt
                        and fallback_clip_end_dt >= window_end_dt
                    )
                    if can_reuse_fallback:
                        source_clip_path = _prefer_native_clip_variant(fallback_artifact)
                    else:
                        resolved_vendor = _normalize_vendor(str(static_source_context.get("vendor") or result_payload.get("vendor") or ""))
                        resolved_host = str(static_source_context.get("host") or "").strip()
                        resolved_sdk_port = int(static_source_context.get("sdkPort") or 0)
                        resolved_nvr_name = str(static_source_context.get("nvrName") or "").strip()
                        resolved_nvr_id = str(static_source_context.get("nvrId") or "").strip()
                        resolved_logical_channel = int(static_source_context.get("logicalChannel") or 1)
                        resolved_camera_id = str(static_source_context.get("cameraId") or "").strip()
                        resolved_camera_name = str(static_source_context.get("cameraName") or "").strip()
                        resolved_username, resolved_password = _resolve_nvr_credentials(
                            nvr_id=resolved_nvr_id,
                            nvr_name=resolved_nvr_name,
                            vendor=resolved_vendor,
                            host=resolved_host,
                            http_port=int(static_source_context.get("httpPort") or 0),
                            sdk_port=resolved_sdk_port,
                            username=str(static_source_context.get("username") or ""),
                            password=None,
                        )
                        bridge = _build_investigation_bridge(resolved_vendor, downloads_dir)
                        on_progress("deep_prepare", "Descargando ventana alrededor del abandono para asociar persona...", 0.08)
                        with _measure_job_phase(
                            job_id,
                            "deep_source_download",
                            channel=resolved_logical_channel,
                            requestedSeconds=round((window_end_dt - window_start_dt).total_seconds(), 2),
                        ):
                            clip_result = _download_investigation_clip(
                                vendor=resolved_vendor,
                                bridge=bridge,
                                host=resolved_host,
                                sdk_port=resolved_sdk_port,
                                username=resolved_username,
                                password=resolved_password,
                                logical_channel=resolved_logical_channel,
                                start_dt=window_start_dt,
                                end_dt=window_end_dt,
                                local_target_dir=downloads_dir,
                                nvr_id=resolved_nvr_id,
                                nvr_name=resolved_nvr_name,
                                camera_id=resolved_camera_id,
                                camera_name=resolved_camera_name,
                                progress_callback=lambda stage, detail: _update_job(job_id, stage=stage, detail=detail),
                            )
                        source_clip_path = _prefer_native_clip_variant(Path(clip_result["final_local_path"]))

                    source_clip_start_offset = max(0.0, float(window_start_offset or 0.0))
                    source_clip_end_offset = max(source_clip_start_offset, float(window_end_offset or 0.0))
                    if prefer_static_window and static_candidate_window:
                        selected_relative_seconds = max(
                            0.0,
                            (
                                float(static_candidate_window.get("startOffsetSeconds", source_clip_start_offset) or 0.0)
                                + float(static_candidate_window.get("endOffsetSeconds", source_clip_end_offset) or 0.0)
                            )
                            / 2.0
                            - source_clip_start_offset,
                        )
                    elif selected_hit:
                        selected_relative_seconds = max(
                            0.0,
                            float(selected_hit.get("absolute_seconds", selected_hit.get("timestamp_seconds", 0.0)) or 0.0)
                            - source_clip_start_offset,
                        )
                    else:
                        selected_relative_seconds = max(
                            0.0,
                            (source_clip_end_offset - source_clip_start_offset) / 2.0,
                        )

                if source_clip_path is None or not source_clip_path.exists():
                    raise RuntimeError("No encontré el clip fuente para ejecutar la búsqueda profunda.")
                source_clip_path = _prefer_native_clip_variant(source_clip_path)

                if source_analysis_mode == "static_discovery_clip":
                    capture = cv2.VideoCapture(str(source_clip_path))
                    if not capture.isOpened():
                        raise RuntimeError(f"No pude abrir el clip corto confirmado: {source_clip_path}")
                    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 1.0)
                    source_total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                    capture.release()
                    source_duration_seconds = (
                        float(source_total_frames) / source_fps if source_fps > 0 and source_total_frames > 0 else 0.0
                    )
                    deep_start_relative = 0.0
                    deep_end_relative = source_duration_seconds or max(
                        5.0,
                        float(source_clip_end_offset or 0.0) - float(source_clip_start_offset or 0.0),
                    )
                    if deep_end_relative <= deep_start_relative:
                        deep_end_relative = deep_start_relative + 5.0
                    source_window_start_offset = float(source_clip_start_offset)
                    source_window_end_offset = float(source_clip_end_offset)
                    source_clip_start_offset = source_window_start_offset + deep_start_relative
                    source_clip_end_offset = source_window_start_offset + deep_end_relative
                    if source_window_end_offset > 0:
                        source_clip_end_offset = min(source_clip_end_offset, source_window_end_offset)
                    on_progress("deep_prepare", "Extrayendo clip exacto alrededor del objeto confirmado...", 0.08)
                    with _measure_job_phase(
                        job_id,
                        "deep_segment_extract",
                        requestedSeconds=round(deep_end_relative - deep_start_relative, 2),
                        mode=source_analysis_mode,
                    ):
                        deep_segment_path = extract_video_segment(
                            source_video=source_clip_path,
                            output_video=deep_dir / "deep_segment.mp4",
                            start_seconds=deep_start_relative,
                            end_seconds=deep_end_relative,
                        )
                else:
                    duration_seconds = float(clip.get("duration_seconds", 0.0) or 0.0)
                    radius = float(payload.investigationRadius or 25.0)
                    deep_start = max(0.0, selected_relative_seconds - radius)
                    deep_end = min(duration_seconds, selected_relative_seconds + radius) if duration_seconds > 0 else selected_relative_seconds + radius
                    if deep_end <= deep_start:
                        deep_end = deep_start + max(5.0, radius)
                    source_clip_start_offset = deep_start
                    source_clip_end_offset = deep_end
                    on_progress("deep_prepare", "Extrayendo ventana profunda alrededor del hallazgo confirmado...", 0.08)
                    with _measure_job_phase(
                        job_id,
                        "deep_segment_extract",
                        requestedSeconds=round(deep_end - deep_start, 2),
                        mode=source_analysis_mode,
                    ):
                        deep_segment_path = extract_video_segment(
                            source_video=source_clip_path,
                            output_video=deep_dir / "deep_segment.mp4",
                            start_seconds=deep_start,
                            end_seconds=deep_end,
                        )

                with _measure_job_phase(
                    job_id,
                    "deep_analysis",
                    frameStep=2,
                    personFrameStep=DEEP_PERSON_DETECTION_FRAME_STEP,
                ):
                    deep_report = run_deep_analysis(
                        query_path=query_path,
                        video_path=deep_segment_path,
                        output_dir=deep_dir / "analysis",
                        similarity_threshold=float(payload.similarityThreshold or 0.58),
                        frame_step=2,
                        person_trigger_mode="always",
                        person_detection_frame_step=DEEP_PERSON_DETECTION_FRAME_STEP,
                        preview_callback_sample_interval=3,
                        max_results=6,
                        save_annotated_video=DEEP_SAVE_ANNOTATED_VIDEO,
                        early_stop_on_person_match=True,
                        on_progress=on_progress,
                        preview_callback=live_preview_callback,
                    )
                deep_matches, first_match = _enrich_deep_matches(deep_report, deep_start=source_clip_start_offset)
                if deep_matches:
                    deep_report["matches"] = deep_matches
                if first_match:
                    deep_report["first_match"] = first_match

                next_payload = dict(result_payload)
                next_payload["mode"] = "guided_pipeline"
                next_payload["next_step"] = "confirm_person"
                next_payload["confirmed_object_hit"] = dict(selected_hit or first_match or {})
                next_payload["deep"] = deep_report
                next_payload["first_match"] = first_match or {}
                next_payload["deep_window_start"] = source_clip_start_offset
                next_payload["deep_window_end"] = source_clip_end_offset
                next_payload["object_moment_clip"] = _build_object_moment_clip_payload(
                    deep_segment_path=deep_segment_path,
                    window_start_seconds=source_clip_start_offset,
                    window_end_seconds=source_clip_end_offset,
                    first_match=first_match,
                )
                next_payload = _rewrite_paths_to_urls(job, next_payload)
                _record_job_metric(
                    job_id,
                    "deep_search_total",
                    time.perf_counter() - runner_started,
                    status="done",
                    matches=len(deep_matches or []),
                )

                with JOBS_LOCK:
                    job2 = JOBS.get(job_id)
                    if not job2:
                        return
                    job2.status = "done"
                    job2.stage = "done"
                    job2.detail = "Búsqueda profunda completada."
                    job2.progress = 1.0
                    job2.result = next_payload
                    job2.updated_at = _now_iso()
                    _persist_job_state(job2)
        except Exception as exc:
            _record_job_metric(
                job_id,
                "deep_search_total",
                time.perf_counter() - runner_started,
                status="error",
            )
            with JOBS_LOCK:
                job2 = JOBS.get(job_id)
                if job2:
                    job2.status = "error"
                    job2.stage = "error"
                    job2.detail = "La búsqueda profunda falló."
                    job2.error = str(exc)
                    job2.progress = min(float(job2.progress or 0.0), 0.99)
                    job2.updated_at = _now_iso()
                    _persist_job_state(job2)

    background.add_task(runner)
    return {"jobId": job_id}


@app.post("/api/investigation/jobs/{job_id}/track-person")
def track_person_in_next_camera(
    job_id: str,
    payload: TrackPersonRequest,
    background: BackgroundTasks,
) -> dict[str, Any]:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if job.status == "running":
            raise HTTPException(status_code=409, detail="El job todavía está ejecutándose.")
        result_payload = dict(job.result or {})
        job_dir = job.job_dir

    normalized_vendor = _normalize_vendor(payload.vendor)
    if normalized_vendor not in ("hikvision", "dahua"):
        raise HTTPException(status_code=400, detail=f"Vendor no soportado: {payload.vendor}")

    try:
        start_dt = _parse_iso_dt(payload.startDt)
        end_dt = _parse_iso_dt(payload.endDt)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Fechas inválidas: {exc}") from exc
    if end_dt <= start_dt:
        raise HTTPException(status_code=400, detail="endDt debe ser mayor que startDt.")

    resolved_user, resolved_pass = _resolve_nvr_credentials(
        nvr_id=payload.nvrId,
        nvr_name=str(payload.nvrName or "").strip(),
        vendor=normalized_vendor,
        host=str(payload.host or "").strip(),
        http_port=int(payload.httpPort or 0),
        sdk_port=int(payload.sdkPort or 0),
        username=payload.username,
        password=payload.password,
    )
    if not resolved_user or not resolved_pass:
        raise HTTPException(status_code=400, detail="No encontré credenciales válidas para rastrear a la persona.")

    discovery_only = bool(payload.discoveryOnly)
    reference_paths: list[Path] = []
    if not discovery_only:
        reference_paths = _collect_person_reference_paths(
            result_payload,
            selected_person_index=payload.selectedPersonIndex,
        )
        explicit_reference_paths = _collect_reference_paths_from_urls(
            job=job,
            urls=payload.referenceUrls,
        )
        for path in explicit_reference_paths:
            if path not in reference_paths:
                reference_paths.append(path)
    if not reference_paths and not discovery_only:
        raise HTTPException(
            status_code=400,
            detail="No encontré crops de persona confirmada en el job. Confirma primero la persona asociada.",
        )

    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        job.status = "running"
        job.stage = "track_person_prepare"
        job.detail = "Preparando rastreo de persona en la siguiente cámara..."
        job.progress = 0.82
        job.updated_at = _now_iso()

    def runner() -> None:
        runner_started = time.perf_counter()
        try:
            with ENGINE_LOCK:
                track_dir = job_dir / "related_person"
                downloads_dir = track_dir / "downloads"
                refs_dir = track_dir / "references"
                scans_dir = track_dir / "scans"
                deep_dir = track_dir / "deep"
                downloads_dir.mkdir(parents=True, exist_ok=True)
                refs_dir.mkdir(parents=True, exist_ok=True)
                scans_dir.mkdir(parents=True, exist_ok=True)
                deep_dir.mkdir(parents=True, exist_ok=True)

                copied_refs: list[dict[str, Any]] = []
                copied_ref_paths: list[Path] = []
                for idx, ref_path in enumerate(reference_paths, start=1):
                    copied = refs_dir / f"reference_{idx:02d}{ref_path.suffix.lower() or '.jpg'}"
                    copied.write_bytes(ref_path.read_bytes())
                    copied_refs.append({"index": idx - 1, "path": str(copied), "source_path": str(ref_path)})
                    copied_ref_paths.append(copied)

                def on_progress(stage: str, detail: str, ratio: float) -> None:
                    base = 0.82
                    span = 0.17
                    if stage in ("download", "server_download", "transfer", "normalize"):
                        base, span = 0.82, 0.06
                    elif stage in ("person_track_coarse", "person_track_refine"):
                        base, span = 0.88, 0.06
                    elif stage in ("deep", "deep_analysis", "person_track_deep"):
                        base, span = 0.94, 0.05
                    progress = base + (span * max(0.0, min(1.0, float(ratio))))
                    _update_job(job_id, stage=stage, detail=detail, progress=progress)

                _update_job(
                    job_id,
                    stage="server_download",
                    detail="Descargando clip de la siguiente cámara para rastrear a la persona...",
                    progress=0.83,
                )

                with _measure_job_phase(
                    job_id,
                    "track_source_download",
                    vendor=normalized_vendor,
                    channel=int(payload.logicalChannel),
                    requestedSeconds=round((end_dt - start_dt).total_seconds(), 2),
                ):
                    if normalized_vendor == "hikvision":
                        bridge = HikvisionBridgeSettings(
                            ssh_host=runtime_config.REMOTE_BRIDGE_HOST,
                            ssh_user=runtime_config.REMOTE_BRIDGE_USER,
                            ssh_key_path=runtime_config.SSH_KEY_PATH,
                            remote_python=runtime_config.REMOTE_BRIDGE_PYTHON,
                            local_download_dir=downloads_dir,
                        )
                        clip_result = download_clip_via_bridge(
                            bridge=bridge,
                            host=str(payload.host).strip(),
                            sdk_port=int(payload.sdkPort),
                            username=resolved_user,
                            password=resolved_pass,
                            logical_channel=int(payload.logicalChannel),
                            start_dt=start_dt,
                            end_dt=end_dt,
                            local_target_dir=downloads_dir,
                            normalize_for_review=False,
                            progress_callback=lambda s, d: on_progress(s, d, 0.6),
                        )
                    else:
                        bridge = DahuaRemoteBridgeSettings(
                            ssh_host=runtime_config.REMOTE_BRIDGE_HOST,
                            ssh_user=runtime_config.REMOTE_BRIDGE_USER,
                            ssh_key_path=runtime_config.SSH_KEY_PATH,
                            remote_python=runtime_config.REMOTE_BRIDGE_PYTHON,
                            remote_sdk_dir=runtime_config.DAHUA_REMOTE_SDK_DIR,
                        )
                        clip_result = _download_dahua_investigation_clip(
                            bridge=bridge,
                            host=str(payload.host).strip(),
                            sdk_port=int(payload.sdkPort),
                            username=resolved_user,
                            password=resolved_pass,
                            logical_channel=int(payload.logicalChannel),
                            start_dt=start_dt,
                            end_dt=end_dt,
                            local_target_dir=downloads_dir,
                            nvr_id=payload.nvrId,
                            nvr_name=str(payload.nvrName or "").strip(),
                            camera_id=str(payload.cameraId or "").strip(),
                            camera_name=str(payload.cameraName or "").strip(),
                            progress_callback=lambda s, d: on_progress(s, d, 0.6),
                        )

                secondary_clip_path = ensure_analysis_clip(
                    Path(clip_result["final_local_path"]),
                    output_dir=downloads_dir,
                )

                if discovery_only or not copied_ref_paths:
                    with _measure_job_phase(job_id, "track_coarse_scan", mode="discovery"):
                        coarse_report = _scan_clip_for_person_candidates(
                            video_path=secondary_clip_path,
                            output_dir=scans_dir / "coarse_person_discovery",
                            sample_every_seconds=max(0.35, float(payload.coarseStepSeconds or 2.0)),
                            stage_label="person_track_coarse",
                            keep_top=16,
                            time_offset_seconds=0.0,
                            on_progress=on_progress,
                        )
                else:
                    with _measure_job_phase(job_id, "track_coarse_scan", mode="reference", references=len(copied_ref_paths)):
                        coarse_report = _scan_clip_for_person_references(
                            reference_paths=copied_ref_paths,
                            video_path=secondary_clip_path,
                            output_dir=scans_dir / "coarse_person",
                            sample_every_seconds=float(payload.coarseStepSeconds or 2.0),
                            similarity_threshold=max(0.28, float(payload.similarityThreshold or 0.45) - 0.08),
                            stage_label="person_track_coarse",
                            keep_top=12,
                            time_offset_seconds=0.0,
                            on_progress=on_progress,
                        )
                coarse_candidates = coarse_report.get("top_hits") or []
                if not isinstance(coarse_candidates, list) or not coarse_candidates:
                    raise RuntimeError(
                        "No encontré personas útiles en la cámara elegida para este rango."
                        if discovery_only
                        else "No encontré a la persona en el rango elegido."
                    )

                best_coarse = coarse_candidates[0]
                secondary_duration = 0.0
                capture = cv2.VideoCapture(str(secondary_clip_path))
                if capture.isOpened():
                    fps = capture.get(cv2.CAP_PROP_FPS) or 1.0
                    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                    secondary_duration = total_frames / fps if fps else 0.0
                capture.release()

                refine_radius = max(float(payload.coarseStepSeconds or 2.0) * 2.0, 3.0)
                refine_start = max(0.0, float(best_coarse.get("absolute_seconds", 0.0)) - refine_radius)
                refine_end = (
                    min(secondary_duration, float(best_coarse.get("absolute_seconds", 0.0)) + refine_radius)
                    if secondary_duration > 0
                    else float(best_coarse.get("absolute_seconds", 0.0)) + refine_radius
                )
                if refine_end <= refine_start:
                    refine_end = refine_start + max(5.0, refine_radius)

                with _measure_job_phase(
                    job_id,
                    "track_refine_segment_extract",
                    requestedSeconds=round(refine_end - refine_start, 2),
                ):
                    refine_segment_path = extract_video_segment(
                        source_video=secondary_clip_path,
                        output_video=scans_dir / "refine_segment.mp4",
                        start_seconds=refine_start,
                        end_seconds=refine_end,
                    )

                if discovery_only or not copied_ref_paths:
                    with _measure_job_phase(job_id, "track_refine_scan", mode="discovery"):
                        refine_report = _scan_clip_for_person_candidates(
                            video_path=refine_segment_path,
                            output_dir=scans_dir / "refine_person_discovery",
                            sample_every_seconds=max(0.25, float(payload.refineStepSeconds or 1.0)),
                            stage_label="person_track_refine",
                            keep_top=16,
                            time_offset_seconds=refine_start,
                            on_progress=on_progress,
                        )
                else:
                    with _measure_job_phase(job_id, "track_refine_scan", mode="reference", references=len(copied_ref_paths)):
                        refine_report = _scan_clip_for_person_references(
                            reference_paths=copied_ref_paths,
                            video_path=refine_segment_path,
                            output_dir=scans_dir / "refine_person",
                            sample_every_seconds=max(0.35, float(payload.refineStepSeconds or 1.0)),
                            similarity_threshold=max(0.25, float(payload.similarityThreshold or 0.45) - 0.10),
                            stage_label="person_track_refine",
                            keep_top=12,
                            time_offset_seconds=refine_start,
                            on_progress=on_progress,
                        )

                refined_first = (
                    refine_report.get("earliest_hit")
                    or best_coarse
                )

                deep_radius = float(payload.investigationRadius or 20.0)
                deep_start = max(0.0, float(refined_first.get("absolute_seconds", 0.0)) - deep_radius)
                deep_end = (
                    min(secondary_duration, float(refined_first.get("absolute_seconds", 0.0)) + deep_radius)
                    if secondary_duration > 0
                    else float(refined_first.get("absolute_seconds", 0.0)) + deep_radius
                )
                if deep_end <= deep_start:
                    deep_end = deep_start + max(5.0, deep_radius)

                with _measure_job_phase(
                    job_id,
                    "track_deep_segment_extract",
                    requestedSeconds=round(deep_end - deep_start, 2),
                ):
                    deep_segment_path = extract_video_segment(
                        source_video=secondary_clip_path,
                        output_video=deep_dir / "deep_segment.mp4",
                        start_seconds=deep_start,
                        end_seconds=deep_end,
                    )

                if discovery_only or not copied_ref_paths:
                    with _measure_job_phase(job_id, "track_deep_scan", mode="discovery"):
                        deep_report = _scan_clip_for_person_candidates(
                            video_path=deep_segment_path,
                            output_dir=deep_dir / "person_analysis_discovery",
                            sample_every_seconds=0.25,
                            stage_label="person_track_deep",
                            keep_top=20,
                            time_offset_seconds=deep_start,
                            on_progress=on_progress,
                        )
                else:
                    with _measure_job_phase(job_id, "track_deep_scan", mode="reference", references=len(copied_ref_paths)):
                        deep_report = _scan_clip_for_person_references(
                            reference_paths=copied_ref_paths,
                            video_path=deep_segment_path,
                            output_dir=deep_dir / "person_analysis",
                            sample_every_seconds=0.35,
                            similarity_threshold=max(0.24, float(payload.similarityThreshold or 0.45) - 0.12),
                            stage_label="person_track_deep",
                            keep_top=16,
                            time_offset_seconds=deep_start,
                            on_progress=on_progress,
                        )
                deep_report["matches"] = deep_report.get("top_hits") or []
                deep_first = (
                    deep_report.get("earliest_hit")
                    or refine_report.get("earliest_hit")
                    or refined_first
                )
                best_reference_path = None
                if copied_ref_paths:
                    best_reference_path = Path(
                        str(
                            (deep_first or best_coarse).get("reference_path")
                            or best_coarse.get("reference_path")
                            or copied_ref_paths[0]
                        )
                    )

                track_payload = {
                    "camera_name": str(payload.logicalChannel),
                    "mode": "person_track",
                    "clip": {
                        "path": str(secondary_clip_path),
                        "duration_seconds": round(float(secondary_duration), 2),
                        "start_dt": start_dt.isoformat(),
                        "end_dt": end_dt.isoformat(),
                    },
                    "tracking_clip": {
                        "path": str(deep_segment_path),
                        "start_seconds": round(float(deep_start), 2),
                        "end_seconds": round(float(deep_end), 2),
                    },
                    "references": copied_refs,
                    "discovery_only": discovery_only,
                    "selected_reference_index": int(best_coarse.get("reference_index", 0) or 0),
                    "selected_reference_path": str(best_reference_path) if best_reference_path else "",
                    "coarse_report": coarse_report,
                    "refine_report": refine_report,
                    "refined_first": refined_first,
                    "deep": deep_report,
                    "first_match": deep_first or refined_first,
                    "next_step": "tracking_review",
                }

                next_payload = dict(result_payload)
                next_payload["person_track"] = track_payload
                next_payload = _rewrite_paths_to_urls(job, next_payload)
                _record_job_metric(
                    job_id,
                    "track_person_total",
                    time.perf_counter() - runner_started,
                    status="done",
                    candidates=len(deep_report.get("matches") or []),
                )

                with JOBS_LOCK:
                    job2 = JOBS.get(job_id)
                    if not job2:
                        return
                    job2.status = "done"
                    job2.stage = "done"
                    job2.detail = "Rastreo de persona completado."
                    job2.progress = 1.0
                    job2.result = next_payload
                    job2.updated_at = _now_iso()
                    _persist_job_state(job2)
        except Exception as exc:
            _record_job_metric(
                job_id,
                "track_person_total",
                time.perf_counter() - runner_started,
                status="error",
            )
            with JOBS_LOCK:
                job2 = JOBS.get(job_id)
                if job2:
                    job2.status = "error"
                    job2.stage = "error"
                    job2.detail = "El rastreo de persona falló."
                    job2.error = str(exc)
                    job2.progress = min(float(job2.progress or 0.0), 0.99)
                    job2.updated_at = _now_iso()
                    _persist_job_state(job2)

    background.add_task(runner)
    return {"jobId": job_id}


@app.get("/api/investigation/artifacts/{job_id}/{artifact_path:path}")
def get_artifact(job_id: str, artifact_path: str) -> FileResponse:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        job_dir = job.job_dir if job else OUTPUT_ROOT / job_id

    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found")

    candidate = (job_dir / artifact_path).resolve()
    if job_dir not in candidate.parents and candidate != job_dir:
        raise HTTPException(status_code=400, detail="Invalid artifact path")
    if not candidate.exists():
        raise HTTPException(status_code=404, detail="Artifact not found")
    media_type = "video/mp4" if candidate.suffix.lower() == ".mp4" else None
    return FileResponse(str(candidate), media_type=media_type)
