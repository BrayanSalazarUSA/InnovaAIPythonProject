# Innova Python Stack Overview

This workspace currently contains two active Python runtimes that matter for
production:

## 1. AI Investigation + Bridge

Path:

- `innova-ai-investigation-runtime`

Responsibilities:

- NVR bridge
- Hikvision integration
- Dahua integration
- camera discovery
- snapshots
- evidence extraction
- AI investigation jobs

In production, this runtime should use the Java backend as the source of truth
for NVR records, camera metadata, and credentials, while `resources/` remains
the local fallback area for development and controlled server-side files.

Important internals:

- `src/innova_investigation/api_server.py`
- `src/innova_investigation/bridges/hikvision.py`
- `src/innova_investigation/bridges/dahua.py`
- `src/innova_investigation/config.py`

## 2. Live View Runtime

Path:

- `innova-live-view-runtime`

Responsibilities:

- per-slot stream startup
- HLS generation
- WebRTC publication
- FFmpeg orchestration
- MediaMTX coordination

In production, this runtime is triggered by the Java backend and should be
treated as a media execution service, not as the primary business API.

Important internals:

- `app/server.py`
- `deployment/nginx/innova-liveview.conf`
- `deployment/mediamtx/mediamtx.yml`
- `deployment/systemd/innova-liveview-streaming.service`

## Legacy folders

These are not the preferred active runtimes anymore:

- `03-busqueda-visual-cctv`
- `04-investigation-mvp`
- `05-camera-health-mvp`

Treat them as legacy references until final production validation is complete.
