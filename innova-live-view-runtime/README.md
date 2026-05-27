# Innova Live View Runtime

This folder is a local, repo-ready copy of the Python live streaming stack that is
currently deployed on the Ubuntu media server.

## What is in here

- `app/server.py`
  - FastAPI service that starts and stops per-slot streams.
- `.env.example`
  - Environment variables used by the live streaming API.
- `deployment/systemd/innova-liveview-streaming.service`
  - Linux systemd unit used on the Ubuntu server.
- `deployment/nginx/innova-liveview.conf`
  - Nginx site configuration used to expose HLS, the live-view API, the
    investigation API, and MediaMTX WebRTC endpoints.
- `deployment/mediamtx/mediamtx.yml`
  - MediaMTX configuration used by the current Ubuntu server.
- `requirements.txt`
  - Minimal Python dependencies inferred from the deployed service.
- `docs/ARCHITECTURE.md`
  - High-level explanation of how the runtime works.
- `docs/DEPLOY_UBUNTU.md`
  - Practical deployment checklist for rebuilding the live-view stack on a new
    Ubuntu server.

## What this service does

This runtime is the media engine behind the dashboard's Live View page.

Flow:

1. The frontend asks the Java backend to start a camera in a slot.
2. The Java backend forwards the request to this Python service.
3. This service builds the correct RTSP source and launches `ffmpeg`.
4. `ffmpeg` either:
   - writes HLS to `/var/www/innova-live-view-runtime/slot-X`, or
   - publishes RTSP to MediaMTX for WebRTC.
5. Nginx exposes the generated media to the browser.

## Relationship to AI Investigation

This folder is separate from:

- `../innova-ai-investigation-runtime`

That other runtime contains:

- NVR bridges
- Hikvision / Dahua SDK integration
- evidence extraction
- AI investigation jobs

This folder only contains the **live streaming runtime**.

## Current production shape

On Ubuntu, the live-view stack is split into:

- Python FastAPI service
- `ffmpeg`
- `MediaMTX`
- `nginx`

It is not a single monolith and it is not inside the AI Investigation runtime.
