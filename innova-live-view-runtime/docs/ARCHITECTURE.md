# Live View Architecture

## Purpose

The live-view runtime is the media backend used by the dashboard's Live View
page. It is independent from the AI Investigation runtime.

## Components

### 1. Python API (`app/server.py`)

This FastAPI service:

- receives start and stop requests per slot
- validates whether the request is HLS passthrough or RTSP-based transcoding
- starts and stops `ffmpeg`
- tracks active runtime processes in `runtime/registry.json`
- exposes `/health` and `/status`
- checks when HLS or WebRTC is truly ready before reporting success

### 2. FFmpeg

`ffmpeg` is the real worker that converts RTSP to streamable output.

Two modes are supported:

- `hls-transcode`
  - RTSP -> HLS playlist + TS segments
- `webrtc-publish`
  - RTSP -> RTSP publish into MediaMTX

### 3. MediaMTX

MediaMTX receives the RTSP publish from `ffmpeg` and exposes WebRTC/WHEP.

It also exposes an API on `127.0.0.1:9997` that the Python service queries to
detect whether a WebRTC path is actually ready.

### 4. Nginx

Nginx publishes:

- `/live-view-runtime/`
- `/live-view-api/`
- `/live-view-webrtc/`
- `/investigation-api/`

This makes the streaming server usable from browsers and from the main backend.

## Runtime directories on Ubuntu

Expected Linux layout:

```text
/opt/innova/live-streaming/
  .env
  .venv/
  app/server.py
  logs/
  runtime/registry.json
  tmp/

/var/www/innova-live-view-runtime/
  slot-0/
  slot-1/
  ...
```

## HLS flow

1. Request arrives with `slot` + `rtspUrl`
2. Python service stops prior process in that slot
3. Python service starts `ffmpeg`
4. `ffmpeg` writes:
   - `index.m3u8`
   - `seg_000001.ts`, etc.
5. Python waits until playlist + segments really exist
6. Browser consumes `/live-view-runtime/slot-X/index.m3u8`

## WebRTC flow

1. Request arrives with `slot` + `rtspUrl`
2. Python service starts `ffmpeg`
3. `ffmpeg` publishes RTSP to `MediaMTX`
4. Python checks MediaMTX API for ready path
5. Browser connects using WHEP URL

## Relationship to the Java backend

The browser should ideally talk to the public dashboard domain.
The Java backend or public proxy then forwards media requests to this live-view
server.

That means:

- media generation stays on Ubuntu
- business logic stays on Java
- browser stays on public domain
