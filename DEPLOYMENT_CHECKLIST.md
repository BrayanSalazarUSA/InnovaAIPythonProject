# Deployment Checklist

This checklist is the final handoff for deploying the stable Python platform on
the new Ubuntu server.

## Scope

Stable services in this folder:

- `innova-ai-investigation-runtime`
- `innova-live-view-runtime`

The Java backend remains the source of truth for:

- users
- roles
- properties
- NVR records
- camera metadata
- credentials

These Python runtimes are execution services, not replacements for that Java
layer.

Before a fresh deploy, also review:

- `innova-ai-investigation-runtime/docs/MODES_OF_OPERATION.md`
- `innova-ai-investigation-runtime/docs/UBUNTU_PRODUCTION_NOTES.md`

## AI Investigation runtime

Required on Ubuntu:

- `/opt/innova/investigation`
- `.venv`
- `.env`
- `yolo11n.pt`
- `resources/nvr_profiles.local.json`
- `resources/keys/elastic-beanstalk.pem`
- Hikvision SDK extracted
- Dahua SDK extracted
- `ffmpeg`
- `java`
- `javac`

Validation before cutover:

- Hikvision discovery
- Dahua discovery
- Hikvision snapshot
- Dahua snapshot
- Hikvision evidence clip download
- Dahua evidence clip download
- one complete AI job end-to-end

## Live View runtime

Required on Ubuntu:

- `/opt/innova/live-streaming`
- Python service running
- `ffmpeg`
- `MediaMTX`
- `nginx`
- `/var/www/innova-live-view-runtime`

Validation before cutover:

- HLS start for at least one camera
- HLS public playback through the main domain
- WebRTC path behavior understood and documented
- stop/cleanup works per slot
- `/health` and `/status` respond correctly

## Before decommissioning the old test server

Do not retire the old server until all of these are true:

- Java backend points only to the new Ubuntu services
- the new Ubuntu `.env` files do not contain old hosts or local Mac paths
- Hikvision and Dahua work on the new Ubuntu server
- evidence extraction is confirmed
- Live View is stable from the public dashboard

## Recommended repo root

Use this folder as the new repo root:

- `innova-python-platform/`
