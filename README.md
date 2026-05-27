# Innova Python Platform

This folder is the consolidated home for the current Python-side production
stack.

## Production relationship with the Java backend

This folder does not replace the main Java backend.

In production, the Java backend remains the system of record for:

- users and roles
- properties
- NVR records
- camera catalog
- credentials and operational metadata

The Python runtimes here should be understood as execution services:

- `innova-ai-investigation-runtime`
  - bridge, evidence extraction, AI jobs
- `innova-live-view-runtime`
  - streaming, HLS, WebRTC, FFmpeg orchestration

That means these Python services should keep working alongside the Java backend,
not instead of it.

## Recommended reading order

If you are trying to understand the stack quickly, start here:

1. `ARCHITECTURE_OVERVIEW.md`
2. `DEPLOYMENT_CHECKLIST.md`
3. `innova-ai-investigation-runtime/docs/MODES_OF_OPERATION.md`
4. `innova-ai-investigation-runtime/docs/UBUNTU_PRODUCTION_NOTES.md`
5. `innova-live-view-runtime/docs/ARCHITECTURE.md`

## Stable runtimes

### 1. AI Investigation + Bridge

Path:

- `innova-ai-investigation-runtime/`

Contains:

- Hikvision bridge
- Dahua bridge
- camera discovery
- snapshot extraction
- evidence / clip extraction
- AI investigation jobs
- Linux deployment docs and SDK setup

Start reading here:

- `innova-ai-investigation-runtime/README.md`
- `innova-ai-investigation-runtime/docs/ARCHITECTURE.md`
- `innova-ai-investigation-runtime/docs/INDEPENDENCE_AUDIT.md`

### 2. Live View Runtime

Path:

- `innova-live-view-runtime/`

Contains:

- Python live-view API
- FFmpeg orchestration
- MediaMTX config
- nginx config
- Ubuntu deployment docs

Start reading here:

- `innova-live-view-runtime/README.md`
- `innova-live-view-runtime/docs/ARCHITECTURE.md`
- `innova-live-view-runtime/docs/DEPLOY_UBUNTU.md`

## Folder intent

This directory is meant to become the clean repo root for the stable Python
platform.

The legacy experiment folders still exist one level above this directory, but
they are no longer the preferred place to understand or evolve the system.

## Suggested repo scope

If you create a fresh repo from here, this folder should be the one to use:

- `innova-python-platform/`

And the stable contents are:

- `innova-ai-investigation-runtime/`
- `innova-live-view-runtime/`
- `README.md`
