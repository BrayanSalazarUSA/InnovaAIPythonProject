# Independence Audit

This document explains whether the modern investigation runtime still depends on
legacy folders such as:

- `03-busqueda-visual-cctv`
- `04-investigation-mvp`
- `05-camera-health-mvp`

## Current conclusion

The active runtime should now be considered:

- `innova-ai-investigation-runtime`

and the live streaming runtime should be considered:

- `innova-live-view-runtime`

The old `03`, `04`, and `05` folders are **legacy references**, not the desired
runtime base.

## Backend ownership reminder

Even after independence cleanup, this runtime still belongs to a larger system.

The Java backend remains responsible for:

- NVR records
- camera metadata
- credentials
- user/session context

This Python runtime should be treated as the execution layer for:

- bridge operations
- evidence extraction
- AI workflows

## What was normalized

The investigation runtime now defaults to its own local resources:

- NVR profiles:
  - `resources/nvr_profiles.local.json`
- SSH key placeholder:
  - `resources/keys/elastic-beanstalk.pem`
- YOLO model:
  - `yolo11n.pt`

Linux server defaults were also normalized:

- bridge host default: `127.0.0.1`
- Hikvision SDK dir default: `/opt/innova/hikvision/current/lib`
- Dahua SDK dir default: `/opt/innova/dahua`

## What still exists from legacy history

These folders may still be useful as:

- historical reference
- prototype code
- experiments
- notes

But they should no longer be required for normal deployment of the modern
runtime if:

1. the `.env` is correct
2. SDKs are extracted under `/opt/innova`
3. the YOLO model is present inside the modern runtime
4. the key file is placed inside `resources/keys`

## Safe posture

Recommended next step:

- keep `03`, `04`, and `05` as archival reference until production validation is
  complete for:
  - Hikvision discovery
  - Dahua discovery
  - snapshots
  - evidence extraction
  - AI jobs

After that, those old folders can be archived outside the active repo or moved
to a `legacy/` container.
