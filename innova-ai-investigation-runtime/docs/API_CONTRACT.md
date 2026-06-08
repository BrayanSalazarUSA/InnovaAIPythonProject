# Contrato API

## Health

- `GET /api/health`

## Investigación inicial

- `POST /api/investigation/first-appearance`
- `POST /api/investigation/static-object-discovery`
- `POST /api/investigation/roi-interaction-search`

Campos esperados:

- `propertyId`
- `nvrId`
- `nvrName`
- `host`
- `httpPort`
- `sdkPort`
- `rtspPort`
- `username`
- `password`
- `cameraId`
- `cameraName`
- `logicalChannel`
- `startDt`
- `endDt`
- `queryImage`
- `caseName`
- `investigationMode`

## ROI interaction search

Campos esperados:

- `propertyId`
- `nvrId`
- `vendor`
- `host`
- `sdkPort`
- `username`
- `cameraId`
- `cameraName`
- `logicalChannel`
- `startDt`
- `endDt`
- `caseName`
- `roi` como JSON normalizado `{ "x": 0, "y": 0, "width": 0.2, "height": 0.2 }`
- `interactionType`, default `possible_vehicle_impact`
- `chunkMinutes`, default `10`
- `sampleEverySeconds`, default `1.5`
- `prePostSeconds`, default `20`

Respuesta final:

- `mode: "roi_interaction_search"`
- `events[]` con timestamp, confianza, razon, objetos detectados, scores, `frameUrl` y `clipUrl`
- `summary` con chunks procesados, eventos encontrados, ROI y limitaciones

## Polling de job

- `GET /api/investigation/jobs/{job_id}`

Respuesta:

- `jobId`
- `status`
- `stage`
- `detail`
- `progress`
- `createdAt`
- `updatedAt`
- `result` cuando termina

## Deep search

- `POST /api/investigation/jobs/{job_id}/deep-search`

## Track person

- `POST /api/investigation/jobs/{job_id}/track-person`

## Artifacts

- `GET /api/investigation/artifacts/{job_id}/{artifact_path}`
