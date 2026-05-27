# Contrato API

## Health

- `GET /api/health`

## Investigación inicial

- `POST /api/investigation/first-appearance`
- `POST /api/investigation/static-object-discovery`

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
