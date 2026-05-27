# Integración con Frontend React

El frontend actual en `FrontendDashboardInnova` ya sabe hablar con este runtime si apuntas la base URL correcta.

## Variables relevantes del frontend

- `VITE_INVESTIGATION_MVP_URL`
- `VITE_INVESTIGATION_MVP_FIRST_APPEARANCE_ENDPOINT`
- `VITE_INVESTIGATION_MVP_STATIC_OBJECT_DISCOVERY_ENDPOINT`
- `VITE_INVESTIGATION_MVP_JOB_ENDPOINT`

## Configuración mínima

Si el runtime corre en:

```text
http://localhost:8512
```

entonces el frontend puede usar:

```env
VITE_INVESTIGATION_MVP_URL=http://localhost:8512/api
```

Con eso seguirá pegándole a:

- `/investigation/first-appearance`
- `/investigation/static-object-discovery`
- `/investigation/jobs/{job_id}`
- `/investigation/jobs/{job_id}/deep-search`
- `/investigation/jobs/{job_id}/track-person`

## Objetivo

Este proyecto se diseñó para reemplazar al `04-investigation-mvp` sin forzar cambios grandes en la UI React.
