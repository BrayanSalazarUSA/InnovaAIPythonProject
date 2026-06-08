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

Y si también quieres que el modo 2 apunte al mismo runtime:

```env
VITE_EVENT_MONITOR_URL=http://localhost:8512/api
```

## Dos formas de trabajo recomendadas

### 1. Frontend local + runtime local

```env
VITE_INVESTIGATION_MVP_URL=http://127.0.0.1:8512/api
VITE_EVENT_MONITOR_URL=http://127.0.0.1:8512/api
```

Luego inicia el runtime con:

```bash
bash scripts/run_api_with_profile.sh macos
```

### 2. Frontend local + runtime Ubuntu real

```env
VITE_INVESTIGATION_MVP_URL=http://18.234.252.123/investigation-api
VITE_EVENT_MONITOR_URL=http://18.234.252.123/investigation-api
```

Esto es útil cuando quieres iterar la UI en `localhost`, pero seguir usando el
bridge, SDKs y runtime reales del servidor Ubuntu.

Con eso seguirá pegándole a:

- `/investigation/first-appearance`
- `/investigation/static-object-discovery`
- `/investigation/jobs/{job_id}`
- `/investigation/jobs/{job_id}/deep-search`
- `/investigation/jobs/{job_id}/track-person`

## Objetivo

Este proyecto se diseñó para reemplazar al `04-investigation-mvp` sin forzar cambios grandes en la UI React.
