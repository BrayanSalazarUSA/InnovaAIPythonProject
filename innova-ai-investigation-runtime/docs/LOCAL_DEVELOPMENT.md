# Desarrollo Local y Perfiles de Entorno

Este runtime ya puede trabajarse de dos formas sin cambiar código:

1. `macOS local`
   - FastAPI corre en tu Mac
   - el frontend local apunta a `http://127.0.0.1:8512/api`
   - el bridge/SDK puede seguir usando el Ubuntu remoto

2. `Ubuntu server`
   - FastAPI corre en `18.234.252.123`
   - el runtime usa SDKs locales del servidor
   - `nginx` publica `/investigation-api`

## Archivos de perfil

- `.env.local-macos.example`
- `.env.ubuntu.example`

Puedes duplicarlos a:

- `.env.local-macos`
- `.env.ubuntu`

## Arranque rápido por perfil

```bash
cd innova-ai-investigation-runtime
bash scripts/run_api_with_profile.sh macos
```

O para Ubuntu:

```bash
cd innova-ai-investigation-runtime
bash scripts/run_api_with_profile.sh ubuntu
```

En `auto` el script elige:

- `Darwin` -> `.env.local-macos`
- `Linux` -> `.env.ubuntu`

## Frontend local

Si quieres probar el wizard React en tu Mac contra el runtime local:

```env
VITE_INVESTIGATION_MVP_URL=http://127.0.0.1:8512/api
VITE_EVENT_MONITOR_URL=http://127.0.0.1:8512/api
```

Si quieres que el frontend local pegue al Ubuntu:

```env
VITE_INVESTIGATION_MVP_URL=http://18.234.252.123/investigation-api
VITE_EVENT_MONITOR_URL=http://18.234.252.123/investigation-api
```

## Nota importante

El modo 1 y el modo 2 pueden correrse localmente, pero cuando el clip histórico
se baja vía Hikvision/Dahua el bridge puede seguir ejecutándose en Ubuntu si el
perfil local define:

- `INNOVA_REMOTE_BRIDGE_HOST=18.234.252.123`

Eso permite iterar localmente en UI y lógica sin redeployar el runtime a cada
ajuste menor.
