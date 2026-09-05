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

El modo 1 y el modo 2 pueden correrse localmente. La diferencia esta en como se
descarga el clip historico desde el NVR.

Para Hikvision, la descarga historica usa nuestra adaptacion Python/ctypes de
HCNetSDK. Esa adaptacion esta local en el runtime, pero el SDK disponible en
este proyecto es `linux64` y carga `libhcnetsdk.so`, asi que necesita ejecutarse
en Linux:

- `INNOVA_REMOTE_BRIDGE_HOST=18.234.252.123`

Eso permite iterar localmente en UI y lógica sin redeployar el runtime a cada
ajuste menor.

Si el bridge Ubuntu esta apagado, hay dos caminos:

- Relevantar el bridge en Ubuntu con `scripts/setup_linux_server.sh`.
- Correr un Linux local con Docker/VM y montar el SDK `linux64`.

Valores utiles para `INNOVA_HIKVISION_DOWNLOAD_MODE`:

- `remote`: usa el bridge SSH Linux. Este es el modo correcto en macOS.
- `local`: ejecuta HCNetSDK directo en el runtime Linux, sin SSH.
- `auto`: usa local primero en Linux y cae al bridge remoto si falla; en macOS usa bridge remoto.

El runner directo del SDK queda disponible como:

```bash
innova-hikvision-sdk-download
```

Ese comando espera las variables `HIK_*` que ya usa el bridge SSH
(`HIK_HOST`, `HIK_SDK_PORT`, `HIK_USER`, `HIK_PASSWORD`, `HIK_CHANNEL`,
`HIK_START_ISO`, `HIK_END_ISO`, `HIK_OUTPUT_NAME`) y opcionalmente
`HIK_SDK_ROOT=/opt/innova/hikvision/current/lib`.

Para Dahua, macOS puede usar el NetSDK Java Mac local si el SDK esta disponible:

```env
INNOVA_DAHUA_DOWNLOAD_MODE=auto
INNOVA_DAHUA_JAVA_SDK_ROOT=/Users/brayansalazaring/Desktop/AIReports/app/General_NetSDK_ChnEng_JAVA_Mac64_IS_V3.060.0000003.0.R.251127
```

Valores utiles para `INNOVA_DAHUA_DOWNLOAD_MODE`:

- `auto`: intenta SDK local Dahua y cae al bridge remoto si falla.
- `local`: usa solo el SDK local Dahua.
- `remote`: usa solo el bridge Linux remoto.

Con esto puedes trabajar en local con Dahua aunque el bridge remoto este apagado,
siempre que el NVR sea alcanzable desde tu Mac y existan grabaciones para el
rango seleccionado.
