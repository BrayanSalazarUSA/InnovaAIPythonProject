# Innova AI Investigation Runtime

Proyecto Python unificado para investigación CCTV con Hikvision y Dahua. Este runtime consolida el motor visual del proyecto `03`, el API operativo que hoy consume el frontend React desde `04`, y las utilidades de health checks de `05`, todo en una sola carpeta lista para mantener y desplegar.

## Qué hace

- Descarga clips históricos desde NVRs Hikvision y Dahua.
- Ejecuta búsqueda visual por imagen de referencia.
- Encuentra primera aparición u objeto estático/abandonado.
- Lanza análisis profundo de personas asociadas.
- Permite rastreo de persona hacia una siguiente cámara.
- Expone artifacts reproducibles para el frontend React.
- Incluye probes de salud para cámaras/NVRs.

## Contrato actual con React

El frontend actual usa estas rutas:

- `POST /api/investigation/first-appearance`
- `POST /api/investigation/static-object-discovery`
- `GET /api/investigation/jobs/{job_id}`
- `POST /api/investigation/jobs/{job_id}/deep-search`
- `POST /api/investigation/jobs/{job_id}/track-person`
- `GET /api/investigation/artifacts/{job_id}/{artifact_path}`

El objetivo de este proyecto es mantener ese contrato para no romper la UI actual.

## Estructura

- `src/innova_investigation/api_server.py`: API FastAPI principal.
- `src/innova_investigation/investigation_api_engine.py`: lógica del flujo de investigación.
- `src/innova_investigation/video_processor.py`: motor visual y detección de personas.
- `src/innova_investigation/similarity_search.py`: matching visual por imagen.
- `src/innova_investigation/bridges/`: bridges Hikvision y Dahua.
- `src/innova_investigation/tools/`: scripts operativos de health check.
- `resources/`: perfiles, grafos, ejemplos y llaves locales.
- `vendor/sdk_archives/`: SDKs Linux necesarios para preparar el servidor.
- `scripts/`: bootstrap local y despliegue Linux.
- `deployment/systemd/`: servicio sugerido para producción.
- `docs/`: documentación operativa y técnica.

## Nota sobre versionado

Los SDKs pesados de Hikvision y Dahua pueden vivir localmente dentro de
`vendor/sdk_archives/`, pero el repo nuevo esta preparado para no trackear los
`.zip` y `.tar.gz` grandes por defecto. La carpeta y la documentacion se
mantienen; los binarios se copian aparte cuando haga falta.

## Arranque local rápido

```bash
cd innova-ai-investigation-runtime
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
python -m innova_investigation
```

API por defecto:

- `http://127.0.0.1:8512/api/health`

## Arranque fácil

Tienes 3 formas rápidas:

```bash
cd innova-ai-investigation-runtime
make run
```

O doble clic en:

- `Start Innova Investigation.command`

O desde terminal:

```bash
bash scripts/run_api.sh
```

## Documentación clave

- [Arquitectura](docs/ARCHITECTURE.md)
- [Módulos](docs/MODULES.md)
- [Contrato API](docs/API_CONTRACT.md)
- [Integración React](docs/FRONTEND_INTEGRATION.md)
- [Despliegue Linux](docs/DEPLOY_LINUX_SERVER.md)
- [SDKs y recursos](docs/SDK_LAYOUT.md)
