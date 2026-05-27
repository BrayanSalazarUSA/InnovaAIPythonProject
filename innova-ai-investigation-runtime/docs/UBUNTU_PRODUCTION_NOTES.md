# Notas de Produccion Ubuntu

Este documento resume los ajustes reales que hubo que hacer en el servidor
Ubuntu nuevo para que `AI Investigation` y `event_monitor` funcionaran en una
prueba de produccion, no solo en local.

Servidor trabajado:

- IP publica: `18.234.252.123`
- instancia: `i-0b17906b776d15e85`
- rol practico: runtime Python de AI + bridge + Live View

## Servicios relevantes

- `innova-ai-investigation.service`
- `innova-liveview-streaming.service`
- `mediamtx.service`
- `nginx.service`

## Configuracion del runtime AI

Archivo de entorno operativo:

- `/opt/innova/investigation/.env`

Valores clave que quedaron alineados:

- `INNOVA_API_HOST=0.0.0.0`
- `INNOVA_API_PORT=8512`
- `INNOVA_BACKEND_API_URL=https://innova-dashboard.com/api`
- `INNOVA_YOLO_MODEL_PATH=/opt/innova/investigation/yolo11n.pt`
- `INNOVA_SSH_KEY_PATH=/opt/innova/investigation/resources/keys/elastic-beanstalk.pem`
- `INNOVA_REMOTE_BRIDGE_HOST=127.0.0.1`
- `INNOVA_REMOTE_BRIDGE_USER=ubuntu`
- `INNOVA_REMOTE_BRIDGE_PYTHON=python3`
- `INNOVA_HIKVISION_REMOTE_SDK_DIR=/opt/innova/hikvision/current/lib`
- `INNOVA_DAHUA_REMOTE_SDK_DIR=/opt/innova/dahua`
- `INNOVA_NVR_PROFILES_PATH=/opt/innova/investigation/resources/nvr_profiles.local.json`

## Correcciones que hubo que hacer

### 1. Event monitor integrado en la API principal

El runtime ya no debe tratar el `event_monitor` como una pieza escondida o
separada mentalmente. Se monto como router dentro del runtime principal para que
la API exponga tambien:

- `/api/event-monitor/health`
- `/api/event-monitor/jobs`
- `/api/event-monitor/events`
- `/api/event-monitor/objects`

### 2. CORS duplicado en `/investigation-api`

Problema:

- FastAPI devolvia `Access-Control-Allow-Origin`
- `nginx` agregaba otro `*`
- el navegador bloqueaba aunque el backend respondiera `200`

Correccion:

- en `/etc/nginx/sites-available/innova-liveview`
- quitar headers CORS manuales dentro de `location /investigation-api/`

Importante:

- esto no debia tocar rutas de `Live View`

### 3. Uploads grandes para modo 1

Problema:

- `first-appearance` fallaba con `413 Request Entity Too Large`

Correccion:

- en `location /investigation-api/` agregar:

```nginx
client_max_body_size 25m;
```

### 4. Host key verification failed

Problema:

- el bridge Hikvision intentaba hacer SSH local a `127.0.0.1`
- faltaba `known_hosts`

Correccion:

- crear `/home/ubuntu/.ssh/known_hosts`
- registrar huellas para:
  - `127.0.0.1`
  - `localhost`
  - IP privada del servidor

### 5. Event monitor resolviendo RTSP real

Problema:

- la UI podia mandar una fuente tipo:
  - `rtsp://USER:PASSWORD@...`
- el monitor fallaba porque intentaba abrir esa cadena literal

Correccion:

- `event_monitor/api.py` ahora resuelve el perfil real del NVR via:
  - `GET /api/nvrs/{id}/investigation-profile`
- con eso construye el RTSP real de Hikvision, Dahua o Uniview

### 6. Despliegue incompleto del modo 2

Problema:

- se habia actualizado `event_monitor/api.py`
- pero el servidor seguia con `event_monitor/models.py` viejo
- por eso `MonitorConfig` no tenia campos como `vendor`, `host`,
  `rtsp_port`, `logical_channel`

Correccion:

- desplegar ambos archivos juntos:
  - `src/innova_investigation/event_monitor/api.py`
  - `src/innova_investigation/event_monitor/models.py`

## Validaciones utiles

Health principal:

```bash
curl http://127.0.0.1:8512/api/health
```

Health modo 2:

```bash
curl http://127.0.0.1:8512/api/event-monitor/health
```

Jobs del modo 2:

```bash
curl http://127.0.0.1:8512/api/event-monitor/jobs
```

## Observacion operativa

El servidor hoy tambien corre `Live View`, por lo tanto el monitoreo continuo
del modo 2 comparte CPU y RAM con muchos procesos `ffmpeg`. Antes de dejar
muchas camaras activas permanentemente, conviene medir carga real y usar
`sampleEverySeconds` mas conservador.
