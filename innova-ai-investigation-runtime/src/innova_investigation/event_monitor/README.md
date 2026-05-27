# Event Monitor MVP

Feature aislada para monitorear una camara RTSP o video, detectar eventos con YOLO y guardar evidencia.

Esta carpeta no reemplaza el flujo actual de investigacion. Es un modo nuevo y simple para aprender/probar:

- personas
- vehiculos
- vehiculos blancos
- personas con camiseta roja aproximada
- modelos custom exportados desde Roboflow/Ultralytics

## Estructura

- `models.py`: contratos de configuracion, reglas y metadata.
- `detectors.py`: wrapper de Ultralytics YOLO.
- `color_rules.py`: reglas simples por color.
- `rules.py`: decide si una deteccion cumple una regla.
- `storage.py`: guarda crop, frame y JSON.
- `monitor.py`: loop principal de lectura/deteccion/eventos.
- `tracking.py`: asigna `track_id` con ByteTrack para no duplicar eventos del mismo objeto.
- `api.py`: API independiente para iniciar/listar/detener jobs.
- `cli.py`: comando para correr desde terminal.

## Ejecutar por terminal

Instala el paquete en modo editable si todavia no lo hiciste:

```bash
cd innova-ai-investigation-runtime
source .venv/bin/activate
pip install -e .
```

Pon un video de prueba en:

```text
resources/videos/sample.mp4
```

Corre:

```bash
innova-event-monitor --config resources/event_monitor/event_rules.example.json
```

La evidencia queda en:

```text
output/event_monitor/demo-camera-1/
```

## Ejecutar API

```bash
innova-event-monitor-api
```

Health:

```bash
curl http://127.0.0.1:8522/api/event-monitor/health
```

Iniciar job:

```bash
curl -X POST http://127.0.0.1:8522/api/event-monitor/jobs \
  -H 'Content-Type: application/json' \
  --data @resources/event_monitor/event_rules.example.json
```

## Ejecutar API y crear el job automaticamente

Para operar como servicio, crea una copia local del ejemplo y pon el RTSP real:

```bash
cp resources/event_monitor/autostart.local.example.json resources/event_monitor/front-lpr.local.json
```

Edita:

```json
"source": "rtsp://USER:PASSWORD@HOST:554/Streaming/Channels/101"
```

Arranca el API con autostart:

```bash
innova-event-monitor-api \
  --autostart-config resources/event_monitor/front-lpr.local.json \
  --restart-on-error
```

Con `max_runtime_seconds: null` el monitor queda continuo hasta presionar Detener o cerrar el proceso. Con `--restart-on-error`, si el stream se corta o el job termina, el servicio intenta levantarlo de nuevo cada 5 segundos.

## Tracking y deduplicacion

El monitor puede usar ByteTrack via `supervision`:

```json
"enable_tracking": true,
"save_once_per_track": true
```

Cuando esta activo, cada persona/vehiculo recibe un `track_id`. El monitor guarda solo el primer evento por combinacion `regla + clase + track_id`, y escribe un indice pequeño en:

```text
output/event_monitor/<camera>/tracks/
```

Esto evita 100 fotos del mismo objeto mientras permanece en escena. Si el objeto desaparece bastante tiempo y vuelve, ByteTrack puede asignarle un nuevo `track_id`, lo cual se trata como una nueva aparicion.

Ademas se mantiene un indice consolidado:

```text
output/event_monitor/<camera>/objects.jsonl
```

Cada objeto incluye `object_id`, `first_seen_utc`, `last_seen_utc`, `duration_seconds`, `best_confidence`, `track_ids`, `attributes` y la mejor evidencia disponible. La UI usa este indice para mostrar una tarjeta por objeto en vez de una tarjeta por deteccion.

Tambien puedes usar variables de entorno:

```bash
INNOVA_EVENT_MONITOR_AUTOSTART_CONFIGS=resources/event_monitor/front-lpr.local.json \
INNOVA_EVENT_MONITOR_RESTART_ON_ERROR=true \
innova-event-monitor-api
```

Listar jobs:

```bash
curl http://127.0.0.1:8522/api/event-monitor/jobs
```

Listar eventos:

```bash
curl 'http://127.0.0.1:8522/api/event-monitor/events?outputDir=output/event_monitor/demo-camera-1&limit=20'
```

## Usar RTSP

Cambia `source` en el JSON:

```json
"source": "rtsp://USER:PASSWORD@HOST:554/Streaming/Channels/101"
```

Para Dahua suele variar el path. Ejemplos comunes:

```text
rtsp://USER:PASSWORD@HOST:554/cam/realmonitor?channel=1&subtype=0
rtsp://USER:PASSWORD@HOST:554/cam/realmonitor?channel=1&subtype=1
```

## Usar datasets/modelos de Roboflow

Roboflow no se "pasa como link" al monitor local. El flujo recomendado es:

1. Exportar dataset desde Roboflow en formato YOLO.
2. Entrenar con Ultralytics o descargar un `.pt` si ya lo tienes.
3. Guardar el modelo en:

```text
resources/event_monitor/models/
```

4. Crear una regla `custom_model` apuntando a ese `.pt`.

Ejemplo:

```json
{
  "name": "garbage_custom_model",
  "type": "custom_model",
  "model_path": "resources/event_monitor/models/garbage_best.pt",
  "class_names": ["trash", "garbage", "garbage_bag"],
  "min_confidence": 0.35
}
```

## Limitaciones del MVP

- Las reglas de color son aproximadas; luz y sombras afectan mucho.
- No hay tracking/deduplicacion avanzada todavia; usamos cooldown por regla.
- No hay busqueda por texto natural todavia. Eso vendra con OpenCLIP + Qdrant.
- No hay integracion directa con Java todavia. Por ahora pegas RTSP/credenciales en el JSON.
