# Arquitectura

## Flujo principal

1. React prepara una investigación con propiedad, NVR, cámara, rango e imagen.
2. FastAPI recibe el request y crea un `job_id`.
3. El bridge descarga el clip desde Hikvision o Dahua.
4. El motor visual ejecuta búsqueda rápida.
5. El job publica estado y artifacts parciales.
6. Si el operador confirma, el frontend llama `deep-search`.
7. El motor profundiza análisis de objeto/persona.
8. Si el operador continúa, el frontend llama `track-person`.
9. El sistema descarga clip de la siguiente cámara y repite el análisis.

## Capas

- `api_server.py`: orquestación HTTP, jobs, paths de output y serialización.
- `investigation_api_engine.py`: análisis de clips, recortes, deep search y preparación de evidencia.
- `video_processor.py`: matching visual y asociación de personas.
- `bridges/hikvision.py`: descarga remota vía HCNetSDK en servidor Linux.
- `bridges/dahua.py`: descarga remota vía NetSDK Linux y ayudas HTTP.

## Dependencias externas

- Backend Java para catálogos/perfiles de NVR.
- Servidor Linux remoto con SDKs extraídos para Hikvision/Dahua.
- Modelo YOLO para detección de personas.
- `ffmpeg` para normalización de clips y snapshots RTSP cuando aplique.
