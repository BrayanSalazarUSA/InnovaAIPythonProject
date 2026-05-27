# Módulos

## `api_server.py`

- Expone el API consumido por React.
- Gestiona `job_id`, progreso, artifacts y polling.
- Normaliza perfiles NVR desde backend y recursos locales.

## `investigation_api_engine.py`

- Ejecuta la búsqueda rápida.
- Refina ventanas de interés.
- Prepara análisis profundo y evidencia de tracking.

## `video_processor.py`

- Usa `SimilaritySearcher` para matching por imagen.
- Gestiona scoring, zonas, personas asociadas y exportables.

## `similarity_search.py`

- Construye firma visual de la imagen de referencia.
- Propone candidatos por template matching, histogramas y features.

## `bridges/hikvision.py`

- Consulta canales/snapshots vía ISAPI.
- Descarga video histórico vía HCNetSDK desde bridge Linux remoto.

## `bridges/dahua.py`

- Consulta snapshots y metadatos vía HTTP.
- Soporta bridge Linux remoto con NetSDK.
- Conserva helpers Java para escenarios locales heredados.

## `tools/*.py`

- Probes livianos de salud para NVR/cámaras.
- Útiles para operación y validación antes de correr investigaciones.
