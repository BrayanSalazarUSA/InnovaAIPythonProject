# Modos de Operacion

Este runtime hoy tiene dos flujos operativos principales. No siempre aparecen
nombrados asi en el codigo, pero funcionalmente asi es como deben entenderse.

## Modo 1: Busqueda de objeto en grabaciones

Este es el flujo investigativo clasico.

### Entrada

- una imagen de referencia
- un NVR/camara
- un rango de tiempo

### API principal

- `POST /api/investigation/first-appearance`
- `POST /api/investigation/static-object-discovery`

### Que hace

1. Resuelve el perfil del NVR desde el backend Java o desde recursos locales.
2. Usa el bridge Hikvision o Dahua para descargar un clip historico.
3. Normaliza el video si hace falta.
4. Hace barrido inicial, refinamiento y, si aplica, analisis profundo.
5. Devuelve artifacts, hallazgos y progreso para la UI React.

### Casos de uso

- "Buscame esta persona/objeto en esta grabacion"
- "Encuentra la primera aparicion"
- "Encuentra un objeto abandonado"

### Modulos mas importantes

- `src/innova_investigation/api_server.py`
- `src/innova_investigation/investigation_api_engine.py`
- `src/innova_investigation/bridges/hikvision.py`
- `src/innova_investigation/bridges/dahua.py`

## Modo 2: Event monitor / busqueda por texto MVP

Este es el flujo de vigilancia continua orientado a reglas simples.

### Entrada

- una camara o fuente RTSP
- reglas tipo:
  - personas
  - vehiculos
  - vehiculos por color
  - camiseta roja

### API principal

- `GET /api/event-monitor/health`
- `GET /api/event-monitor/jobs`
- `POST /api/event-monitor/jobs`
- `GET /api/event-monitor/jobs/{job_id}`
- `POST /api/event-monitor/jobs/{job_id}/stop`
- `GET /api/event-monitor/events`
- `GET /api/event-monitor/objects`
- `GET /api/event-monitor/artifact`

### Que hace

1. Recibe una configuracion de monitor.
2. Si el `source` viene con placeholder `USER:PASSWORD`, intenta resolver el
   RTSP real desde el backend Java usando el perfil del NVR.
3. Abre el stream RTSP y va muestreando frames.
4. Ejecuta inferencia y reglas.
5. Guarda evidencias, eventos y objetos en disco.
6. La UI consulta despues esos resultados por camara o de forma global.

### Casos de uso

- "Deja monitoreando esta camara para vehiculos"
- "Traeme eventos de vehiculos blancos de esta camara"
- "Si no selecciono camara, traeme todos los eventos guardados"

### Modulos mas importantes

- `src/innova_investigation/event_monitor/api.py`
- `src/innova_investigation/event_monitor/models.py`
- `src/innova_investigation/event_monitor/storage.py`

## Diferencia corta entre ambos

- `Modo 1`
  - trabaja sobre grabaciones historicas
  - parte de una imagen de referencia
  - descarga clips desde el NVR

- `Modo 2`
  - trabaja como monitoreo continuo
  - parte de reglas simples
  - construye un repositorio de eventos e imagenes

## Modo 3: Impact / Interaction Search

Este es el flujo para investigar un objeto fijo seleccionado por el operador,
por ejemplo un vehiculo estacionado junto a un dumpster.

### Entrada

- NVR/camara
- rango historico
- ROI normalizada marcada por el usuario
- tipo de interaccion, por defecto `possible_vehicle_impact`

### API principal

- `POST /api/investigation/roi-interaction-search`
- `GET /api/investigation/jobs/{job_id}`

### Que hace

1. Descarga microclips historicos por chunks desde el NVR.
2. Analiza solo la ROI expandida para reducir costo.
3. Busca cambios/movimiento y objetos cercanos como vehiculos o personas.
4. Guarda candidatos con frame anotado, clip corto, timestamp y razon.
5. Devuelve eventos para revision humana, no una conclusion legal automatica.

### Casos de uso

- "Quien golpeo el vehiculo estacionado?"
- "Que movimiento hubo alrededor de este carro?"
- "Alguien interactuo con este objeto fijo?"

## Relacion con el backend Java

Los dos modos dependen del backend Java como fuente de verdad para:

- propiedades
- NVRs
- camaras
- credenciales
- metadata operativa

Este runtime Python no reemplaza esa capa. La consume.
