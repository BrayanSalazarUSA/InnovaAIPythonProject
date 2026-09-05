# Despliegue Linux

## Arquitectura recomendada

Mantener este runtime como un servicio separado del bridge de Live View:

- `innova-liveview-streaming.service`: HLS / WebRTC / ffmpeg para mosaico y Live View.
- `innova-ai-investigation.service`: discovery de NVR, snapshots, investigación y bridge SDK.

Ambos pueden vivir en el mismo Ubuntu, pero no conviene fusionarlos en un solo proceso.

## 1. Copiar proyecto al servidor

Ubicación sugerida y consistente:

```bash
/opt/innova/investigation
```

## 2. Crear entorno

```bash
cd /opt/innova/investigation
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

## 3. Preparar SDKs

El proyecto incluye los archivos en `vendor/sdk_archives/`.

Ejecuta:

```bash
cd /opt/innova/investigation
bash scripts/setup_linux_server.sh
```

Esto extrae:

- Hikvision en `/opt/innova/hikvision`
- Dahua en `/opt/innova/dahua`

Ademas, el script deja listo un alias estable para Hikvision:

- `/opt/innova/hikvision/current`

## 4. Variables de entorno

```bash
cp .env.example .env
```

Ajusta:

- `INNOVA_BACKEND_API_URL`
- `INNOVA_SSH_KEY_PATH`
- `INNOVA_REMOTE_BRIDGE_HOST`
- `INNOVA_REMOTE_BRIDGE_USER`
- `INNOVA_HIKVISION_REMOTE_SDK_DIR`
- `INNOVA_DAHUA_REMOTE_SDK_DIR`
- `INNOVA_YOLO_MODEL_PATH`

Valores Linux recomendados:

- `INNOVA_SSH_KEY_PATH=/opt/innova/investigation/resources/keys/elastic-beanstalk.pem`
- `INNOVA_REMOTE_BRIDGE_HOST=127.0.0.1`
- `INNOVA_REMOTE_BRIDGE_USER=ubuntu`
- `INNOVA_HIKVISION_REMOTE_SDK_DIR=/opt/innova/hikvision/current/lib`
- `INNOVA_HIKVISION_LOCAL_SDK_DIR=/opt/innova/hikvision/current/lib`
- `INNOVA_HIKVISION_DOWNLOAD_MODE=local`
- `INNOVA_DAHUA_REMOTE_SDK_DIR=/opt/innova/dahua`
- `INNOVA_YOLO_MODEL_PATH=/opt/innova/investigation/yolo11n.pt`

## 5. Paquetes del sistema recomendados

Instala tambien:

- `python3`
- `python3-venv`
- `ffmpeg`
- `unzip`
- `openjdk-17-jre`
- `openjdk-17-jdk`

`ffmpeg` es importante para extraccion y reprocesamiento de evidencia.
Java/Javac es importante para algunos caminos operativos del SDK Dahua.

## 6. Arrancar manualmente

```bash
source .venv/bin/activate
export $(grep -v '^#' .env | xargs)
python -m innova_investigation
```

## 7. Probar el runner Hikvision/HCNetSDK

La adaptacion Python del HCNetSDK tambien se puede ejecutar directo en Linux
para validar que el SDK, las librerias y las credenciales estan bien antes de
usar el API:

```bash
source .venv/bin/activate
export HIK_SDK_ROOT=/opt/innova/hikvision/current/lib
export HIK_EVIDENCE_ROOT=/tmp/innova_hikvision
export HIK_HOST=<nvr-host>
export HIK_SDK_PORT=8000
export HIK_USER=<usuario>
export HIK_PASSWORD=<password>
export HIK_CHANNEL=1
export HIK_START_ISO="2026-09-03 08:05:00"
export HIK_END_ISO="2026-09-03 08:05:20"
export HIK_OUTPUT_NAME="test_hikvision.mp4"
innova-hikvision-sdk-download
```

La salida correcta es JSON con `ok: true`, `remote_path`, `sdk_channel` y
`size_bytes`.

## 8. Systemd

Copia `deployment/systemd/innova-ai-investigation.service` a `/etc/systemd/system/`.

Luego:

```bash
sudo systemctl daemon-reload
sudo systemctl enable innova-ai-investigation
sudo systemctl start innova-ai-investigation
sudo systemctl status innova-ai-investigation
```
