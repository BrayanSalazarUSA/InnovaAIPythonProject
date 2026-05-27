# Deploying Live View on Ubuntu

## Required packages

Install these first:

- Python 3
- `python3-venv`
- `ffmpeg`
- `nginx`
- `curl`

MediaMTX must also be installed manually or copied into `/opt/innova/mediamtx`.

## Suggested filesystem layout

```text
/opt/innova/live-streaming
/opt/innova/mediamtx
/var/www/innova-live-view-runtime
```

## Python service setup

1. Copy this folder to the server as `/opt/innova/live-streaming`
2. Create a virtualenv:

```bash
cd /opt/innova/live-streaming
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

3. Create `.env` from `.env.example`

4. Install systemd unit:

```bash
sudo cp deployment/systemd/innova-liveview-streaming.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable innova-liveview-streaming
sudo systemctl restart innova-liveview-streaming
```

## MediaMTX setup

1. Place MediaMTX binary in:

```text
/opt/innova/mediamtx/mediamtx
```

2. Copy config:

```bash
sudo cp deployment/mediamtx/mediamtx.yml /opt/innova/mediamtx/mediamtx.yml
```

3. Run MediaMTX as a service or supervised process.

Important ports:

- `8554` RTSP publish
- `8889` WebRTC/WHEP
- `9997` MediaMTX API

## Nginx setup

1. Copy the provided site:

```bash
sudo cp deployment/nginx/innova-liveview.conf /etc/nginx/sites-available/innova-liveview
sudo ln -sf /etc/nginx/sites-available/innova-liveview /etc/nginx/sites-enabled/innova-liveview
sudo nginx -t
sudo systemctl reload nginx
```

2. Ensure the HLS output directory exists:

```bash
sudo mkdir -p /var/www/innova-live-view-runtime
sudo chown -R ubuntu:ubuntu /var/www/innova-live-view-runtime
```

## Health checks

Python API:

```bash
curl http://127.0.0.1:8600/health
```

MediaMTX API:

```bash
curl http://127.0.0.1:9997/v3/paths/list
```

Public HLS example:

```bash
curl http://SERVER_IP/live-view-runtime/slot-0/index.m3u8
```

## Operational notes

- `logs/slot-X-ffmpeg.log` stores HLS worker logs
- `logs/slot-X-webrtc.log` stores WebRTC publish logs
- `runtime/registry.json` tracks active slot processes
- `tmp/webrtc-slot-X` is temporary per-slot working space

## Important limitation

This folder reflects the live-view runtime currently running on Ubuntu.
It does not include:

- the Java backend proxy
- frontend mosaic logic
- AI Investigation runtime

Those stay in other projects/services.
