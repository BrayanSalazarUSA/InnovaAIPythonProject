# SDKs y recursos

## SDKs incluidos en el proyecto

En `vendor/sdk_archives/` quedaron copiados:

- `EN-HCNetSDKV6.1.9.4_build20220412_linux64.zip`
- `General_NetSDK_Eng_Linux64_IS_V3.060.0000003.0.R.251127.tar.gz`

Estos son los paquetes base para preparar el bridge Linux remoto.

La adaptacion Python/ctypes de Hikvision vive en:

- `src/innova_investigation/bridges/hikvision.py`
- `src/innova_investigation/tools/hikvision_sdk_download.py`

El primer archivo mantiene el bridge SSH que ejecuta HCNetSDK en Linux. El
segundo expone el mismo runner como comando (`innova-hikvision-sdk-download`)
para pruebas directas en Ubuntu, Docker o VM Linux.

## Recursos operativos

- `resources/nvr_profiles.local.json`: perfiles locales de NVR.
- `resources/dahua_devices.local.json`: inventario local Dahua.
- `resources/camera_graphs/`: grafos/notas de flujo entre cámaras.
- `resources/examples/`: imágenes de ejemplo para pruebas.
- `resources/keys/`: coloca aquí llaves locales si decides operar desde esta carpeta.

## Nota sobre seguridad

No se incluyen llaves privadas reales en el repo/carpeta. Deben copiarse manualmente y mantenerse fuera de control de versiones.
