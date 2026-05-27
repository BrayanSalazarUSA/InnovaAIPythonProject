# Dahua SDK Probe

Fecha de validacion: 2026-04-17

## Equipo probado

- Host: `170.55.166.214`
- Vendor: `Dahua`
- SDK port: `37777`
- RTSP port: `8085`
- Usuario: `sanket`
- Channel probado: `12`

## Resultado

- Carga del SDK Java Mac: `OK`
- Login SDK: `OK`
- Canales reportados por el equipo: `64`
- Query de grabaciones en canal 12 para `2026-04-17 00:00:00 -> 2026-04-17 23:59:59`: `OK`
- Cantidad de bloques reportados: `11`

## Hallazgos importantes

1. El SDK oficial Java para Mac funciona en Apple Silicon despues de corregir el loader local del paquete.
2. Este NVR si expone el servicio SDK por `37777`, asi que si es integrable de forma parecida a Hikvision.
3. La siguiente pieza para el proyecto no es RTSP, sino un `dahua_bridge` para:
   - login
   - query de grabaciones
   - download por tiempo
   - luego integrarlo al flujo de IA

## Comando de prueba

```bash
cd <ruta-del-sdk-java-dahua>
java -Djava.library.path=/tmp:libs/mac64 \
  -cp /tmp/dahua-sdk-test/classes:libs/jna.jar:res \
  com.netsdk.demo.custom.DahuaInvestigationSmokeTest \
  170.55.166.214 37777 sanket 'B00kk33p3r?' 12 \
  '2026-04-17 00:00:00' '2026-04-17 23:59:59'
```
