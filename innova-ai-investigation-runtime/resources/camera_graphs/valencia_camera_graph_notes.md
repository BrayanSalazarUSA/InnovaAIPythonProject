# Valencia Camera Graph Notes

Archivo base:

- `resources/camera_graphs/valencia_camera_graph.json`

## Que contiene

- inventario inicial de camaras por NVR
- building graph inferido desde el plano
- camera handoff rules para seguimiento multicamara
- backlog de validaciones pendientes

## Lo mas importante

- `B14_1` y `B14_2` se interpretan como camaras del edificio 14
- las camaras del mismo edificio deben revisarse primero
- luego se revisan edificios vecinos segun el grafo
- las conexiones estan marcadas con `high`, `medium` o `low`

## NVRs ya mapeados

- `Valencia NVR2` -> puerto HTTP `82`, SDK `8002`
- `Valencia NVR3` -> puerto HTTP `83`, SDK `8003`
- `Valencia NVR1` -> perfil guardado, pero aun pendiente de validar por ISAPI

## Siguiente uso practico

Este JSON ya sirve como semilla para:

1. sugerir la siguiente camara a revisar
2. limitar la busqueda multicamara a vecinas probables
3. reconstruir una ruta probable de la persona
