# Detector YOLO - Monitoreo y Visión Artificial

Sistema de detección y monitoreo basado en YOLO/ONNX para EPP, control de accesos y pose.

##  Configuración de Modelos de IA

Los archivos pesados de entrenamiento (`.pt` y `.onnx`) no están incluidos directamente en el repositorio de GitHub para mantener la ligereza del código.

### Instrucciones para descargar los modelos:
1. Descarga la carpeta de modelos completa desde este enlace:
    **[Descargar carpeta models de Google Drive/OneDrive](https://drive.google.com/drive/folders/1VmBUkEJhUsI0W393zMU7dczEuM-B2qhf?usp=sharing)**
2. Descomprime los archivos (si están en ZIP) y colócalos dentro de la carpeta `models/` en la raíz de este proyecto.

Asegúrate de que la estructura de carpetas quede así:
```text
detector_yolo/
├── models/
│   ├── epp_yolo.onnx
│   ├── epp_yolo.pt
│   ├── face_detection_yunet_2023mar.onnx
│   ├── face_recognition_sface_2021dec.onnx
│   ├── yolo11n-pose.onnx
│   └── yolo11s-pose.pt
├── main.py
└── ...