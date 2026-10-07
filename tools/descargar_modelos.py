"""Descarga y verifica los modelos de rostros (YuNet + SFace) en models/.

Uso (desde la carpeta del proyecto, con el venv activo):
    python -m tools.descargar_modelos

- Si un modelo ya existe y es válido, no lo vuelve a descargar.
- Al final carga ambos modelos con OpenCV y hace una inferencia de prueba.
- Código de salida 0 = todo correcto, 1 = algo falló (el mensaje explica qué).
"""
import os
import sys
import urllib.error
import urllib.request

import cv2
import numpy as np

import config

BASE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models"
MODELOS = [
    {
        "nombre": "YuNet (detector de rostros)",
        "ruta": config.YUNET_PATH,
        "url": f"{BASE_URL}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "min_bytes": 100_000,  # pesa unos cientos de KB
    },
    {
        "nombre": "SFace (reconocimiento facial)",
        "ruta": config.SFACE_PATH,
        "url": f"{BASE_URL}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "min_bytes": 10_000_000,  # pesa decenas de MB
    },
]


def _tamano_ok(modelo):
    ruta = modelo["ruta"]
    return ruta.is_file() and ruta.stat().st_size >= modelo["min_bytes"]


def _descargar(modelo):
    ruta = modelo["ruta"]
    ruta.parent.mkdir(parents=True, exist_ok=True)
    parcial = ruta.with_suffix(ruta.suffix + ".part")
    print(f"  Descargando {modelo['nombre']} ...")
    req = urllib.request.Request(modelo["url"], headers={"User-Agent": "detectepp/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(parcial, "wb") as f:
        total = int(resp.headers.get("Content-Length") or 0)
        leido = 0
        while True:
            bloque = resp.read(1024 * 256)
            if not bloque:
                break
            f.write(bloque)
            leido += len(bloque)
            if total:
                print(f"\r  {leido / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    os.replace(parcial, ruta)
    if not _tamano_ok(modelo):
        raise RuntimeError(
            f"{ruta.name} se descargó pero su tamaño ({ruta.stat().st_size} bytes) es "
            "demasiado pequeño: probablemente no es el modelo real."
        )


def verificar():
    """Carga los modelos y corre una inferencia de prueba. Lanza excepción si falla."""
    detector = cv2.FaceDetectorYN.create(str(config.YUNET_PATH), "", (320, 320))
    detector.setInputSize((320, 320))
    detector.detect(np.zeros((320, 320, 3), np.uint8))  # imagen negra: sin rostros

    reconocedor = cv2.FaceRecognizerSF.create(str(config.SFACE_PATH), "")
    vector = reconocedor.feature(np.zeros((112, 112, 3), np.uint8))
    if vector.shape != (1, 128):
        raise RuntimeError(f"El vector facial tiene forma inesperada: {vector.shape}")


def main():
    print(f"OpenCV {cv2.__version__}")
    if int(cv2.__version__.split(".")[0]) >= 5:
        print("AVISO: tienes OpenCV 5.x; el proyecto necesita 4.x (pip install -r requirements.txt).")

    print(f"Carpeta de modelos: {config.MODELS_DIR}")
    for modelo in MODELOS:
        if _tamano_ok(modelo):
            print(f"  OK  {modelo['ruta'].name} ya existe.")
            continue
        try:
            _descargar(modelo)
            print(f"  OK  {modelo['ruta'].name} descargado.")
        except (urllib.error.URLError, OSError, RuntimeError) as e:
            print(f"\n  ERROR descargando {modelo['nombre']}: {e}")
            print(f"  Descárgalo a mano desde:\n    {modelo['url']}\n  y guárdalo como:\n    {modelo['ruta']}")
            return 1

    try:
        verificar()
    except Exception as e:  # cv2.error, RuntimeError...
        print(f"\nERROR: los modelos no cargan correctamente: {e}")
        print("Borra la carpeta models/ (solo los .onnx) y ejecuta este comando otra vez.")
        return 1

    print("\nTodo correcto: YuNet y SFace cargan y producen resultados.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
