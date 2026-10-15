"""Descarga YOLO11s-pose y lo exporta a ONNX (640 y 960) en la carpeta models/.

Hace falta una sola vez (los .onnx pesan ~40 MB y no se pueden copiar desde el chat).
Usa ultralytics + torch, que ya están en el venv; la primera vez instala solo ``onnx`` si falta.

Uso (desde la carpeta del proyecto, con internet):
    venv\\Scripts\\python -m tools.exportar_pose
"""
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config


def main() -> int:
    from ultralytics import YOLO
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    pt = config.MODELS_DIR / "yolo11s-pose.pt"
    modelo = YOLO(str(pt) if pt.exists() else "yolo11s-pose.pt")   # se descarga de GitHub si no está
    for lado in (640, 960):
        destino = config.MODELS_DIR / f"yolo11s-pose-{lado}.onnx"
        if destino.exists():
            print(f"Ya existe {destino.name}")
            continue
        salida = Path(modelo.export(format="onnx", imgsz=lado, opset=12, simplify=True))
        shutil.move(str(salida), str(destino))
        print(f"Listo: {destino}")
    descargado = Path("yolo11s-pose.pt")
    if descargado.exists() and not pt.exists():
        shutil.move(str(descargado), str(pt))
    return 0


if __name__ == "__main__":
    sys.exit(main())
