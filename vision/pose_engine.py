"""Detección de personas y su esqueleto (17 puntos COCO) con YOLO-pose en ONNX, sin torch.

Modelo en ``config.POSE_MODELO_PATH`` (ver opciones en config.py: el ``s`` detecta varias personas y
también lejos; el ``n`` es más rápido pero pierde personas lejanas).
"""
from pathlib import Path
from typing import List, Optional

import numpy as np

import config
from core.riesgos import PersonaPose
from vision.yolo_onnx import ModeloYoloOnnx


class ModeloPoseNoEncontrado(Exception):
    """Falta el archivo del modelo de pose o no es un modelo de pose."""


# Si falta el modelo configurado, se usa el primero de estos que exista (con un aviso en pantalla).
ALTERNATIVAS = ("yolo11s-pose-640.onnx", "yolo11n-pose.onnx")


class PoseEngine:
    def __init__(self, ruta: Optional[Path] = None, modelo=None):
        """``modelo`` (solo pruebas): objeto con ``detectar(frame, conf)`` como ``ModeloYoloOnnx``."""
        self.aviso: Optional[str] = None     # texto para la pantalla si se usa un modelo de reemplazo
        if modelo is None:
            ruta = Path(ruta or config.POSE_MODELO_PATH)
            if not ruta.exists():
                for nombre in ALTERNATIVAS:
                    alternativa = config.MODELS_DIR / nombre
                    if alternativa.exists():
                        self.aviso = (f"No está {ruta.name}; se usa {nombre}"
                                      + (" (detecta peor a las personas lejanas). Para el modelo preciso ejecuta: "
                                         "venv\\Scripts\\python -m tools.exportar_pose" if "11n" in nombre else "."))
                        ruta = alternativa
                        break
            if not ruta.exists():
                raise ModeloPoseNoEncontrado(
                    f"No se encontró el modelo de pose en {ruta}.\n"
                    "Copia ahí el modelo YOLO-pose en ONNX para activar el Monitoreo.")
            try:
                modelo = ModeloYoloOnnx(ruta)
            except Exception as e:
                raise ModeloPoseNoEncontrado(f"No se pudo abrir {ruta.name}: {e}") from e
            if modelo.tarea != "pose" or tuple(modelo.kpt_shape or ()) != (17, 3):
                raise ModeloPoseNoEncontrado(
                    f"{ruta.name} no es un modelo de pose de 17 puntos (tarea: {modelo.tarea}).")
        self.modelo = modelo

    @property
    def lado(self) -> int:
        return int(getattr(self.modelo, "lado", 640))

    def detectar(self, frame: np.ndarray) -> List[PersonaPose]:
        personas = []
        for c in self.modelo.detectar(frame, config.MON_CONF_PERSONA):
            if c.puntos is None:
                continue
            personas.append(PersonaPose((c.x1, c.y1, c.x2, c.y2), c.confianza, np.asarray(c.puntos, np.float32)))
        return personas
