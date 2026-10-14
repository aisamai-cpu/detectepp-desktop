"""Detección de EPP con un modelo YOLO entrenado (ultralytics, CPU).

El modelo se coloca en ``config.EPP_MODELO_PATH``. Se prefiere el ``.onnx`` (corre sin torch,
ver ``vision.yolo_onnx``); si solo existe el ``.pt`` se usa ultralytics + torch con los
hilos limitados a ``config.IA_HILOS_CPU``. Los nombres de clase del modelo (``Hardhat``, ``Safety Vest``,
``NO-Mask``…) se traducen a los nombres que usa la base de datos (``casco``,
``chaleco``, ``mascarilla``…) con ``config.EPP_ALIAS``.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

import config
from vision.yolo_onnx import ModeloYoloOnnx

_PREFIJOS_NEGATIVOS = ("no ", "sin ")


class ModeloEppNoEncontrado(Exception):
    """Falta el archivo del modelo o la librería ultralytics."""


@dataclass(frozen=True)
class DeteccionEpp:
    clase: str                       # nombre canónico (casco, chaleco…) o el del modelo si no tiene alias
    presente: bool                   # False en clases de ausencia del modelo (p. ej. "NO-Hardhat")
    confianza: float
    caja: Tuple[int, int, int, int]  # x, y, ancho, alto en píxeles del frame recibido


def _limpiar(nombre: str) -> str:
    return " ".join(str(nombre).lower().replace("_", " ").replace("-", " ").split())


def normalizar_clase(nombre: str) -> Tuple[str, bool]:
    """``"NO-Hardhat"`` -> ``("casco", False)``; ``"Safety Vest"`` -> ``("chaleco", True)``."""
    limpio = _limpiar(nombre)
    presente = True
    for prefijo in _PREFIJOS_NEGATIVOS:
        if limpio.startswith(prefijo):
            limpio, presente = limpio[len(prefijo):], False
            break
    alias = {_limpiar(k): v for k, v in config.EPP_ALIAS.items()}
    return alias.get(limpio, limpio), presente


class EppEngine:
    def __init__(self, ruta: Optional[Path] = None, modelo=None):
        """``modelo`` (solo pruebas): objeto con ``names`` y ``predict()`` como el YOLO de ultralytics."""
        if modelo is None:
            modelo = _cargar(Path(ruta or config.EPP_MODELO_PATH))
        self.modelo = modelo
        nombres = modelo.names
        self.nombres = dict(enumerate(nombres)) if isinstance(nombres, (list, tuple)) else dict(nombres)
        # Prendas que este modelo sabe reconocer (por su clase positiva o la "NO-xxx").
        self.prendas_detectables = {normalizar_clase(n)[0] for n in self.nombres.values()}

    def detectar(self, frame: np.ndarray) -> List[DeteccionEpp]:
        if isinstance(self.modelo, ModeloYoloOnnx):
            cajas = self.modelo.detectar(frame, config.EPP_CONF_MIN)
            if not cajas:
                return []
            xyxy = np.array([(c.x1, c.y1, c.x2, c.y2) for c in cajas], dtype=float)
            clases = np.array([c.clase for c in cajas], dtype=float)
            confs = np.array([c.confianza for c in cajas], dtype=float)
        else:
            r = self.modelo.predict(frame, imgsz=config.EPP_IMGSZ, conf=config.EPP_CONF_MIN,
                                    device="cpu", verbose=False)[0]
            cajas = r.boxes
            if cajas is None or len(cajas) == 0:
                return []
            xyxy = np.asarray(cajas.xyxy, dtype=float).reshape(-1, 4)
            clases = np.asarray(cajas.cls, dtype=float).reshape(-1)
            confs = np.asarray(cajas.conf, dtype=float).reshape(-1)
        detecciones = []
        for (x1, y1, x2, y2), c, conf in zip(xyxy, clases, confs):
            clase, presente = normalizar_clase(self.nombres.get(int(c), str(int(c))))
            detecciones.append(DeteccionEpp(clase, presente, float(conf),
                                            (int(x1), int(y1), int(x2 - x1), int(y2 - y1))))
        return detecciones


def _cargar(ruta: Path):
    """``.onnx`` si existe (rápido, sin torch); si no, el ``.pt`` con ultralytics."""
    onnx, pt = ruta.with_suffix(".onnx"), ruta.with_suffix(".pt")
    if onnx.exists():
        try:
            return ModeloYoloOnnx(onnx)
        except Exception as e:
            raise ModeloEppNoEncontrado(f"No se pudo abrir {onnx.name}: {e}") from e
    if not pt.exists():
        raise ModeloEppNoEncontrado(
            f"No se encontró el modelo de EPP ({onnx.name} ni {pt.name}) en {ruta.parent}.\n"
            "Copia ahí el modelo YOLO entrenado para activar la detección de EPP.")
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as e:
        raise ModeloEppNoEncontrado("Falta la librería ultralytics: pip install ultralytics") from e
    torch.set_num_threads(config.IA_HILOS_CPU)   # por defecto torch usa TODOS los núcleos y congela la ventana
    return YOLO(str(pt))
