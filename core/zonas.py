"""Zonas restringidas por cámara (rectángulos en coordenadas 0..1 de la imagen ya girada). Sin PyQt.

Archivo: ``config.ZONAS_PATH`` -> ``{"<id de cámara>": [{"nombre": "Zona 1", "x1": .., "y1": .., "x2": .., "y2": ..}]}``
"""
import json
import logging
from typing import List, Tuple

import config

logger = logging.getLogger(__name__)

Zona = Tuple[str, float, float, float, float]


def _leer() -> dict:
    try:
        return json.loads(config.ZONAS_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError):
        logger.exception("No se pudo leer %s; se ignoran las zonas", config.ZONAS_PATH)
        return {}


def _escribir(datos: dict):
    config.ZONAS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = config.ZONAS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(config.ZONAS_PATH)


def de_camara(camara_id: str) -> List[Zona]:
    return [(z["nombre"], z["x1"], z["y1"], z["x2"], z["y2"]) for z in _leer().get(camara_id, [])]


def agregar(camara_id: str, x1: float, y1: float, x2: float, y2: float) -> Zona:
    """Agrega un rectángulo (se ordenan las esquinas y se recorta a 0..1). Devuelve la zona creada."""
    x1, x2 = sorted((min(max(x1, 0.0), 1.0), min(max(x2, 0.0), 1.0)))
    y1, y2 = sorted((min(max(y1, 0.0), 1.0), min(max(y2, 0.0), 1.0)))
    datos = _leer()
    lista = datos.setdefault(camara_id, [])
    usados = {z["nombre"] for z in lista}
    n = 1
    while f"Zona {n}" in usados:
        n += 1
    zona = {"nombre": f"Zona {n}", "x1": x1, "y1": y1, "x2": x2, "y2": y2}
    lista.append(zona)
    _escribir(datos)
    return (zona["nombre"], x1, y1, x2, y2)


def borrar(camara_id: str) -> int:
    datos = _leer()
    n = len(datos.pop(camara_id, []))
    if n:
        _escribir(datos)
    return n
