"""Prepara las muestras de un rostro nuevo para guardarlo como empleado.

Las muestras ya fueron capturadas mientras la persona estaba frente a la cámara
(ver ``core.intrusos.DetectorIntrusos``); aquí solo se comprueba que sirvan:

- todas deben parecerse a la primera (misma persona; si no, se descartan las que no);
- si el rostro ya pertenece a un empleado registrado, se avisa y NO se registra.

No importa PyQt.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

import config
from vision.face_engine import Galeria


@dataclass
class MuestrasListas:
    vectores: List[np.ndarray] = field(default_factory=list)
    recortes: List[np.ndarray] = field(default_factory=list)
    duplicado: Optional[Tuple[int, float]] = None    # (id_empleado, similitud) si ya estaba registrado
    error: str = ""


def preparar(muestras: Sequence[Tuple[np.ndarray, np.ndarray]], galeria: Galeria) -> MuestrasListas:
    """``muestras``: lista de (vector, recorte). Devuelve las coherentes o el motivo del rechazo."""
    salida = MuestrasListas()
    if not muestras:
        salida.error = "No hay muestras del rostro."
        return salida
    ref = np.asarray(muestras[0][0], dtype=np.float32).reshape(-1)
    for vector, recorte in muestras:
        v = np.asarray(vector, dtype=np.float32).reshape(-1)
        if float(v @ ref) >= config.REGISTRO_COHERENCIA:
            salida.vectores.append(v)
            salida.recortes.append(recorte)
    if len(salida.vectores) < config.REGISTRO_MUESTRAS_MIN:
        salida.error = "Las capturas no corresponden a una sola persona. Espera a una nueva alerta."
        return salida
    for v in salida.vectores:     # ninguna muestra debe coincidir con un empleado ya registrado
        emp_id, similitud = galeria.buscar(v)
        if emp_id is not None:
            salida.duplicado = (emp_id, similitud)
            break
    return salida
