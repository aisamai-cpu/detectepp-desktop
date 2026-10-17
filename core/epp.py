"""Evaluación del EPP de un empleado contra lo que exige su área (sin PyQt).

Una sola imagen puede fallar (mano tapando el casco, mala pose), así que cada
prenda se decide por mayoría en las últimas ``config.EPP_VENTANA`` detecciones.
"""
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

import config
from core.catalogo_epp import EQUIVALENCIAS, NO_VALIDAS


@dataclass(frozen=True)
class EvaluacionEpp:
    requeridos: List[str]        # solo las prendas que el modelo puede detectar (las que se evalúan)
    presentes: List[str]
    faltantes: List[str]
    porcentaje: float
    aprobado: bool
    no_detectables: List[str] = ()   # exigidas por el área pero el modelo aún no las conoce (no cuentan)


def prendas_vistas(detecciones) -> set:
    """Prendas con al menos una detección positiva en la imagen."""
    return {d.clase for d in detecciones if d.presente}


def _opciones(prenda: str) -> set:
    """La prenda y las que también la cumplen (``respirador`` cumple ``mascarilla``)."""
    return {prenda, *EQUIVALENCIAS.get(prenda, ())}


def evaluar(vistas: set, requeridos: Iterable[str], detectables: Optional[set] = None) -> EvaluacionEpp:
    """``detectables``: prendas que el modelo conoce (None = todas). Las demás no cuentan."""
    todas = list(requeridos)
    if detectables is None:
        evaluables, no_detectables = todas, []
    else:
        evaluables = [p for p in todas if _opciones(p) & detectables]
        no_detectables = [p for p in todas if not _opciones(p) & detectables]
    presentes = [p for p in evaluables if _opciones(p) & vistas]
    faltantes = [p for p in evaluables if not _opciones(p) & vistas]
    porcentaje = 100.0 * len(presentes) / len(evaluables) if evaluables else 100.0
    return EvaluacionEpp(evaluables, presentes, faltantes, porcentaje,
                         porcentaje >= config.EPP_UMBRAL_APROBACION, no_detectables)


class SeguimientoEpp:
    """Acumula las últimas detecciones y evalúa por mayoría."""

    def __init__(self):
        self._historial = deque(maxlen=max(1, config.EPP_VENTANA))

    def reiniciar(self):
        self._historial.clear()

    @property
    def cantidad(self) -> int:
        """Análisis acumulados (como máximo ``config.EPP_VENTANA``)."""
        return len(self._historial)

    def agregar(self, detecciones):
        self._historial.append(prendas_vistas(detecciones or []))

    def evaluar(self, requeridos: Iterable[str], detectables: Optional[set] = None) -> EvaluacionEpp:
        n = len(self._historial)
        conteo = {}
        for vistas in self._historial:
            for p in vistas:
                conteo[p] = conteo.get(p, 0) + 1
        estables = {p for p, c in conteo.items() if c * 2 > n}
        return evaluar(estables, requeridos, detectables)


class AlertaEpp:
    """Decide cuándo avisar que un empleado no cumple el protocolo de EPP.

    - Solo con la ventana completa (``config.EPP_VENTANA`` análisis): sin decisiones con 1 imagen.
    - El resultado debe seguir INCOMPLETO ``config.EPP_ALERTA_SEGUNDOS`` seguidos.
    - Una sola alerta por empleado. Se rearma tras ``config.EPP_ALERTA_ENFRIAMIENTO_S``
      o cuando el empleado completa su EPP (si luego se quita algo, vuelve a avisar).
    """

    def __init__(self, reloj: Callable[[], float] = time.monotonic):
        self._reloj = reloj
        self._desde: Optional[float] = None   # desde cuándo está incompleto sin interrupción
        self._emp = None                      # empleado al que corresponde ``_desde``
        self._avisado: Dict[object, float] = {}   # empleado -> instante de su última alerta

    def reiniciar(self):
        """Olvida el conteo en curso (cambio de pestaña o de cámara). Las alertas dadas se recuerdan."""
        self._desde = None
        self._emp = None

    def procesar(self, emp_id, ev: Optional[EvaluacionEpp], analisis: int) -> bool:
        """True justo cuando hay que lanzar la alerta para ``emp_id``."""
        ahora = self._reloj()
        if emp_id is None or ev is None or not ev.requeridos:
            self.reiniciar()
            return False
        if ev.aprobado:
            self.reiniciar()
            self._avisado.pop(emp_id, None)   # completó su EPP: la próxima falta vuelve a avisar
            return False
        if analisis < max(1, config.EPP_VENTANA):
            return False
        if ahora - self._avisado.get(emp_id, -1e18) < config.EPP_ALERTA_ENFRIAMIENTO_S:
            return False
        if self._desde is None or self._emp != emp_id:
            self._desde, self._emp = ahora, emp_id
        if ahora - self._desde >= config.EPP_ALERTA_SEGUNDOS:
            self._avisado[emp_id] = ahora
            self.reiniciar()
            return True
        return False


# ----------------------------------------------------------------------------
# Qué cajas se dibujan en el video según el área
# ----------------------------------------------------------------------------
OK, FALTA, NO_VALIDA, PERSONA = "ok", "falta", "no_valida", "persona"


def clasificar_caja(clase: str, presente: bool, requeridos: Optional[Iterable[str]]):
    """Cómo dibujar una detección en un área. Devuelve (tipo, texto) o None si no se dibuja.

    - Solo se muestran las prendas que exige el área (y las que también las cumplen: respirador por mascarilla).
    - Una prenda que ocupa el lugar de otra exigida sin cumplirla (mascarilla donde se exige respirador)
      se muestra en ámbar como no válida; si falta ("sin mascarilla"), se muestra "sin respirador".
    - ``requeridos`` None (sin empleado identificado): se muestra todo, como antes.
    """
    if clase == "persona":
        return PERSONA, "persona"
    if requeridos is None:
        return (OK, clase) if presente else (FALTA, f"sin {clase}")
    req = list(requeridos)
    for exigida in req:
        if clase == exigida or clase in EQUIVALENCIAS.get(exigida, ()):
            return (OK, clase) if presente else (FALTA, f"sin {exigida}")
    for exigida in req:
        if clase in NO_VALIDAS.get(exigida, ()):
            return (NO_VALIDA, f"{clase} (se exige {exigida})") if presente else (FALTA, f"sin {exigida}")
    return None
