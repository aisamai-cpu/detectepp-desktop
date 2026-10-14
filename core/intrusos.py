"""Detección de rostros desconocidos que permanecen frente a la cámara.

Si una persona NO registrada se queda ``INTRUSO_SEGUNDOS`` seguidos frente a la cámara,
se pide una alerta una sola vez (``procesar`` devuelve True). Mientras tanto se van
guardando algunas muestras del rostro (vector + recorte): si el operador decide agregarla
a la base de datos, la persona YA no tiene que seguir frente a la cámara ni moverse.

Hasta que alguien resuelva la alerta (``resolver``) no vuelve a disparar, y después hay un
enfriamiento para no repetirla cada pocos segundos con la misma persona.

No importa PyQt. Recibe los ``Resultado`` de ``vision.identificador``.
"""
import time
from typing import Callable, List, Tuple

import numpy as np

import config


class DetectorIntrusos:
    def __init__(self, reloj: Callable[[], float] = time.monotonic):
        self._reloj = reloj
        self._desde = None          # desde cuándo hay un desconocido seguido
        self._hasta = -1e9          # enfriamiento: no disparar antes de este instante
        self._ultima_muestra = -1e9
        self.pendiente = False      # hay una alerta sin resolver
        self.muestras: List[Tuple[np.ndarray, np.ndarray]] = []   # (vector, recorte) del desconocido actual

    def reiniciar(self):
        """Olvida el conteo en curso (cambio de pestaña o de cámara)."""
        self._desde = None
        if not self.pendiente:
            self.muestras = []

    def resolver(self):
        """La alerta se atendió (agregada o reportada): libera y activa el enfriamiento."""
        self.pendiente = False
        self._desde = None
        self.muestras = []
        self._hasta = self._reloj() + config.INTRUSO_ENFRIAMIENTO_S

    def _guardar_muestra(self, res, ahora):
        if (res.vector is None or res.recorte is None or not res.calidad_ok
                or len(self.muestras) >= config.REGISTRO_MUESTRAS
                or ahora - self._ultima_muestra < config.REGISTRO_SEPARACION_S):
            return
        self.muestras.append((np.asarray(res.vector, dtype=np.float32).reshape(-1).copy(), res.recorte.copy()))
        self._ultima_muestra = ahora

    def procesar(self, res) -> bool:
        """True justo cuando se cumple el tiempo (y hay muestras suficientes para registrar)."""
        ahora = self._reloj()
        if res is None or not res.desconocido:
            self._desde = None
            if not self.pendiente:
                self.muestras = []      # se fue antes de tiempo: no se mezclan personas distintas
            return False
        if ahora < self._hasta and not self.pendiente:
            return False
        self._guardar_muestra(res, ahora)    # también tras la alerta, hasta completar el cupo
        if self.pendiente:
            return False
        if self._desde is None:
            self._desde = ahora
        if (ahora - self._desde >= config.INTRUSO_SEGUNDOS and res.recorte is not None
                and len(self.muestras) >= config.REGISTRO_MUESTRAS_MIN):
            self.pendiente = True
            self._desde = None
            return True
        return False
