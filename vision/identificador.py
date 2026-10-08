"""Identificación estable de la persona principal frente a una cámara.

Combina ``FaceEngine`` + ``Galeria`` y añade lo que la interfaz necesita para no
parpadear: una identidad solo se da por buena tras ``FACE_CONFIRMACIONES``
análisis seguidos iguales, y se olvida tras ``FACE_PERDIDA_INFERENCIAS`` análisis
sin rostro. Solo se atiende a UNA persona por cámara (el rostro más grande).

No importa PyQt.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

import config
from vision.face_engine import FaceEngine, Galeria, Rostro


@dataclass(eq=False)
class Resultado:
    rostro: Optional[Rostro] = None       # coordenadas sobre el frame analizado
    calidad_ok: bool = False
    aviso: str = ""                       # texto para el usuario ("Acércate más", ...)
    emp_id: Optional[int] = None          # empleado CONFIRMADO (None = nadie / desconocido)
    similitud: float = 0.0
    desconocido: bool = False             # rostro bueno confirmado que no está en la galería
    vector: Optional[np.ndarray] = None   # último vector válido (para registrar)
    recorte: Optional[np.ndarray] = None  # recorte del rostro (para la foto del empleado)

    @property
    def hay_rostro(self) -> bool:
        return self.rostro is not None


class Identificador:
    def __init__(self, motor: FaceEngine, galeria: Optional[Galeria] = None):
        self.motor = motor
        self.galeria = galeria if galeria is not None else Galeria()
        self._candidato = None          # emp_id o "?" (desconocido) visto en el último análisis
        self._repeticiones = 0
        self._sin_rostro = 0
        self._emp_id: Optional[int] = None
        self._desconocido = False
        self._similitud = 0.0

    def reiniciar(self):
        """Olvida la persona actual (p. ej. al cambiar de cámara)."""
        self._candidato, self._repeticiones, self._sin_rostro = None, 0, 0
        self._emp_id, self._desconocido, self._similitud = None, False, 0.0

    def procesar(self, frame: np.ndarray) -> Resultado:
        rostro = self.motor.principal(frame)
        if rostro is None:
            self._sin_rostro += 1
            if self._sin_rostro >= config.FACE_PERDIDA_INFERENCIAS:
                self.reiniciar()
                self._sin_rostro = config.FACE_PERDIDA_INFERENCIAS
            # mientras no se "pierde" del todo se conserva la identidad previa
            return Resultado(emp_id=self._emp_id, desconocido=self._desconocido, similitud=self._similitud)

        self._sin_rostro = 0
        calidad_ok, aviso = self.motor.evaluar_calidad(frame, rostro)
        if not calidad_ok:
            # Con mala calidad no se decide nada: ni se confirma ni se desconfirma.
            return Resultado(rostro, False, aviso, self._emp_id, self._similitud, self._desconocido)

        vector = self.motor.embedding(frame, rostro)
        emp_id, similitud = self.galeria.buscar(vector)
        candidato = emp_id if emp_id is not None else "?"
        if candidato == self._candidato:
            self._repeticiones += 1
        else:
            self._candidato, self._repeticiones = candidato, 1

        if self._repeticiones >= config.FACE_CONFIRMACIONES:
            self._emp_id = emp_id
            self._desconocido = emp_id is None
            self._similitud = similitud

        return Resultado(
            rostro, True, "OK", self._emp_id, similitud, self._desconocido,
            vector=vector, recorte=self.motor.recortar(frame, rostro),
        )
