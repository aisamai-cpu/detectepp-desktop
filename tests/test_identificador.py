"""Pruebas del Identificador (sin modelos reales): confirmación, pérdida y calidad."""
import unittest

import numpy as np

from vision.face_engine import Galeria, Rostro
from vision.identificador import Identificador


def vec(i):
    v = np.zeros(128, np.float32)
    v[i] = 1.0
    return v


class MotorFalso:
    """``guion``: lista de elementos por llamada a principal(): None (sin rostro),
    ('ok', indice_vector) o ('mala', None) (rostro de baja calidad)."""

    def __init__(self, guion):
        self.guion = list(guion)
        self.actual = None

    def principal(self, frame):
        self.actual = self.guion.pop(0) if self.guion else None
        if self.actual is None:
            return None
        return Rostro(10, 10, 100, 100, 0.99, np.zeros(15, np.float32))

    def evaluar_calidad(self, frame, rostro):
        return (False, "Acércate más a la cámara") if self.actual[0] == "mala" else (True, "OK")

    def embedding(self, frame, rostro):
        return vec(self.actual[1])

    @staticmethod
    def recortar(frame, rostro, margen=0.25):
        return frame[:10, :10].copy()


FRAME = np.zeros((200, 200, 3), np.uint8)


def identificador(guion):
    galeria = Galeria()
    galeria.agregar(7, vec(0))   # empleado 7
    galeria.agregar(8, vec(1))   # empleado 8
    return Identificador(MotorFalso(guion), galeria)


class TestIdentificador(unittest.TestCase):
    def test_exige_confirmaciones_antes_de_dar_identidad(self):
        ident = identificador([("ok", 0), ("ok", 0)])
        self.assertIsNone(ident.procesar(FRAME).emp_id)     # 1.ª vez: aún no
        r = ident.procesar(FRAME)
        self.assertEqual(r.emp_id, 7)                       # 2.ª vez: confirmado
        self.assertFalse(r.desconocido)
        self.assertIsNotNone(r.vector)
        self.assertIsNotNone(r.recorte)

    def test_desconocido_confirmado(self):
        ident = identificador([("ok", 50), ("ok", 50)])
        ident.procesar(FRAME)
        r = ident.procesar(FRAME)
        self.assertIsNone(r.emp_id)
        self.assertTrue(r.desconocido)

    def test_cambio_de_persona_requiere_reconfirmar(self):
        ident = identificador([("ok", 0), ("ok", 0), ("ok", 1), ("ok", 1)])
        ident.procesar(FRAME); ident.procesar(FRAME)
        self.assertEqual(ident.procesar(FRAME).emp_id, 7)   # el 8 aún no está confirmado
        self.assertEqual(ident.procesar(FRAME).emp_id, 8)

    def test_pierde_identidad_tras_varios_analisis_sin_rostro(self):
        ident = identificador([("ok", 0), ("ok", 0), None, None, None, None])
        ident.procesar(FRAME); ident.procesar(FRAME)
        for _ in range(3):
            r = ident.procesar(FRAME)
            self.assertEqual(r.emp_id, 7)                   # se conserva un instante
            self.assertFalse(r.hay_rostro)
        self.assertIsNone(ident.procesar(FRAME).emp_id)     # 4.º seguido: se olvida

    def test_mala_calidad_no_confirma_ni_borra(self):
        ident = identificador([("ok", 0), ("ok", 0), ("mala", None)])
        ident.procesar(FRAME); ident.procesar(FRAME)
        r = ident.procesar(FRAME)
        self.assertFalse(r.calidad_ok)
        self.assertEqual(r.aviso, "Acércate más a la cámara")
        self.assertEqual(r.emp_id, 7)
        self.assertIsNone(r.vector)

    def test_galeria_vacia_da_desconocido(self):
        ident = Identificador(MotorFalso([("ok", 3), ("ok", 3)]), Galeria())
        ident.procesar(FRAME)
        self.assertTrue(ident.procesar(FRAME).desconocido)

    def test_reiniciar_olvida_todo(self):
        ident = identificador([("ok", 0), ("ok", 0), ("ok", 0)])
        ident.procesar(FRAME); ident.procesar(FRAME)
        ident.reiniciar()
        self.assertIsNone(ident.procesar(FRAME).emp_id)


if __name__ == "__main__":
    unittest.main()
