"""Pruebas de la preparación de muestras de un rostro nuevo (vision.registro.preparar)."""
import unittest
from unittest import mock

import numpy as np

import config
from vision import registro
from vision.face_engine import Galeria


def persona(semilla=0):
    """Vector base de una 'persona'; sus muestras se parecen mucho a él."""
    rng = np.random.default_rng(semilla)
    v = rng.normal(size=128).astype(np.float32)
    return v / np.linalg.norm(v)


def variante(base, k):
    rng = np.random.default_rng(1000 + k)
    ruido = rng.normal(size=128).astype(np.float32)
    v = base + 0.25 * ruido / np.linalg.norm(ruido)
    return v / np.linalg.norm(v)


def muestras(base, n=5):
    return [(variante(base, k), np.full((20, 20, 3), 100 + k, np.uint8)) for k in range(n)]


class TestPreparar(unittest.TestCase):
    def setUp(self):
        self.base = persona(1)

    def test_muestras_de_una_persona_nueva_quedan_listas(self):
        r = registro.preparar(muestras(self.base), Galeria())
        self.assertEqual(r.error, "")
        self.assertIsNone(r.duplicado)
        self.assertEqual(len(r.vectores), 5)
        self.assertEqual(r.recortes[0][0, 0, 0], 100)             # la foto del empleado es la primera captura

    def test_sin_muestras_da_error(self):
        self.assertTrue(registro.preparar([], Galeria()).error)

    def test_descarta_las_que_no_se_parecen_a_la_primera(self):
        mezcla = muestras(self.base, 4) + muestras(persona(99), 1)
        r = registro.preparar(mezcla, Galeria())
        self.assertEqual(len(r.vectores), 4)
        self.assertEqual(r.error, "")

    def test_si_quedan_menos_del_minimo_rechaza(self):
        mezcla = muestras(self.base, 2) + muestras(persona(98), 3)
        r = registro.preparar(mezcla, Galeria())
        self.assertIn("una sola persona", r.error)

    def test_empleado_ya_registrado_se_detecta(self):
        galeria = Galeria()
        galeria.agregar(7, self.base)
        r = registro.preparar(muestras(self.base), galeria)
        self.assertEqual(r.duplicado[0], 7)

    def test_persona_nueva_con_galeria_de_otros_no_es_duplicado(self):
        galeria = Galeria()
        galeria.agregar(7, persona(50))
        galeria.agregar(8, persona(51))
        self.assertIsNone(registro.preparar(muestras(self.base), galeria).duplicado)

    def test_el_minimo_es_configurable(self):
        with mock.patch.object(config, "REGISTRO_MUESTRAS_MIN", 5):
            self.assertTrue(registro.preparar(muestras(self.base, 4), Galeria()).error)


if __name__ == "__main__":
    unittest.main()
