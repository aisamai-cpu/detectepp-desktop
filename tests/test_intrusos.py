"""Pruebas del detector de rostros desconocidos (sin interfaz)."""
import unittest
from unittest import mock

import config
from core.intrusos import DetectorIntrusos
from vision.identificador import Resultado
import numpy as np


def res(desconocido=True, recorte=True, calidad=True, k=0):
    return Resultado(desconocido=desconocido, calidad_ok=calidad,
                     vector=np.full(128, 0.1 + k, np.float32) if recorte else None,
                     recorte=np.full((10, 10, 3), k, np.uint8) if recorte else None)


class Reloj:
    t = 100.0

    def __call__(self):
        return self.t


class TestDetectorIntrusos(unittest.TestCase):
    def setUp(self):
        self.reloj = Reloj()
        self.det = DetectorIntrusos(self.reloj)
        # Las pruebas de tiempo no dependen del muestreo: se desactiva su separación y su mínimo
        self._parches = [mock.patch.object(config, "REGISTRO_SEPARACION_S", 0.0),
                         mock.patch.object(config, "REGISTRO_MUESTRAS_MIN", 1)]
        for p in self._parches:
            p.start()

    def tearDown(self):
        for p in self._parches:
            p.stop()

    def avanzar(self, segundos, r=None, **kw):
        self.reloj.t += segundos
        return self.det.procesar(res() if r is None else r, **kw)

    def test_dispara_tras_el_tiempo_configurado(self):
        self.assertFalse(self.avanzar(0))
        self.assertFalse(self.avanzar(config.INTRUSO_SEGUNDOS - 0.1))
        self.assertTrue(self.avanzar(0.2))

    def test_dispara_una_sola_vez_hasta_resolver(self):
        self.avanzar(0)
        self.assertTrue(self.avanzar(config.INTRUSO_SEGUNDOS + 1))
        for _ in range(5):
            self.assertFalse(self.avanzar(10))
        self.assertTrue(self.det.pendiente)

    def test_se_reinicia_si_deja_de_ser_desconocido(self):
        S = config.INTRUSO_SEGUNDOS
        self.avanzar(0)
        self.avanzar(S * 0.6)
        self.avanzar(0.1, res(desconocido=False))              # pasó alguien conocido
        self.avanzar(0.1)                                      # el conteo empieza de nuevo
        self.assertFalse(self.avanzar(S * 0.8))
        self.assertTrue(self.avanzar(S * 0.3))

    def test_sin_resultado_no_cuenta(self):
        self.avanzar(0)
        self.det.procesar(None)
        self.reloj.t += 10
        self.assertFalse(self.avanzar(0))                        # reinició
        self.assertTrue(self.avanzar(config.INTRUSO_SEGUNDOS + 0.1))

    def test_necesita_una_foto_del_rostro_para_disparar(self):
        self.avanzar(0)
        self.assertFalse(self.avanzar(config.INTRUSO_SEGUNDOS + 1, res(recorte=False)))
        self.assertTrue(self.avanzar(0.1))

    def test_enfriamiento_tras_resolver(self):
        self.avanzar(0)
        self.assertTrue(self.avanzar(config.INTRUSO_SEGUNDOS + 1))
        self.det.resolver()
        self.assertFalse(self.det.pendiente)
        self.avanzar(0)
        self.assertFalse(self.avanzar(config.INTRUSO_SEGUNDOS + 1))          # aún en enfriamiento
        self.avanzar(config.INTRUSO_ENFRIAMIENTO_S)
        self.assertFalse(self.avanzar(0))
        self.assertTrue(self.avanzar(config.INTRUSO_SEGUNDOS + 0.1))

    def test_reiniciar_borra_el_conteo(self):
        S = config.INTRUSO_SEGUNDOS
        self.avanzar(0)
        self.avanzar(S * 0.8)
        self.det.reiniciar()
        self.assertFalse(self.avanzar(S * 0.5))                # sin reinicio ya habría disparado

    # ---------------- muestras para registrar sin que la persona siga en cámara ----------------
    def test_guarda_muestras_separadas_con_tope(self):
        with mock.patch.object(config, "REGISTRO_SEPARACION_S", 0.15), mock.patch.object(config, "REGISTRO_MUESTRAS", 4):
            for k in range(30):
                self.avanzar(0.05, res(k=k))
        self.assertEqual(len(self.det.muestras), 4)                       # tope
        vector, recorte = self.det.muestras[0]
        self.assertEqual(vector.shape, (128,))
        self.assertEqual(recorte.shape, (10, 10, 3))

    def test_no_dispara_sin_el_minimo_de_muestras_buenas(self):
        with mock.patch.object(config, "REGISTRO_MUESTRAS_MIN", 3):
            self.avanzar(0, res(calidad=False))
            for _ in range(5):                                            # borroso: no sirve como muestra
                self.assertFalse(self.avanzar(config.INTRUSO_SEGUNDOS, res(calidad=False)))
            self.assertEqual(self.det.muestras, [])
            self.assertFalse(self.avanzar(0.1, res(k=1)))                 # 1 muestra
            self.assertFalse(self.avanzar(0.1, res(k=2)))                 # 2
            self.assertTrue(self.avanzar(0.1, res(k=3)))                  # 3: ya puede alertar

    def test_conserva_las_muestras_si_la_persona_se_va_con_la_alerta_pendiente(self):
        self.avanzar(0)
        self.assertTrue(self.avanzar(config.INTRUSO_SEGUNDOS + 1))
        self.det.procesar(None)                                            # se fue de la cámara
        self.assertTrue(self.det.muestras)
        self.det.resolver()
        self.assertEqual(self.det.muestras, [])

    def test_si_se_va_antes_de_tiempo_se_olvidan_las_muestras(self):
        self.avanzar(0)
        self.avanzar(0.5)
        self.assertTrue(self.det.muestras)
        self.det.procesar(res(desconocido=False))
        self.assertEqual(self.det.muestras, [])


if __name__ == "__main__":
    unittest.main()
