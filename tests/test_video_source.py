"""Pruebas de VideoSource: reconexión, parada limpia, archivo real y detección USB."""
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

import config
from core import video_source as vs
from core.camaras import Camara
from tests.helpers import BaseConCarpetaTemporal


def imagen(valor=100):
    return np.full((48, 64, 3), valor, np.uint8)


class CapFalsa:
    """Captura simulada. ``guion``: lista de frames (ndarray) o None (fallo); al agotarse,
    ``al_final`` decide: 'fallar' o 'repetir' (frames infinitos)."""

    def __init__(self, guion=(), al_final="repetir", abierta=True):
        self.guion = list(guion)
        self.al_final = al_final
        self.abierta = abierta
        self.liberada = False
        self.lecturas = 0

    def isOpened(self):
        return self.abierta and not self.liberada

    def read(self):
        self.lecturas += 1
        time.sleep(0.002)
        if self.guion:
            item = self.guion.pop(0)
        elif self.al_final == "repetir":
            item = imagen()
        else:
            item = None
        return (True, item) if item is not None else (False, None)

    def release(self):
        self.liberada = True

    def get(self, prop):
        return 30.0

    def set(self, prop, valor):
        return True


def esperar(condicion, tiempo=3.0):
    limite = time.monotonic() + tiempo
    while time.monotonic() < limite:
        if condicion():
            return True
        time.sleep(0.01)
    return False


class FuenteBase(BaseConCarpetaTemporal):
    def setUp(self):
        super().setUp()
        self._p = [
            mock.patch.object(config, "CAMARA_SIN_FRAMES_S", 0.3),
            mock.patch.object(vs, "_ESPERAS_REINTENTO", (0.01,)),
        ]
        for p in self._p:
            p.start()
        self.cam = Camara(id="t", nombre="Prueba", tipo="usb", valor="0")
        self.fuente = None

    def tearDown(self):
        if self.fuente:
            self.fuente.detener()
        for p in self._p:
            p.stop()
        super().tearDown()

    def crear(self, abrir, camara=None):
        self.fuente = vs.VideoSource(camara or self.cam, abrir=abrir)
        return self.fuente


class TestVideoSource(FuenteBase):
    def test_antes_de_iniciar_no_hay_frame(self):
        f = self.crear(lambda c: CapFalsa())
        self.assertEqual(f.leer(), (None, 0))
        self.assertEqual(f.estado, vs.DETENIDA)

    def test_entrega_frames_nuevos_y_se_detiene(self):
        f = self.crear(lambda c: CapFalsa())
        f.iniciar()
        self.assertTrue(esperar(lambda: f.leer()[1] >= 5))
        frame, n = f.leer()
        self.assertEqual(frame.shape, (48, 64, 3))
        self.assertEqual(f.estado, vs.OK)
        t0 = time.monotonic()
        f.detener()
        self.assertLess(time.monotonic() - t0, 1.0)
        self.assertEqual(f.estado, vs.DETENIDA)

    def test_reconecta_cuando_la_camara_deja_de_dar_imagen(self):
        caps = [CapFalsa([imagen(), imagen()], al_final="fallar"), CapFalsa()]
        aperturas = []

        def abrir(c):
            aperturas.append(1)
            return caps[min(len(aperturas) - 1, 1)]

        f = self.crear(abrir)
        f.iniciar()
        self.assertTrue(esperar(lambda: f.leer()[1] >= 2 + 5, tiempo=5))
        self.assertGreaterEqual(len(aperturas), 2)
        self.assertTrue(caps[0].liberada)  # la captura vieja se libera
        self.assertEqual(f.estado, vs.OK)

    def test_reintenta_si_al_principio_no_abre(self):
        intentos = []

        def abrir(c):
            intentos.append(1)
            return CapFalsa(abierta=False) if len(intentos) < 3 else CapFalsa()

        f = self.crear(abrir)
        f.iniciar()
        self.assertEqual(f.estado, vs.CONECTANDO)
        self.assertTrue(esperar(lambda: f.leer()[1] >= 1))
        self.assertGreaterEqual(len(intentos), 3)

    def test_una_excepcion_al_abrir_no_mata_el_hilo(self):
        intentos = []

        def abrir(c):
            intentos.append(1)
            if len(intentos) < 2:
                raise OSError("driver roto")
            return CapFalsa()

        f = self.crear(abrir)
        f.iniciar()
        self.assertTrue(esperar(lambda: f.leer()[1] >= 1))

    def test_detener_interrumpe_la_espera_entre_reintentos(self):
        with mock.patch.object(vs, "_ESPERAS_REINTENTO", (30.0,)):
            f = self.crear(lambda c: CapFalsa(abierta=False))
            f.iniciar()
            time.sleep(0.1)
            t0 = time.monotonic()
            f.detener()
            self.assertLess(time.monotonic() - t0, 1.0)

    def test_reiniciar_no_duplica_hilos(self):
        f = self.crear(lambda c: CapFalsa())
        f.iniciar()
        f.iniciar()
        hilos = [t for t in threading.enumerate() if t.name == "video-t"]
        self.assertEqual(len(hilos), 1)

    def test_mensajes_para_cada_estado(self):
        f = self.crear(lambda c: CapFalsa())
        for estado in (vs.CONECTANDO, vs.OK, vs.RECONECTANDO, vs.DETENIDA):
            f._estado = estado
            self.assertIn("Prueba", f.mensaje)


class TestArchivoReal(FuenteBase):
    def test_video_de_archivo_se_repite_en_bucle(self):
        ruta = Path(self._tmp.name) / "demo.avi"
        w = cv2.VideoWriter(str(ruta), cv2.VideoWriter_fourcc(*"MJPG"), 60.0, (64, 48))
        if not w.isOpened():
            self.skipTest("Este OpenCV no puede escribir AVI MJPG")
        for i in range(10):
            w.write(imagen(10 + i * 20))
        w.release()

        cam = Camara(id="a", nombre="Demo", tipo="archivo", valor=str(ruta))
        f = vs.VideoSource(cam)  # usa abrir_captura real
        self.fuente = f
        f.iniciar()
        self.assertTrue(esperar(lambda: f.leer()[1] >= 25, tiempo=6))  # más de 10 => dio la vuelta
        self.assertEqual(f.leer()[0].shape, (48, 64, 3))
        self.assertEqual(f.estado, vs.OK)

    def test_archivo_inexistente_no_cuelga_y_queda_reconectando_o_conectando(self):
        cam = Camara(id="a", nombre="Demo", tipo="archivo", valor=str(Path(self._tmp.name) / "no.avi"))
        f = vs.VideoSource(cam)
        self.fuente = f
        f.iniciar()
        time.sleep(0.3)
        self.assertEqual(f.leer(), (None, 0))
        self.assertIn(f.estado, (vs.CONECTANDO, vs.RECONECTANDO))


class TestDeteccionYApertura(unittest.TestCase):
    def test_detectar_usb_devuelve_solo_las_que_dan_imagen(self):
        caps = {}

        def abrir(c):
            i = int(c.valor)
            if i == 3:
                raise OSError("sin permisos")
            cap = CapFalsa(abierta=i in (0, 2))
            caps[i] = cap
            return cap

        encontradas = vs.detectar_usb(max_indices=5, abrir=abrir)
        self.assertEqual([e["indice"] for e in encontradas], [0, 2])
        self.assertEqual((encontradas[0]["ancho"], encontradas[0]["alto"]), (64, 48))
        self.assertTrue(all(c.liberada for c in caps.values()))  # nunca deja cámaras abiertas

    def test_abrir_captura_usb_usa_el_indice(self):
        with mock.patch.object(vs.cv2, "VideoCapture") as VC:
            VC.return_value.isOpened.return_value = True
            vs.abrir_captura(Camara("x", "USB", "usb", "2", ancho=1280, alto=720))
        args = VC.call_args[0]
        self.assertEqual(args[0], 2)
        if os.name == "nt":
            self.assertEqual(args[1], cv2.CAP_DSHOW)
        VC.return_value.set.assert_any_call(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        VC.return_value.set.assert_any_call(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def test_abrir_captura_url_usa_ffmpeg_y_buffer_minimo(self):
        with mock.patch.object(vs.cv2, "VideoCapture") as VC:
            vs.abrir_captura(Camara("x", "IP", "url", "rtsp://192.168.1.5/s"))
        args = VC.call_args[0]
        self.assertEqual(args[0], "rtsp://192.168.1.5/s")
        self.assertEqual(args[1], cv2.CAP_FFMPEG)
        VC.return_value.set.assert_any_call(cv2.CAP_PROP_BUFFERSIZE, 1)


if __name__ == "__main__":
    unittest.main()
