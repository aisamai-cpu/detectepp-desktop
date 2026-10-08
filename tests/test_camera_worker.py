"""Pruebas de CameraWorker._paso() con fuente y motor falsos (sin lanzar el hilo ni la cámara)."""
import unittest
from unittest import mock

import numpy as np

try:
    from camera_worker import CameraWorker
    PYQT = True
except ImportError:  # PyQt6 no instalado
    PYQT = False

import config
from core import camaras, database, face_store
from tests.helpers import BaseConCarpetaTemporal
from tests.test_identificador import FRAME, MotorFalso, vec


class FuenteFalsa:
    def __init__(self, camara):
        self.camara = camara
        self.rotacion = camara.rotacion
        self.estado = "OK"
        self.mensaje = "en vivo"
        self.numero = 0
        self.iniciada = self.detenida = False
        self.frame = np.zeros((40, 60, 3), np.uint8)

    def iniciar(self):
        self.iniciada = True

    def detener(self):
        self.detenida = True

    def nueva_imagen(self):
        self.numero += 1

    def leer(self):
        return (self.frame, self.numero) if self.numero else (None, 0)


class Reloj:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@unittest.skipUnless(PYQT, "requiere PyQt6")
class TestWorker(BaseConCarpetaTemporal):
    def setUp(self):
        super().setUp()
        config.ensure_dirs()
        database.init_db()
        self.fuentes = []
        self.reloj = Reloj()
        self.emitidos = []
        self.estados = []

    def crear(self, guion=()):
        def fabrica(cam):
            f = FuenteFalsa(cam)
            self.fuentes.append(f)
            return f
        w = CameraWorker(fabrica_fuente=fabrica, motor=MotorFalso(guion), reloj=self.reloj)
        w.frame_listo.connect(lambda f, r: self.emitidos.append((f, r)))
        w.estado_camara.connect(lambda e, m: self.estados.append(e))
        w._preparar()
        return w

    def avanzar(self, w, dt=0.2):
        self.reloj.t += dt
        self.fuentes[-1].nueva_imagen()
        return w._paso()

    def test_sin_frame_nuevo_no_emite(self):
        w = self.crear()
        self.assertFalse(w._paso())
        self.assertEqual(self.emitidos, [])
        self.assertEqual(self.estados, ["OK"])  # el estado sí se informa una vez

    def test_reconoce_y_emite_resultado(self):
        face_store.registrar_empleado("Ana", 30, "Química", "x.jpg", [vec(0)])
        w = self.crear([("ok", 0), ("ok", 0)])
        self.avanzar(w); self.avanzar(w)
        frame, res = self.emitidos[-1]
        self.assertEqual(res.emp_id, 1)
        self.assertEqual(frame.shape, (40, 60, 3))

    def test_limita_la_frecuencia_de_analisis(self):
        w = self.crear([("ok", 0)] * 5)
        self.avanzar(w, dt=0.2)                      # analiza
        self.avanzar(w, dt=0.01)                     # demasiado pronto: reutiliza
        self.assertEqual(len(w._identificador.motor.guion), 4)
        self.assertIs(self.emitidos[0][1] is not None, True)
        self.assertIs(self.emitidos[1][1], self.emitidos[0][1])

    def test_fuera_de_biometria_no_analiza(self):
        w = self.crear([("ok", 0)])
        w.set_mode("EPP")
        self.avanzar(w)
        self.assertIsNone(self.emitidos[-1][1])
        self.assertEqual(len(w._identificador.motor.guion), 1)

    def test_un_fallo_del_motor_no_mata_el_hilo(self):
        w = self.crear([("ok", 0)])
        with mock.patch.object(w._identificador, "procesar", side_effect=RuntimeError("boom")), \
                self.assertLogs("camera_worker", "ERROR"):
            self.avanzar(w)
        self.assertIsNone(self.emitidos[-1][1])
        self.assertTrue(self.avanzar(w))             # sigue funcionando

    def test_usb_se_espeja_y_url_no(self):
        w = self.crear()
        self.fuentes[-1].frame[:, 0] = 255            # columna izquierda blanca
        self.avanzar(w)
        self.assertEqual(int(self.emitidos[-1][0][0, -1, 0]), 255)   # USB: pasa a la derecha
        w.cambiar_camara(camaras.agregar("Cel", "url", "http://1.2.3.4:8080/video"))
        w._atender_pendientes()
        self.fuentes[-1].frame[:, 0] = 255
        self.avanzar(w)
        self.assertEqual(int(self.emitidos[-1][0][0, 0, 0]), 255)    # IP: sin espejo

    def test_cambiar_camara_cierra_la_anterior(self):
        w = self.crear()
        primera = self.fuentes[-1]
        w.cambiar_camara(camaras.agregar("Cel", "url", "http://1.2.3.4:8080/video"))
        w._atender_pendientes()
        self.assertTrue(primera.detenida)
        self.assertTrue(self.fuentes[-1].iniciada)
        self.assertEqual(w._camara.nombre, "Cel")

    def test_reconectar_abre_una_fuente_nueva(self):
        w = self.crear()
        w.reconectar()
        w._atender_pendientes()
        self.assertEqual(len(self.fuentes), 2)
        self.assertTrue(self.fuentes[0].detenida)

    def test_reconexion_automatica_opcional(self):
        w = self.crear()
        with mock.patch.object(config, "CAMARA_RECONEXION_MIN", 5):
            self.reloj.t += 200
            w._atender_pendientes()
            self.assertEqual(len(self.fuentes), 1)   # aún no
            self.reloj.t += 200
            w._atender_pendientes()
            self.assertEqual(len(self.fuentes), 2)
        w._atender_pendientes()
        self.assertEqual(len(self.fuentes), 2)       # desactivada por defecto: no repite

    def test_rotar_actualiza_la_fuente_en_vivo(self):
        w = self.crear()
        w.rotar(90)
        w._atender_pendientes()
        self.assertEqual(self.fuentes[-1].rotacion, 90)

    def test_recargar_galeria_toma_empleados_nuevos(self):
        w = self.crear()
        self.assertEqual(len(w._identificador.galeria), 0)
        face_store.registrar_empleado("Luis", 40, "Química", "y.jpg", [vec(2)])
        w.recargar_galeria()
        w._atender_pendientes()
        self.assertEqual(len(w._identificador.galeria), 1)

    def test_sin_modelos_avisa_y_sigue_con_video(self):
        errores = []
        w = CameraWorker(fabrica_fuente=lambda c: FuenteFalsa(c), reloj=self.reloj)  # sin motor: no hay .onnx
        w.error_modelos.connect(errores.append)
        w.frame_listo.connect(lambda f, r: self.emitidos.append((f, r)))
        w._preparar()
        self.assertEqual(len(errores), 1)
        self.assertIn("descargar_modelos", errores[0])
        w._fuente.nueva_imagen()
        self.assertTrue(w._paso())
        self.assertIsNone(self.emitidos[-1][1])


if __name__ == "__main__":
    unittest.main()
