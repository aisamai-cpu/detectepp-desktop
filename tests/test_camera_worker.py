"""Pruebas de CameraWorker._paso() con fuente y motor falsos (sin lanzar el hilo ni la cámara)."""
import unittest
from unittest import mock

import time

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
        self._fps = 0.0
        self.iniciada = self.detenida = False
        self.frame = np.zeros((40, 60, 3), np.uint8)

    @property
    def fps(self):
        return self._fps

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
        w = CameraWorker(fabrica_fuente=fabrica, motor=MotorFalso(guion), reloj=self.reloj, analisis_sincrono=True)
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
        w = CameraWorker(fabrica_fuente=lambda c: FuenteFalsa(c), reloj=self.reloj, analisis_sincrono=True)  # sin motor inyectado
        w.error_modelos.connect(errores.append)
        w.frame_listo.connect(lambda f, r: self.emitidos.append((f, r)))
        # Los .onnx reales pueden existir en la máquina del usuario: se apunta a rutas que no existen.
        with mock.patch.object(config, "YUNET_PATH", config.MODELS_DIR / "no_existe_yunet.onnx"), \
                mock.patch.object(config, "SFACE_PATH", config.MODELS_DIR / "no_existe_sface.onnx"):
            w._preparar()
        self.assertEqual(len(errores), 1)
        self.assertIn("descargar_modelos", errores[0])
        w._fuente.nueva_imagen()
        self.assertTrue(w._paso())
        self.assertIsNone(self.emitidos[-1][1])

    # -- fluidez ----------------------------------------------------------------
    def test_control_de_flujo_descarta_frames_mientras_la_interfaz_no_dibuja(self):
        w = self.crear()
        w.control_flujo = True
        self.avanzar(w, dt=0.01)                 # se envía
        self.avanzar(w, dt=0.01)                 # la interfaz no ha confirmado: se descarta
        self.avanzar(w, dt=0.01)
        self.assertEqual(len(self.emitidos), 1)
        w.frame_mostrado()
        self.avanzar(w, dt=0.01)
        self.assertEqual(len(self.emitidos), 2)

    def test_control_de_flujo_no_se_atasca_si_la_interfaz_nunca_confirma(self):
        w = self.crear()
        w.control_flujo = True
        self.avanzar(w, dt=0.01)
        self.avanzar(w, dt=0.6)                  # pasó el tiempo de seguridad
        self.assertEqual(len(self.emitidos), 2)

    def test_sin_control_de_flujo_se_envia_todo(self):
        w = self.crear()
        for _ in range(5):
            self.avanzar(w, dt=0.01)
        self.assertEqual(len(self.emitidos), 5)

    def test_informa_fps_una_vez_por_segundo(self):
        fps = []
        w = self.crear()
        w.fps_camara.connect(lambda lectura, pantalla: fps.append((lectura, pantalla)))
        self.fuentes[-1]._fps = 9.0
        for _ in range(12):
            self.avanzar(w, dt=0.1)              # ~10 imágenes por segundo
        self.assertEqual(len(fps), 1)
        self.assertAlmostEqual(fps[0][1], 10.0, delta=1.5)

    def test_el_analisis_en_hilo_no_bloquea_el_video(self):
        """Con análisis lento, _paso() sigue entregando imágenes sin esperarlo."""
        import threading
        import time as t
        liberar = threading.Event()
        w = CameraWorker(fabrica_fuente=lambda c: FuenteFalsa(c), motor=MotorFalso([("ok", 0)] * 3),
                         reloj=time.monotonic)
        original = w._identificador.procesar if w._identificador else None
        w.frame_listo.connect(lambda f, r: self.emitidos.append((f, r)))
        w._preparar()
        recibidos = []

        def lento(frame):
            recibidos.append(1)
            liberar.wait(3)
            return None
        w._identificador.procesar = lento
        try:
            fuente = w._fuente
            inicio = t.monotonic()
            for _ in range(10):
                fuente.nueva_imagen()
                w._paso()
                t.sleep(config.FACE_INTERVALO_S / 4 if hasattr(config, "FACE_INTERVALO_S") else 0.03)
            self.assertLess(t.monotonic() - inicio, 2.0)     # no esperó a la IA (bloqueada)
            self.assertEqual(len(self.emitidos), 10)
            self.assertEqual(len(recibidos), 1)              # el análisis lento se quedó con 1 frame
        finally:
            liberar.set()
            w._salir.set()
            with w._cond:
                w._cond.notify_all()
            w._hilo_analisis.join(timeout=3)
            w._cerrar_fuente()


if __name__ == "__main__":
    unittest.main()
