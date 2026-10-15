"""Hilo de cámara en modo MONITOREO con un modelo de pose falso (sin ONNX real ni cámara)."""
import unittest
from unittest import mock

import numpy as np

import config
from core import database
from core.riesgos import ResultadoPose
from tests.esqueletos import cuerpo
from tests.helpers import BaseConCarpetaTemporal
from vision.identificador import Resultado
from vision.yolo_onnx import Caja

try:
    from tests.test_camera_worker import FuenteFalsa, Reloj
    from camera_worker import CameraWorker, MODO_MONITOREO, resultado_valido
    from vision.pose_engine import ModeloPoseNoEncontrado, PoseEngine
    PYQT = True
except ImportError:
    PYQT = False


class PoseFalsa:
    """Imita ModeloYoloOnnx de pose: devuelve las personas que se le indiquen."""
    tarea, kpt_shape, lado = "pose", (17, 3), 640

    def __init__(self):
        self.personas = [cuerpo(cx=30, cy=20, T=8)]

    def detectar(self, frame, conf):
        return [Caja(*p.caja, 0, p.confianza, p.puntos) for p in self.personas]


@unittest.skipUnless(PYQT, "requiere PyQt6")
class TestWorkerMonitoreo(BaseConCarpetaTemporal):
    def setUp(self):
        super().setUp()
        config.ensure_dirs()
        database.init_db()
        self.reloj = Reloj()
        self.emitidos, self.eventos, self.errores = [], [], []
        self.pose = PoseFalsa()

    def crear(self, motor_pose="falso"):
        def fabrica(cam):
            self.fuente = FuenteFalsa(cam)
            return self.fuente
        motor = PoseEngine(modelo=self.pose) if motor_pose == "falso" else motor_pose
        w = CameraWorker(fabrica_fuente=fabrica, motor=mock.Mock(), reloj=self.reloj,
                         analisis_sincrono=True, motor_pose=motor)
        w.frame_listo.connect(lambda f, r: self.emitidos.append(r))
        w.evento_riesgo.connect(lambda ev, fr: self.eventos.append((ev, fr)))
        w.error_pose.connect(self.errores.append)
        w._preparar()
        w.set_mode(MODO_MONITOREO)
        return w

    def avanzar(self, w, n=1, dt=0.2):
        for _ in range(n):
            self.reloj.t += dt
            self.fuente.nueva_imagen()
            w._paso()

    def test_emite_resultado_pose_con_personas(self):
        w = self.crear()
        self.avanzar(w, 3)
        r = self.emitidos[-1]
        self.assertIsInstance(r, ResultadoPose)
        self.assertEqual([e.id for e in r.estados], ["P1"])
        self.assertAlmostEqual(r.frecuencia, 5.0, places=1)

    def test_varias_personas(self):
        self.pose.personas = [cuerpo(cx=10, cy=20, T=6), cuerpo(cx=50, cy=20, T=6)]
        w = self.crear()
        self.avanzar(w, 2)
        self.assertEqual({e.id for e in self.emitidos[-1].estados}, {"P1", "P2"})

    def test_evento_viaja_por_su_propia_senal_con_el_frame(self):
        self.pose.personas = [cuerpo(cx=30, cy=20, T=8, brazos="arriba")]
        w = self.crear()
        with mock.patch.object(config, "MON_BRAZOS_S", 1.0):
            self.avanzar(w, 8)
        self.assertEqual([e.tipo for e, _ in self.eventos], ["BRAZOS_ELEVADOS"])
        self.assertEqual(self.eventos[0][1].frame.shape, (40, 60, 3))
        self.assertEqual(len(self.eventos[0][1].personas), 1)

    def test_girar_reinicia_el_monitor(self):
        w = self.crear()
        self.avanzar(w, 2)
        self.pose.personas = [cuerpo(cx=30, cy=20, T=8), cuerpo(cx=55, cy=20, T=8)]
        self.avanzar(w, 1)
        self.assertEqual({e.id for e in self.emitidos[-1].estados}, {"P1", "P2"})
        w.rotar(90)
        self.avanzar(w, 2)
        self.assertEqual({e.id for e in self.emitidos[-1].estados}, {"P1", "P2"})   # ids desde cero
        self.assertEqual(w.monitor.seguidor._siguiente, 3)

    def test_vuelta_del_video_reinicia_el_monitor(self):
        w = self.crear()
        self.avanzar(w, 2)
        gen = w._generacion
        self.fuente.cortes = 1
        self.avanzar(w, 1)
        self.assertEqual(w._generacion, gen + 1)

    def test_sin_modelo_avisa_y_sigue_el_video(self):
        w = self.crear(motor_pose=None)
        with self.assertLogs("camera_worker", "ERROR"):
            self.avanzar(w, 2)
        self.assertEqual(len(self.errores), 1)
        self.assertEqual(self.emitidos, [None, None])

    def test_filtro_por_modo(self):
        self.assertTrue(resultado_valido("EPP", []))
        self.assertFalse(resultado_valido("EPP", ResultadoPose((), 0, (1, 1))))
        self.assertTrue(resultado_valido(MODO_MONITOREO, ResultadoPose((), 0, (1, 1))))
        self.assertFalse(resultado_valido(MODO_MONITOREO, []))
        self.assertFalse(resultado_valido("BIOMETRIA", ResultadoPose((), 0, (1, 1))))


@unittest.skipUnless(PYQT, "requiere PyQt6")
class TestPoseEngine(BaseConCarpetaTemporal):
    def test_sin_archivo(self):
        with self.assertRaises(ModeloPoseNoEncontrado):
            PoseEngine()

    def test_modelo_que_no_es_de_pose(self):
        from tests.test_yolo_onnx import ONNX, crear_onnx, salida_detect
        if not ONNX:
            self.skipTest("requiere onnx")
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        crear_onnx(config.POSE_MODELO_PATH, salida_detect(), {"names": "{0: 'a', 1: 'b'}", "task": "detect",
                                                              "imgsz": "[64, 64]"})
        with self.assertRaises(ModeloPoseNoEncontrado) as e:
            PoseEngine()
        self.assertIn("no es un modelo de pose", str(e.exception))

    def test_convierte_a_personas(self):
        motor = PoseEngine(modelo=PoseFalsa())
        personas = motor.detectar(np.zeros((40, 60, 3), np.uint8))
        self.assertEqual(len(personas), 1)
        self.assertEqual(personas[0].puntos.shape, (17, 3))


if __name__ == "__main__":
    unittest.main()


class TestHerramientaCalibracion(unittest.TestCase):
    def test_comparar(self):
        from core.riesgos import Evento
        from tools.probar_monitoreo import comparar
        ev = lambda t, g: Evento(t, g, "P1", 0, "", True, (0, 0, 1, 1))
        self.assertEqual(comparar([ev("CAIDA", "CRITICO")], ["CAIDA"]), (True, [], []))
        self.assertEqual(comparar([ev("SOS", "CRITICO")], []), (False, [], ["SOS"]))
        self.assertEqual(comparar([ev("BRAZOS_ELEVADOS", "ERGONOMIA")], []), (True, [], []))
        self.assertEqual(comparar([], ["CAIDA"]), (False, ["CAIDA"], []))


@unittest.skipUnless(PYQT, "requiere PyQt6")
class TestModeloDeReemplazo(BaseConCarpetaTemporal):
    def test_usa_el_modelo_n_si_falta_el_configurado_y_avisa(self):
        from tests.test_yolo_onnx import ONNX, crear_onnx, salida_pose
        if not ONNX:
            self.skipTest("requiere onnx")
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        crear_onnx(config.MODELS_DIR / "yolo11n-pose.onnx", salida_pose(),
                   {"names": "{0: 'person'}", "task": "pose", "imgsz": "[64, 64]", "kpt_shape": "[17, 3]"})
        motor = PoseEngine()
        self.assertIn("yolo11n-pose.onnx", motor.aviso)
        self.assertIn("tools.exportar_pose", motor.aviso)


@unittest.skipUnless(PYQT, "requiere PyQt6")
class TestHiloDeAnalisisReal(BaseConCarpetaTemporal):
    """Como en la app: el análisis corre en su PROPIO hilo (no en modo síncrono de prueba).
    Cubre el fallo en que el pedido viajaba con los argumentos en otro orden y nada se analizaba."""

    def setUp(self):
        super().setUp()
        config.ensure_dirs()
        database.init_db()
        self.reloj = Reloj()
        self.emitidos = []

    def crear(self, **motores):
        from tests.test_camera_worker import MotorFalso
        def fabrica(cam):
            self.fuente = FuenteFalsa(cam)
            return self.fuente
        w = CameraWorker(fabrica_fuente=fabrica, motor=MotorFalso([("ok", 0)] * 50), reloj=self.reloj,
                         analisis_sincrono=False, **motores)
        w.frame_listo.connect(lambda f, r: self.emitidos.append(r))
        w._preparar()                    # arranca el hilo de análisis
        self.addCleanup(self._detener, w)
        return w

    def _detener(self, w):
        w._salir.set()
        with w._cond:
            w._cond.notify_all()
        w._hilo_analisis.join(timeout=3)

    def hasta_resultado(self, w, tipo):
        from tests.helpers import esperar
        def hay():
            self.reloj.t += 0.3
            self.fuente.nueva_imagen()
            w._paso()
            return any(isinstance(r, tipo) for r in self.emitidos)
        return esperar(hay, timeout=5, paso=0.02)

    def test_biometria(self):
        from vision.identificador import Resultado
        w = self.crear()
        self.assertTrue(self.hasta_resultado(w, Resultado))

    def test_epp(self):
        from tests.test_epp import YoloFalso
        from vision.epp_engine import EppEngine
        w = self.crear(motor_epp=EppEngine(modelo=YoloFalso([(0, 0, 10, 10, 0, 0.8)])))
        w.set_mode("EPP")
        self.assertTrue(self.hasta_resultado(w, list))

    def test_monitoreo(self):
        w = self.crear(motor_pose=PoseEngine(modelo=PoseFalsa()))
        w.set_mode(MODO_MONITOREO)
        self.assertTrue(self.hasta_resultado(w, ResultadoPose))
