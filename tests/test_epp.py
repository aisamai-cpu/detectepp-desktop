"""Pruebas de la detección de EPP con YOLO (modelo falso, sin ultralytics ni cámara)."""
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

import config
from core import database, epp
from tests.helpers import BaseConCarpetaTemporal
from vision.epp_engine import DeteccionEpp, EppEngine, ModeloEppNoEncontrado, normalizar_clase

try:
    from tests.test_camera_worker import FuenteFalsa, Reloj
    from camera_worker import CameraWorker
    PYQT = True
except ImportError:  # PyQt6 no instalado
    PYQT = False


class YoloFalso:
    """Imita ``ultralytics.YOLO``: ``names`` y ``predict()`` con ``boxes.xyxy/cls/conf``."""
    names = {0: "Hardhat", 1: "NO-Safety Vest", 2: "Person", 3: "Safety Vest"}

    def __init__(self, cajas=()):
        self.cajas = list(cajas)    # (x1, y1, x2, y2, clase, confianza)
        self.llamadas = []

    def predict(self, frame, **kwargs):
        self.llamadas.append(kwargs)
        a = np.array(self.cajas, dtype=float).reshape(-1, 6)
        return [SimpleNamespace(boxes=_Cajas(a))]


class _Cajas:
    def __init__(self, a):
        self.xyxy, self.cls, self.conf = a[:, :4], a[:, 4], a[:, 5]

    def __len__(self):
        return len(self.cls)


def det(clase, presente=True):
    return DeteccionEpp(clase, presente, 0.9, (0, 0, 10, 10))


class TestNormalizar(unittest.TestCase):
    def test_alias_y_clases_negativas(self):
        self.assertEqual(normalizar_clase("Hardhat"), ("casco", True))
        self.assertEqual(normalizar_clase("NO-Hardhat"), ("casco", False))
        self.assertEqual(normalizar_clase("safety_vest"), ("chaleco", True))
        self.assertEqual(normalizar_clase("NO-Safety Vest"), ("chaleco", False))
        self.assertEqual(normalizar_clase("Person"), ("persona", True))

    def test_clase_sin_alias_queda_en_minusculas(self):
        self.assertEqual(normalizar_clase("Safety Cone"), ("safety cone", True))


class TestEngine(BaseConCarpetaTemporal):
    def test_sin_archivo_avisa(self):
        with self.assertRaises(ModeloEppNoEncontrado):
            EppEngine()

    def test_convierte_cajas_a_detecciones(self):
        motor = EppEngine(modelo=YoloFalso([(10, 20, 50, 80, 0, 0.9), (0, 0, 5, 5, 1, 0.6)]))
        d = motor.detectar(np.zeros((100, 100, 3), np.uint8))
        self.assertEqual(d[0], DeteccionEpp("casco", True, 0.9, (10, 20, 40, 60)))
        self.assertEqual((d[1].clase, d[1].presente), ("chaleco", False))
        self.assertEqual(motor.modelo.llamadas[0]["conf"], config.EPP_CONF_MIN)
        self.assertEqual(motor.modelo.llamadas[0]["device"], "cpu")

    def test_sin_detecciones(self):
        self.assertEqual(EppEngine(modelo=YoloFalso()).detectar(np.zeros((10, 10, 3), np.uint8)), [])


class TestEvaluacion(unittest.TestCase):
    def test_porcentaje_y_faltantes(self):
        ev = epp.evaluar({"casco", "lentes"}, ["casco", "botas", "chaleco", "lentes"])
        self.assertEqual(ev.presentes, ["casco", "lentes"])
        self.assertEqual(ev.faltantes, ["botas", "chaleco"])
        self.assertEqual(ev.porcentaje, 50.0)
        self.assertFalse(ev.aprobado)

    def test_completo_aprueba(self):
        self.assertTrue(epp.evaluar({"casco"}, ["casco"]).aprobado)

    def test_negativa_no_cuenta_como_puesta(self):
        self.assertEqual(epp.prendas_vistas([det("casco", False), det("chaleco")]), {"chaleco"})

    def test_mayoria_ignora_un_fallo_aislado(self):
        s = epp.SeguimientoEpp()
        for d in ([det("casco")], [det("casco")], [], [det("casco")], [det("casco")]):
            s.agregar(d)
        self.assertEqual(s.evaluar(["casco"]).presentes, ["casco"])
        s.reiniciar()
        s.agregar([det("casco")])
        s.agregar([])
        self.assertEqual(s.evaluar(["casco"]).faltantes, ["casco"])   # empate = no confirmado


class TestRequeridos(BaseConCarpetaTemporal):
    def test_lee_epp_del_area(self):
        database.init_db()
        self.assertEqual(database.get_epp_requeridos("Bodega y despacho"), ["casco", "chaleco", "botas", "guantes"])
        self.assertEqual(database.get_epp_requeridos("No existe"), [])


@unittest.skipUnless(PYQT, "requiere PyQt6")
class TestWorkerEpp(BaseConCarpetaTemporal):
    def setUp(self):
        super().setUp()
        config.ensure_dirs()
        database.init_db()
        self.reloj = Reloj()
        self.emitidos = []
        self.errores = []

    def crear(self, motor_epp=None):
        self.fuente = None

        def fabrica(cam):
            self.fuente = FuenteFalsa(cam)
            return self.fuente
        w = CameraWorker(fabrica_fuente=fabrica, motor=mock.Mock(), reloj=self.reloj,
                         analisis_sincrono=True, motor_epp=motor_epp)
        w.frame_listo.connect(lambda f, r: self.emitidos.append(r))
        w.error_epp.connect(self.errores.append)
        w._preparar()
        w.set_mode("EPP")
        return w

    def avanzar(self, w, dt=0.5):
        self.reloj.t += dt
        self.fuente.nueva_imagen()
        w._paso()

    def test_en_modo_epp_emite_detecciones(self):
        w = self.crear(EppEngine(modelo=YoloFalso([(0, 0, 10, 10, 0, 0.8)])))
        self.avanzar(w)
        self.assertEqual([d.clase for d in self.emitidos[-1]], ["casco"])

    def test_sin_modelo_avisa_una_vez_y_sigue_el_video(self):
        w = self.crear()
        with self.assertLogs("camera_worker", "ERROR"):
            self.avanzar(w)
        self.avanzar(w)
        self.assertEqual(len(self.errores), 1)
        self.assertEqual(self.emitidos, [None, None])

    def test_resultado_tardio_de_biometria_no_se_emite_en_modo_epp(self):
        w = self.crear(EppEngine(modelo=YoloFalso([])))

        def analisis_tardio(frame, generacion, modo="BIOMETRIA", t=None):
            w._ultimo_resultado = object()   # un Resultado de rostro que terminó justo al cambiar de modo
        w._analizar = analisis_tardio
        self.avanzar(w)
        self.assertIsNone(self.emitidos[-1])

    def test_al_volver_a_biometria_no_se_mezclan_resultados(self):
        w = self.crear(EppEngine(modelo=YoloFalso([(0, 0, 10, 10, 0, 0.8)])))
        self.avanzar(w)
        w.set_mode("MONITOREO")
        self.avanzar(w)
        self.assertIsNone(self.emitidos[-1])
