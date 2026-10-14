"""Pruebas de la alerta de EPP incompleto (lógica y panel)."""
import os
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

import config
from core import database, face_store
from core.epp import AlertaEpp, evaluar
from tests.helpers import BaseConCarpetaTemporal
from vision.epp_engine import DeteccionEpp

try:
    from PyQt6.QtWidgets import QApplication
    import main as app_main
    PYQT = True
except ImportError:
    PYQT = False

REQ = ["casco", "chaleco"]
INCOMPLETO = evaluar({"casco"}, REQ)
COMPLETO = evaluar({"casco", "chaleco"}, REQ)


class Reloj:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class TestAlertaEpp(unittest.TestCase):
    def setUp(self):
        self._p = [mock.patch.object(config, "EPP_VENTANA", 5),
                   mock.patch.object(config, "EPP_ALERTA_SEGUNDOS", 3.0),
                   mock.patch.object(config, "EPP_ALERTA_ENFRIAMIENTO_S", 60.0)]
        for p in self._p:
            p.start()
        self.reloj = Reloj()
        self.a = AlertaEpp(self.reloj)

    def tearDown(self):
        for p in self._p:
            p.stop()

    def pasar(self, segundos, ev=INCOMPLETO, emp=1, analisis=5):
        """Avanza el reloj de a 0.5 s; devuelve cuántas alertas salieron."""
        n = 0
        for _ in range(int(segundos / 0.5) + 1):
            n += self.a.procesar(emp, ev, analisis)
            self.reloj.t += 0.5
        return n

    def test_alerta_tras_3_segundos_incompleto(self):
        self.assertFalse(self.a.procesar(1, INCOMPLETO, 5))
        self.reloj.t += 2.9
        self.assertFalse(self.a.procesar(1, INCOMPLETO, 5))
        self.reloj.t += 0.2
        self.assertTrue(self.a.procesar(1, INCOMPLETO, 5))

    def test_no_alerta_sin_ventana_completa(self):
        self.assertEqual(self.pasar(10, analisis=4), 0)

    def test_completo_no_alerta(self):
        self.assertEqual(self.pasar(10, ev=COMPLETO), 0)

    def test_interrupcion_reinicia_el_conteo(self):
        self.pasar(2)
        self.a.procesar(1, COMPLETO, 5)
        self.assertFalse(self.a.procesar(1, INCOMPLETO, 5))
        self.reloj.t += 2.0
        self.assertFalse(self.a.procesar(1, INCOMPLETO, 5))

    def test_una_sola_alerta_durante_el_enfriamiento(self):
        self.assertEqual(self.pasar(50), 1)

    def test_vuelve_a_alertar_tras_el_enfriamiento(self):
        self.assertEqual(self.pasar(70), 2)

    def test_completar_epp_rearma_la_alerta(self):
        self.assertEqual(self.pasar(4), 1)
        self.a.procesar(1, COMPLETO, 5)
        self.assertEqual(self.pasar(4), 1)

    def test_otro_empleado_alerta_aparte(self):
        self.assertEqual(self.pasar(4, emp=1), 1)
        self.assertEqual(self.pasar(4, emp=2), 1)

    def test_sin_empleado_o_sin_requeridos_no_alerta(self):
        self.assertEqual(self.pasar(10, emp=None), 0)
        self.assertEqual(self.pasar(10, ev=evaluar(set(), [])), 0)


@unittest.skipUnless(PYQT, "requiere PyQt6")
class TestPanelAlertaEpp(BaseConCarpetaTemporal):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        super().setUp()
        config.ensure_dirs()
        database.init_db()
        emp_id = face_store.registrar_empleado("José Peña", 33, "Bodega y despacho", "f.jpg", [np.full(128, 0.1, np.float32)])
        self.ui = app_main.MainSystemWidget(window_control_parent=mock.Mock())
        self.ui.worker = mock.Mock()
        self.reloj = Reloj()
        self.ui.alerta_epp = AlertaEpp(self.reloj)
        self.ui.recargar_empleados()
        self.ui.current_emp = self.ui.empleados[emp_id]
        self.ui.tabs.setCurrentIndex(1)

    def tearDown(self):
        self.ui.deleteLater()
        super().tearDown()

    def analizar(self, clases, segundos):
        for _ in range(int(segundos / 0.3) + 1):
            dets = [DeteccionEpp(c, True, 0.9, (0, 0, 10, 10)) for c in clases]
            self.ui._aplicar_epp(dets)
            self.reloj.t += 0.3

    def test_incompleto_lanza_alerta_con_sonido_y_tarjeta(self):
        with mock.patch.object(app_main.AlertManager, "trigger_alert") as alerta:
            self.analizar(["casco", "botas"], 5)
        alerta.assert_called_once()
        tipo, area, detalle = alerta.call_args.args
        self.assertEqual((tipo, area), ("EPP_INCOMPLETO", "Bodega y despacho"))
        self.assertIn("José Peña", detalle)
        self.assertIn("chaleco", detalle)
        self.assertTrue(alerta.call_args.kwargs.get("sonido"))
        self.assertFalse(self.ui.card_epp_alerta.isHidden())

    def test_alerta_se_guarda_en_la_base(self):
        self.analizar(["casco"], 5)
        tipos = [a["tipo"] if isinstance(a, dict) else a[1] for a in database.get_alertas()]
        self.assertIn("EPP_INCOMPLETO", tipos)

    def test_completo_no_alerta_y_oculta_tarjeta(self):
        with mock.patch.object(app_main.AlertManager, "trigger_alert") as alerta:
            self.analizar(["casco"], 5)
            self.analizar(["casco", "botas", "chaleco", "guantes"], 3)
        alerta.assert_called_once()
        self.assertTrue(self.ui.card_epp_alerta.isHidden())


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(PYQT, "requiere PyQt6")
class TestCambioDePestana(BaseConCarpetaTemporal):
    """Frames en cola con el resultado del modo anterior no deben tumbar la app."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        super().setUp()
        database.init_db()
        self.ui = app_main.MainSystemWidget(window_control_parent=mock.Mock())
        self.ui.worker = mock.Mock()
        self.ui.resize(1000, 700)

    def tearDown(self):
        self.ui.deleteLater()
        super().tearDown()

    def test_resultado_de_biometria_llega_en_pestana_epp(self):
        from tests.test_ui_biometria import resultado
        self.ui.tabs.setCurrentIndex(1)
        self.ui.on_frame(np.zeros((120, 160, 3), np.uint8), resultado(emp_id=None))
        self.assertIsNone(self.ui._ultimas_epp)
        self.ui.worker.frame_mostrado.assert_called()

    def test_detecciones_epp_llegan_en_pestana_biometria(self):
        self.ui.tabs.setCurrentIndex(0)
        self.ui.on_frame(np.zeros((120, 160, 3), np.uint8), [DeteccionEpp("casco", True, 0.9, (0, 0, 5, 5))])
        self.assertIsNone(self.ui.resultado)


@unittest.skipUnless(PYQT, "requiere PyQt6")
class TestCamaraGirada(BaseConCarpetaTemporal):
    """Con la cámara girada (imagen vertical) el recuadro del video no debe crecer frame a frame."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        super().setUp()
        database.init_db()
        self.ui = app_main.MainSystemWidget(window_control_parent=mock.Mock())
        self.ui.worker = mock.Mock()
        self.ui.resize(1280, 720)
        self.ui.show()

    def tearDown(self):
        self.ui.close()
        self.ui.deleteLater()
        super().tearDown()

    def test_tamano_estable_en_todas_las_pestanas(self):
        vertical = np.zeros((1280, 720, 3), np.uint8)
        for idx, etiqueta in ((0, self.ui.video_label_bio), (1, self.ui.video_label_epp),
                              (2, self.ui.video_label_accidents)):
            with self.subTest(pestana=idx):
                self.ui.tabs.setCurrentIndex(idx)
                self.ui.on_frame(np.zeros((720, 1280, 3), np.uint8), None)
                self.app.processEvents()
                antes = etiqueta.size()
                for _ in range(6):
                    self.ui.on_frame(vertical, None)
                    self.app.processEvents()
                self.assertEqual(etiqueta.size(), antes)
                self.assertLessEqual(etiqueta.pixmap().height(), etiqueta.height())
