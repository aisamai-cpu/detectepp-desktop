"""Pestaña Monitoreo (sin ventana visible): esqueletos, emergencias, Atendido, zonas y cambios de pestaña."""
import os
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

import config
from core import camaras, database, zonas
from core.riesgos import CRITICO, ERGONOMIA, Evento, FotoEvento, MonitorRiesgos, ResultadoPose
from tests.esqueletos import cuerpo
from tests.helpers import BaseConCarpetaTemporal

try:
    from PyQt6.QtWidgets import QApplication
    import main as app_main
    import ui_monitoreo
    PYQT = True
except ImportError:
    PYQT = False


def resultado_con(*personas, frecuencia=5.0):
    est, _ = MonitorRiesgos().procesar(list(personas), 1.0, (1280, 720))
    return ResultadoPose(tuple(est), frecuencia, (1280, 720))


def evento(tipo="CAIDA", grupo=CRITICO):
    return Evento(tipo, grupo, "P1", 0.0, f"{tipo} (P1)", True, (500, 200, 800, 600))


@unittest.skipUnless(PYQT, "requiere PyQt6")
class TestPanelMonitoreo(BaseConCarpetaTemporal):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        super().setUp()
        database.init_db()
        self._sonido = mock.patch("ui_monitoreo.play_alert")
        self.play = self._sonido.start()
        self.ui = app_main.MainSystemWidget(window_control_parent=mock.Mock())
        self.ui.worker = mock.Mock()
        self.panel = self.ui.panel_monitoreo
        self.panel.worker = self.ui.worker
        self.ui.resize(1280, 720)
        self.ui.show()
        self.ui.tabs.setCurrentIndex(2)
        self.app.processEvents()

    def tearDown(self):
        self._sonido.stop()
        self.ui.close()
        self.ui.deleteLater()
        super().tearDown()

    def frame(self):
        return np.zeros((720, 1280, 3), np.uint8)

    def test_muestra_personas_con_su_estado(self):
        res = resultado_con(cuerpo(cx=300), cuerpo(cx=900, brazos="arriba"))
        self.ui.on_frame(self.frame(), res)
        self.assertIsNotNone(self.panel.video.pixmap())
        texto = self.panel.lbl_personas.text()
        self.assertIn("P1", texto)
        self.assertIn("P2", texto)
        self.assertIn("Brazos arriba", texto)
        self.assertIn("2 personas", self.panel.lbl_estado.text())

    def test_aviso_de_analisis_lento(self):
        self.ui.on_frame(self.frame(), resultado_con(cuerpo(), frecuencia=1.5))
        self.assertIn("Análisis lento", self.panel.lbl_estado.text())

    def test_resultado_de_otro_modo_no_rompe_la_pestana(self):
        self.ui.on_frame(self.frame(), [])                 # detecciones de EPP en cola
        self.ui.tabs.setCurrentIndex(1)
        self.ui.on_frame(self.frame(), resultado_con(cuerpo()))   # pose en la pestaña EPP
        self.ui.worker.frame_mostrado.assert_called()

    def test_emergencia_tarjeta_aviso_en_barra_y_atendido(self):
        foto = FotoEvento(self.frame(), (cuerpo(),))
        self.panel.on_evento(evento(), foto)
        self.assertEqual(len(self.panel.tarjetas), 1)
        self.assertFalse(self.ui.btn_emergencias.isHidden())
        self.assertIn("1 emergencia", self.ui.btn_emergencias.text())
        self.assertEqual(self.panel.lista_eventos.count(), 1)
        self.panel._repetir_sonido()
        self.play.assert_called()                           # sigue sonando hasta "Atendido"

        ev_id = next(iter(self.panel.tarjetas))
        self.panel.atender(ev_id)
        self.assertEqual(self.panel.tarjetas, {})
        self.assertTrue(self.ui.btn_emergencias.isHidden())
        self.assertIsNotNone(database.get_evento(ev_id)["atendido_en"])
        self.play.reset_mock()
        self.panel._repetir_sonido()
        self.play.assert_not_called()

    def test_ergonomia_va_a_la_lista_sin_tarjeta(self):
        self.panel.on_evento(evento("BRAZOS_ELEVADOS", ERGONOMIA), None)
        self.assertEqual(self.panel.tarjetas, {})
        self.assertEqual(self.panel.lista_eventos.count(), 1)

    def test_al_reiniciar_la_app_recarga_emergencias_sin_sonar(self):
        self.panel.on_evento(evento("SOS"), None)
        otro = ui_monitoreo.PanelMonitoreo(mock.Mock(), camaras.activa())
        self.assertEqual(len(otro.tarjetas), 1)
        self.assertFalse(next(iter(otro.tarjetas.values())).suena)
        otro.deleteLater()

    def test_dibujar_zona_la_guarda_y_la_envia_al_monitor(self):
        self.panel.btn_zona.setChecked(True)
        self.assertTrue(self.panel.video.dibujando)
        self.panel.video.zona_dibujada.emit(0.1, 0.5, 0.4, 0.9)
        self.assertFalse(self.panel.video.dibujando)
        cam = camaras.activa()
        self.assertEqual(len(zonas.de_camara(cam.id)), 1)
        self.ui.worker.set_zonas.assert_called_with(zonas.de_camara(cam.id))
        self.ui.on_frame(self.frame(), resultado_con(cuerpo()))   # se dibuja sin errores

    def test_esc_cancela_el_dibujo(self):
        self.panel.btn_zona.setChecked(True)
        self.panel.video.zona_dibujada.emit(-1, -1, -1, -1)
        self.assertEqual(zonas.de_camara(camaras.activa().id), [])

    def test_coordenadas_del_raton_descuentan_las_franjas_negras(self):
        self.ui.on_frame(np.zeros((1280, 720, 3), np.uint8), None)   # imagen vertical: franjas a los lados
        a = self.panel.video.area_imagen
        self.assertGreater(a.x(), 0)
        self.assertEqual(self.panel.video.a_normalizado(a.x(), a.y()), (0.0, 0.0))
        x, y = self.panel.video.a_normalizado(a.x() + a.width() / 2, a.y() + a.height())
        self.assertAlmostEqual(x, 0.5, places=2)
        self.assertEqual(y, 1.0)

    def test_girar_la_camara_borra_sus_zonas(self):
        cam = camaras.activa()
        zonas.agregar(cam.id, 0.1, 0.1, 0.5, 0.5)
        camaras.rotar(cam.id, 90)
        with mock.patch.object(ui_monitoreo.QMessageBox, "information") as aviso:
            self.ui.sincronizar_camara_activa()
        aviso.assert_called_once()
        self.assertEqual(zonas.de_camara(cam.id), [])
        self.ui.worker.rotar.assert_called_once_with(90)

    def test_reporte_del_dia(self):
        with mock.patch.object(ui_monitoreo.QDesktopServices, "openUrl") as abrir:
            self.panel._reporte()
        abrir.assert_called_once()
        self.assertTrue(any(config.REPORTES_DIR.glob("reporte_*.html")))


if __name__ == "__main__":
    unittest.main()
