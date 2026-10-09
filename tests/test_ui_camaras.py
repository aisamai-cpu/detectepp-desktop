"""Pruebas de la ventana «Cámaras» (sin cámaras reales, sin ventana visible)."""
import os
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6.QtWidgets import QApplication, QMessageBox
    import main as app_main
    from ui_camaras import CamarasDialog
    PYQT = True
except ImportError:
    PYQT = False

from core import camaras
from tests.helpers import BaseConCarpetaTemporal, esperar
from tests.test_camera_worker import FuenteFalsa

URL = "http://192.168.1.50:8080/videofeed"


@unittest.skipUnless(PYQT, "requiere PyQt6")
class TestDialogoCamaras(BaseConCarpetaTemporal):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        super().setUp()
        self.fuentes = []
        self.usb_hallado = []

        def fabrica(cam):
            f = FuenteFalsa(cam)
            self.fuentes.append(f)
            return f

        self.dlg = CamarasDialog(fabrica_fuente=fabrica, detectar=lambda: self.usb_hallado)
        self.avisos = []
        self.dlg.cambiado.connect(lambda: self.avisos.append(1))

    def tearDown(self):
        self.dlg.close()
        self.dlg.deleteLater()
        super().tearDown()

    def agregar(self, nombre="Celular", valor=URL, tipo=1):
        self.dlg.combo_tipo.setCurrentIndex(tipo)
        self.dlg.entrada_nombre.setText(nombre)
        self.dlg.entrada_valor.setText(valor)
        self.dlg.agregar()

    def test_lista_marca_la_camara_en_uso(self):
        self.assertEqual(self.dlg.lista.count(), 1)
        self.assertIn("en uso", self.dlg.lista.item(0).text())
        self.assertFalse(self.dlg.btn_eliminar.isEnabled())      # no se puede dejar sin cámaras
        self.assertFalse(self.dlg.btn_activar.isEnabled())       # ya es la activa

    def test_agregar_ip_la_selecciona_y_avisa(self):
        self.agregar()
        self.assertEqual(self.dlg.lista.count(), 2)
        self.assertEqual(camaras.listar()[-1].valor, URL)
        self.assertEqual(self.dlg._id_seleccionado(), camaras.listar()[-1].id)
        self.assertEqual(len(self.avisos), 1)
        self.assertEqual(self.dlg.entrada_nombre.text(), "")

    def test_agregar_invalido_muestra_el_error_y_no_guarda(self):
        self.agregar(valor="hola mundo")
        self.assertIn("IP del celular", self.dlg.lbl_mensaje.text())
        self.assertEqual(len(camaras.listar()), 1)
        self.assertEqual(self.avisos, [])

    def test_agregar_solo_con_la_ip_del_celular(self):
        self.agregar("Celular", "192.168.0.123")
        self.assertEqual(camaras.listar()[-1].valor, "http://192.168.0.123:8080/video")
        self.assertIn("http://192.168.0.123:8080/video", self.dlg.lbl_mensaje.text())

    def test_agregar_usb_y_archivo(self):
        self.agregar("Externa", "1", tipo=0)
        self.agregar("Demo", "C:\\videos\\demo.mp4", tipo=2)
        tipos = [c.tipo for c in camaras.listar()]
        self.assertEqual(tipos, ["usb", "usb", "archivo"])

    def test_activar_cambia_la_activa(self):
        self.agregar()
        self.assertTrue(self.dlg.btn_activar.isEnabled())
        self.dlg.activar()
        self.assertEqual(camaras.activa().nombre, "Celular")
        self.assertIn("en uso", self.dlg.lista.item(1).text())
        self.assertEqual(len(self.avisos), 2)

    def test_girar_persiste(self):
        self.agregar()
        self.dlg.girar()
        self.dlg.girar()
        self.assertEqual(camaras.listar()[-1].rotacion, 180)

    def test_eliminar_pide_confirmacion(self):
        self.agregar()
        with mock.patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.No):
            self.dlg.eliminar()
        self.assertEqual(len(camaras.listar()), 2)
        with mock.patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes):
            self.dlg.eliminar()
        self.assertEqual(len(camaras.listar()), 1)
        self.assertEqual(self.dlg.lista.count(), 1)

    def test_no_prueba_una_usb_que_esta_en_uso(self):
        self.dlg.alternar_prueba()
        self.assertEqual(self.fuentes, [])
        self.assertIn("en uso", self.dlg.lbl_mensaje.text())

    def test_prueba_muestra_imagen_y_se_detiene(self):
        self.agregar()
        self.dlg.alternar_prueba()
        self.assertEqual(len(self.fuentes), 1)
        self.assertTrue(self.fuentes[0].iniciada)
        self.fuentes[0].nueva_imagen()
        self.dlg._actualizar_vista()
        self.assertFalse(self.dlg.vista.pixmap().isNull())
        self.dlg.alternar_prueba()
        self.assertTrue(self.fuentes[0].detenida)
        self.assertEqual(self.dlg.btn_probar.text(), "▶ Probar")

    def test_cambiar_de_seleccion_detiene_la_prueba(self):
        self.agregar()
        self.dlg.alternar_prueba()
        self.dlg.lista.setCurrentRow(0)
        self.assertTrue(self.fuentes[0].detenida)

    def test_cerrar_detiene_la_prueba(self):
        self.agregar()
        self.dlg.alternar_prueba()
        self.dlg.close()
        self.assertTrue(self.fuentes[0].detenida)

    def test_buscar_usb_rellena_la_primera_sin_registrar(self):
        self.usb_hallado = [{"indice": 0, "ancho": 640, "alto": 480}, {"indice": 1, "ancho": 1280, "alto": 720}]
        self.dlg.combo_tipo.setCurrentIndex(0)
        self.dlg.buscar_usb()
        self.assertTrue(esperar(lambda: (self.app.processEvents() or True) and self.dlg.entrada_valor.text() == "1"))
        self.assertEqual(self.dlg.entrada_nombre.text(), "Cámara USB 1")
        self.assertTrue(self.dlg.btn_buscar.isEnabled())

    def test_buscar_usb_sin_resultados(self):
        self.dlg.buscar_usb()
        self.assertTrue(esperar(lambda: (self.app.processEvents() or True) and self.dlg.btn_buscar.isEnabled()))
        self.assertIn("No se encontró", self.dlg.lbl_mensaje.text())

    def test_buscar_usb_si_el_detector_falla_no_cuelga(self):
        def malo():
            raise OSError("driver roto")
        self.dlg._detectar = malo
        with self.assertLogs("ui_camaras", "ERROR"):
            self.dlg.buscar_usb()
            self.assertTrue(esperar(lambda: (self.app.processEvents() or True) and self.dlg.btn_buscar.isEnabled()))


@unittest.skipUnless(PYQT, "requiere PyQt6")
class TestSincronizacionConLaPantallaPrincipal(BaseConCarpetaTemporal):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        super().setUp()
        from core import database
        import config
        config.ensure_dirs()
        database.init_db()
        self.ui = app_main.MainSystemWidget(window_control_parent=mock.Mock())
        self.ui.worker = mock.Mock()

    def tearDown(self):
        self.ui.deleteLater()
        super().tearDown()

    def test_activar_otra_camara_reabre_en_el_worker(self):
        cel = camaras.agregar("Celular", "url", URL, activar_ahora=True)
        self.ui._tras_gestion_camaras()
        self.ui.worker.cambiar_camara.assert_called_once()
        self.assertEqual(self.ui.worker.cambiar_camara.call_args[0][0].id, cel.id)
        self.assertEqual(self.ui.combo_camara.currentData(), cel.id)

    def test_sin_cambios_no_toca_el_worker(self):
        self.ui._tras_gestion_camaras()
        self.ui.worker.cambiar_camara.assert_not_called()
        self.ui.worker.rotar.assert_not_called()

    def test_girar_la_activa_solo_rota(self):
        camaras.rotar(camaras.activa().id, 270)
        self.ui._tras_gestion_camaras()
        self.ui.worker.rotar.assert_called_once_with(270)
        self.ui.worker.cambiar_camara.assert_not_called()

    def test_eliminar_la_activa_pasa_a_otra(self):
        cel = camaras.agregar("Celular", "url", URL, activar_ahora=True)
        self.ui._tras_gestion_camaras()
        camaras.eliminar(cel.id)
        self.ui._tras_gestion_camaras()
        self.assertEqual(self.ui.worker.cambiar_camara.call_args[0][0].id, "cam-integrada")


if __name__ == "__main__":
    unittest.main()
