"""Prueba de humo de la interfaz (sin ventana visible): resultado de rostro -> panel y marcación."""
import os
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

try:
    from PyQt6.QtWidgets import QApplication
    import main as app_main
    PYQT = True
except ImportError:
    PYQT = False

import config
from core import camaras, database, face_store
from tests.helpers import BaseConCarpetaTemporal
from tests.test_identificador import vec
from vision.face_engine import Rostro
from vision.identificador import Resultado


def resultado(emp_id=None, desconocido=False, calidad=True, rostro=True, aviso="OK", vector=None):
    r = Rostro(20, 20, 100, 100, 0.99, np.zeros(15, np.float32)) if rostro else None
    return Resultado(r, calidad, aviso, emp_id, 0.8, desconocido, vector=vector,
                     recorte=np.full((50, 50, 3), 120, np.uint8) if rostro else None)


@unittest.skipUnless(PYQT, "requiere PyQt6")
class TestUI(BaseConCarpetaTemporal):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        super().setUp()
        config.ensure_dirs()
        database.init_db()
        self.emp_id = face_store.registrar_empleado("José Peña", 33, "Área Química", "f.jpg", [vec(0)])
        self.ui = app_main.MainSystemWidget(window_control_parent=mock.Mock())
        self.ui.resize(1000, 700)
        self.ui.worker = mock.Mock()   # no se toca la cámara real

    def tearDown(self):
        self.ui.deleteLater()
        super().tearDown()

    def test_combo_lista_las_camaras_y_cambia(self):
        cel = camaras.agregar("Celular", "url", "http://1.2.3.4:8080/videofeed")
        self.ui.recargar_camaras()
        self.assertEqual(self.ui.combo_camara.count(), 2)
        self.ui.combo_camara.setCurrentIndex(1)
        self.ui.on_cambiar_camara(1)
        self.assertEqual(camaras.activa().id, cel.id)
        self.ui.worker.cambiar_camara.assert_called_once()

    def test_girar_guarda_y_avisa_al_worker(self):
        self.ui.on_girar_camara()
        self.assertEqual(camaras.activa().rotacion, 90)
        self.ui.worker.rotar.assert_called_with(90)

    def test_empleado_reconocido_rellena_y_marca_una_sola_vez(self):
        self.ui._aplicar_resultado(resultado(emp_id=self.emp_id))
        self.assertEqual(self.ui.input_nombre.text(), "José Peña")
        self.assertEqual(self.ui.current_emp["id"], self.emp_id)
        self.assertFalse(self.ui.btn_registrar.isEnabled())
        self.ui._aplicar_resultado(resultado(emp_id=self.emp_id))
        with database._db() as c:
            self.assertEqual(c.execute("SELECT COUNT(*) FROM asistencia").fetchone()[0], 1)

    def test_otro_empleado_marca_aunque_el_primero_marco_hace_poco(self):
        otro = face_store.registrar_empleado("Ana", 28, "Área Química", "g.jpg", [vec(1)])
        self.ui.recargar_empleados()
        self.ui._aplicar_resultado(resultado(emp_id=self.emp_id))
        self.ui._aplicar_resultado(resultado(emp_id=otro))
        with database._db() as c:
            self.assertEqual(c.execute("SELECT COUNT(*) FROM asistencia").fetchone()[0], 2)

    def test_sin_rostro_limpia_lo_autocompletado_y_no_marca(self):
        self.ui._aplicar_resultado(resultado(emp_id=self.emp_id))
        self.ui._marcado_en.clear()
        self.ui._aplicar_resultado(resultado(emp_id=self.emp_id, rostro=False))  # identidad retenida, sin rostro
        with database._db() as c:
            self.assertEqual(c.execute("SELECT COUNT(*) FROM asistencia").fetchone()[0], 1)
        self.ui._aplicar_resultado(resultado(rostro=False))
        self.assertEqual(self.ui.input_nombre.text(), "")
        self.assertIsNone(self.ui.current_emp)

    def test_rostro_desconocido_permite_registrar_y_registra_con_vector(self):
        self.ui._aplicar_resultado(resultado(desconocido=True, vector=vec(5)))
        self.assertTrue(self.ui.btn_registrar.isEnabled())
        self.ui.input_nombre.setText("María Núñez")
        self.ui.input_edad.setText("41")
        with mock.patch.object(app_main.QMessageBox, "information") as info:
            self.ui.guardar_rostro()
        info.assert_called_once()
        emp = [e for e in database.get_todos_empleados() if e["nombre"] == "María Núñez"][0]
        self.assertEqual(database.contar_embeddings(emp["id"]), 1)
        self.assertTrue(os.path.exists(emp["foto_path"]))
        self.ui.worker.recargar_galeria.assert_called_once()
        self.assertIn(emp["id"], self.ui.empleados)

    def test_no_registra_sin_rostro_nitido_ni_a_alguien_ya_registrado(self):
        with mock.patch.object(app_main.QMessageBox, "warning") as aviso:
            self.ui.guardar_rostro()                                         # sin análisis
            self.ui._aplicar_resultado(resultado(calidad=False, aviso="Imagen borrosa"))
            self.ui.guardar_rostro()
            self.ui._aplicar_resultado(resultado(emp_id=self.emp_id, vector=vec(0)))
            self.ui.input_nombre.setText("Otro"); self.ui.input_edad.setText("30")
            self.ui.guardar_rostro()
        self.assertEqual(aviso.call_count, 3)
        self.assertEqual(len(database.get_todos_empleados()), 1)

    def test_edad_invalida(self):
        self.ui._aplicar_resultado(resultado(desconocido=True, vector=vec(5)))
        self.ui.input_nombre.setText("X"); self.ui.input_edad.setText("200")
        with mock.patch.object(app_main.QMessageBox, "warning") as aviso:
            self.ui.guardar_rostro()
        aviso.assert_called_once()
        self.assertEqual(len(database.get_todos_empleados()), 1)

    def test_dibuja_el_cuadro_con_nombre_sin_fallar(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        self.ui.show()
        self.app.processEvents()
        self.ui.update_video_frame(frame, resultado(emp_id=self.emp_id))
        self.ui.update_video_frame(frame, resultado(desconocido=True))
        self.ui.update_video_frame(frame, resultado(calidad=False, aviso="Acércate más a la cámara"))
        self.ui.update_video_frame(frame, None)
        self.assertFalse(self.ui.video_label_bio.pixmap().isNull())

    def test_muestra_los_fps_y_confirma_cada_frame_al_worker(self):
        self.ui.on_estado_camara("OK", "«Celular» en vivo")
        self.ui.on_fps_camara(9.4, 8.6)
        self.assertIn("9 fps", self.ui.lbl_cam_estado.text())
        self.ui.on_frame(np.zeros((48, 64, 3), np.uint8), None)
        self.ui.worker.frame_mostrado.assert_called_once()
        self.ui.on_estado_camara("RECONECTANDO", "sin señal")
        self.assertNotIn("fps", self.ui.lbl_cam_estado.text())

    def test_estado_de_camara_se_muestra(self):
        self.ui.on_estado_camara("RECONECTANDO", "«Celular» sin señal: reintentando…")
        self.assertIn("sin señal", self.ui.lbl_cam_estado.text())
        self.assertIn("sin señal", self.ui.video_label_bio.text())


if __name__ == "__main__":
    unittest.main()
