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
from tests.test_registro import persona, variante
from core.intrusos import DetectorIntrusos
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
        self.assertIn("José Peña", self.ui.lbl_bio_status.text())
        self.assertEqual(self.ui.current_emp["id"], self.emp_id)
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

    def test_sin_rostro_no_marca_y_olvida_al_empleado(self):
        self.ui._aplicar_resultado(resultado(emp_id=self.emp_id))
        self.ui._marcado_en.clear()
        self.ui._aplicar_resultado(resultado(emp_id=self.emp_id, rostro=False))  # identidad retenida, sin rostro
        with database._db() as c:
            self.assertEqual(c.execute("SELECT COUNT(*) FROM asistencia").fetchone()[0], 1)
        self.ui._aplicar_resultado(resultado(rostro=False))
        self.assertIsNone(self.ui.current_emp)

    def test_ya_no_existe_el_boton_ni_el_formulario_de_registro(self):
        for nombre in ("btn_registrar", "input_nombre", "input_edad", "combo_bio_area", "barra_registro", "sesion"):
            self.assertFalse(hasattr(self.ui, nombre), nombre)

    def test_el_panel_no_cambia_de_tamano_al_cambiar_los_mensajes(self):
        # causa del "temblor" del scroll: el contenido cambiaba de alto con cada mensaje
        self.ui.show()
        app_main.QApplication.processEvents()
        alto = self.ui.lbl_bio_status.height()
        contenedor = self.ui.lbl_bio_status.parentWidget()
        alto_panel = contenedor.sizeHint().height()
        for i in range(8):
            self.ui._aplicar_resultado(resultado(desconocido=bool(i % 2), calidad=bool(i % 3), aviso="Imagen borrosa: quédate quieto y mejora la luz"))
            app_main.QApplication.processEvents()
            self.assertEqual(self.ui.lbl_bio_status.height(), alto)
            self.assertEqual(contenedor.sizeHint().height(), alto_panel)

    # ---------------- tarjeta de ingreso del empleado reconocido ----------------
    def test_reconocer_a_un_empleado_muestra_su_captura_datos_y_hora(self):
        self.assertTrue(self.ui.card_ingreso.isHidden())
        self.ui._aplicar_resultado(resultado(emp_id=self.emp_id))
        self.assertFalse(self.ui.card_ingreso.isHidden())
        self.assertFalse(self.ui.lbl_ingreso_foto.pixmap().isNull())
        texto = self.ui.lbl_ingreso_datos.text()
        for esperado in ("José Peña", "33 años", "Área Química", "Hora de ingreso"):
            self.assertIn(esperado, texto)
        with database._db() as c:
            hora = c.execute("SELECT timestamp FROM asistencia").fetchone()[0]
        self.assertIn(hora.split(" ")[1], texto)                 # la hora mostrada es la guardada
        self.assertIn("INGRESO", self.ui.lbl_ingreso_titulo.text())

    def test_en_modo_salida_la_tarjeta_dice_hora_de_salida(self):
        self.ui.radio_salida.setChecked(True)
        self.ui._aplicar_resultado(resultado(emp_id=self.emp_id))
        self.assertIn("Hora de salida", self.ui.lbl_ingreso_datos.text())
        self.assertIn("SALIDA", self.ui.lbl_ingreso_titulo.text())

    def test_sin_rostro_presente_no_se_muestra_tarjeta(self):
        self.ui._aplicar_resultado(resultado(emp_id=self.emp_id, rostro=False))
        self.assertTrue(self.ui.card_ingreso.isHidden())

    def test_la_alarma_de_desconocido_es_a_los_2_segundos(self):
        self.assertEqual(config.INTRUSO_SEGUNDOS, 2.0)

    # ---------------- rostro desconocido / posible intruso ----------------
    def preparar_reloj(self):
        class Reloj:
            t = 500.0
            def __call__(self):
                return self.t
        self.reloj = Reloj()
        self.ui.intrusos = DetectorIntrusos(self.reloj)

    def desconocido_durante(self, segundos, pasos=6):
        for _ in range(pasos):
            self.ui._aplicar_resultado(resultado(desconocido=True, vector=vec(5)))
            self.reloj.t += segundos / pasos
        self.ui._aplicar_resultado(resultado(desconocido=True, vector=vec(5)))

    def alertas(self):
        return [a["tipo"] for a in database.get_alertas()]

    def test_desconocido_3_segundos_muestra_tarjeta_foto_alerta_y_sonido(self):
        self.preparar_reloj()
        with mock.patch("core.alerts.play_alert") as sonido:
            self.desconocido_durante(config.INTRUSO_SEGUNDOS - 1)
            self.assertTrue(self.ui.card_intruso.isHidden())
            sonido.assert_not_called()
            self.desconocido_durante(2)
        self.assertFalse(self.ui.card_intruso.isHidden())
        self.assertIn("ROSTRO NUEVO IDENTIFICADO", self.ui.lbl_intruso_titulo.text())
        self.assertFalse(self.ui.lbl_intruso_foto.pixmap().isNull())
        sonido.assert_called_once()
        self.assertEqual(self.alertas(), ["ROSTRO_DESCONOCIDO"])
        fotos = list(config.INTRUSOS_DIR.glob("desconocido_*.jpg"))
        self.assertEqual(len(fotos), 1)
        self.assertIn(fotos[0].name, database.get_alertas()[0]["detalle"])

    def test_no_repite_la_alerta_mientras_siga_pendiente(self):
        self.preparar_reloj()
        with mock.patch("core.alerts.play_alert") as sonido:
            self.desconocido_durante(config.INTRUSO_SEGUNDOS + 1)
            self.desconocido_durante(30)
        sonido.assert_called_once()
        self.assertEqual(self.alertas(), ["ROSTRO_DESCONOCIDO"])

    def test_un_empleado_conocido_o_el_cambio_de_pestana_reinician_el_conteo(self):
        self.preparar_reloj()
        self.desconocido_durante(config.INTRUSO_SEGUNDOS - 1)
        self.ui._aplicar_resultado(resultado(emp_id=self.emp_id))
        self.desconocido_durante(config.INTRUSO_SEGUNDOS - 1)
        self.ui.tabs.setCurrentIndex(1)
        self.ui.tabs.setCurrentIndex(0)
        self.desconocido_durante(config.INTRUSO_SEGUNDOS - 1)
        self.assertTrue(self.ui.card_intruso.isHidden())
        self.assertEqual(self.alertas(), [])

    def test_reportar_registra_el_reporte_sin_sonido_y_oculta_la_tarjeta(self):
        self.preparar_reloj()
        with mock.patch("core.alerts.play_alert") as sonido:
            self.desconocido_durante(config.INTRUSO_SEGUNDOS + 1)
            sonido.reset_mock()
            self.ui.reportar_intruso()
        sonido.assert_not_called()
        self.assertTrue(self.ui.card_intruso.isHidden())
        self.assertEqual(self.alertas(), ["INTRUSO_REPORTADO", "ROSTRO_DESCONOCIDO"])
        self.assertEqual(len(list(config.INTRUSOS_DIR.glob("desconocido_*.jpg"))), 1)   # la evidencia se conserva
        self.assertIn("ajena", self.ui.lbl_marcacion_info.text())

    def test_tras_resolver_hay_enfriamiento_y_luego_puede_volver_a_alertar(self):
        self.preparar_reloj()
        self.desconocido_durante(config.INTRUSO_SEGUNDOS + 1)
        self.ui.reportar_intruso()
        self.desconocido_durante(config.INTRUSO_SEGUNDOS + 1)
        self.assertTrue(self.ui.card_intruso.isHidden())
        self.reloj.t += config.INTRUSO_ENFRIAMIENTO_S
        self.desconocido_durante(config.INTRUSO_SEGUNDOS + 1)
        self.assertFalse(self.ui.card_intruso.isHidden())
        self.assertEqual(self.alertas().count("ROSTRO_DESCONOCIDO"), 2)

    # ---------------- agregar a la base de datos desde la tarjeta (sin que la persona siga en cámara) ----------------
    def desconocido_con_muestras(self, base):
        """Un desconocido frente a la cámara el tiempo necesario; cada análisis trae un vector distinto de la misma persona."""
        for k in range(8):
            self.ui._aplicar_resultado(resultado(desconocido=True, vector=variante(base, k)))
            self.reloj.t += (config.INTRUSO_SEGUNDOS + 1) / 7

    def dialogo_falso(self, nombre="Rosa Vera", edad=35, area="Área Química", acepta=True):
        d = mock.Mock()
        d.exec.return_value = app_main.QDialog.DialogCode.Accepted if acepta else app_main.QDialog.DialogCode.Rejected
        d.datos.return_value = (nombre, edad, area)
        return d

    def test_agregar_registra_al_empleado_con_las_capturas_y_la_persona_ya_se_pudo_ir(self):
        self.preparar_reloj()
        self.desconocido_con_muestras(persona(9))
        self.ui._aplicar_resultado(resultado(rostro=False))                    # la persona YA se fue de la cámara
        self.assertFalse(self.ui.card_intruso.isHidden())
        with mock.patch.object(app_main, "DialogoNuevoEmpleado", return_value=self.dialogo_falso()), \
                mock.patch.object(app_main.QMessageBox, "information") as info:
            self.ui.agregar_intruso()
        info.assert_called_once()
        emp = [e for e in database.get_todos_empleados() if e["nombre"] == "Rosa Vera"][0]
        self.assertEqual(database.contar_embeddings(emp["id"]), config.REGISTRO_MUESTRAS)
        self.assertTrue(os.path.exists(emp["foto_path"]))
        self.ui.worker.recargar_galeria.assert_called_once()
        self.assertTrue(self.ui.card_intruso.isHidden())
        self.assertEqual(list(config.INTRUSOS_DIR.glob("desconocido_*.jpg")), [])    # ya es empleado: se borra la foto
        self.assertFalse(self.ui.card_ingreso.isHidden())                             # y se muestra su ingreso
        self.assertIn("Rosa Vera", self.ui.lbl_ingreso_datos.text())

    def test_cancelar_el_dialogo_no_borra_la_alerta_pendiente(self):
        self.preparar_reloj()
        self.desconocido_con_muestras(persona(9))
        with mock.patch.object(app_main, "DialogoNuevoEmpleado", return_value=self.dialogo_falso(acepta=False)):
            self.ui.agregar_intruso()
        self.assertFalse(self.ui.card_intruso.isHidden())
        self.assertEqual(len(database.get_todos_empleados()), 1)
        self.assertEqual(len(list(config.INTRUSOS_DIR.glob("desconocido_*.jpg"))), 1)

    def test_agregar_a_alguien_ya_registrado_se_rechaza_con_su_nombre(self):
        base = persona(9)
        face_store.registrar_empleado("Pedro Gil", 50, "Área Química", "p.jpg", [base])
        self.ui.recargar_empleados()
        self.preparar_reloj()
        self.desconocido_con_muestras(base)
        with mock.patch.object(app_main.QMessageBox, "warning") as aviso:
            self.ui.agregar_intruso()
        self.assertIn("Pedro Gil", aviso.call_args[0][2])
        self.assertEqual(len(database.get_todos_empleados()), 2)
        self.assertTrue(self.ui.card_intruso.isHidden())

    def test_capturas_de_personas_distintas_no_se_registran(self):
        self.preparar_reloj()
        otra = persona(77)
        for k in range(8):
            base = persona(9) if k < 2 else persona(60 + k)
            self.ui._aplicar_resultado(resultado(desconocido=True, vector=base))
            self.reloj.t += (config.INTRUSO_SEGUNDOS + 1) / 7
        self.ui.agregar_intruso()
        self.assertEqual(len(database.get_todos_empleados()), 1)
        self.assertIn("una sola persona", self.ui.lbl_intruso_info.text())
        self.assertFalse(self.ui.card_intruso.isHidden())

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
