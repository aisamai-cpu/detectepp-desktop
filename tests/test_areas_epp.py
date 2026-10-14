"""Áreas de trabajo de limpieza: catálogo, cambio de área, prendas no detectables y reinicio de datos."""
import os
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

import config
from core import catalogo_epp, database, face_store
from core.epp import evaluar
from tests.helpers import BaseConCarpetaTemporal
from tools import reiniciar_datos
from vision.epp_engine import DeteccionEpp, EppEngine, normalizar_clase

try:
    from PyQt6.QtWidgets import QApplication
    import main as app_main
    from ui_empleado import DialogoNuevoEmpleado
    PYQT = True
except ImportError:
    PYQT = False


def emp(nombre="Ana Ruiz", area="Bodega y despacho"):
    return face_store.registrar_empleado(nombre, 30, area, "a.jpg", [np.full(128, 0.1, np.float32)])


class TestCatalogo(BaseConCarpetaTemporal):
    def test_catalogo_reemplaza_areas_de_prueba(self):
        database.init_db()
        areas = database.get_areas()
        self.assertEqual(areas, [a for a, _ in catalogo_epp.AREAS])
        for vieja in catalogo_epp.AREAS_DE_PRUEBA:
            self.assertNotIn(vieja, areas)

    def test_base_antigua_se_actualiza(self):
        config.ensure_dirs()
        conn = database.get_connection()
        database._migracion_v1(conn)
        database._migracion_v2(conn)
        conn.execute("PRAGMA user_version = 2")
        conn.commit()
        conn.close()
        database.init_db()
        self.assertEqual(database.get_epp_requeridos("Área Química"), [])
        self.assertEqual(database.get_epp_requeridos("Limpieza en altura"),
                         ["casco", "arnés", "guantes", "lentes", "botas"])

    def test_todas_las_prendas_del_catalogo_son_canonicas(self):
        for _, epp in catalogo_epp.AREAS:
            self.assertTrue(set(epp) <= set(catalogo_epp.PRENDAS), epp)

    def test_cambiar_area_del_empleado(self):
        database.init_db()
        e = emp()
        database.set_area_empleado(e, "Limpieza en altura")
        self.assertEqual(database.get_todos_empleados()[0]["area"], "Limpieza en altura")


class TestAlias(unittest.TestCase):
    def test_nombres_en_ingles_de_prendas_nuevas(self):
        casos = {"Respirator": "respirador", "Face Shield": "careta", "Apron": "mandil",
                 "Coverall": "traje", "Hairnet": "cofia", "Ear Muffs": "orejeras", "Harness": "arnés"}
        for clase, esperado in casos.items():
            self.assertEqual(normalizar_clase(clase), (esperado, True))
        self.assertEqual(normalizar_clase("NO-Harness"), ("arnés", False))

    def test_clases_en_espanol_del_modelo_de_colab(self):
        self.assertEqual(normalizar_clase("sin_casco"), ("casco", False))
        self.assertEqual(normalizar_clase("sin_mascarilla"), ("mascarilla", False))
        self.assertEqual(normalizar_clase("lentes"), ("lentes", True))
        self.assertEqual(normalizar_clase("persona"), ("persona", True))


class TestEquivalenciasYNoDetectables(unittest.TestCase):
    def test_respirador_cumple_mascarilla_y_traje_cumple_bata(self):
        ev = evaluar({"respirador", "traje"}, ["mascarilla", "bata", "mandil"])
        self.assertEqual(ev.faltantes, [])

    def test_careta_no_reemplaza_lentes(self):
        self.assertEqual(evaluar({"careta"}, ["lentes"]).faltantes, ["lentes"])

    def test_prendas_que_el_modelo_no_conoce_no_cuentan(self):
        ev = evaluar({"guantes"}, ["guantes", "cofia", "botas"], detectables={"guantes", "botas"})
        self.assertEqual(ev.requeridos, ["guantes", "botas"])
        self.assertEqual(ev.no_detectables, ["cofia"])
        self.assertEqual(ev.faltantes, ["botas"])
        self.assertEqual(ev.porcentaje, 50.0)

    def test_detectable_por_equivalente(self):
        ev = evaluar({"respirador"}, ["mascarilla"], detectables={"respirador"})
        self.assertEqual((ev.requeridos, ev.no_detectables, ev.aprobado), (["mascarilla"], [], True))

    def test_engine_informa_prendas_detectables(self):
        modelo = mock.Mock(names={0: "Hardhat", 1: "NO-Safety Vest", 2: "Person"})
        self.assertEqual(EppEngine(modelo=modelo).prendas_detectables, {"casco", "chaleco", "persona"})


class TestReiniciarDatos(BaseConCarpetaTemporal):
    def test_respalda_y_borra_todo_menos_areas(self):
        database.init_db()
        e = emp()
        database.registrar_marcacion(e, "ENTRADA")
        database.registrar_alerta("EPP_INCOMPLETO", "Bodega y despacho", "x")
        config.FOTOS_DIR.mkdir(parents=True, exist_ok=True)
        (config.FOTOS_DIR / "emp_1.jpg").write_bytes(b"x")
        (config.INTRUSOS_DIR / "d.jpg").write_bytes(b"y")

        r = reiniciar_datos.reiniciar()

        self.assertEqual(database.get_todos_empleados(), [])
        self.assertEqual(database.get_embeddings(), [])
        self.assertEqual(database.get_alertas(), [])
        self.assertEqual(len(database.get_areas()), len(catalogo_epp.AREAS))
        self.assertEqual(list(config.FOTOS_DIR.iterdir()), [])
        self.assertTrue((r["respaldo"] / "detectepp.db").exists())
        self.assertTrue((r["respaldo"] / "fotos_empleados" / "emp_1.jpg").exists())
        self.assertTrue((r["respaldo"] / "intrusos" / "d.jpg").exists())
        self.assertEqual(emp("Nuevo"), 1)   # los ids empiezan de nuevo

    def test_sin_confirmar_no_borra(self):
        database.init_db()
        emp()
        with mock.patch("builtins.print"):
            self.assertEqual(reiniciar_datos.main(entrada=lambda _: "no"), 1)
        self.assertEqual(len(database.get_todos_empleados()), 1)


@unittest.skipUnless(PYQT, "requiere PyQt6")
class TestUiAreas(BaseConCarpetaTemporal):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        super().setUp()
        database.init_db()
        self.emp_id = emp()
        self.ui = app_main.MainSystemWidget(window_control_parent=mock.Mock())
        self.ui.worker = mock.Mock()
        self.ui.recargar_empleados()
        self.ui.current_emp = self.ui.empleados[self.emp_id]
        self.ui.tabs.setCurrentIndex(1)

    def tearDown(self):
        self.ui.deleteLater()
        super().tearDown()

    def detectar(self, *clases):
        for _ in range(config.EPP_VENTANA):
            self.ui._aplicar_epp([DeteccionEpp(c, True, 0.9, (0, 0, 5, 5)) for c in clases])

    def test_selector_muestra_el_area_del_empleado(self):
        self.ui.actualizar_vista_epp()
        self.assertEqual(self.ui.combo_area_epp.currentText(), "Bodega y despacho")
        self.assertEqual(self.ui.combo_area_epp.count(), len(catalogo_epp.AREAS))

    def test_cambiar_area_guarda_y_reevalua(self):
        self.detectar("casco", "chaleco", "botas", "guantes")
        self.assertIn("COMPLETO", self.ui.lbl_status_epp.text())
        i = self.ui.combo_area_epp.findText("Limpieza en altura")
        self.ui.combo_area_epp.setCurrentIndex(i)
        self.ui.on_cambiar_area_epp(i)
        self.assertEqual(database.get_todos_empleados()[0]["area"], "Limpieza en altura")
        self.assertIn("INCOMPLETO", self.ui.lbl_status_epp.text())
        self.assertIn("arnés", self.ui.lbl_status_epp.text())

    def test_prendas_no_entrenadas_salen_en_gris_y_no_alertan(self):
        self.ui.on_prendas_epp({"casco", "chaleco", "botas"})    # el modelo no conoce guantes
        with mock.patch.object(app_main.AlertManager, "trigger_alert") as alerta:
            self.detectar("casco", "chaleco", "botas")
        self.assertIn("COMPLETO", self.ui.lbl_status_epp.text())
        self.assertNotIn("INCOMPLETO", self.ui.lbl_status_epp.text())
        self.assertIn("guantes (no detectable aún)", self.ui.lbl_epp_detalle.text())
        alerta.assert_not_called()

    def test_formulario_de_empleado_lee_las_areas_de_la_base(self):
        d = DialogoNuevoEmpleado()
        self.assertEqual([d.area.itemText(i) for i in range(d.area.count())], database.get_areas())
        d.deleteLater()


if __name__ == "__main__":
    unittest.main()
