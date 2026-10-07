"""Pruebas de la Fase 0 (solo biblioteca estándar, sin pytest).

Ejecutar desde la carpeta del proyecto:
    python -m unittest discover -s tests -t . -v
"""
import sqlite3
import tempfile
from contextlib import closing
import unittest
from pathlib import Path
from unittest import mock

import config
from core import alerts, database, sound


class BaseConCarpetaTemporal(unittest.TestCase):
    """Redirige todas las rutas de config a una carpeta temporal."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        self._parches = [
            mock.patch.object(config, "DATA_DIR", base / "data"),
            mock.patch.object(config, "DB_PATH", base / "data" / "detectepp.db"),
            mock.patch.object(config, "FOTOS_DIR", base / "data" / "fotos"),
            mock.patch.object(config, "LOGS_DIR", base / "data" / "logs"),
            mock.patch.object(config, "ALERT_LOG_PATH", base / "data" / "logs" / "alert.log"),
            mock.patch.object(config, "SONIDO_ALERTA_ACTIVO", False),
        ]
        for p in self._parches:
            p.start()

    def tearDown(self):
        for p in self._parches:
            p.stop()
        self._tmp.cleanup()


class TestBaseDeDatos(BaseConCarpetaTemporal):
    def test_carpeta_vacia_crea_todo(self):
        database.init_db()
        with closing(database.get_connection()) as conn:
            tablas = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            version = conn.execute("PRAGMA user_version").fetchone()[0]
        self.assertTrue({"empleados", "asistencia", "areas_epp", "alertas"} <= tablas)
        self.assertEqual(version, database.VERSION_ACTUAL)
        self.assertTrue(config.FOTOS_DIR.is_dir())

    def test_init_db_es_idempotente(self):
        database.init_db()
        database.add_empleado("Ana", 30, "Área Química", "x.jpg")
        database.init_db()
        self.assertEqual(len(database.get_todos_empleados()), 1)

    def test_migra_base_antigua_sin_perder_datos(self):
        config.ensure_dirs()
        viejo = sqlite3.connect(str(config.DB_PATH))
        viejo.executescript("""
            CREATE TABLE empleados (id INTEGER PRIMARY KEY AUTOINCREMENT, nombre TEXT NOT NULL,
                edad INTEGER NOT NULL, area TEXT NOT NULL, foto_path TEXT NOT NULL);
            CREATE TABLE asistencia (id INTEGER PRIMARY KEY AUTOINCREMENT, empleado_id INTEGER NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, tipo_marcacion TEXT NOT NULL,
                epp_porcentaje REAL NOT NULL, estado TEXT NOT NULL);
            INSERT INTO empleados (nombre, edad, area, foto_path) VALUES ('Luis', 40, 'Área Maquinaria', 'a.jpg');
        """)
        viejo.commit()
        viejo.close()

        database.init_db()

        empleados = database.get_todos_empleados()
        self.assertEqual([e["nombre"] for e in empleados], ["Luis"])
        self.assertIn("fecha_registro", empleados[0])
        self.assertEqual(database.get_alertas(), [])  # la tabla alertas ya existe

    def test_marcacion_no_inventa_epp(self):
        database.init_db()
        emp = database.add_empleado("Ana", 30, "Área Química", "x.jpg")
        database.registrar_marcacion(emp, "ENTRADA")
        with closing(database.get_connection()) as conn:
            fila = conn.execute("SELECT epp_porcentaje, estado FROM asistencia").fetchone()
        self.assertEqual((fila["epp_porcentaje"], fila["estado"]), (0.0, "PENDIENTE"))

    def test_tipo_marcacion_invalido(self):
        database.init_db()
        with self.assertRaises(ValueError):
            database.registrar_marcacion(1, "ALMUERZO")


class TestAlertas(BaseConCarpetaTemporal):
    def test_alerta_guarda_en_bd_y_log(self):
        database.init_db()
        alerta_id = alerts.AlertManager.trigger_alert("EPP", "Área Química", "Falta mascarilla")
        self.assertIsNotNone(alerta_id)
        self.assertEqual(database.get_alertas()[0]["detalle"], "Falta mascarilla")
        self.assertIn("Falta mascarilla", config.ALERT_LOG_PATH.read_text(encoding="utf-8"))

    def test_alerta_no_explota_si_falla_la_bd(self):
        # Sin init_db: la tabla no existe. Debe devolver None y seguir.
        config.ensure_dirs()
        self.assertIsNone(alerts.AlertManager.trigger_alert("SOS", "Área X", "prueba"))
        self.assertTrue(config.ALERT_LOG_PATH.exists())


class TestSonido(BaseConCarpetaTemporal):
    def test_desactivado_no_suena(self):
        self.assertFalse(sound.play_alert())

    def test_nunca_lanza_excepcion(self):
        with mock.patch.object(config, "SONIDO_ALERTA_ACTIVO", True), \
             mock.patch.object(sound, "_reproducir", side_effect=RuntimeError("sin audio")):
            self.assertTrue(sound.play_alert(bloquear=True))
            # el candado debe quedar libre para la siguiente alerta
            self.assertTrue(sound.play_alert(bloquear=True))

    def test_no_superpone_alertas(self):
        with mock.patch.object(config, "SONIDO_ALERTA_ACTIVO", True):
            sound._en_curso.acquire()
            try:
                self.assertFalse(sound.play_alert(bloquear=True))
            finally:
                sound._en_curso.release()

    def test_selecciona_backend_por_plataforma(self):
        for plataforma, esperado in (("win32", "_windows"), ("darwin", "_macos"), ("linux", "_linux")):
            with mock.patch.object(sys_platform(), "platform", plataforma), \
                 mock.patch.object(sound, esperado) as backend:
                sound._reproducir(2)
                backend.assert_called_once_with(2)


def sys_platform():
    import sys
    return sys


if __name__ == "__main__":
    unittest.main()
