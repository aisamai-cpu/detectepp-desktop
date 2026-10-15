"""Eventos de Monitoreo: BD v4, foto difuminada, duplicados, Atendido, retención, zonas, reporte y reinicio."""
import datetime
from unittest import mock

import cv2
import numpy as np

import config
from core import database, reporte, zonas
from core.eventos import GestorEventos, difuminar_cabezas
from core.riesgos import CRITICO, ERGONOMIA, Evento, FotoEvento
from tests.esqueletos import cuerpo
from tests.helpers import BaseConCarpetaTemporal
from tools import reiniciar_datos


class Reloj:
    def __init__(self):
        self.t = datetime.datetime(2026, 10, 14, 9, 0, 0)

    def __call__(self):
        return self.t


def evento(tipo="CAIDA", grupo=CRITICO, persona="P1", sonido=True):
    return Evento(tipo, grupo, persona, 0.0, f"{tipo} ({persona})", sonido, (200, 100, 400, 500))


def foto():
    frame = np.random.default_rng(0).integers(0, 255, (720, 1280, 3), dtype=np.uint8)
    return FotoEvento(frame, (cuerpo(cx=300, cy=300), cuerpo(cx=900, cy=300)))


class TestBaseV4(BaseConCarpetaTemporal):
    def test_migracion_desde_v3(self):
        config.ensure_dirs()
        conn = database.get_connection()
        for v in (1, 2, 3):
            database.MIGRACIONES[v](conn)
        conn.execute("PRAGMA user_version = 3")
        conn.commit()
        conn.close()
        database.init_db()
        self.assertEqual(database.eventos_dia("2026-10-14"), [])
        c = database.get_connection()
        try:
            self.assertEqual(c.execute("PRAGMA user_version").fetchone()[0], 4)
        finally:
            c.close()      # en Windows un archivo abierto no se puede borrar al terminar la prueba

    def test_borrar_registros_con_eventos_enlazados_a_alertas(self):
        database.init_db()
        alerta = database.registrar_alerta("CAIDA", "Cam", "x")
        database.registrar_evento("CAIDA", CRITICO, "Cam", "P1", "x", alerta_id=alerta)
        filas = database.borrar_registros()      # no debe fallar por la clave foránea
        self.assertEqual((filas["eventos_riesgo"], filas["alertas"]), (1, 1))


class TestGestor(BaseConCarpetaTemporal):
    def setUp(self):
        super().setUp()
        database.init_db()
        self.reloj = Reloj()
        self.g = GestorEventos(reloj=self.reloj)

    def test_registra_con_foto_alerta_y_sonido(self):
        with mock.patch("core.eventos.AlertManager.trigger_alert", wraps=self._alerta) as alerta:
            r = self.g.registrar(evento(), foto(), "Cámara celular")
        alerta.assert_called_once_with("CAIDA", "Cámara celular", "CAIDA (P1)", sonido=True)
        fila = database.get_evento(r.id)
        self.assertEqual((fila["tipo"], fila["camara"]), ("CAIDA", "Cámara celular"))
        self.assertEqual(database.get_alertas(1)[0]["id"], fila["alerta_id"])
        self.assertTrue(r.foto_path.endswith(".jpg"))
        self.assertIn("2026-10-14", r.foto_path)
        self.assertEqual(cv2.imread(r.foto_path).shape, (720, 1280, 3))

    @staticmethod
    def _alerta(tipo, area, detalle, sonido=True):
        return database.registrar_alerta(tipo, area, detalle)   # como AlertManager, sin sonido ni log

    def test_emergencia_duplicada_sin_atender_se_descarta(self):
        self.assertIsNotNone(self.g.registrar(evento(persona="P1"), None, "Cam"))
        self.reloj.t += datetime.timedelta(seconds=20)
        self.assertIsNone(self.g.registrar(evento(persona="P4"), None, "Cam"))   # mismo caído con otro id

    def test_tras_atender_una_nueva_emergencia_si_se_registra(self):
        r = self.g.registrar(evento(), None, "Cam")
        self.reloj.t += datetime.timedelta(seconds=30)
        self.assertIsNotNone(self.g.atender(r.id))
        self.assertIsNotNone(database.get_evento(r.id)["atendido_en"])
        self.assertIsNotNone(self.g.registrar(evento(), None, "Cam"))

    def test_inmovil_tras_caida_abierta_queda_enlazado_y_sin_sonido_extra(self):
        caida = self.g.registrar(evento("CAIDA"), None, "Cam")
        with mock.patch("core.eventos.AlertManager.trigger_alert", return_value=None) as alerta:
            inm = self.g.registrar(evento("PERSONA_INMOVIL"), None, "Cam")
        self.assertFalse(alerta.call_args.kwargs["sonido"])
        self.assertFalse(inm.sonido)
        self.assertEqual(database.get_evento(inm.id)["relacionado_id"], caida.id)

    def test_abiertos_para_recargar_al_iniciar(self):
        self.g.registrar(evento("SOS"), None, "Cam")
        self.g.registrar(evento("BRAZOS_ELEVADOS", ERGONOMIA), None, "Cam")
        self.assertEqual([a["tipo"] for a in self.g.abiertos()], ["SOS"])

    def test_retencion_borra_fotos_viejas_y_limpia_rutas(self):
        viejo = self.g.registrar(evento(), foto(), "Cam")
        self.g.atender(viejo.id)
        self.reloj.t += datetime.timedelta(days=config.MON_RETENCION_DIAS + 1)
        nuevo = self.g.registrar(evento(), foto(), "Cam")
        self.assertEqual(self.g.limpiar_retencion(), 1)
        self.assertIsNone(database.get_evento(viejo.id)["foto_path"])
        self.assertIsNotNone(database.get_evento(nuevo.id)["foto_path"])


class TestDifuminado(BaseConCarpetaTemporal):
    def test_cabeza_difuminada_con_y_sin_puntos(self):
        rng = np.random.default_rng(1)
        img = rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)
        con_puntos = cuerpo(cx=300, cy=400)
        sin_cabeza = cuerpo(cx=900, cy=400, ocultar=(0, 1, 2, 3, 4))
        salida = difuminar_cabezas(img.copy(), [con_puntos, sin_cabeza])

        def varianza(im, x, y):
            return float(im[y - 10:y + 10, x - 10:x + 10].astype(float).var())
        nariz = con_puntos.puntos[0, :2].astype(int)
        self.assertLess(varianza(salida, *nariz), varianza(img, *nariz) / 4)
        x1, y1, x2, y2 = (int(v) for v in sin_cabeza.caja)
        cx, cy = (x1 + x2) // 2, y1 + int(0.15 * (y2 - y1))
        self.assertLess(varianza(salida, cx, cy), varianza(img, cx, cy) / 4)
        # los pies no se tocan
        self.assertEqual(varianza(salida, 300, 560), varianza(img, 300, 560))


class TestZonas(BaseConCarpetaTemporal):
    def test_agregar_listar_borrar(self):
        z = zonas.agregar("cam-1", 0.8, 0.9, 0.2, 0.1)
        self.assertEqual(z, ("Zona 1", 0.2, 0.1, 0.8, 0.9))
        zonas.agregar("cam-1", 0, 0, 0.1, 0.1)
        self.assertEqual([n for n, *_ in zonas.de_camara("cam-1")], ["Zona 1", "Zona 2"])
        self.assertEqual(zonas.de_camara("otra"), [])
        self.assertEqual(zonas.borrar("cam-1"), 2)
        self.assertEqual(zonas.de_camara("cam-1"), [])


class TestReporte(BaseConCarpetaTemporal):
    def test_reporte_del_dia(self):
        database.init_db()
        from core import face_store
        emp = face_store.registrar_empleado("Ana <Ruiz>", 30, "Bodega y despacho", "a.jpg",
                                            [np.full(128, 0.1, np.float32)])
        hoy = datetime.date.today().strftime("%Y-%m-%d")
        database.registrar_marcacion(emp, "ENTRADA")
        database.registrar_alerta("EPP_INCOMPLETO", "Bodega y despacho",
                                  "Ana <Ruiz> no cumple el protocolo de EPP (50 %). Falta: guantes, casco")
        g = GestorEventos()
        r = g.registrar(evento("CAIDA"), None, "Cámara celular")
        g.atender(r.id)
        g.registrar(evento("BRAZOS_ELEVADOS", ERGONOMIA), None, "Cámara celular")
        ruta = reporte.generar(hoy)
        texto = ruta.read_text(encoding="utf-8")
        self.assertIn("Ana &lt;Ruiz&gt;", texto)          # escapado
        self.assertIn("guantes, casco", texto)
        self.assertIn("Caída", texto)
        self.assertIn("Brazos elevados", texto)
        self.assertIn("0 %", texto)                         # 1 entrada, con alerta de EPP


class TestReinicio(BaseConCarpetaTemporal):
    def test_respalda_carpetas_de_eventos(self):
        database.init_db()
        GestorEventos().registrar(evento(), foto(), "Cam")
        r = reiniciar_datos.reiniciar()
        self.assertEqual(database.eventos_dia(datetime.date.today().strftime("%Y-%m-%d")), [])
        self.assertTrue(any((r["respaldo"] / "eventos").rglob("*.jpg")))
