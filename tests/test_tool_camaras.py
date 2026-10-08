"""Pruebas de la herramienta de consola tools/probar_camaras.py (sin abrir cámaras)."""
import io
from contextlib import redirect_stdout

from core import camaras
from tests.helpers import BaseConCarpetaTemporal
from tools import probar_camaras


def ejecutar(*argv):
    salida = io.StringIO()
    with redirect_stdout(salida):
        codigo = probar_camaras.main(list(argv))
    return codigo, salida.getvalue()


class TestHerramientaCamaras(BaseConCarpetaTemporal):
    def test_listar_marca_la_activa(self):
        codigo, texto = ejecutar("listar")
        self.assertEqual(codigo, 0)
        self.assertIn("* cam-integrada", texto)

    def test_agregar_ip_oculta_la_clave_y_se_puede_activar(self):
        codigo, texto = ejecutar("agregar-url", "rtsp://admin:miclave@192.168.1.20/s", "--nombre", "Pasillo", "--activar")
        self.assertEqual(codigo, 0)
        self.assertNotIn("miclave", texto)
        _, listado = ejecutar("listar")
        self.assertNotIn("miclave", listado)
        self.assertEqual(camaras.activa().nombre, "Pasillo")

    def test_errores_dan_mensaje_y_codigo_1(self):
        codigo, texto = ejecutar("agregar-url", "192.168.1.20/video", "--nombre", "Mala")
        self.assertEqual(codigo, 1)
        self.assertIn("http://", texto)
        codigo, texto = ejecutar("eliminar", "cam-integrada")
        self.assertEqual(codigo, 1)
        self.assertIn("al menos una", texto)
        codigo, _ = ejecutar("activar", "no-existe")
        self.assertEqual(codigo, 1)

    def test_agregar_usb_activar_y_eliminar(self):
        ejecutar("agregar-usb", "1", "--nombre", "Externa")
        externa = [c for c in camaras.listar() if c.nombre == "Externa"][0]
        self.assertEqual(ejecutar("activar", externa.id)[0], 0)
        self.assertEqual(camaras.activa().id, externa.id)
        self.assertEqual(ejecutar("eliminar", externa.id)[0], 0)
        self.assertEqual(camaras.activa().id, "cam-integrada")
