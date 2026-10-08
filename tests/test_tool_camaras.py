"""Pruebas de la herramienta de consola tools/probar_camaras.py (sin abrir cámaras)."""
import io
from contextlib import redirect_stderr, redirect_stdout

import numpy as np

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

class TestRotacionEnLaHerramienta(BaseConCarpetaTemporal):
    def test_agregar_con_rotar_y_comando_rotar(self):
        codigo, texto = ejecutar("agregar-url", "http://192.168.1.50:8080/video", "--nombre", "Celular", "--rotar", "90")
        self.assertEqual(codigo, 0)
        self.assertIn("giro 90", texto)
        cam = [c for c in camaras.listar() if c.nombre == "Celular"][0]
        self.assertEqual(ejecutar("rotar", cam.id, "270")[0], 0)
        self.assertEqual(camaras.obtener(cam.id).rotacion, 270)

    def test_rotacion_invalida_la_rechaza_argparse(self):
        with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
            probar_camaras.main(["rotar", "cam-integrada", "45"])

    def test_rotar_camara_inexistente(self):
        self.assertEqual(ejecutar("rotar", "no-existe", "90")[0], 1)


class TestAjustarAPantalla(BaseConCarpetaTemporal):
    def test_vertical_1080x1920_cabe_completo_y_sin_deformarse(self):
        img = np.zeros((1920, 1080, 3), np.uint8)
        v = probar_camaras.ajustar_a_pantalla(img)
        self.assertLessEqual(v.shape[0], probar_camaras.MAX_ALTO_VENTANA)
        self.assertLessEqual(v.shape[1], probar_camaras.MAX_ANCHO_VENTANA)
        self.assertAlmostEqual(v.shape[1] / v.shape[0], 1080 / 1920, places=2)

    def test_imagen_pequena_no_se_agranda_y_es_copia(self):
        img = np.zeros((300, 400, 3), np.uint8)
        v = probar_camaras.ajustar_a_pantalla(img)
        self.assertEqual(v.shape, img.shape)
        v[0, 0] = 255
        self.assertEqual(img[0, 0, 0], 0)  # dibujar sobre la vista no daña el frame original