"""Pruebas del registro de cámaras (data/camaras.json)."""
import json

import config
from core import camaras
from tests.helpers import BaseConCarpetaTemporal


class TestRegistroCamaras(BaseConCarpetaTemporal):
    def test_primera_vez_crea_la_camara_integrada(self):
        lista = camaras.listar()
        self.assertEqual(len(lista), 1)
        self.assertEqual((lista[0].tipo, lista[0].valor), ("usb", "0"))
        self.assertEqual(camaras.activa().id, lista[0].id)
        self.assertTrue(config.CAMARAS_PATH.exists())

    def test_agregar_usb_normaliza_el_indice(self):
        cam = camaras.agregar("Externa", "usb", " 01 ")
        self.assertEqual(cam.valor, "1")
        self.assertEqual(len(camaras.listar()), 2)

    def test_agregar_ip_y_persistencia(self):
        cam = camaras.agregar("Celular Ana", "url", "http://192.168.1.50:8080/video", activar_ahora=True)
        self.assertEqual(camaras.activa().id, cam.id)
        self.assertEqual(camaras.obtener(cam.id).valor, "http://192.168.1.50:8080/video")

    def test_validaciones(self):
        casos = [
            ("", "usb", "0"),                      # sin nombre
            ("X", "bluetooth", "0"),               # tipo inválido
            ("X", "usb", "abc"),                   # índice no numérico
            ("X", "url", "192.168.1.5/video"),     # sin protocolo
            ("X", "archivo", "   "),               # sin ruta
            ("N" * 61, "usb", "0"),                # nombre muy largo
        ]
        for nombre, tipo, valor in casos:
            with self.subTest(nombre=nombre[:10], tipo=tipo, valor=valor):
                with self.assertRaises(ValueError):
                    camaras.agregar(nombre, tipo, valor)
        with self.assertRaises(ValueError):
            camaras.agregar("X", "usb", "1", ancho=640)  # ancho sin alto
        self.assertEqual(len(camaras.listar()), 1)       # nada se guardó

    def test_activar_inexistente(self):
        with self.assertRaises(KeyError):
            camaras.activar("no-existe")

    def test_eliminar_la_activa_pasa_a_otra(self):
        otra = camaras.agregar("Otra", "usb", "1", activar_ahora=True)
        camaras.eliminar(otra.id)
        self.assertEqual(camaras.activa().id, "cam-integrada")

    def test_no_se_puede_eliminar_la_ultima(self):
        with self.assertRaises(ValueError):
            camaras.eliminar("cam-integrada")
        with self.assertRaises(KeyError):
            camaras.eliminar("no-existe")

    def test_credenciales_ocultas_en_pantalla_pero_guardadas(self):
        url = "rtsp://admin:secreto123@192.168.1.5:554/stream1"
        cam = camaras.agregar("Pasillo", "url", url)
        self.assertNotIn("secreto123", cam.valor_visible)
        self.assertNotIn("secreto123", cam.resumen())
        self.assertIn("192.168.1.5", cam.resumen())
        self.assertEqual(camaras.obtener(cam.id).valor, url)  # la URL real sí se conserva para conectar

    def test_archivo_corrupto_se_recrea_con_copia(self):
        config.ensure_dirs()
        config.CAMARAS_PATH.write_text("{esto no es json", encoding="utf-8")
        self.assertEqual(len(camaras.listar()), 1)
        copias = list(config.CAMARAS_PATH.parent.glob("camaras.corrupto-*.json"))
        self.assertEqual(len(copias), 1)

    def test_camara_activa_inexistente_se_repara(self):
        camaras.listar()
        datos = json.loads(config.CAMARAS_PATH.read_text(encoding="utf-8"))
        datos["activa"] = "fantasma"
        config.CAMARAS_PATH.write_text(json.dumps(datos), encoding="utf-8")
        self.assertEqual(camaras.activa().id, "cam-integrada")

    def test_ignora_campos_desconocidos_de_versiones_futuras(self):
        camaras.listar()
        datos = json.loads(config.CAMARAS_PATH.read_text(encoding="utf-8"))
        datos["camaras"][0]["area"] = "Área Química"  # campo que aún no existe
        config.CAMARAS_PATH.write_text(json.dumps(datos), encoding="utf-8")
        self.assertEqual(camaras.activa().nombre, "Cámara integrada")

    def test_ocultar_credenciales_sin_credenciales(self):
        self.assertEqual(camaras.ocultar_credenciales("http://192.168.1.5:8080/video"), "http://192.168.1.5:8080/video")
        self.assertEqual(camaras.ocultar_credenciales("0"), "0")

class TestRotacion(BaseConCarpetaTemporal):
    def test_por_defecto_sin_giro_y_archivos_viejos_siguen_funcionando(self):
        self.assertEqual(camaras.activa().rotacion, 0)  # camaras.json sin el campo "rotacion"

    def test_agregar_con_rotacion_y_resumen(self):
        cam = camaras.agregar("Celular", "url", "http://192.168.1.5:8080/video", rotacion=90)
        self.assertEqual(camaras.obtener(cam.id).rotacion, 90)
        self.assertIn("giro 90", cam.resumen())
        self.assertNotIn("giro", camaras.obtener("cam-integrada").resumen())

    def test_rotar_persiste(self):
        camaras.rotar("cam-integrada", 180)
        self.assertEqual(camaras.obtener("cam-integrada").rotacion, 180)

    def test_rotacion_invalida(self):
        with self.assertRaises(ValueError):
            camaras.rotar("cam-integrada", 45)
        with self.assertRaises(ValueError):
            camaras.agregar("X", "usb", "1", rotacion=91)
        with self.assertRaises(KeyError):
            camaras.rotar("no-existe", 90)
        self.assertEqual(len(camaras.listar()), 1)