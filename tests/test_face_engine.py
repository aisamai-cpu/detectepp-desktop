"""Pruebas del motor de rostros SIN necesitar los modelos .onnx (usan objetos falsos).

    python -m unittest discover -s tests -t . -v
"""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import config
from vision import face_engine as fe


def fila(x, y, w, h, score=0.99):
    """Una fila con el formato de YuNet: caja (4) + 5 puntos (10) + score."""
    return [x, y, w, h] + [0.0] * 10 + [score]


class DetectorFalso:
    def __init__(self, filas):
        self.filas = filas
        self.tam_entrada = None
        self.forma_vista = None

    def setInputSize(self, tam):
        self.tam_entrada = tam

    def detect(self, imagen):
        self.forma_vista = imagen.shape[:2]
        return 1, (np.array(self.filas, dtype=np.float32) if self.filas else None)


class ReconocedorFalso:
    def __init__(self, vector):
        self.vector = np.asarray(vector, dtype=np.float32)
        self.ultima_fila = None

    def alignCrop(self, imagen, fila_rostro):
        self.ultima_fila = fila_rostro
        return np.zeros((112, 112, 3), np.uint8)

    def feature(self, alineado):
        return self.vector.reshape(1, -1)


def vector_aleatorio(semilla):
    v = np.random.default_rng(semilla).normal(size=fe.DIM_EMBEDDING).astype(np.float32)
    return v / np.linalg.norm(v)


def frame(alto, ancho):
    return np.full((alto, ancho, 3), 120, np.uint8)


class TestDeteccion(unittest.TestCase):
    def test_reduce_frame_grande_y_devuelve_coordenadas_originales(self):
        det = DetectorFalso([fila(100, 50, 80, 80)])  # coords del frame REDUCIDO
        motor = fe.FaceEngine(det, ReconocedorFalso(vector_aleatorio(1)))
        rostros = motor.detectar(frame(720, 1280))  # escala = 640/1280 = 0.5
        self.assertEqual(det.forma_vista, (360, 640))
        self.assertEqual(det.tam_entrada, (640, 360))
        self.assertEqual(rostros[0].caja, (200, 100, 160, 160))

    def test_no_reduce_si_ya_es_pequeno(self):
        det = DetectorFalso([fila(100, 50, 80, 80)])
        motor = fe.FaceEngine(det, ReconocedorFalso(vector_aleatorio(1)))
        rostros = motor.detectar(frame(480, 640))
        self.assertEqual(det.forma_vista, (480, 640))
        self.assertEqual(rostros[0].caja, (100, 50, 80, 80))

    def test_principal_es_el_mas_grande(self):
        det = DetectorFalso([fila(10, 10, 50, 50), fila(200, 100, 120, 120), fila(400, 50, 90, 90)])
        motor = fe.FaceEngine(det, ReconocedorFalso(vector_aleatorio(1)))
        self.assertEqual(motor.principal(frame(480, 640)).caja, (200, 100, 120, 120))

    def test_descarta_score_bajo_y_sin_rostros(self):
        motor = fe.FaceEngine(DetectorFalso([fila(10, 10, 100, 100, score=0.5)]), ReconocedorFalso(vector_aleatorio(1)))
        self.assertEqual(motor.detectar(frame(480, 640)), [])
        vacio = fe.FaceEngine(DetectorFalso([]), ReconocedorFalso(vector_aleatorio(1)))
        self.assertIsNone(vacio.principal(frame(480, 640)))

    def test_caja_se_recorta_a_los_limites_del_frame(self):
        motor = fe.FaceEngine(DetectorFalso([fila(-20, -10, 100, 100)]), ReconocedorFalso(vector_aleatorio(1)))
        r = motor.principal(frame(480, 640))
        self.assertEqual((r.x, r.y, r.w, r.h), (0, 0, 80, 90))

    def test_frame_vacio(self):
        motor = fe.FaceEngine(DetectorFalso([]), ReconocedorFalso(vector_aleatorio(1)))
        self.assertEqual(motor.detectar(None), [])


class TestCalidadYEmbedding(unittest.TestCase):
    def setUp(self):
        self.motor = fe.FaceEngine(DetectorFalso([]), ReconocedorFalso(vector_aleatorio(1)))

    def _rostro(self, w):
        return fe.Rostro(100, 100, w, w, 0.99, np.zeros(15, np.float32))

    def test_rostro_pequeno_se_rechaza(self):
        ok, msg = self.motor.evaluar_calidad(frame(480, 640), self._rostro(40))
        self.assertFalse(ok)
        self.assertIn("Acércate", msg)

    def test_imagen_lisa_es_borrosa(self):
        ok, msg = self.motor.evaluar_calidad(frame(480, 640), self._rostro(120))
        self.assertFalse(ok)
        self.assertIn("borrosa", msg)

    def test_imagen_con_detalle_pasa(self):
        ruido = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
        ok, _ = self.motor.evaluar_calidad(ruido, self._rostro(120))
        self.assertTrue(ok)

    def test_embedding_normalizado_y_recibe_fila_original(self):
        sin_normalizar = vector_aleatorio(2) * 7.5
        rec = ReconocedorFalso(sin_normalizar)
        motor = fe.FaceEngine(DetectorFalso([]), rec)
        rostro = self._rostro(120)
        v = motor.embedding(frame(480, 640), rostro)
        self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=5)
        self.assertIs(rec.ultima_fila, rostro.fila)

    def test_embedding_invalido_lanza_error(self):
        motor = fe.FaceEngine(DetectorFalso([]), ReconocedorFalso(np.zeros(fe.DIM_EMBEDDING)))
        with self.assertRaises(RuntimeError):
            motor.embedding(frame(480, 640), self._rostro(120))

    def test_recortar_respeta_limites(self):
        img = frame(480, 640)
        rostro = fe.Rostro(5, 5, 100, 100, 0.99, np.zeros(15, np.float32))
        recorte = fe.FaceEngine.recortar(img, rostro, margen=0.5)
        self.assertEqual(recorte.shape, (5 + 100 + 50, 5 + 100 + 50, 3))


class TestGaleria(unittest.TestCase):
    def setUp(self):
        self.ana, self.luis = vector_aleatorio(10), vector_aleatorio(20)
        self.g = fe.Galeria()
        self.g.cargar([(1, self.ana), (2, self.luis)])

    def test_identifica_a_la_persona_correcta(self):
        emp_id, sim = self.g.buscar(self.luis)
        self.assertEqual(emp_id, 2)
        self.assertAlmostEqual(sim, 1.0, places=4)

    def test_desconocido_devuelve_none(self):
        emp_id, sim = self.g.buscar(vector_aleatorio(999))
        self.assertIsNone(emp_id)
        self.assertLess(sim, config.FACE_UMBRAL_COSENO)

    def test_galeria_vacia(self):
        self.assertEqual(fe.Galeria().buscar(self.ana), (None, 0.0))

    def test_varias_muestras_por_empleado_usa_la_mejor(self):
        g = fe.Galeria()
        g.agregar(7, vector_aleatorio(30))
        g.agregar(7, self.ana)  # segunda muestra del mismo empleado
        self.assertEqual(g.buscar(self.ana)[0], 7)
        self.assertEqual(len(g), 1)

    def test_quitar_empleado(self):
        self.g.quitar(2)
        self.assertIsNone(self.g.buscar(self.luis)[0])
        self.assertEqual(self.g.buscar(self.ana)[0], 1)

    def test_agregar_despues_de_buscar_actualiza_la_matriz(self):
        self.g.buscar(self.ana)  # fuerza la construcción de la matriz
        nuevo = vector_aleatorio(40)
        self.g.agregar(3, nuevo)
        self.assertEqual(self.g.buscar(nuevo)[0], 3)

    def test_umbral_explicito(self):
        parecido = self.ana * 0.5 + vector_aleatorio(50) * 0.5
        parecido /= np.linalg.norm(parecido)
        _, sim = self.g.mejor(parecido)
        self.assertIsNone(self.g.buscar(parecido, umbral=sim + 0.01)[0])
        self.assertEqual(self.g.buscar(parecido, umbral=sim - 0.01)[0], 1)


class TestSerializacionYModelos(unittest.TestCase):
    def test_ida_y_vuelta_bytes(self):
        v = vector_aleatorio(5)
        datos = fe.embedding_a_bytes(v)
        self.assertEqual(len(datos), 512)
        np.testing.assert_allclose(fe.bytes_a_embedding(datos), v, atol=1e-6)

    def test_bytes_invalidos(self):
        with self.assertRaises(ValueError):
            fe.bytes_a_embedding(b"123")
        with self.assertRaises(ValueError):
            fe.embedding_a_bytes(np.zeros(10))

    def test_modelo_faltante_da_mensaje_claro(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(config, "YUNET_PATH", Path(tmp) / "no_existe.onnx"):
                with self.assertRaises(fe.ModeloNoEncontrado) as ctx:
                    fe.FaceEngine()
        self.assertIn("descargar_modelos", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
