"""Modelos YOLO en ONNX sin torch: metadatos, decodificación, escala y NMS (modelo ONNX diminuto falso)."""
import builtins
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from vision import yolo_onnx
from vision.epp_engine import EppEngine, ModeloEppNoEncontrado

try:
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    ONNX = True
except ImportError:
    ONNX = False

LADO = 64


def salida_detect():
    """3 candidatos (cx, cy, w, h en la imagen de 64x64) y 2 clases. El 2.º se solapa con el 1.º."""
    cajas = np.array([[32, 32, 20, 20], [33, 32, 20, 20], [10, 10, 8, 8]], np.float32)
    clases = np.array([[0.9, 0.1], [0.6, 0.1], [0.05, 0.8]], np.float32)
    return np.concatenate([cajas, clases], axis=1).T[None]          # (1, 6, 3)


def salida_pose():
    caja = np.array([[32, 32, 20, 40, 0.9]], np.float32)
    puntos = np.tile(np.array([[32, 20, 0.95]], np.float32), (17, 1)).reshape(1, -1)
    return np.concatenate([caja, puntos], axis=1).T[None]            # (1, 56, 1)


def crear_onnx(ruta, salida, meta):
    """Grafo: salida = constante + 0 * media(entrada). Lleva los metadatos de ultralytics."""
    entrada = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, LADO, LADO])
    out = helper.make_tensor_value_info("output0", TensorProto.FLOAT, list(salida.shape))
    nodos = [
        helper.make_node("ReduceMean", ["images"], ["m"], keepdims=0),
        helper.make_node("Mul", ["m", "cero"], ["z"]),
        helper.make_node("Add", ["const", "z"], ["output0"]),
    ]
    inits = [numpy_helper.from_array(salida.astype(np.float32), "const"),
             numpy_helper.from_array(np.array(0, np.float32), "cero")]
    grafo = helper.make_graph(nodos, "falso", [entrada], [out], initializer=inits)
    modelo = helper.make_model(grafo, opset_imports=[helper.make_opsetid("", 12)])
    modelo.ir_version = 8
    for k, v in meta.items():
        modelo.metadata_props.add(key=k, value=v)
    onnx.save(modelo, str(ruta))


def sin_onnxruntime():
    real = builtins.__import__

    def importar(nombre, *a, **k):
        if nombre == "onnxruntime":
            raise ImportError(nombre)
        return real(nombre, *a, **k)
    return mock.patch("builtins.__import__", importar)


@unittest.skipUnless(ONNX, "requiere el paquete onnx para crear el modelo de prueba")
class TestYoloOnnx(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.det = self.dir / "epp_yolo.onnx"
        crear_onnx(self.det, salida_detect(),
                   {"names": "{0: 'casco', 1: 'sin_chaleco'}", "task": "detect", "imgsz": f"[{LADO}, {LADO}]"})
        self.pose = self.dir / "pose.onnx"
        crear_onnx(self.pose, salida_pose(), {"names": "{0: 'person'}", "task": "pose",
                                              "imgsz": f"[{LADO}, {LADO}]", "kpt_shape": "[17, 3]"})

    def tearDown(self):
        self._tmp.cleanup()

    def test_lee_metadatos(self):
        meta = yolo_onnx.leer_metadatos(self.det)
        self.assertEqual(meta["task"], "detect")
        self.assertIn("casco", meta["names"])

    def _modelos(self, ruta):
        """El mismo modelo con OpenCV DNN y, si está instalado, con onnxruntime."""
        with sin_onnxruntime():
            yield yolo_onnx.ModeloYoloOnnx(ruta)
        try:
            import onnxruntime  # noqa: F401
            yield yolo_onnx.ModeloYoloOnnx(ruta)
        except ImportError:
            pass

    def test_detecta_escala_al_frame_y_aplica_nms(self):
        frame = np.zeros((128, 256, 3), np.uint8)   # 4x más ancho que alto: letterbox con relleno arriba/abajo
        for m in self._modelos(self.det):
            with self.subTest(backend=m.backend):
                cajas = sorted(m.detectar(frame, conf=0.5), key=lambda c: c.clase)
                self.assertEqual([c.clase for c in cajas], [0, 1])          # el solapado se descarta
                casco = cajas[0]
                # r = 64/256 = 0.25, relleno vertical = (64 - 32) / 2 = 16
                self.assertAlmostEqual(casco.x1, (22 - 0) / 0.25, delta=1)
                self.assertAlmostEqual(casco.y1, (22 - 16) / 0.25, delta=1)
                self.assertAlmostEqual(casco.confianza, 0.9, places=3)

    def test_umbral_de_confianza(self):
        for m in self._modelos(self.det):
            self.assertEqual(m.detectar(np.zeros((64, 64, 3), np.uint8), conf=0.95), [])

    def test_pose_devuelve_17_puntos_en_pixeles_del_frame(self):
        for m in self._modelos(self.pose):
            with self.subTest(backend=m.backend):
                cajas = m.detectar(np.zeros((128, 128, 3), np.uint8), conf=0.5)
                self.assertEqual(len(cajas), 1)
                self.assertEqual(cajas[0].puntos.shape, (17, 3))
                self.assertAlmostEqual(float(cajas[0].puntos[0, 1]), 40, delta=1)   # 20 * (128/64)

    def test_epp_engine_prefiere_onnx_y_traduce_clases(self):
        with sin_onnxruntime():
            motor = EppEngine(ruta=self.dir / "epp_yolo.pt")
        self.assertIsInstance(motor.modelo, yolo_onnx.ModeloYoloOnnx)
        dets = motor.detectar(np.zeros((64, 64, 3), np.uint8))
        self.assertEqual({(d.clase, d.presente) for d in dets}, {("casco", True), ("chaleco", False)})
        self.assertEqual(motor.prendas_detectables, {"casco", "chaleco"})


class TestSinModelo(unittest.TestCase):
    def test_sin_onnx_ni_pt_avisa(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ModeloEppNoEncontrado) as e:
                EppEngine(ruta=Path(d) / "epp_yolo.onnx")
        self.assertIn("epp_yolo.onnx", str(e.exception))


if __name__ == "__main__":
    unittest.main()
