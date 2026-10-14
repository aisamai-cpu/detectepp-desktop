"""Modelos YOLO exportados a ONNX, ejecutados SIN torch.

Por qué: importar torch en Windows tarda 10-30 s y bloquea la ventana, y torch usa todos
los núcleos de la CPU por defecto. Con ONNX la inferencia corre en onnxruntime (si está
instalado) o en OpenCV DNN; ambos sueltan el GIL mientras calculan, así la interfaz sigue fluida.

Sirve para detección (EPP) y para pose (esqueleto de personas, Monitoreo).
Los nombres de clase, el tamaño de entrada y la tarea se leen de los metadatos que
ultralytics guarda dentro del .onnx al exportar. No importa PyQt.
"""
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import config


class ModeloOnnxInvalido(Exception):
    """El archivo no es un ONNX de YOLO exportado con ultralytics."""


# ----------------------------------------------------------------------------
# Metadatos (lectura mínima del protobuf de ONNX, sin depender del paquete onnx)
# ----------------------------------------------------------------------------
def _varint(datos: bytes, i: int) -> Tuple[int, int]:
    valor, desplazamiento = 0, 0
    while True:
        b = datos[i]
        i += 1
        valor |= (b & 0x7F) << desplazamiento
        if not b & 0x80:
            return valor, i
        desplazamiento += 7


def _campos(datos: bytes):
    """Recorre los campos de primer nivel de un mensaje protobuf: (número, tipo, valor)."""
    i, n = 0, len(datos)
    while i < n:
        clave, i = _varint(datos, i)
        numero, tipo = clave >> 3, clave & 7
        if tipo == 0:
            valor, i = _varint(datos, i)
        elif tipo == 1:
            valor, i = datos[i:i + 8], i + 8
        elif tipo == 2:
            largo, i = _varint(datos, i)
            valor, i = datos[i:i + largo], i + largo
        elif tipo == 5:
            valor, i = datos[i:i + 4], i + 4
        else:
            raise ValueError(f"tipo protobuf no soportado: {tipo}")
        yield numero, tipo, valor


def leer_metadatos(ruta) -> Dict[str, str]:
    """``metadata_props`` del modelo ONNX (campo 14 de ModelProto) como diccionario."""
    metadatos = {}
    for numero, tipo, valor in _campos(Path(ruta).read_bytes()):
        if numero == 14 and tipo == 2:
            entrada = {n: v for n, t, v in _campos(valor) if t == 2}
            if 1 in entrada:
                metadatos[entrada[1].decode("utf-8")] = entrada.get(2, b"").decode("utf-8")
    return metadatos


# ----------------------------------------------------------------------------
# Resultado
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Caja:
    x1: float
    y1: float
    x2: float
    y2: float
    clase: int
    confianza: float
    puntos: Optional[np.ndarray] = None   # pose: (17, 3) con x, y, confianza en píxeles del frame


def letterbox(img: np.ndarray, lado: int):
    """Reduce manteniendo la proporción y rellena con gris (igual que ultralytics)."""
    h, w = img.shape[:2]
    r = min(lado / h, lado / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    dx, dy = (lado - nw) / 2, (lado - nh) / 2
    if (w, h) != (nw, nh):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    arriba, abajo = int(round(dy - 0.1)), int(round(dy + 0.1))
    izq, der = int(round(dx - 0.1)), int(round(dx + 0.1))
    img = cv2.copyMakeBorder(img, arriba, abajo, izq, der, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, r, (izq, arriba)


class ModeloYoloOnnx:
    def __init__(self, ruta, hilos: Optional[int] = None):
        ruta = Path(ruta)
        meta = leer_metadatos(ruta)
        if "names" not in meta:
            raise ModeloOnnxInvalido(f"{ruta.name} no tiene los metadatos de ultralytics (names)")
        nombres = ast.literal_eval(meta["names"])
        self.names: Dict[int, str] = dict(enumerate(nombres)) if isinstance(nombres, list) else dict(nombres)
        self.tarea = meta.get("task", "detect")
        imgsz = ast.literal_eval(meta.get("imgsz", "[640, 640]"))
        self.lado = int(imgsz[0] if isinstance(imgsz, (list, tuple)) else imgsz)
        self.kpt_shape = tuple(ast.literal_eval(meta["kpt_shape"])) if "kpt_shape" in meta else None
        hilos = int(hilos or config.IA_HILOS_CPU)

        self._sesion = self._net = None
        try:
            import onnxruntime as ort
            opciones = ort.SessionOptions()
            opciones.intra_op_num_threads = hilos
            opciones.inter_op_num_threads = 1
            self._sesion = ort.InferenceSession(str(ruta), opciones, providers=["CPUExecutionProvider"])
            self._entrada = self._sesion.get_inputs()[0].name
            self.backend = "onnxruntime"
        except ImportError:
            self._net = cv2.dnn.readNetFromONNX(str(ruta))
            self.backend = "opencv"

    def _inferir(self, blob: np.ndarray) -> np.ndarray:
        if self._sesion is not None:
            return self._sesion.run(None, {self._entrada: blob})[0]
        self._net.setInput(blob)
        return self._net.forward()

    def detectar(self, frame: np.ndarray, conf: float, iou: float = 0.45) -> List[Caja]:
        img, r, (dx, dy) = letterbox(frame, self.lado)
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (self.lado, self.lado), swapRB=True)
        salida = np.asarray(self._inferir(blob))[0].T          # (N, 4 + clases [+ puntos])
        cxcywh = salida[:, :4]
        if self.tarea == "pose":
            puntajes = salida[:, 4]
            clases = np.zeros(len(salida), dtype=int)
            puntos = salida[:, 5:]
        else:
            matriz = salida[:, 4:4 + len(self.names)]
            clases = matriz.argmax(axis=1)
            puntajes = matriz[np.arange(len(matriz)), clases]
            puntos = None
        sel = puntajes >= conf
        if not sel.any():
            return []
        cxcywh, puntajes, clases = cxcywh[sel], puntajes[sel], clases[sel]
        puntos = puntos[sel] if puntos is not None else None
        xywh = np.column_stack([cxcywh[:, 0] - cxcywh[:, 2] / 2, cxcywh[:, 1] - cxcywh[:, 3] / 2,
                                cxcywh[:, 2], cxcywh[:, 3]])
        indices = cv2.dnn.NMSBoxesBatched(xywh.tolist(), puntajes.tolist(), clases.tolist(), conf, iou)
        h, w = frame.shape[:2]
        cajas = []
        for k in np.asarray(indices).reshape(-1):
            x, y, bw, bh = xywh[k]
            x1 = float(np.clip((x - dx) / r, 0, w))
            y1 = float(np.clip((y - dy) / r, 0, h))
            x2 = float(np.clip((x + bw - dx) / r, 0, w))
            y2 = float(np.clip((y + bh - dy) / r, 0, h))
            kp = None
            if puntos is not None:
                kp = puntos[k].reshape(-1, 3).astype(np.float32).copy()
                kp[:, 0] = (kp[:, 0] - dx) / r
                kp[:, 1] = (kp[:, 1] - dy) / r
            cajas.append(Caja(x1, y1, x2, y2, int(clases[k]), float(puntajes[k]), kp))
        return cajas
