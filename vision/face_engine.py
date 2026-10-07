"""Motor de rostros: detección (YuNet) + vector facial (SFace) + galería.

Este módulo NO importa PyQt: se puede probar y reutilizar sin interfaz.

Flujo típico:
    motor = FaceEngine()
    rostro = motor.principal(frame)              # el rostro más grande, o None
    vector = motor.embedding(frame, rostro)      # 128 números, normalizado
    galeria.buscar(vector)                       # -> (id_empleado | None, similitud)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import cv2
import numpy as np

import config

DIM_EMBEDDING = 128


class ModeloNoEncontrado(FileNotFoundError):
    """Falta un archivo .onnx en models/."""


@dataclass(eq=False)
class Rostro:
    """Un rostro detectado, con coordenadas en el frame ORIGINAL."""

    x: int
    y: int
    w: int
    h: int
    score: float
    fila: np.ndarray  # 15 valores de YuNet (caja, 5 puntos faciales, score) en coords originales

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def caja(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)


# ----------------------------------------------------------------------------
# Motor
# ----------------------------------------------------------------------------
class FaceEngine:
    """Envuelve YuNet (detector) y SFace (reconocedor).

    ``detector`` y ``reconocedor`` solo se pasan en las pruebas (objetos falsos).
    """

    def __init__(self, detector=None, reconocedor=None):
        self._detector = detector if detector is not None else self._crear_detector()
        self._reconocedor = reconocedor if reconocedor is not None else self._crear_reconocedor()
        self._tam_entrada: Optional[Tuple[int, int]] = None

    # -- creación de modelos -------------------------------------------------
    @staticmethod
    def _verificar_archivo(ruta):
        if not ruta.is_file():
            raise ModeloNoEncontrado(
                f"Falta el modelo {ruta.name}. Ejecuta: python -m tools.descargar_modelos"
            )

    @classmethod
    def _crear_detector(cls):
        cls._verificar_archivo(config.YUNET_PATH)
        return cv2.FaceDetectorYN.create(
            str(config.YUNET_PATH), "", (320, 320), config.FACE_SCORE_MIN, 0.3, 5000
        )

    @classmethod
    def _crear_reconocedor(cls):
        cls._verificar_archivo(config.SFACE_PATH)
        return cv2.FaceRecognizerSF.create(str(config.SFACE_PATH), "")

    # -- detección -----------------------------------------------------------
    def detectar(self, frame: np.ndarray) -> list:
        """Devuelve los rostros del frame, del más grande al más pequeño.

        El frame se reduce internamente (``FACE_INFERENCIA_MAX_LADO``) para que
        una cámara 1080p no sature el CPU; las coordenadas se devuelven ya
        escaladas al tamaño original.
        """
        if frame is None or frame.size == 0:
            return []
        alto, ancho = frame.shape[:2]
        escala = min(1.0, config.FACE_INFERENCIA_MAX_LADO / max(alto, ancho))
        if escala < 1.0:
            pequeno = cv2.resize(
                frame, (round(ancho * escala), round(alto * escala)), interpolation=cv2.INTER_AREA
            )
        else:
            pequeno = frame

        ph, pw = pequeno.shape[:2]
        if self._tam_entrada != (pw, ph):
            self._detector.setInputSize((pw, ph))
            self._tam_entrada = (pw, ph)

        _, filas = self._detector.detect(pequeno)
        if filas is None:
            return []

        rostros = []
        for fila in filas:
            f = np.array(fila, dtype=np.float32).reshape(-1)
            f[:14] /= escala  # caja y puntos faciales vuelven a coordenadas originales
            score = float(f[14])
            if score < config.FACE_SCORE_MIN:
                continue
            x0 = max(0, int(round(f[0])))
            y0 = max(0, int(round(f[1])))
            x1 = min(ancho, int(round(f[0] + f[2])))
            y1 = min(alto, int(round(f[1] + f[3])))
            if x1 - x0 <= 0 or y1 - y0 <= 0:
                continue
            rostros.append(Rostro(x0, y0, x1 - x0, y1 - y0, score, f))
        rostros.sort(key=lambda r: r.area, reverse=True)
        return rostros

    def principal(self, frame: np.ndarray) -> Optional[Rostro]:
        """El rostro más grande (el más cercano a la cámara), o None."""
        rostros = self.detectar(frame)
        return rostros[0] if rostros else None

    # -- calidad -------------------------------------------------------------
    def evaluar_calidad(self, frame: np.ndarray, rostro: Rostro) -> Tuple[bool, str]:
        """¿Sirve este rostro para registrarlo? Devuelve (ok, mensaje para el usuario)."""
        if rostro.w < config.FACE_TAM_MIN:
            return False, "Acércate más a la cámara"
        recorte = frame[rostro.y: rostro.y + rostro.h, rostro.x: rostro.x + rostro.w]
        gris = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
        if cv2.Laplacian(gris, cv2.CV_64F).var() < config.FACE_NITIDEZ_MIN:
            return False, "Imagen borrosa: quédate quieto y mejora la luz"
        return True, "OK"

    # -- vector facial -------------------------------------------------------
    def embedding(self, frame: np.ndarray, rostro: Rostro) -> np.ndarray:
        """Vector de 128 números normalizado (norma 1) que identifica el rostro."""
        alineado = self._reconocedor.alignCrop(frame, rostro.fila)
        vector = np.asarray(self._reconocedor.feature(alineado), dtype=np.float32).reshape(-1)
        norma = float(np.linalg.norm(vector))
        if vector.size != DIM_EMBEDDING or norma == 0.0:
            raise RuntimeError("El modelo no produjo un vector facial válido")
        return vector / norma

    @staticmethod
    def recortar(frame: np.ndarray, rostro: Rostro, margen: float = 0.25) -> np.ndarray:
        """Recorte del rostro con algo de margen (para guardar la foto del empleado)."""
        alto, ancho = frame.shape[:2]
        mx, my = int(rostro.w * margen), int(rostro.h * margen)
        x0, y0 = max(0, rostro.x - mx), max(0, rostro.y - my)
        x1, y1 = min(ancho, rostro.x + rostro.w + mx), min(alto, rostro.y + rostro.h + my)
        return frame[y0:y1, x0:x1].copy()


# ----------------------------------------------------------------------------
# Galería: vectores de los empleados registrados, en memoria
# ----------------------------------------------------------------------------
class Galeria:
    """Guarda varios vectores por empleado y busca el más parecido.

    Los vectores están normalizados, así que la similitud coseno es un simple
    producto punto: una sola multiplicación de matrices por consulta, sin leer
    nada del disco.
    """

    def __init__(self):
        self._ids: list = []
        self._vectores: list = []
        self._matriz: Optional[np.ndarray] = None  # se reconstruye al cambiar

    def __len__(self) -> int:
        return len(set(self._ids))

    def cargar(self, registros: Iterable[Tuple[int, np.ndarray]]):
        """Reemplaza el contenido por ``(id_empleado, vector)`` de la base de datos."""
        self._ids, self._vectores, self._matriz = [], [], None
        for emp_id, vector in registros:
            self.agregar(emp_id, vector)

    def agregar(self, emp_id: int, vector: np.ndarray):
        self._ids.append(int(emp_id))
        self._vectores.append(np.asarray(vector, dtype=np.float32).reshape(-1))
        self._matriz = None

    def quitar(self, emp_id: int):
        mantener = [i for i, e in enumerate(self._ids) if e != int(emp_id)]
        self._ids = [self._ids[i] for i in mantener]
        self._vectores = [self._vectores[i] for i in mantener]
        self._matriz = None

    def mejor(self, vector: np.ndarray) -> Tuple[Optional[int], float]:
        """Empleado más parecido y su similitud, SIN aplicar umbral."""
        if not self._ids:
            return None, 0.0
        if self._matriz is None:
            self._matriz = np.vstack(self._vectores)
        similitudes = self._matriz @ np.asarray(vector, dtype=np.float32).reshape(-1)
        mejor = int(np.argmax(similitudes))
        return self._ids[mejor], float(similitudes[mejor])

    def buscar(self, vector: np.ndarray, umbral: Optional[float] = None) -> Tuple[Optional[int], float]:
        """(id_empleado, similitud) si supera el umbral; si no, (None, similitud)."""
        if umbral is None:
            umbral = config.FACE_UMBRAL_COSENO
        emp_id, similitud = self.mejor(vector)
        if emp_id is not None and similitud >= umbral:
            return emp_id, similitud
        return None, similitud


# ----------------------------------------------------------------------------
# Serialización para SQLite (BLOB de 512 bytes)
# ----------------------------------------------------------------------------
def embedding_a_bytes(vector: np.ndarray) -> bytes:
    v = np.asarray(vector, dtype=np.float32).reshape(-1)
    if v.size != DIM_EMBEDDING:
        raise ValueError(f"Un vector facial debe tener {DIM_EMBEDDING} valores, no {v.size}")
    return v.tobytes()


def bytes_a_embedding(datos: bytes) -> np.ndarray:
    v = np.frombuffer(datos, dtype=np.float32).copy()
    if v.size != DIM_EMBEDDING:
        raise ValueError(f"BLOB inválido: {v.size} valores en lugar de {DIM_EMBEDDING}")
    norma = float(np.linalg.norm(v))
    return v / norma if norma else v
