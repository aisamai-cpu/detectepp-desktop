"""Fuente de video intercambiable: cámara USB, cámara IP/celular (RTSP, HTTP) o archivo.

Una ``VideoSource`` lee en un hilo propio y SIEMPRE conserva solo el frame más
reciente. Así la IA y la interfaz nunca procesan imágenes atrasadas (el problema
típico de RTSP, que acumula retraso), y si la cámara se desconecta el hilo
reintenta solo, sin congelar ni cerrar la aplicación.

Uso:
    fuente = VideoSource(camaras.activa())
    fuente.iniciar()
    frame, n = fuente.leer()      # (None, 0) mientras no haya imagen
    fuente.estado                 # CONECTANDO / OK / RECONECTANDO / DETENIDA
    fuente.detener()

No importa PyQt. El frame devuelto NO debe modificarse: haz ``frame.copy()`` antes de dibujar.
"""
import logging
import os
import threading
import time
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

import config
from core.camaras import Camara

# RTSP por TCP: con Wi-Fi o celulares evita la imagen rota por paquetes UDP perdidos.
# Debe definirse antes de la primera apertura con FFmpeg.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

logger = logging.getLogger(__name__)

CONECTANDO = "CONECTANDO"
OK = "OK"
RECONECTANDO = "RECONECTANDO"
DETENIDA = "DETENIDA"

_ESPERAS_REINTENTO = (1.0, 2.0, 4.0, 5.0)  # segundos entre intentos; el último se repite


# ----------------------------------------------------------------------------
# Apertura de la captura según el tipo de cámara
# ----------------------------------------------------------------------------
def abrir_captura(camara: Camara):
    """Crea el ``cv2.VideoCapture`` adecuado. Puede devolver una captura cerrada."""
    if camara.tipo == "usb":
        indice = int(camara.valor)
        cap = cv2.VideoCapture(indice, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(indice)
        if cap.isOpened() and camara.ancho and camara.alto:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, camara.ancho)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camara.alto)
        return cap

    if camara.tipo == "url":
        try:  # tiempos de espera (OpenCV >= 4.6): una URL caída no bloquea para siempre
            cap = cv2.VideoCapture(
                camara.valor, cv2.CAP_FFMPEG,
                [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000],
            )
        except TypeError:
            cap = cv2.VideoCapture(camara.valor, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    return cv2.VideoCapture(camara.valor)  # archivo


def detectar_usb(max_indices: int = 5, abrir: Optional[Callable] = None) -> list:
    """Prueba los índices 0..max_indices-1 y devuelve ``[{indice, ancho, alto}]`` de las que dan imagen.

    Es lento (cada índice vacío tarda un poco): NO llamarla desde el hilo de la interfaz.
    Una cámara que ya está abierta por otra fuente no aparecerá (Windows no permite abrirla dos veces).
    """
    abrir = abrir or abrir_captura
    encontradas = []
    for i in range(max_indices):
        cap = None
        try:
            cap = abrir(Camara(id=f"usb-{i}", nombre=f"USB {i}", tipo="usb", valor=str(i)))
            if cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None:
                    alto, ancho = frame.shape[:2]
                    encontradas.append({"indice": i, "ancho": ancho, "alto": alto})
        except Exception:
            logger.debug("Índice USB %s no disponible", i, exc_info=True)
        finally:
            if cap is not None:
                cap.release()
    return encontradas


# ----------------------------------------------------------------------------
# Fuente
# ----------------------------------------------------------------------------
class VideoSource:
    def __init__(self, camara: Camara, abrir: Optional[Callable] = None):
        """``abrir`` solo se reemplaza en las pruebas (devuelve un objeto tipo VideoCapture)."""
        self.camara = camara
        self._abrir = abrir or abrir_captura
        self._hilo: Optional[threading.Thread] = None
        self._detener = threading.Event()
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._numero = 0
        self._estado = DETENIDA

    # -- API pública ---------------------------------------------------------
    @property
    def estado(self) -> str:
        return self._estado

    @property
    def mensaje(self) -> str:
        """Texto para mostrar al usuario según el estado."""
        nombre = self.camara.nombre
        return {
            CONECTANDO: f"Conectando a «{nombre}»…",
            OK: f"«{nombre}» en vivo",
            RECONECTANDO: f"«{nombre}» sin señal: reintentando…",
            DETENIDA: f"«{nombre}» detenida",
        }[self._estado]

    def iniciar(self):
        if self._hilo is not None and self._hilo.is_alive():
            return
        self._detener.clear()
        self._frame, self._numero = None, 0
        self._estado = CONECTANDO
        self._hilo = threading.Thread(target=self._bucle, daemon=True, name=f"video-{self.camara.id}")
        self._hilo.start()

    def detener(self, espera: float = 3.0):
        self._detener.set()
        if self._hilo is not None:
            self._hilo.join(timeout=espera)
        self._hilo = None
        self._estado = DETENIDA

    def leer(self) -> Tuple[Optional[np.ndarray], int]:
        """``(frame más reciente, número de frame)``. El número sube con cada imagen nueva."""
        with self._lock:
            return self._frame, self._numero

    # -- hilo lector ---------------------------------------------------------
    def _bucle(self):
        cap = None
        intento = 0
        frames_en_apertura = 0
        ultimo_frame = time.monotonic()
        es_archivo = self.camara.tipo == "archivo"
        pausa_archivo = 0.033

        try:
            while not self._detener.is_set():
                if cap is None:
                    cap = self._abrir_seguro()
                    if cap is None:
                        if self._numero > 0:
                            self._estado = RECONECTANDO
                        self._esperar(_ESPERAS_REINTENTO[min(intento, len(_ESPERAS_REINTENTO) - 1)])
                        intento += 1
                        continue
                    intento = 0
                    frames_en_apertura = 0
                    ultimo_frame = time.monotonic()
                    if es_archivo:
                        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                        pausa_archivo = 1.0 / max(1.0, min(fps, 120.0))

                ok, frame = cap.read()
                if ok and frame is not None:
                    with self._lock:
                        self._frame = frame
                        self._numero += 1
                    self._estado = OK
                    ultimo_frame = time.monotonic()
                    frames_en_apertura += 1
                    if es_archivo:
                        time.sleep(pausa_archivo)  # un archivo se lee más rápido que en vivo
                    continue

                if es_archivo and frames_en_apertura > 0:
                    cap.release()  # fin del video: reabrir y repetir sin avisar
                    cap = None
                    continue

                if time.monotonic() - ultimo_frame > config.CAMARA_SIN_FRAMES_S:
                    logger.warning("«%s»: %.0f s sin imagen, reconectando", self.camara.nombre, config.CAMARA_SIN_FRAMES_S)
                    cap.release()
                    cap = None
                    self._estado = RECONECTANDO
                else:
                    time.sleep(0.02)
        finally:
            if cap is not None:
                cap.release()

    def _abrir_seguro(self):
        try:
            cap = self._abrir(self.camara)
        except Exception:
            logger.exception("Error abriendo «%s»", self.camara.nombre)
            return None
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            logger.info("No se pudo abrir «%s» (%s)", self.camara.nombre, self.camara.valor_visible)
            return None
        return cap

    def _esperar(self, segundos: float):
        self._detener.wait(segundos)  # se interrumpe al detener()
