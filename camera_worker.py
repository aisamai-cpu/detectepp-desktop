"""Hilo de cámara: lee de una ``VideoSource``, analiza el rostro y avisa a la interfaz.

Todo el trabajo pesado ocurre aquí, nunca en el hilo de la interfaz. La lógica de
cada vuelta está en ``_paso()`` para poder probarla sin lanzar el hilo.
"""
import logging
import threading
import time
from typing import Callable, Optional

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

import config
from core import camaras, face_store
from core.camaras import Camara
from core.video_source import VideoSource
from vision.face_engine import FaceEngine, ModeloNoEncontrado
from vision.identificador import Identificador, Resultado

logger = logging.getLogger(__name__)

MODO_BIOMETRIA = "BIOMETRIA"


class CameraWorker(QThread):
    # frame (BGR, ya girado/reducido; NO modificar) y resultado del análisis (o None)
    frame_listo = pyqtSignal(object, object)
    # estado de la cámara (CONECTANDO/OK/RECONECTANDO/DETENIDA) y texto para mostrar
    estado_camara = pyqtSignal(str, str)
    # texto del problema cuando faltan los modelos de IA (la cámara sigue funcionando)
    error_modelos = pyqtSignal(str)

    def __init__(self, parent=None, fabrica_fuente: Optional[Callable[[Camara], VideoSource]] = None,
                 motor: Optional[FaceEngine] = None, reloj: Callable[[], float] = time.monotonic):
        super().__init__(parent)
        self._fabrica = fabrica_fuente or VideoSource
        self._motor_inyectado = motor
        self._reloj = reloj
        self._salir = threading.Event()
        self._lock = threading.Lock()

        self.mode = MODO_BIOMETRIA
        self._camara: Optional[Camara] = None
        self._fuente: Optional[VideoSource] = None
        self._identificador: Optional[Identificador] = None

        self._pendiente_cambio: Optional[Camara] = None
        self._pendiente_reconexion = False
        self._pendiente_galeria = True
        self._pendiente_rotacion: Optional[int] = None

        self._ultimo_numero = 0
        self._ultimo_analisis = 0.0
        self._ultimo_resultado: Optional[Resultado] = None
        self._ultimo_estado = None
        self._inicio_fuente = 0.0

    # -- API para la interfaz (segura entre hilos) ----------------------------
    def set_mode(self, mode: str):
        self.mode = mode
        if self._identificador is not None and mode != MODO_BIOMETRIA:
            self._ultimo_resultado = None

    def cambiar_camara(self, camara: Camara):
        with self._lock:
            self._pendiente_cambio = camara

    def reconectar(self):
        """Reabre la cámara actual: vacía cualquier retraso acumulado."""
        with self._lock:
            self._pendiente_reconexion = True

    def recargar_galeria(self):
        """Vuelve a leer los vectores faciales de la base de datos (tras registrar a alguien)."""
        with self._lock:
            self._pendiente_galeria = True

    def rotar(self, grados: int):
        with self._lock:
            self._pendiente_rotacion = grados

    def stop(self):
        self._salir.set()
        self.wait(5000)

    # -- hilo ------------------------------------------------------------------
    def run(self):
        try:
            self._preparar()
            while not self._salir.is_set():
                if not self._paso():
                    time.sleep(0.01)
        except Exception:
            logger.exception("El hilo de cámara terminó por un error inesperado")
        finally:
            self._cerrar_fuente()

    def _preparar(self):
        try:
            motor = self._motor_inyectado or FaceEngine()
            self._identificador = Identificador(motor)
        except ModeloNoEncontrado as e:
            logger.error("%s", e)
            self.error_modelos.emit(str(e))
            self._identificador = None
        self._camara = camaras.activa()
        self._abrir_fuente()

    def _abrir_fuente(self):
        self._cerrar_fuente()
        self._fuente = self._fabrica(self._camara)
        self._fuente.iniciar()
        self._ultimo_numero = 0
        self._ultimo_resultado = None
        self._inicio_fuente = self._reloj()
        if self._identificador is not None:
            self._identificador.reiniciar()

    def _cerrar_fuente(self):
        if self._fuente is not None:
            self._fuente.detener()
            self._fuente = None

    def _atender_pendientes(self):
        with self._lock:
            cambio, self._pendiente_cambio = self._pendiente_cambio, None
            reconexion, self._pendiente_reconexion = self._pendiente_reconexion, False
            galeria, self._pendiente_galeria = self._pendiente_galeria, False
            rotacion, self._pendiente_rotacion = self._pendiente_rotacion, None

        if galeria and self._identificador is not None:
            try:
                self._identificador.galeria = face_store.cargar_galeria()
                self._identificador.reiniciar()
            except Exception:
                logger.exception("No se pudo cargar la galería de rostros")
        if rotacion is not None and self._fuente is not None:
            self._fuente.rotacion = rotacion
            self._camara.rotacion = rotacion
        if cambio is not None:
            self._camara = cambio
            self._abrir_fuente()
        elif reconexion or self._toca_reconexion_automatica():
            self._abrir_fuente()

    def _toca_reconexion_automatica(self) -> bool:
        minutos = config.CAMARA_RECONEXION_MIN
        return bool(minutos) and (self._reloj() - self._inicio_fuente) >= minutos * 60

    def _paso(self) -> bool:
        """Una vuelta del bucle. Devuelve True si hubo un frame nuevo."""
        self._atender_pendientes()

        estado = (self._fuente.estado, self._fuente.mensaje)
        if estado != self._ultimo_estado:
            self._ultimo_estado = estado
            self.estado_camara.emit(*estado)

        frame, numero = self._fuente.leer()
        if frame is None or numero == self._ultimo_numero:
            return False
        self._ultimo_numero = numero

        if self._camara.tipo == "usb":
            frame = cv2.flip(frame, 1)  # las cámaras locales se ven como un espejo

        resultado = self._ultimo_resultado
        if self.mode == MODO_BIOMETRIA and self._identificador is not None:
            ahora = self._reloj()
            if ahora - self._ultimo_analisis >= config.FACE_INTERVALO_S:
                self._ultimo_analisis = ahora
                try:
                    resultado = self._identificador.procesar(frame)
                except Exception:
                    logger.exception("Falló el análisis del rostro")
                    resultado = None
                self._ultimo_resultado = resultado
        else:
            resultado = None

        self.frame_listo.emit(frame, resultado)
        return True
