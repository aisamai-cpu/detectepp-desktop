"""Hilo de cámara: lee de una ``VideoSource``, analiza el rostro y avisa a la interfaz.

Para que el video se vea fluido, la imagen y el análisis van por caminos separados:

- este hilo solo toma el frame más reciente y se lo entrega a la interfaz de inmediato;
- un segundo hilo (``analisis``) hace la IA (≈50-100 ms por análisis) sobre el último
  frame disponible. El video nunca espera a la IA: el cuadro del rostro se actualiza
  cuando termina cada análisis;
- si la interfaz va más lenta que la cámara, se descartan frames en vez de acumularlos
  (control de flujo), así no aparece retraso.

La lógica de cada vuelta está en ``_paso()`` para poder probarla sin lanzar el hilo.
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
    # fps reales: (lectura de la cámara, imágenes entregadas a la pantalla), una vez por segundo
    fps_camara = pyqtSignal(float, float)

    def __init__(self, parent=None, fabrica_fuente: Optional[Callable[[Camara], VideoSource]] = None,
                 motor: Optional[FaceEngine] = None, reloj: Callable[[], float] = time.monotonic,
                 analisis_sincrono: bool = False):
        """``analisis_sincrono`` (solo pruebas): analiza dentro de ``_paso()`` en vez de en otro hilo."""
        super().__init__(parent)
        self._analisis_sincrono = analisis_sincrono
        self.control_flujo = False        # la interfaz lo activa y confirma con frame_mostrado()
        self._ui_pendiente = False
        self._t_emision = 0.0
        self._cond = threading.Condition()
        self._frame_a_analizar: Optional[np.ndarray] = None
        self._hilo_analisis: Optional[threading.Thread] = None
        self._generacion = 0              # sube al cambiar de cámara: descarta análisis viejos
        self._lock_ident = threading.Lock()
        self._t_fps = 0.0
        self._n_emitidos = 0
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

    def frame_mostrado(self):
        """La interfaz avisa que ya dibujó el último frame (permite enviar el siguiente)."""
        self._ui_pendiente = False

    def stop(self):
        self._salir.set()
        with self._cond:
            self._cond.notify_all()
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
            self._salir.set()
            with self._cond:
                self._cond.notify_all()
            if self._hilo_analisis is not None:
                self._hilo_analisis.join(timeout=3)
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
        if self._identificador is not None and not self._analisis_sincrono:
            self._hilo_analisis = threading.Thread(target=self._bucle_analisis, daemon=True, name="analisis-rostro")
            self._hilo_analisis.start()

    def _abrir_fuente(self):
        self._cerrar_fuente()
        self._fuente = self._fabrica(self._camara)
        self._fuente.iniciar()
        self._ultimo_numero = 0
        self._ultimo_resultado = None
        self._generacion += 1
        self._inicio_fuente = self._reloj()
        if self._identificador is not None:
            with self._lock_ident:
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
                galeria = face_store.cargar_galeria()
                with self._lock_ident:
                    self._identificador.galeria = galeria
                    self._identificador.reiniciar()
                self._generacion += 1
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

        resultado = None
        if self.mode == MODO_BIOMETRIA and self._identificador is not None:
            ahora = self._reloj()
            if ahora - self._ultimo_analisis >= config.FACE_INTERVALO_S:
                self._ultimo_analisis = ahora
                if self._analisis_sincrono:
                    self._analizar(frame, self._generacion)
                else:
                    with self._cond:
                        self._frame_a_analizar = frame   # solo importa el más reciente
                        self._cond.notify()
            resultado = self._ultimo_resultado

        ahora = self._reloj()
        if not (self.control_flujo and self._ui_pendiente and ahora - self._t_emision < 0.5):
            # (si la interfaz aún no dibujó el anterior se descarta este frame en vez de acumular)
            self._ui_pendiente = True
            self._t_emision = ahora
            self._n_emitidos += 1
            self.frame_listo.emit(frame, resultado)
        self._informar_fps(ahora)
        return True

    def _informar_fps(self, ahora: float):
        if self._t_fps == 0.0:
            self._t_fps = ahora
        elif ahora - self._t_fps >= 1.0:
            pantalla = self._n_emitidos / (ahora - self._t_fps)
            self.fps_camara.emit(float(self._fuente.fps), pantalla)
            self._t_fps, self._n_emitidos = ahora, 0

    # -- análisis (hilo propio) ----------------------------------------------------
    def _bucle_analisis(self):
        while not self._salir.is_set():
            with self._cond:
                if self._frame_a_analizar is None:
                    self._cond.wait(0.2)
                frame, self._frame_a_analizar = self._frame_a_analizar, None
            if frame is not None:
                self._analizar(frame, self._generacion)

    def _analizar(self, frame: np.ndarray, generacion: int):
        try:
            with self._lock_ident:
                resultado = self._identificador.procesar(frame)
        except Exception:
            logger.exception("Falló el análisis del rostro")
            resultado = None
        if generacion == self._generacion and self.mode == MODO_BIOMETRIA:
            self._ultimo_resultado = resultado
