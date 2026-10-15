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
from core.riesgos import FotoEvento, MonitorRiesgos, ResultadoPose
from core.video_source import VideoSource
from vision.epp_engine import EppEngine, ModeloEppNoEncontrado
from vision.face_engine import FaceEngine, ModeloNoEncontrado
from vision.identificador import Identificador, Resultado
from vision.pose_engine import ModeloPoseNoEncontrado, PoseEngine

logger = logging.getLogger(__name__)

MODO_BIOMETRIA = "BIOMETRIA"
MODO_EPP = "EPP"
MODO_MONITOREO = "MONITOREO"

# Tipo de resultado que corresponde a cada modo: al cambiar de pestaña puede quedar
# un análisis del modo anterior y NUNCA debe llegar a otra pestaña.
TIPO_RESULTADO = {MODO_BIOMETRIA: Resultado, MODO_EPP: list, MODO_MONITOREO: ResultadoPose}


def resultado_valido(modo: str, resultado) -> bool:
    tipo = TIPO_RESULTADO.get(modo)
    return resultado is not None and tipo is not None and isinstance(resultado, tipo)


class CameraWorker(QThread):
    # frame (BGR, ya girado/reducido; NO modificar) y resultado del análisis (o None)
    frame_listo = pyqtSignal(object, object)
    # estado de la cámara (CONECTANDO/OK/RECONECTANDO/DETENIDA) y texto para mostrar
    estado_camara = pyqtSignal(str, str)
    # texto del problema cuando faltan los modelos de IA (la cámara sigue funcionando)
    error_modelos = pyqtSignal(str)
    # fps reales: (lectura de la cámara, imágenes entregadas a la pantalla), una vez por segundo
    fps_camara = pyqtSignal(float, float)
    # texto del problema cuando falta el modelo YOLO de EPP (se avisa una sola vez)
    error_epp = pyqtSignal(str)
    # prendas que el modelo YOLO de EPP sabe reconocer (set), al cargarlo
    prendas_epp = pyqtSignal(object)
    # texto del problema cuando falta el modelo de pose (Monitoreo; se avisa una sola vez)
    error_pose = pyqtSignal(str)
    # aviso no grave del modelo de pose (p. ej. se usa un modelo de reemplazo menos preciso)
    aviso_pose = pyqtSignal(str)
    # evento de riesgo nuevo (core.riesgos.Evento) y core.riesgos.FotoEvento. Señal aparte para que
    # el control de flujo de frames nunca descarte un evento.
    evento_riesgo = pyqtSignal(object, object)

    def __init__(self, parent=None, fabrica_fuente: Optional[Callable[[Camara], VideoSource]] = None,
                 motor: Optional[FaceEngine] = None, reloj: Callable[[], float] = time.monotonic,
                 analisis_sincrono: bool = False, motor_epp: Optional[EppEngine] = None,
                 motor_pose: Optional[PoseEngine] = None):
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
        self._epp: Optional[EppEngine] = motor_epp   # se carga al entrar por primera vez al modo EPP
        self._epp_intentado = motor_epp is not None
        self._pose: Optional[PoseEngine] = motor_pose  # se carga al entrar por primera vez a Monitoreo
        self._pose_intentado = motor_pose is not None
        self.monitor = MonitorRiesgos()                 # solo se usa desde el hilo de análisis
        self._gen_monitor = -1
        self._t_pose: list = []                         # instantes de los últimos análisis de pose
        self._cortes_fuente = 0
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
    def set_zonas(self, zonas):
        """Zonas restringidas de la cámara activa: [(nombre, x1, y1, x2, y2)] en 0..1."""
        self.monitor.zonas = list(zonas)

    def set_mode(self, mode: str):
        if mode != self.mode:
            self._ultimo_resultado = None   # el resultado de un modo no sirve en otro
        self.mode = mode

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
        if not self._analisis_sincrono:
            self._hilo_analisis = threading.Thread(target=self._bucle_analisis, daemon=True, name="analisis-ia")
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
            self._generacion += 1          # coordenadas nuevas: lo analizado antes ya no sirve
            self._ultimo_resultado = None
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
        cortes = getattr(self._fuente, "cortes", 0)
        if cortes != self._cortes_fuente:      # el video volvió a empezar o se reconectó: la escena "salta"
            self._cortes_fuente = cortes
            self._generacion += 1

        if self._camara.tipo == "usb":
            frame = cv2.flip(frame, 1)  # las cámaras locales se ven como un espejo

        resultado = None
        modo = self.mode
        intervalo = {MODO_BIOMETRIA: config.FACE_INTERVALO_S, MODO_EPP: config.EPP_INTERVALO_S,
                     MODO_MONITOREO: config.MON_INTERVALO_S}.get(modo)
        if intervalo is not None and self._puede_analizar(modo):
            ahora = self._reloj()
            if ahora - self._ultimo_analisis >= intervalo:
                self._ultimo_analisis = ahora
                if self._analisis_sincrono:
                    self._analizar(frame, self._generacion, modo, ahora)
                else:
                    with self._cond:
                        # solo importa el más reciente; viaja con su generación y su instante de captura
                        self._frame_a_analizar = (frame, self._generacion, modo, ahora)   # mismo orden que _analizar()
                        self._cond.notify()
            resultado = self._ultimo_resultado
            if not resultado_valido(modo, resultado):
                resultado = None   # quedó del modo anterior (un análisis terminó justo al cambiar)

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
                pedido, self._frame_a_analizar = self._frame_a_analizar, None
            if pedido is not None:
                self._analizar(*pedido)

    def _puede_analizar(self, modo: str) -> bool:
        if modo == MODO_BIOMETRIA:
            return self._identificador is not None
        if modo == MODO_EPP:
            return self._epp is not None or not self._epp_intentado
        if modo == MODO_MONITOREO:
            return self._pose is not None or not self._pose_intentado
        return False

    def _cargar_pose(self):
        self._pose_intentado = True
        try:
            self._pose = PoseEngine()
            if self._pose.aviso:
                logger.warning("%s", self._pose.aviso)
                self.aviso_pose.emit(self._pose.aviso)
        except ModeloPoseNoEncontrado as e:
            logger.error("%s", e)
            self.error_pose.emit(str(e))
        except Exception as e:
            logger.exception("No se pudo cargar el modelo de pose")
            self.error_pose.emit(f"No se pudo cargar el modelo de pose: {e}")

    def _analizar_pose(self, frame: np.ndarray, generacion: int, t: float) -> Optional[ResultadoPose]:
        if not self._pose_intentado:
            self._cargar_pose()
        if self._pose is None:
            return None
        if generacion != self._gen_monitor:      # otra cámara, giro o salto del video: se empieza de cero
            self._gen_monitor = generacion
            self.monitor.reiniciar()
            self._t_pose.clear()
        personas = self._pose.detectar(frame)
        h, w = frame.shape[:2]
        estados, eventos = self.monitor.procesar(personas, t, (w, h))
        self._t_pose = [x for x in self._t_pose if t - x <= 3.0] + [t]
        frecuencia = (len(self._t_pose) - 1) / (self._t_pose[-1] - self._t_pose[0]) if len(self._t_pose) > 1 else 0.0
        if eventos:
            foto = FotoEvento(frame.copy(), tuple(personas))
            for ev in eventos:
                self.evento_riesgo.emit(ev, foto)
        return ResultadoPose(tuple(estados), frecuencia, (w, h))

    def _cargar_epp(self):
        """Carga el YOLO de EPP la primera vez que se necesita (importar ultralytics tarda)."""
        self._epp_intentado = True
        try:
            self._epp = EppEngine()
            self.prendas_epp.emit(set(self._epp.prendas_detectables))
        except ModeloEppNoEncontrado as e:
            logger.error("%s", e)
            self.error_epp.emit(str(e))
        except Exception as e:
            logger.exception("No se pudo cargar el modelo de EPP")
            self.error_epp.emit(f"No se pudo cargar el modelo de EPP: {e}")

    def _analizar(self, frame: np.ndarray, generacion: int, modo: str = MODO_BIOMETRIA, t: Optional[float] = None):
        try:
            if modo == MODO_MONITOREO:
                resultado = self._analizar_pose(frame, generacion, self._reloj() if t is None else t)
            elif modo == MODO_EPP:
                if not self._epp_intentado:
                    self._cargar_epp()
                if self._epp is None:
                    return
                resultado = self._epp.detectar(frame)
            else:
                with self._lock_ident:
                    resultado = self._identificador.procesar(frame)
        except Exception:
            logger.exception("Falló el análisis (%s)", modo)
            resultado = None
        if generacion == self._generacion and self.mode == modo:
            self._ultimo_resultado = resultado
