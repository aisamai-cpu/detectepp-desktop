"""Gestión de eventos de Monitoreo: foto (con cabezas difuminadas), BD, alarma, "Atendido", retención. Sin PyQt.

La UI llama a ``GestorEventos.registrar(evento, foto, camara)`` cuando el worker emite ``evento_riesgo``.
"""
import datetime
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

import config
from core import database
from core.alerts import AlertManager
from core.postura import ESQUELETO, NARIZ, OJO_D, OJO_I, OREJA_D, OREJA_I
from core.riesgos import CRITICO, FotoEvento, PersonaPose, TIPOS

logger = logging.getLogger(__name__)

_CABEZA = (NARIZ, OJO_I, OJO_D, OREJA_I, OREJA_D)


@dataclass(frozen=True)
class EventoRegistrado:
    id: int
    tipo: str
    grupo: str
    hora: str            # 'AAAA-MM-DD HH:MM:SS'
    persona: str
    detalle: str
    foto_path: Optional[str]
    sonido: bool         # si debe sonar (y, si es crítico, repetirse hasta "Atendido")


# ----------------------------------------------------------------------------
# Foto
# ----------------------------------------------------------------------------
def _difuminar(img: np.ndarray, x1, y1, x2, y2):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return
    region = img[y1:y2, x1:x2]
    k = max(3, (min(x2 - x1, y2 - y1) // 3) | 1)
    img[y1:y2, x1:x2] = cv2.GaussianBlur(region, (k, k), 0)


def difuminar_cabezas(img: np.ndarray, personas, detector_rostros=None) -> np.ndarray:
    """Difumina la cabeza de cada persona (por sus puntos; si no hay, el 30 % superior de su caja)
    y, si hay detector de rostros, cualquier rostro que encuentre (personas que la pose no vio)."""
    for p in personas:
        x1, y1, x2, y2 = p.caja
        pts = np.asarray(p.puntos).reshape(17, 3)
        cabeza = pts[list(_CABEZA)]
        cabeza = cabeza[cabeza[:, 2] >= config.MON_PUNTO_CONF_MIN]
        hombros = pts[[5, 6]]
        hombros = hombros[hombros[:, 2] >= config.MON_PUNTO_CONF_MIN]
        if len(cabeza):
            c = cabeza[:, :2].mean(axis=0)
            ancho_hombros = float(np.ptp(hombros[:, 0])) if len(hombros) == 2 else 0.0
            r = max(0.6 * ancho_hombros, 0.15 * max(x2 - x1, y2 - y1), 10.0)   # generoso: mejor de más
            _difuminar(img, c[0] - r, c[1] - r, c[0] + r, c[1] + r)
        else:
            _difuminar(img, x1, y1, x2, y1 + 0.3 * (y2 - y1))
    if detector_rostros is not None:
        try:
            h, w = img.shape[:2]
            detector_rostros.setInputSize((w, h))
            _, caras = detector_rostros.detect(img)
            for f in (caras if caras is not None else []):
                x, y, cw, ch = f[:4]
                _difuminar(img, x - 0.2 * cw, y - 0.2 * ch, x + 1.2 * cw, y + 1.2 * ch)
        except cv2.error:
            logger.debug("Detector de rostros no disponible para difuminar", exc_info=True)
    return img


def dibujar_foto(foto: FotoEvento, evento_caja, difuminar: bool, detector_rostros=None) -> np.ndarray:
    img = np.ascontiguousarray(foto.frame).copy()
    if difuminar:
        difuminar_cabezas(img, foto.personas, detector_rostros)
    for p in foto.personas:
        pts = np.asarray(p.puntos).reshape(17, 3)
        for a, b in ESQUELETO:
            if pts[a, 2] >= config.MON_PUNTO_CONF_MIN and pts[b, 2] >= config.MON_PUNTO_CONF_MIN:
                cv2.line(img, tuple(int(v) for v in pts[a, :2]), tuple(int(v) for v in pts[b, :2]),
                         (0, 230, 118), 2, cv2.LINE_AA)
    x1, y1, x2, y2 = (int(v) for v in evento_caja)
    cv2.rectangle(img, (x1, y1), (x2, y2), (68, 23, 255), 3)   # rojo; sin texto (cv2 no dibuja tildes)
    return img


def _crear_detector_rostros():
    if not config.MON_DIFUMINAR_ROSTROS or not Path(config.YUNET_PATH).exists():
        return None
    try:
        return cv2.FaceDetectorYN.create(str(config.YUNET_PATH), "", (320, 320), 0.6, 0.3, 5000)
    except cv2.error:
        logger.warning("No se pudo cargar YuNet para difuminar rostros en las fotos de eventos")
        return None


# ----------------------------------------------------------------------------
# Gestor
# ----------------------------------------------------------------------------
class GestorEventos:
    def __init__(self, reloj=datetime.datetime.now):
        self._ahora = reloj
        self._detector = None
        self._detector_intentado = False

    def _detector_rostros(self):
        if not self._detector_intentado:
            self._detector_intentado = True
            self._detector = _crear_detector_rostros()
        return self._detector

    def _guardar_foto(self, evento, foto: FotoEvento, ahora: datetime.datetime) -> Optional[str]:
        try:
            img = dibujar_foto(foto, evento.caja, config.MON_DIFUMINAR_ROSTROS,
                               self._detector_rostros() if config.MON_DIFUMINAR_ROSTROS else None)
            carpeta = config.EVENTOS_DIR / ahora.strftime("%Y-%m-%d")
            carpeta.mkdir(parents=True, exist_ok=True)
            ruta = carpeta / f"{evento.tipo}_{ahora.strftime('%H%M%S')}_{evento.persona}.jpg"
            ok, buf = cv2.imencode(".jpg", img)
            if not ok:
                return None
            buf.tofile(str(ruta))
            return str(ruta)
        except Exception:
            logger.exception("No se pudo guardar la foto del evento %s", evento.tipo)
            return None

    def registrar(self, evento, foto: Optional[FotoEvento], camara: str) -> Optional[EventoRegistrado]:
        """Guarda el evento. Devuelve None si se descartó por ser un duplicado de una emergencia abierta."""
        ahora = self._ahora()
        hora = ahora.strftime("%Y-%m-%d %H:%M:%S")
        limite = (ahora - datetime.timedelta(seconds=config.MON_ENFRIAMIENTO_S)).strftime("%Y-%m-%d %H:%M:%S")
        sonido, relacionado = evento.sonido, None
        if evento.grupo == CRITICO:
            abiertos = database.eventos_abiertos(limite)
            if any(a["tipo"] == evento.tipo for a in abiertos):
                logger.info("Evento %s descartado: ya hay uno igual sin atender", evento.tipo)
                return None
            if evento.tipo == "PERSONA_INMOVIL":
                caidas = [a for a in abiertos if a["tipo"] == "CAIDA"]
                if caidas:
                    relacionado = caidas[-1]["id"]
                    sonido = False      # la alarma de la caída ya está sonando
        foto_path = self._guardar_foto(evento, foto, ahora) if foto is not None else None
        alerta_id = AlertManager.trigger_alert(evento.tipo, camara, evento.detalle, sonido=sonido)
        try:
            evento_id = database.registrar_evento(evento.tipo, evento.grupo, camara, evento.persona, evento.detalle,
                                                  foto_path, alerta_id, relacionado, timestamp=hora)
        except Exception:
            logger.exception("No se pudo guardar el evento %s en la base de datos", evento.tipo)
            return None
        return EventoRegistrado(evento_id, evento.tipo, evento.grupo, hora, evento.persona, evento.detalle,
                                foto_path, sonido)

    def atender(self, evento_id: int) -> Optional[str]:
        return database.atender_evento(evento_id)

    def abiertos(self, horas: int = 24) -> List[dict]:
        desde = (self._ahora() - datetime.timedelta(hours=horas)).strftime("%Y-%m-%d %H:%M:%S")
        return database.eventos_abiertos(desde)

    def limpiar_retencion(self) -> int:
        """Borra las carpetas de fotos de eventos con más de MON_RETENCION_DIAS días. Devuelve cuántas."""
        limite = (self._ahora() - datetime.timedelta(days=config.MON_RETENCION_DIAS)).date()
        borradas = 0
        if config.EVENTOS_DIR.exists():
            for carpeta in config.EVENTOS_DIR.iterdir():
                try:
                    fecha = datetime.datetime.strptime(carpeta.name, "%Y-%m-%d").date()
                except ValueError:
                    continue
                if carpeta.is_dir() and fecha < limite:
                    shutil.rmtree(carpeta, ignore_errors=True)
                    borradas += 1
        try:
            database.quitar_fotos_anteriores(limite.strftime("%Y-%m-%d"))
        except Exception:
            logger.exception("No se pudieron limpiar las rutas de fotos viejas")
        return borradas


def nombre_tipo(tipo: str) -> str:
    return TIPOS.get(tipo, (None, tipo))[1]
