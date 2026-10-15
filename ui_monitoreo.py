"""Pestaña "Monitoreo de Riesgos y SOS": video con esqueletos, emergencias, eventos, zonas y reporte.

La lógica (reglas, eventos, BD) vive en core/; aquí solo se dibuja y se conecta con la interfaz.
"""
import datetime
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PyQt6.QtCore import QRect, QRectF, Qt, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QDesktopServices, QFont, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMessageBox, QPushButton,
                             QScrollArea, QSizePolicy, QVBoxLayout, QWidget)

import config
from core import database, reporte, zonas
from core.eventos import EventoRegistrado, GestorEventos, nombre_tipo
from core.postura import ESQUELETO
from core.riesgos import AMARILLO, CRITICO, ROJO, ResultadoPose
from core.sound import play_alert

logger = logging.getLogger(__name__)

COLORES = {"verde": QColor("#00e676"), AMARILLO: QColor("#ffb74d"), ROJO: QColor("#ff1744")}
ESTILO_BOTON = ("QPushButton { background-color: #2b2b3b; color: white; border: none; padding: 7px 10px; "
                "border-radius: 4px; } QPushButton:hover { background-color: #3b3b4b; } "
                "QPushButton:checked { background-color: #ff1744; }")


class VideoMonitoreo(QLabel):
    """Muestra el video y permite dibujar zonas (clic y arrastre) cuando el modo dibujo está activo."""
    zona_dibujada = pyqtSignal(float, float, float, float)   # x1, y1, x2, y2 en 0..1 del frame

    def __init__(self):
        super().__init__("Cargando cámara de Monitoreo…")
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)   # ver nota en EPP (main.py)
        self.setStyleSheet("background-color: #000; color: #888;")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.dibujando = False
        self._inicio = None
        self._actual = None
        self.area_imagen = QRect()        # dónde quedó la imagen dentro del recuadro (sin franjas negras)

    def activar_dibujo(self, activo: bool):
        self.dibujando = activo
        self._inicio = self._actual = None
        self.setCursor(Qt.CursorShape.CrossCursor if activo else Qt.CursorShape.ArrowCursor)
        if activo:
            self.setFocus()

    def a_normalizado(self, x: float, y: float):
        a = self.area_imagen
        if a.width() <= 0 or a.height() <= 0:
            return None
        return (min(max((x - a.x()) / a.width(), 0.0), 1.0), min(max((y - a.y()) / a.height(), 0.0), 1.0))

    def mousePressEvent(self, e):
        if self.dibujando and e.button() == Qt.MouseButton.LeftButton:
            self._inicio = self._actual = e.position()
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self.dibujando and self._inicio is not None:
            self._actual = e.position()
            self.update()
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self.dibujando and self._inicio is not None and e.button() == Qt.MouseButton.LeftButton:
            a = self.a_normalizado(self._inicio.x(), self._inicio.y())
            b = self.a_normalizado(e.position().x(), e.position().y())
            self._inicio = self._actual = None
            if a and b and abs(a[0] - b[0]) > 0.02 and abs(a[1] - b[1]) > 0.02:
                self.zona_dibujada.emit(a[0], a[1], b[0], b[1])
        super().mouseReleaseEvent(e)

    def keyPressEvent(self, e):
        if self.dibujando and e.key() == Qt.Key.Key_Escape:
            self.zona_dibujada.emit(-1, -1, -1, -1)    # cancelar
            return
        super().keyPressEvent(e)

    def paintEvent(self, e):
        super().paintEvent(e)
        if self.dibujando and self._inicio is not None and self._actual is not None:
            p = QPainter(self)
            p.setPen(QPen(QColor("#ff1744"), 2, Qt.PenStyle.DashLine))
            p.setBrush(QBrush(QColor(255, 23, 68, 50)))
            p.drawRect(QRectF(self._inicio, self._actual).normalized())
            p.end()


class TarjetaEmergencia(QFrame):
    atendida = pyqtSignal(int)

    def __init__(self, ev_id: int, tipo: str, persona: str, hora: str, foto: Optional[str], suena: bool):
        super().__init__()
        self.ev_id, self.hora, self.suena = ev_id, hora, suena
        self.setObjectName("cardEmergencia")
        self.setStyleSheet("QFrame#cardEmergencia { background-color: #4a1018; border: 2px solid #ff1744; "
                           "border-radius: 8px; } QLabel { border: none; background: transparent; }")
        fila = QHBoxLayout(self)
        self.lbl_foto = QLabel()
        self.lbl_foto.setFixedSize(90, 68)
        if foto and Path(foto).exists():
            self.lbl_foto.setPixmap(QPixmap(foto).scaled(90, 68, Qt.AspectRatioMode.KeepAspectRatio,
                                                         Qt.TransformationMode.SmoothTransformation))
        else:
            self.lbl_foto.setText("sin foto")
            self.lbl_foto.setStyleSheet("color: #aaa; font-size: 10px;")
        texto = QVBoxLayout()
        titulo = QLabel(f"⚠ {nombre_tipo(tipo).upper()} – {persona}")
        titulo.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        titulo.setStyleSheet("color: #ffffff;")
        titulo.setWordWrap(True)
        self.lbl_tiempo = QLabel()
        self.lbl_tiempo.setStyleSheet("color: #ffcdd2; font-size: 12px;")
        btn = QPushButton("✔ Atendido")
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setStyleSheet("QPushButton { background-color: #ffffff; color: #b71c1c; font-weight: bold; border: none; "
                          "padding: 6px 12px; border-radius: 4px; } QPushButton:hover { background-color: #ffcdd2; }")
        btn.clicked.connect(lambda: self.atendida.emit(self.ev_id))
        texto.addWidget(titulo)
        texto.addWidget(self.lbl_tiempo)
        texto.addWidget(btn, alignment=Qt.AlignmentFlag.AlignLeft)
        fila.addWidget(self.lbl_foto)
        fila.addLayout(texto, 1)
        self.actualizar()

    def actualizar(self):
        try:
            inicio = datetime.datetime.strptime(self.hora, "%Y-%m-%d %H:%M:%S")
            seg = int((datetime.datetime.now() - inicio).total_seconds())
        except ValueError:
            seg = 0
        hace = f"{seg // 60} min {seg % 60} s" if seg >= 60 else f"{seg} s"
        self.lbl_tiempo.setText(f"{self.hora[11:]} · hace {hace} · sin atender")


class PanelMonitoreo(QWidget):
    # cantidad de emergencias sin atender (para el aviso de la barra superior)
    emergencias_cambiaron = pyqtSignal(int)

    def __init__(self, worker, camara_activa, parent=None, gestor: Optional[GestorEventos] = None):
        super().__init__(parent)
        self.worker = worker
        self._camara = camara_activa
        self.gestor = gestor or GestorEventos()
        self.tarjetas: Dict[int, TarjetaEmergencia] = {}
        self._error_pose: Optional[str] = None
        self._ultimo: Optional[ResultadoPose] = None
        self._zonas: List = []
        self._construir()

        self._timer_sonido = QTimer(self)
        self._timer_sonido.timeout.connect(self._repetir_sonido)
        self._timer_sonido.start(int(config.MON_REPETIR_SONIDO_S * 1000))
        self._timer_reloj = QTimer(self)
        self._timer_reloj.timeout.connect(self._cada_segundo)
        self._timer_reloj.start(1000)

        self.worker.evento_riesgo.connect(self.on_evento)
        self.worker.error_pose.connect(self.on_error_pose)
        self.worker.aviso_pose.connect(self.on_aviso_pose)
        self._iniciar_datos()

    # ------------------------------------------------------------------ UI
    def _construir(self):
        capa = QHBoxLayout(self)
        capa.setContentsMargins(0, 0, 0, 0)
        capa.setSpacing(0)
        self.video = VideoMonitoreo()
        self.video.zona_dibujada.connect(self._zona_dibujada)

        panel = QWidget()
        panel.setStyleSheet("background-color: #181825;")
        col = QVBoxLayout(panel)
        col.setContentsMargins(18, 18, 18, 18)
        col.setSpacing(10)

        titulo = QLabel("MONITOREO DE RIESGOS Y SOS")
        titulo.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        titulo.setStyleSheet("color: #ff1744;")
        self.lbl_estado = QLabel("● Esperando la cámara…")
        self.lbl_estado.setWordWrap(True)
        self.lbl_estado.setStyleSheet("color: #aaa; font-weight: bold;")

        self.caja_emergencias = QVBoxLayout()
        self.caja_emergencias.setSpacing(8)

        lbl_personas_t = QLabel("PERSONAS EN CÁMARA")
        lbl_personas_t.setStyleSheet("color: #888; font-size: 11px; font-weight: bold;")
        self.lbl_personas = QLabel("—")
        self.lbl_personas.setTextFormat(Qt.TextFormat.RichText)
        self.lbl_personas.setWordWrap(True)
        self.lbl_personas.setStyleSheet("color: #ddd; font-size: 12px;")

        lbl_eventos_t = QLabel("EVENTOS DE HOY  (doble clic: ver foto)")
        lbl_eventos_t.setStyleSheet("color: #888; font-size: 11px; font-weight: bold;")
        self.lista_eventos = QListWidget()
        self.lista_eventos.setMinimumHeight(150)
        self.lista_eventos.setStyleSheet("QListWidget { background-color: #11111b; color: #ddd; border: 1px solid #2b2b3b; "
                                         "font-size: 12px; } QListWidget::item { padding: 4px; }")
        self.lista_eventos.itemDoubleClicked.connect(self._abrir_foto)

        botones = QHBoxLayout()
        self.btn_zona = QPushButton("▭ Dibujar zona")
        self.btn_zona.setCheckable(True)
        self.btn_zona.setToolTip("Clic y arrastre sobre el video para marcar una zona restringida (Esc cancela)")
        self.btn_zona.toggled.connect(self._modo_dibujo)
        self.btn_borrar_zonas = QPushButton("Borrar zonas")
        self.btn_borrar_zonas.clicked.connect(self._borrar_zonas)
        self.btn_reporte = QPushButton("📄 Reporte del día")
        self.btn_reporte.clicked.connect(self._reporte)
        for b in (self.btn_zona, self.btn_borrar_zonas, self.btn_reporte):
            b.setStyleSheet(ESTILO_BOTON)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            botones.addWidget(b)

        self.lbl_aviso = QLabel("")
        self.lbl_aviso.setWordWrap(True)
        self.lbl_aviso.setStyleSheet("color: #ffb74d; font-size: 11px;")
        self.lbl_aviso.hide()
        ayuda = QLabel("Para caídas y posturas: cámara a 2–4 m, de lado o en diagonal, con el cuerpo completo "
                       "a la vista. Las personas se muestran como P1, P2…; no se identifican.")
        ayuda.setWordWrap(True)
        ayuda.setStyleSheet("color: #777; font-size: 11px;")

        col.addWidget(titulo)
        col.addWidget(self.lbl_estado)
        col.addLayout(self.caja_emergencias)
        col.addSpacing(4)
        col.addWidget(lbl_personas_t)
        col.addWidget(self.lbl_personas)
        col.addSpacing(4)
        col.addWidget(lbl_eventos_t)
        col.addWidget(self.lista_eventos, 1)
        col.addLayout(botones)
        col.addWidget(self.lbl_aviso)
        col.addWidget(ayuda)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { background-color: #181825; border-left: 1px solid #2b2b3b; }")
        scroll.setWidget(panel)
        capa.addWidget(self.video, stretch=65)
        capa.addWidget(scroll, stretch=35)

    # ------------------------------------------------------------------ datos
    def _iniciar_datos(self):
        try:
            self.gestor.limpiar_retencion()
        except Exception:
            logger.exception("No se pudo aplicar la retención de fotos de eventos")
        try:
            for fila in self.gestor.abiertos():
                self._agregar_tarjeta(fila["id"], fila["tipo"], fila["persona"], fila["timestamp"],
                                      fila["foto_path"], suena=False)   # recargadas al iniciar: sin volver a sonar
        except Exception:
            logger.exception("No se pudieron recargar las emergencias abiertas")
        self._recargar_lista()
        self._aplicar_zonas()

    def _recargar_lista(self):
        self.lista_eventos.clear()
        try:
            filas = database.eventos_dia(datetime.date.today().strftime("%Y-%m-%d"), limite=20)
        except Exception:
            logger.exception("No se pudieron leer los eventos del día")
            return
        for f in filas:
            self._item_evento(f, al_final=True)

    def _item_evento(self, f: dict, al_final: bool):
        if f["grupo"] == CRITICO:
            estado = "Pendiente"
            if f.get("atendido_en"):
                try:
                    a = datetime.datetime.strptime(f["atendido_en"], "%Y-%m-%d %H:%M:%S")
                    b = datetime.datetime.strptime(f["timestamp"], "%Y-%m-%d %H:%M:%S")
                    estado = f"Atendido en {int((a - b).total_seconds())} s"
                except ValueError:
                    estado = "Atendido"
            icono = "🚨"
        else:
            estado, icono = "Informativo", "⚠" if f["grupo"] == "ERGONOMIA" else "⛔"
        item = QListWidgetItem(f"{f['timestamp'][11:]}  {icono} {nombre_tipo(f['tipo'])} · {f['persona']} — {estado}")
        item.setData(Qt.ItemDataRole.UserRole, f.get("foto_path"))
        item.setToolTip(f["detalle"])
        if al_final:
            self.lista_eventos.addItem(item)
        else:
            self.lista_eventos.insertItem(0, item)
            while self.lista_eventos.count() > 20:
                self.lista_eventos.takeItem(self.lista_eventos.count() - 1)

    def _abrir_foto(self, item: QListWidgetItem):
        ruta = item.data(Qt.ItemDataRole.UserRole)
        if ruta and Path(ruta).exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(ruta)))
        else:
            QMessageBox.information(self, "Foto", "Foto no disponible (no se guardó o se borró por antigüedad).")

    # ------------------------------------------------------------------ emergencias
    def _agregar_tarjeta(self, ev_id, tipo, persona, hora, foto, suena: bool):
        if ev_id in self.tarjetas:
            return
        t = TarjetaEmergencia(ev_id, tipo, persona, hora, foto, suena)
        t.atendida.connect(self.atender)
        self.caja_emergencias.addWidget(t)
        self.tarjetas[ev_id] = t
        self.emergencias_cambiaron.emit(len(self.tarjetas))

    def atender(self, ev_id: int):
        try:
            self.gestor.atender(ev_id)
        except Exception:
            logger.exception("No se pudo marcar el evento %s como atendido", ev_id)
        t = self.tarjetas.pop(ev_id, None)
        if t is not None:
            t.setParent(None)
            t.deleteLater()
        self.emergencias_cambiaron.emit(len(self.tarjetas))
        self._recargar_lista()

    def _repetir_sonido(self):
        if any(t.suena for t in self.tarjetas.values()):
            play_alert()

    def _cada_segundo(self):
        for t in self.tarjetas.values():
            t.actualizar()

    def on_evento(self, evento, foto):
        """Señal del worker: guarda el evento (foto, BD, alarma) y lo muestra."""
        try:
            reg: Optional[EventoRegistrado] = self.gestor.registrar(evento, foto, self._camara.nombre)
        except Exception:
            logger.exception("No se pudo registrar el evento %s", getattr(evento, "tipo", "?"))
            return
        if reg is None:
            return
        if reg.grupo == CRITICO:
            self._agregar_tarjeta(reg.id, reg.tipo, reg.persona, reg.hora, reg.foto_path, suena=reg.sonido)
        self._item_evento({"timestamp": reg.hora, "grupo": reg.grupo, "tipo": reg.tipo, "persona": reg.persona,
                           "detalle": reg.detalle, "foto_path": reg.foto_path, "atendido_en": None}, al_final=False)

    def on_aviso_pose(self, texto: str):
        self.lbl_aviso.setText(f"ℹ {texto}")
        self.lbl_aviso.show()

    def on_error_pose(self, texto: str):
        self._error_pose = texto
        self.lbl_estado.setText(f"⚠ Modelo de pose no disponible\n{texto}")
        self.lbl_estado.setStyleSheet("color: #ffb74d; font-weight: bold;")

    # ------------------------------------------------------------------ zonas
    def camara_cambiada(self, camara, girada: bool):
        """La llama la ventana ANTES de que el worker gire o cambie de cámara."""
        if girada and zonas.de_camara(camara.id):
            zonas.borrar(camara.id)
            QMessageBox.information(self, "Zonas restringidas",
                                    "Las zonas de esta cámara se borraron porque giraste la imagen.\n"
                                    "Vuelve a dibujarlas si las necesitas.")
        self._camara = camara
        self._aplicar_zonas()

    def _aplicar_zonas(self):
        try:
            self._zonas = zonas.de_camara(self._camara.id)
        except Exception:
            logger.exception("No se pudieron leer las zonas")
            self._zonas = []
        self.worker.set_zonas(self._zonas)
        self.btn_borrar_zonas.setEnabled(bool(self._zonas))

    def _modo_dibujo(self, activo: bool):
        self.video.activar_dibujo(activo)

    def _zona_dibujada(self, x1, y1, x2, y2):
        self.btn_zona.setChecked(False)
        if x1 < 0:
            return
        zonas.agregar(self._camara.id, x1, y1, x2, y2)
        self._aplicar_zonas()

    def _borrar_zonas(self):
        zonas.borrar(self._camara.id)
        self._aplicar_zonas()

    # ------------------------------------------------------------------ reporte
    def _reporte(self):
        try:
            ruta = reporte.generar()
        except Exception as e:
            logger.exception("No se pudo generar el reporte")
            QMessageBox.warning(self, "Reporte", f"No se pudo generar el reporte:\n{e}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(ruta)))

    # ------------------------------------------------------------------ video
    def al_entrar(self):
        self._ultimo = None
        self._aplicar_zonas()

    def mostrar(self, frame: np.ndarray, resultado: Optional[ResultadoPose]):
        if resultado is not None:
            self._ultimo = resultado
            self._actualizar_panel(resultado)
        lbl_w, lbl_h = self.video.width(), self.video.height()
        if lbl_w <= 0 or lbl_h <= 0 or frame is None:
            return
        frame = np.ascontiguousarray(frame)
        h, w, ch = frame.shape
        img = QImage(frame.data, w, h, ch * w, QImage.Format.Format_BGR888)
        pix = QPixmap.fromImage(img).scaled(lbl_w, lbl_h, Qt.AspectRatioMode.KeepAspectRatio,
                                            Qt.TransformationMode.FastTransformation)
        self.video.area_imagen = QRect((lbl_w - pix.width()) // 2, (lbl_h - pix.height()) // 2,
                                       pix.width(), pix.height())
        self._dibujar(pix, w, h, self._ultimo)
        self.video.setPixmap(pix)

    def _dibujar(self, pix: QPixmap, w: int, h: int, res: Optional[ResultadoPose]):
        esc = pix.width() / max(1, w)
        p = QPainter(pix)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        for nombre, x1, y1, x2, y2 in self._zonas:
            r = QRectF(x1 * pix.width(), y1 * pix.height(), (x2 - x1) * pix.width(), (y2 - y1) * pix.height())
            p.setPen(QPen(QColor("#ff1744"), 2))
            p.setBrush(QBrush(QColor(255, 23, 68, 45)))
            p.drawRect(r)
            p.setPen(QColor("#ffffff"))
            p.setFont(QFont("Arial", 9, QFont.Weight.Bold))
            p.drawText(r.adjusted(4, 2, 0, 0), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
                       f"{nombre} · restringida")
        if res is not None:
            parpadeo = int(time.monotonic() * 2) % 2 == 0
            for e in res.estados:
                color = COLORES.get(e.semaforo, COLORES["verde"])
                pts = np.asarray(e.puntos).reshape(17, 3)
                p.setPen(QPen(color, max(2, int(3 * esc + 0.5))))
                for a, b in ESQUELETO:
                    if pts[a, 2] >= config.MON_PUNTO_CONF_MIN and pts[b, 2] >= config.MON_PUNTO_CONF_MIN:
                        p.drawLine(int(pts[a, 0] * esc), int(pts[a, 1] * esc), int(pts[b, 0] * esc), int(pts[b, 1] * esc))
                x1, y1, x2, y2 = (v * esc for v in e.caja)
                if e.emergencia:
                    if parpadeo:
                        p.setPen(QPen(COLORES[ROJO], 5))
                        p.setBrush(Qt.BrushStyle.NoBrush)
                        p.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))
                texto = f"{e.id} · {e.etiqueta}"
                p.setFont(QFont("Arial", 11 if not e.emergencia else 13, QFont.Weight.Bold))
                ancho = p.fontMetrics().horizontalAdvance(texto) + 10
                ty = max(0.0, y1 - 22)
                p.fillRect(QRectF(x1, ty, ancho, 20), QColor(0, 0, 0, 180))
                p.setPen(color)
                p.drawText(QRectF(x1 + 5, ty, ancho, 20), Qt.AlignmentFlag.AlignVCenter, texto)
        p.end()

    def _actualizar_panel(self, res: ResultadoPose):
        if self._error_pose:
            return
        n = len(res.estados)
        if 0 < res.frecuencia < config.MON_FRECUENCIA_MIN:
            self.lbl_estado.setText(f"⚠ Análisis lento ({res.frecuencia:.1f}/s): caídas y SOS pueden fallar · "
                                    f"{n} persona{'s' if n != 1 else ''}")
            self.lbl_estado.setStyleSheet("color: #ffb74d; font-weight: bold;")
        elif n == 0:
            self.lbl_estado.setText(f"● Activo · sin personas a la vista · {res.frecuencia:.0f} análisis/s")
            self.lbl_estado.setStyleSheet("color: #00e676; font-weight: bold;")
        else:
            self.lbl_estado.setText(f"● Activo · {n} persona{'s' if n != 1 else ''} · {res.frecuencia:.0f} análisis/s")
            self.lbl_estado.setStyleSheet("color: #00e676; font-weight: bold;")
        filas = []
        for e in sorted(res.estados, key=lambda s: int(s.id[1:]) if s.id[1:].isdigit() else 0):
            color = COLORES.get(e.semaforo, COLORES["verde"]).name()
            extra = []
            if e.seg_amarillo >= 1:
                extra.append(f"amarillo {_duracion(e.seg_amarillo)}")
            if e.seg_rojo >= 1:
                extra.append(f"rojo {_duracion(e.seg_rojo)}")
            detalle = e.etiqueta if e.etiqueta != "cuerpo incompleto" else "cuerpo incompleto: no se evalúan piernas"
            filas.append(f"<span style='color:{color}'>●</span> <b>{e.id}</b> · {detalle}"
                         + (f" <span style='color:#888'>({' · '.join(extra)})</span>" if extra else ""))
        self.lbl_personas.setText("<br>".join(filas) if filas else "—")


def _duracion(seg: float) -> str:
    seg = int(seg)
    return f"{seg // 60} min {seg % 60} s" if seg >= 60 else f"{seg} s"
