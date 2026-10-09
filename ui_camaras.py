"""Ventana «Cámaras»: ver, agregar, probar, girar, activar y eliminar fuentes de video.

Solo usa ``core.camaras`` (el registro) y ``core.video_source`` (para la vista previa).
No toca la cámara que la aplicación ya está usando: Windows no permite abrir una
cámara USB dos veces, así que esa se ve en la pantalla principal.
"""
import logging
import threading
from typing import Callable, Optional

import numpy as np
from PyQt6.QtCore import QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (QComboBox, QDialog, QFileDialog, QGroupBox, QHBoxLayout, QLabel,
                             QLineEdit, QListWidget, QMessageBox, QPushButton, QVBoxLayout)

import config
from core import camaras
from core.video_source import VideoSource, detectar_usb

logger = logging.getLogger(__name__)

_TIPOS = (("usb", "USB / integrada"), ("url", "IP / celular / seguridad"), ("archivo", "Archivo de video"))
_AYUDA = {
    "usb": "Índice de la cámara: 0 es la integrada; la externa suele ser 1.",
    "url": "Celular con IP Webcam: escribe solo su IP (ej. 192.168.0.123). Cámara de seguridad: dirección completa rtsp://usuario:clave@IP:554/ruta",
    "archivo": "Ruta de un video grabado (sirve como demo de respaldo si falla la cámara).",
}
_ESTILO = """
QDialog { background-color: #1e1e2e; }
QLabel, QGroupBox { color: #dddddd; }
QGroupBox { border: 1px solid #3b3b52; border-radius: 6px; margin-top: 10px; padding-top: 10px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; color: #00e676; }
QListWidget, QLineEdit, QComboBox { background-color: #2b2b3b; color: white; border: 1px solid #3b3b52; padding: 4px; }
QListWidget::item:selected { background-color: #00e676; color: black; }
QPushButton { background-color: #2b2b3b; color: white; border: none; padding: 7px 14px; border-radius: 4px; }
QPushButton:hover { background-color: #3b3b4b; }
QPushButton:disabled { color: #666; }
"""


class CamarasDialog(QDialog):
    cambiado = pyqtSignal()                 # el registro de cámaras cambió (agregar/eliminar/activar/girar)
    _usb_detectadas = pyqtSignal(list)

    def __init__(self, parent=None, fabrica_fuente: Callable = VideoSource,
                 detectar: Callable = detectar_usb):
        super().__init__(parent)
        self.setWindowTitle("Cámaras")
        self.setMinimumSize(760, 560)
        self.setStyleSheet(_ESTILO)
        self._fabrica = fabrica_fuente
        self._detectar = detectar
        self._fuente: Optional[VideoSource] = None
        self._temporizador = QTimer(self)
        self._temporizador.setInterval(50)
        self._temporizador.timeout.connect(self._actualizar_vista)
        self._ultimo_numero = 0
        self._buscando = False
        self._usb_detectadas.connect(self._mostrar_usb_detectadas)

        self._construir()
        self.recargar()

    # ------------------------------------------------------------------ UI
    def _construir(self):
        raiz = QVBoxLayout(self)

        fila = QHBoxLayout()
        self.lista = QListWidget()
        self.lista.currentRowChanged.connect(self._actualizar_botones)
        self.lista.itemDoubleClicked.connect(lambda _i: self.activar())
        fila.addWidget(self.lista, 3)

        columna = QVBoxLayout()
        self.btn_activar = QPushButton("Usar esta cámara")
        self.btn_probar = QPushButton("▶ Probar")
        self.btn_girar = QPushButton("⟳ Girar 90°")
        self.btn_eliminar = QPushButton("Eliminar")
        self.btn_activar.clicked.connect(self.activar)
        self.btn_probar.clicked.connect(self.alternar_prueba)
        self.btn_girar.clicked.connect(self.girar)
        self.btn_eliminar.clicked.connect(self.eliminar)
        for b in (self.btn_activar, self.btn_probar, self.btn_girar, self.btn_eliminar):
            columna.addWidget(b)
        columna.addStretch()
        fila.addLayout(columna, 1)
        raiz.addLayout(fila, 2)

        self.vista = QLabel("Selecciona una cámara y pulsa «Probar» para ver su imagen aquí.")
        self.vista.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.vista.setMinimumHeight(200)
        self.vista.setStyleSheet("background-color: #000; color: #aaa; border-radius: 6px;")
        raiz.addWidget(self.vista, 3)

        grupo = QGroupBox("Agregar cámara")
        caja = QVBoxLayout(grupo)
        linea = QHBoxLayout()
        self.combo_tipo = QComboBox()
        for clave, texto in _TIPOS:
            self.combo_tipo.addItem(texto, clave)
        self.combo_tipo.currentIndexChanged.connect(self._tipo_cambiado)
        self.entrada_nombre = QLineEdit()
        self.entrada_nombre.setPlaceholderText("Nombre (ej. Celular de Carlos)")
        self.entrada_valor = QLineEdit()
        linea.addWidget(self.combo_tipo, 2)
        linea.addWidget(self.entrada_nombre, 3)
        linea.addWidget(self.entrada_valor, 4)
        caja.addLayout(linea)

        self.lbl_ayuda = QLabel()
        self.lbl_ayuda.setWordWrap(True)
        self.lbl_ayuda.setStyleSheet("color: #aaa; font-size: 11px;")
        caja.addWidget(self.lbl_ayuda)

        botones = QHBoxLayout()
        self.btn_buscar = QPushButton("Buscar cámaras USB")
        self.btn_examinar = QPushButton("Examinar…")
        self.btn_agregar = QPushButton("Agregar")
        self.btn_agregar.setStyleSheet("QPushButton { background-color: #00e676; color: black; font-weight: bold; }")
        self.btn_buscar.clicked.connect(self.buscar_usb)
        self.btn_examinar.clicked.connect(self._examinar)
        self.btn_agregar.clicked.connect(self.agregar)
        botones.addWidget(self.btn_buscar)
        botones.addWidget(self.btn_examinar)
        botones.addStretch()
        botones.addWidget(self.btn_agregar)
        caja.addLayout(botones)
        raiz.addWidget(grupo)

        self.lbl_mensaje = QLabel("")
        self.lbl_mensaje.setWordWrap(True)
        self.lbl_mensaje.setStyleSheet("color: #00e676;")
        raiz.addWidget(self.lbl_mensaje)

        self._tipo_cambiado()

    def _tipo_cambiado(self):
        tipo = self.combo_tipo.currentData()
        self.lbl_ayuda.setText(_AYUDA[tipo])
        self.entrada_valor.setPlaceholderText({"usb": "1", "url": "192.168.0.123",
                                               "archivo": "C:\\videos\\demo.mp4"}[tipo])
        self.btn_buscar.setVisible(tipo == "usb")
        self.btn_examinar.setVisible(tipo == "archivo")

    # ------------------------------------------------------------- lista
    def recargar(self, seleccionar: Optional[str] = None):
        actual = seleccionar or self._id_seleccionado()
        activa = camaras.activa().id
        self.lista.blockSignals(True)
        self.lista.clear()
        self._ids = []
        for cam in camaras.listar():
            marca = "  ★ en uso" if cam.id == activa else ""
            self.lista.addItem(cam.resumen() + marca)
            self._ids.append(cam.id)
        fila = self._ids.index(actual) if actual in self._ids else self._ids.index(activa)
        self.lista.setCurrentRow(fila)
        self.lista.blockSignals(False)
        self._actualizar_botones()

    def _id_seleccionado(self) -> Optional[str]:
        fila = self.lista.currentRow()
        return self._ids[fila] if 0 <= fila < len(getattr(self, "_ids", [])) else None

    def _actualizar_botones(self, *_):
        cam_id = self._id_seleccionado()
        hay = cam_id is not None
        es_activa = hay and cam_id == camaras.activa().id
        self.btn_activar.setEnabled(hay and not es_activa)
        self.btn_girar.setEnabled(hay)
        self.btn_eliminar.setEnabled(hay and self.lista.count() > 1)
        self.btn_probar.setEnabled(hay)
        if self._fuente is not None:
            self.detener_prueba()

    def _aviso(self, texto: str, error: bool = False):
        self.lbl_mensaje.setStyleSheet(f"color: {'#ff5252' if error else '#00e676'};")
        self.lbl_mensaje.setText(texto)

    # ----------------------------------------------------------- acciones
    def activar(self):
        cam_id = self._id_seleccionado()
        if cam_id is None:
            return
        camaras.activar(cam_id)
        self.recargar()
        self._aviso(f"«{camaras.obtener(cam_id).nombre}» es ahora la cámara en uso.")
        self.cambiado.emit()

    def girar(self):
        cam_id = self._id_seleccionado()
        if cam_id is None:
            return
        cam = camaras.obtener(cam_id)
        nueva = (cam.rotacion + 90) % 360
        camaras.rotar(cam_id, nueva)
        if self._fuente is not None:
            self._fuente.rotacion = nueva
        self.recargar(cam_id)
        self._aviso(f"«{cam.nombre}»: giro {nueva}°.")
        self.cambiado.emit()

    def eliminar(self):
        cam_id = self._id_seleccionado()
        if cam_id is None:
            return
        cam = camaras.obtener(cam_id)
        if QMessageBox.question(self, "Eliminar cámara", f"¿Eliminar «{cam.nombre}»?") != QMessageBox.StandardButton.Yes:
            return
        self.detener_prueba()
        try:
            camaras.eliminar(cam_id)
        except ValueError as e:
            self._aviso(str(e), error=True)
            return
        self.recargar()
        self._aviso(f"«{cam.nombre}» eliminada.")
        self.cambiado.emit()

    def agregar(self):
        tipo = self.combo_tipo.currentData()
        try:
            cam = camaras.agregar(self.entrada_nombre.text(), tipo, self.entrada_valor.text())
        except ValueError as e:
            self._aviso(str(e), error=True)
            return
        self.entrada_nombre.clear()
        self.entrada_valor.clear()
        self.recargar(cam.id)
        self._aviso(f"«{cam.nombre}» agregada ({cam.valor_visible}). Pulsa «Probar» para ver su imagen.")
        self.cambiado.emit()

    def _examinar(self):
        ruta, _ = QFileDialog.getOpenFileName(self, "Video", "", "Videos (*.mp4 *.avi *.mkv *.mov);;Todos (*.*)")
        if ruta:
            self.entrada_valor.setText(ruta)

    # --------------------------------------------------------- USB
    def buscar_usb(self):
        if self._buscando:
            return
        self._buscando = True
        self.btn_buscar.setEnabled(False)
        self._aviso("Buscando cámaras USB… (puede tardar unos segundos)")

        def trabajo():
            try:
                encontradas = self._detectar()
            except Exception:
                logger.exception("Falló la búsqueda de cámaras USB")
                encontradas = []
            try:
                self._usb_detectadas.emit(encontradas)
            except RuntimeError:
                pass  # la ventana se cerró mientras buscaba
        threading.Thread(target=trabajo, daemon=True, name="buscar-usb").start()

    def _mostrar_usb_detectadas(self, encontradas: list):
        self._buscando = False
        self.btn_buscar.setEnabled(True)
        if not encontradas:
            self._aviso("No se encontró ninguna cámara USB libre. (La que la aplicación ya usa no aparece: está ocupada.)", True)
            return
        registradas = {c.valor for c in camaras.listar() if c.tipo == "usb"}
        nuevas = [e for e in encontradas if str(e["indice"]) not in registradas]
        texto = ", ".join(f"índice {e['indice']} ({e['ancho']}x{e['alto']})" for e in encontradas)
        if nuevas:
            self.entrada_valor.setText(str(nuevas[0]["indice"]))
            if not self.entrada_nombre.text():
                self.entrada_nombre.setText(f"Cámara USB {nuevas[0]['indice']}")
            self._aviso(f"Encontradas: {texto}. Se rellenó la primera sin registrar; pulsa «Agregar».")
        else:
            self._aviso(f"Encontradas: {texto}. Todas ya están registradas.")

    # --------------------------------------------------------- vista previa
    def alternar_prueba(self):
        if self._fuente is not None:
            self.detener_prueba()
            return
        cam_id = self._id_seleccionado()
        if cam_id is None:
            return
        cam = camaras.obtener(cam_id)
        if cam.tipo == "usb" and cam_id == camaras.activa().id:
            self._aviso("Esta cámara está en uso por el sistema: se ve en la pantalla principal. "
                        "Para probarla aquí, activa otra cámara primero.", True)
            return
        self._ultimo_numero = 0
        self._fuente = self._fabrica(cam)
        self._fuente.iniciar()
        self.vista.setText(f"Conectando a «{cam.nombre}»…")
        self.btn_probar.setText("■ Detener")
        self._temporizador.start()

    def detener_prueba(self):
        self._temporizador.stop()
        if self._fuente is not None:
            self._fuente.detener()
            self._fuente = None
        self.btn_probar.setText("▶ Probar")
        self.vista.clear()
        self.vista.setText("Selecciona una cámara y pulsa «Probar» para ver su imagen aquí.")

    def _actualizar_vista(self):
        if self._fuente is None:
            return
        frame, numero = self._fuente.leer()
        if frame is None:
            self.vista.setText(self._fuente.mensaje)
            return
        if numero == self._ultimo_numero:
            return
        self._ultimo_numero = numero
        frame = np.ascontiguousarray(frame)
        alto, ancho = frame.shape[:2]
        imagen = QImage(frame.data, ancho, alto, 3 * ancho, QImage.Format.Format_BGR888)
        self.vista.setPixmap(QPixmap.fromImage(imagen).scaled(
            self.vista.width(), self.vista.height(),
            Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation))

    # --------------------------------------------------------- cierre
    def done(self, resultado):
        self.detener_prueba()
        super().done(resultado)

    def closeEvent(self, evento):
        self.detener_prueba()
        super().closeEvent(evento)
