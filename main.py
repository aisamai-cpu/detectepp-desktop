import logging
import os
import sys
import cv2
import numpy as np
import time
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QTabWidget, 
                             QComboBox, QRadioButton, QMessageBox, QDialog,
                             QFrame, QTextEdit, QStackedWidget, QSizePolicy,
                             QScrollArea)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap, QFont, QPainter, QPen, QColor

import config
from core import camaras, face_store
from core.alerts import AlertManager
from core.intrusos import DetectorIntrusos
from core.database import (init_db, get_todos_empleados, registrar_marcacion, get_epp_requeridos,
                           get_areas, set_area_empleado)
from core.epp import AlertaEpp, SeguimientoEpp, clasificar_caja, OK, FALTA, NO_VALIDA
from camera_worker import CameraWorker, MODO_BIOMETRIA, MODO_EPP, MODO_MONITOREO, resultado_valido
from ui_camaras import CamarasDialog
from ui_empleado import DialogoNuevoEmpleado
from ui_monitoreo import PanelMonitoreo
from vision import registro

logger = logging.getLogger(__name__)

# ------------------- PANTALLA DE BIENVENIDA (PANTALLA COMPLETA) -------------------
class WelcomeScreen(QWidget):
    def __init__(self, on_enter_callback, on_privacy_callback, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background-color: #1a1a24;")

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Contenedor central
        center_layout = QVBoxLayout()
        center_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Título
        title = QLabel("DETECTEPP")
        title.setFont(QFont("Arial", 42, QFont.Weight.Bold))
        title.setStyleSheet("color: #00e676; letter-spacing: 2px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Subtítulo
        subtitle = QLabel("Sistema Inteligente de Control de EPP y Seguridad Industrial")
        subtitle.setFont(QFont("Arial", 14))
        subtitle.setStyleSheet("color: #ffffff; margin-top: 10px; margin-bottom: 30px;")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Botón Ingresar
        btn_enter = QPushButton("INGRESAR AL SISTEMA")
        btn_enter.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        btn_enter.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_enter.setStyleSheet("""
            QPushButton {
                background-color: #00e676;
                color: #000000;
                padding: 15px 40px;
                border-radius: 8px;
                border: none;
            }
            QPushButton:hover {
                background-color: #00c853;
            }
        """)
        btn_enter.clicked.connect(on_enter_callback)

        center_layout.addWidget(title)
        center_layout.addWidget(subtitle)
        center_layout.addWidget(btn_enter, alignment=Qt.AlignmentFlag.AlignCenter)

        # Pie de página (Políticas de Privacidad)
        footer_layout = QHBoxLayout()
        footer_layout.setContentsMargins(20, 20, 30, 20)
        footer_layout.addStretch()

        btn_privacy = QPushButton("Políticas de Privacidad")
        btn_privacy.setFont(QFont("Arial", 10))
        btn_privacy.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_privacy.setStyleSheet("""
            QPushButton {
                color: #8888aa;
                background: transparent;
                border: none;
                text-decoration: underline;
            }
            QPushButton:hover {
                color: #00e676;
            }
        """)
        btn_privacy.clicked.connect(on_privacy_callback)
        footer_layout.addWidget(btn_privacy)

        main_layout.addStretch()
        main_layout.addLayout(center_layout)
        main_layout.addStretch()
        main_layout.addLayout(footer_layout)


# ------------------- VENTANA DE POLÍTICAS DE PRIVACIDAD -------------------
class PrivacyWidget(QWidget):
    def __init__(self, on_back_callback, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background-color: #1a1a24;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(50, 40, 50, 40)

        title = QLabel("Políticas de Privacidad y Tratamiento de Datos")
        title.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        title.setStyleSheet("color: #00e676; margin-bottom: 20px;")

        privacy_text = QTextEdit()
        privacy_text.setReadOnly(True)
        privacy_text.setStyleSheet("""
            QTextEdit {
                background-color: #252538;
                color: #dddddd;
                border: 1px solid #3b3b52;
                border-radius: 8px;
                padding: 20px;
                font-size: 13px;
                line-height: 1.5;
            }
        """)
        privacy_text.setHtml("""
            <h3 style='color:#00e676;'>1. Captura y Uso de Datos Biométricos</h3>
            <p>El sistema <b>DETECTEPP</b> realiza captura de video e imágenes faciales en tiempo real para:</p>
            <ul>
                <li>Control de marcación de asistencia y verificación de identidad.</li>
                <li>Monitoreo del uso adecuado de Equipos de Protección Personal (EPP).</li>
                <li>Detección automatizada de emergencias mediante gestos de auxilio (SOS).</li>
            </ul>
            <h3 style='color:#00e676;'>2. Monitoreo de posturas y accidentes</h3>
            <p>La pestaña de Monitoreo analiza la <b>postura</b> de las personas (esqueleto de 17 puntos) para detectar
            caídas, personas inmóviles, pedidos de auxilio y posturas que pueden causar lesiones. <b>No identifica</b>
            a nadie: las personas aparecen como P1, P2… Solo se guarda una foto por evento, con la <b>cabeza
            difuminada</b>, y se borra automáticamente a los 30 días. No se graba video continuo.</p>
            <h3 style='color:#00e676;'>3. Almacenamiento y Seguridad</h3>
            <p>Todas las imágenes capturadas y los datos de empleados se guardan de forma local en la base de datos interna de la empresa. No se transmiten ni comparten con servidores externos de terceros.</p>
        """)

        btn_back = QPushButton("Volver")
        btn_back.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        btn_back.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_back.setStyleSheet("""
            QPushButton {
                background-color: #2b2b3b;
                color: #ffffff;
                padding: 10px 30px;
                border-radius: 6px;
                border: none;
            }
            QPushButton:hover {
                background-color: #3b3b4b;
            }
        """)
        btn_back.clicked.connect(on_back_callback)

        layout.addWidget(title)
        layout.addWidget(privacy_text)
        layout.addSpacing(15)
        layout.addWidget(btn_back, alignment=Qt.AlignmentFlag.AlignLeft)


# ------------------- VENTANA PRINCIPAL (SISTEMA) -------------------
class MainSystemWidget(QWidget):
    def __init__(self, window_control_parent):
        super().__init__()
        self.parent_window = window_control_parent

        self.resultado = None          # último análisis de rostro (vision.identificador.Resultado)
        self.has_face = False
        self.current_emp = None         # dict del empleado confirmado frente a la cámara
        self.empleados = {}             # id -> dict, para mostrar nombres sin tocar la BD en cada frame
        self._marcado_en = {}           # id -> instante de su última marcación (enfriamiento por empleado)
        self._estado_cam = "CONECTANDO"
        self._cam_en_worker = camaras.activa()   # la cámara que el worker tiene abierta (para detectar cambios)

        self.intrusos = DetectorIntrusos()   # ¿un desconocido lleva INTRUSO_SEGUNDOS frente a la cámara?
        self._intruso = None            # alerta pendiente: {"foto": ruta, "hora": texto, "camara": nombre}
        self._mensaje_cam = "Conectando…"
        self._fps_cam = None
        self.seguimiento_epp = SeguimientoEpp()   # mayoría de las últimas detecciones YOLO
        self._ultimas_epp = None        # última lista de detecciones ya contada (llega repetida en varios frames)
        self._error_epp = None          # texto si falta el modelo YOLO de EPP
        self._prendas_modelo = None     # prendas que el YOLO cargado sabe detectar (None = aún no se sabe)
        self._epp_requeridos = None     # EPP del área del empleado actual (filtra las cajas del video)
        self._epp_no_detectables = []   # prendas exigidas que el modelo aún no conoce (aviso sobre el video)
        self.alerta_epp = AlertaEpp()   # ¿el EPP lleva EPP_ALERTA_SEGUNDOS incompleto? (una alerta por empleado)
        self._epp_alerta_emp = None     # id del empleado que muestra la tarjeta roja de EPP

        self.worker = CameraWorker()
        self.worker.control_flujo = True   # si la pantalla va más lenta que la cámara, se descartan frames (sin retraso)
        self.worker.frame_listo.connect(self.on_frame)
        self.worker.estado_camara.connect(self.on_estado_camara)
        self.worker.fps_camara.connect(self.on_fps_camara)
        self.worker.error_modelos.connect(self.on_error_modelos)
        self.worker.error_epp.connect(self.on_error_epp)
        self.worker.prendas_epp.connect(self.on_prendas_epp)

        self.recargar_empleados()
        self.init_ui()
        self.recargar_camaras()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Barra Superior
        top_bar = QFrame()
        top_bar.setStyleSheet("background-color: #11111b; min-height: 40px; max-height: 40px;")
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(15, 0, 10, 0)

        app_title = QLabel("DETECTEPP - Sistema de Control Biométrico e Industrial")
        app_title.setStyleSheet("color: #00e676; font-weight: bold; font-size: 14px;")

        btn_minimize = QPushButton("—")
        btn_minimize.setFixedSize(35, 25)
        btn_minimize.setStyleSheet("QPushButton { background-color: #2b2b3b; color: white; border: none; font-weight: bold; } QPushButton:hover { background-color: #3b3b4b; }")
        btn_minimize.clicked.connect(self.parent_window.showMinimized)

        btn_restore_disabled = QPushButton("❐")
        btn_restore_disabled.setFixedSize(35, 25)
        btn_restore_disabled.setEnabled(False)
        btn_restore_disabled.setStyleSheet("background-color: #1b1b26; color: #555; border: none;")

        btn_close = QPushButton("✕")
        btn_close.setFixedSize(35, 25)
        btn_close.setStyleSheet("QPushButton { background-color: #2b2b3b; color: white; border: none; font-weight: bold; } QPushButton:hover { background-color: #ff1744; }")
        btn_close.clicked.connect(self.parent_window.close)

        estilo_barra = "QPushButton { background-color: #2b2b3b; color: white; border: none; padding: 3px 10px; } QPushButton:hover { background-color: #3b3b4b; }"
        self.combo_camara = QComboBox()
        self.combo_camara.setMinimumWidth(190)
        self.combo_camara.setStyleSheet("background-color: #2b2b3b; color: white; padding: 2px 6px;")
        self.combo_camara.setToolTip("Cámara activa")
        self.combo_camara.activated.connect(self.on_cambiar_camara)

        btn_gestionar = QPushButton("⚙ Cámaras")
        btn_gestionar.setStyleSheet(estilo_barra)
        btn_gestionar.setToolTip("Agregar, probar o eliminar cámaras (USB, celular, IP, archivo)")
        btn_gestionar.clicked.connect(self.abrir_gestion_camaras)

        btn_girar = QPushButton("⟳ Girar")
        btn_girar.setStyleSheet(estilo_barra)
        btn_girar.setToolTip("Gira la imagen 90° (útil si el celular se ve de lado)")
        btn_girar.clicked.connect(self.on_girar_camara)

        btn_reconectar = QPushButton("↻ Reconectar")
        btn_reconectar.setStyleSheet(estilo_barra)
        btn_reconectar.setToolTip("Reabre la cámara y elimina el retraso acumulado")
        btn_reconectar.clicked.connect(self.worker.reconectar)

        self.btn_emergencias = QPushButton("")
        self.btn_emergencias.setStyleSheet("QPushButton { background-color: #ff1744; color: white; border: none; "
                                           "font-weight: bold; padding: 3px 10px; border-radius: 3px; }")
        self.btn_emergencias.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_emergencias.setToolTip("Ir a la pestaña de Monitoreo")
        self.btn_emergencias.hide()

        self.lbl_cam_estado = QLabel("● Conectando…")
        self.lbl_cam_estado.setStyleSheet("color: #ffb74d; font-size: 12px; padding: 0 8px;")

        top_layout.addWidget(app_title)
        top_layout.addStretch()
        top_layout.addWidget(self.btn_emergencias)
        top_layout.addWidget(self.lbl_cam_estado)
        top_layout.addWidget(self.combo_camara)
        top_layout.addWidget(btn_gestionar)
        top_layout.addWidget(btn_girar)
        top_layout.addWidget(btn_reconectar)
        top_layout.addSpacing(10)
        top_layout.addWidget(btn_minimize)
        top_layout.addWidget(btn_restore_disabled)
        top_layout.addWidget(btn_close)

        main_layout.addWidget(top_bar)

        # Pestañas
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane { border: none; background: #1e1e2e; }
            QTabBar::tab { background: #2b2b3b; color: #aaa; padding: 12px 24px; font-weight: bold; }
            QTabBar::tab:selected { background: #00e676; color: #000; }
        """)

        # 1. Biometría
        self.tab_biometria = QWidget()
        self.setup_biometria_tab()

        # 2. Detección EPP (65% / 35%)
        self.tab_epp = QWidget()
        self.setup_epp_tab()

        # 3. Monitoreo de riesgos y SOS (65% / 35%): ver ui_monitoreo.py
        self.panel_monitoreo = PanelMonitoreo(self.worker, camaras.activa())
        self.tab_accidents = self.panel_monitoreo
        self.video_label_accidents = self.panel_monitoreo.video
        self.panel_monitoreo.emergencias_cambiaron.connect(self._mostrar_emergencias)
        self.btn_emergencias.clicked.connect(lambda: self.tabs.setCurrentIndex(2))
        self._mostrar_emergencias(len(self.panel_monitoreo.tarjetas))

        self.tabs.addTab(self.tab_biometria, "👤 Biometría Facial y Registro")
        self.tabs.addTab(self.tab_epp, "🛡️ Detección de EPP")
        self.tabs.addTab(self.tab_accidents, "🚨 Monitoreo de Riesgos y SOS")
        self.tabs.currentChanged.connect(self.on_tab_changed)

        main_layout.addWidget(self.tabs)

    def start_camera(self):
        if not self.worker.isRunning():
            self.worker.start()

    def setup_biometria_tab(self):
        layout = QHBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)

        self.video_label_bio = QLabel("Cargando Cámara...")
        self.video_label_bio.setMinimumSize(640, 480)
        self.video_label_bio.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)   # ver nota en EPP
        self.video_label_bio.setStyleSheet("background-color: #000; border-radius: 8px;")
        self.video_label_bio.setAlignment(Qt.AlignmentFlag.AlignCenter)

        panel = QVBoxLayout()
        title = QLabel("CONTROL Y REGISTRO BIOMÉTRICO")
        title.setFont(QFont("Arial", 13, QFont.Weight.Bold))
        title.setStyleSheet("color: #00e676;")

        lbl_tipo = QLabel("Tipo de Marcación:")
        lbl_tipo.setStyleSheet("color: #aaa; font-weight: bold;")

        radio_style = """
            QRadioButton { color: #ffffff; font-weight: bold; font-size: 14px; padding: 4px; }
            QRadioButton::indicator { width: 18px; height: 18px; border-radius: 9px; }
            QRadioButton::indicator:checked { background-color: #00e676; border: 2px solid #ffffff; }
            QRadioButton::indicator:unchecked { background-color: #444455; border: 2px solid #888; }
        """
        self.radio_entrada = QRadioButton("ENTRADA")
        self.radio_salida = QRadioButton("SALIDA")
        self.radio_entrada.setChecked(True)
        self.radio_entrada.setStyleSheet(radio_style)
        self.radio_salida.setStyleSheet(radio_style)

        tipo_layout = QHBoxLayout()
        tipo_layout.addWidget(self.radio_entrada)
        tipo_layout.addWidget(self.radio_salida)

        self.lbl_bio_status = QLabel("Estado: Buscando rostro...")
        self.lbl_bio_status.setStyleSheet("color: #fff; font-size: 12px;")
        # Alto fijo y texto que no cambia el tamaño: evita que el panel "tiemble" al cambiar los mensajes.
        self.lbl_bio_status.setWordWrap(True)
        self.lbl_bio_status.setFixedHeight(54)
        self.lbl_bio_status.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self._estado_bio = None

        # Tarjeta de rostro desconocido (aparece tras INTRUSO_SEGUNDOS con un desconocido frente a la cámara)
        self.card_intruso = QFrame()
        self.card_intruso.setStyleSheet("QFrame#cardIntruso { background-color: #2a1d1d; border: 2px solid #ff5252; border-radius: 8px; }")
        self.card_intruso.setObjectName("cardIntruso")
        caja = QVBoxLayout(self.card_intruso)
        self.lbl_intruso_titulo = QLabel("⚠ ROSTRO NUEVO IDENTIFICADO")
        self.lbl_intruso_titulo.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.lbl_intruso_titulo.setStyleSheet("color: #ff5252; border: none;")
        caja.addWidget(self.lbl_intruso_titulo)
        cuerpo = QHBoxLayout()
        self.lbl_intruso_foto = QLabel()
        self.lbl_intruso_foto.setFixedSize(120, 120)
        self.lbl_intruso_foto.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_intruso_foto.setStyleSheet("background-color: #000; border: 1px solid #555;")
        derecha = QVBoxLayout()
        self.lbl_intruso_info = QLabel("")
        self.lbl_intruso_info.setWordWrap(True)
        self.lbl_intruso_info.setStyleSheet("color: #ddd; font-size: 12px; border: none;")
        self.btn_intruso_agregar = QPushButton("Agregar a la base de datos")
        self.btn_intruso_agregar.setStyleSheet("QPushButton { background-color: #00e676; color: #000; padding: 7px; border-radius: 5px; font-weight: bold; } QPushButton:hover { background-color: #00c853; }")
        self.btn_intruso_agregar.clicked.connect(self.agregar_intruso)
        self.btn_intruso_reportar = QPushButton("Reportar: no pertenece a la empresa")
        self.btn_intruso_reportar.setStyleSheet("QPushButton { background-color: #ff5252; color: #fff; padding: 7px; border-radius: 5px; font-weight: bold; } QPushButton:hover { background-color: #ff1744; }")
        self.btn_intruso_reportar.clicked.connect(self.reportar_intruso)
        derecha.addWidget(self.lbl_intruso_info)
        derecha.addWidget(self.btn_intruso_agregar)
        derecha.addWidget(self.btn_intruso_reportar)
        cuerpo.addWidget(self.lbl_intruso_foto)
        cuerpo.addLayout(derecha, 1)
        caja.addLayout(cuerpo)
        self.card_intruso.hide()

        # Tarjeta del último empleado reconocido (captura + datos + hora de ingreso/salida)
        self.card_ingreso = QFrame()
        self.card_ingreso.setObjectName("cardIngreso")
        self.card_ingreso.setStyleSheet("QFrame#cardIngreso { background-color: #1b2a22; border: 2px solid #00e676; border-radius: 8px; }")
        caja_i = QVBoxLayout(self.card_ingreso)
        self.lbl_ingreso_titulo = QLabel("✔ INGRESO REGISTRADO")
        self.lbl_ingreso_titulo.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.lbl_ingreso_titulo.setStyleSheet("color: #00e676; border: none;")
        caja_i.addWidget(self.lbl_ingreso_titulo)
        cuerpo_i = QHBoxLayout()
        self.lbl_ingreso_foto = QLabel()
        self.lbl_ingreso_foto.setFixedSize(110, 110)
        self.lbl_ingreso_foto.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_ingreso_foto.setStyleSheet("background-color: #000; border: 1px solid #555;")
        self.lbl_ingreso_datos = QLabel("")
        self.lbl_ingreso_datos.setTextFormat(Qt.TextFormat.RichText)
        self.lbl_ingreso_datos.setWordWrap(True)
        self.lbl_ingreso_datos.setStyleSheet("color: #ddd; font-size: 12px; border: none;")
        cuerpo_i.addWidget(self.lbl_ingreso_foto)
        cuerpo_i.addWidget(self.lbl_ingreso_datos, 1)
        caja_i.addLayout(cuerpo_i)
        self.card_ingreso.hide()

        self.lbl_marcacion_info = QLabel("")
        self.lbl_marcacion_info.setWordWrap(True)
        self.lbl_marcacion_info.setStyleSheet("color: #00e676; font-size: 13px; font-weight: bold;")

        panel.addWidget(title)
        panel.addSpacing(10)
        panel.addWidget(lbl_tipo)
        panel.addLayout(tipo_layout)
        panel.addSpacing(10)
        panel.addWidget(self.lbl_bio_status)
        panel.addSpacing(10)
        panel.addWidget(self.card_intruso)
        panel.addSpacing(8)
        panel.addWidget(self.card_ingreso)
        panel.addSpacing(8)
        panel.addWidget(self.lbl_marcacion_info)
        panel.addStretch()

        layout.addWidget(self.video_label_bio, stretch=3)
        # El panel va dentro de un área con desplazamiento: en pantallas pequeñas, si las tarjetas
        # no caben, aparece una barra de scroll en vez de recortar el contenido.
        contenedor = QWidget()
        contenedor.setLayout(panel)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { background: transparent; } QScrollArea > QWidget > QWidget { background: transparent; }")
        scroll.setWidget(contenedor)
        layout.addWidget(scroll, stretch=2)
        self.tab_biometria.setLayout(layout)

    def setup_epp_tab(self):
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 65% LADO IZQUIERDO
        self.video_label_epp = QLabel("Cargando Cámara EPP...")
        # Ignored: el tamaño lo decide el layout, no la imagen. Con Expanding, una imagen vertical
        # (cámara girada) agrandaba el recuadro en cada frame: zoom y paneles moviéndose.
        self.video_label_epp.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.video_label_epp.setStyleSheet("background-color: #000;")
        self.video_label_epp.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # 35% LADO DERECHO
        panel_widget = QWidget()
        panel_widget.setStyleSheet("background-color: #181825; border-left: 1px solid #2b2b3b;")
        panel_epp = QVBoxLayout(panel_widget)
        panel_epp.setContentsMargins(20, 20, 20, 20)

        title_epp = QLabel("ANÁLISIS DE EPP")
        title_epp.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title_epp.setStyleSheet("color: #00e676;")

        self.lbl_epp_emp_name = QLabel("Empleado: Ninguno Identificado")
        self.lbl_epp_emp_name.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.lbl_epp_emp_name.setStyleSheet("color: #ffffff;")

        self.lbl_epp_area = QLabel("Área de trabajo asignada:")
        self.lbl_epp_area.setStyleSheet("color: #aaa; font-size: 13px;")
        # Se pone sola en el área del empleado; si hoy trabaja en otra, el supervisor la cambia aquí
        self.combo_area_epp = QComboBox()
        self.combo_area_epp.setPlaceholderText("Seleccione el área…")
        self.combo_area_epp.setStyleSheet("QComboBox { padding: 6px; background-color: #2b2b3b; color: white; border-radius: 4px; }")
        self.combo_area_epp.activated.connect(self.on_cambiar_area_epp)   # solo cambios hechos por el usuario
        self.recargar_areas_epp()

        self.lbl_status_epp = QLabel("ESTADO EPP: ESPERANDO BIOMETRÍA")
        self.lbl_status_epp.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        self.lbl_status_epp.setStyleSheet("color: #ffb74d;")
        self.lbl_status_epp.setWordWrap(True)

        self.lbl_epp_detalle = QLabel("")   # una línea por EPP requerido: ✔ / ✘
        self.lbl_epp_detalle.setFont(QFont("Arial", 12))
        self.lbl_epp_detalle.setStyleSheet("color: #ffffff;")
        self.lbl_epp_detalle.setTextFormat(Qt.TextFormat.RichText)

        panel_epp.addWidget(title_epp)
        panel_epp.addSpacing(15)
        panel_epp.addWidget(self.lbl_epp_emp_name)
        panel_epp.addWidget(self.lbl_epp_area)
        panel_epp.addWidget(self.combo_area_epp)
        panel_epp.addSpacing(15)
        panel_epp.addWidget(self.lbl_status_epp)
        panel_epp.addSpacing(10)
        panel_epp.addWidget(self.lbl_epp_detalle)

        # Tarjeta roja: el empleado no cumple el protocolo de EPP (misma alarma que Biometría)
        self.card_epp_alerta = QFrame()
        self.card_epp_alerta.setObjectName("cardEppAlerta")
        self.card_epp_alerta.setStyleSheet("QFrame#cardEppAlerta { background-color: #2a1d1d; border: 2px solid #ff1744; border-radius: 8px; }")
        caja_alerta = QVBoxLayout(self.card_epp_alerta)
        titulo_alerta = QLabel("⚠ NO CUMPLE PROTOCOLO DE EPP")
        titulo_alerta.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        titulo_alerta.setStyleSheet("color: #ff1744; border: none; background: transparent;")
        self.lbl_epp_alerta = QLabel("")
        self.lbl_epp_alerta.setWordWrap(True)
        self.lbl_epp_alerta.setStyleSheet("color: #ddd; font-size: 12px; border: none; background: transparent;")
        caja_alerta.addWidget(titulo_alerta)
        caja_alerta.addWidget(self.lbl_epp_alerta)
        self.card_epp_alerta.hide()

        panel_epp.addSpacing(15)
        panel_epp.addWidget(self.card_epp_alerta)
        panel_epp.addStretch()

        layout.addWidget(self.video_label_epp, stretch=65)
        layout.addWidget(panel_widget, stretch=35)
        self.tab_epp.setLayout(layout)

    def _mostrar_emergencias(self, n: int):
        """Aviso en la barra superior (visible desde cualquier pestaña) mientras haya emergencias sin atender."""
        self.btn_emergencias.setText(f"⚠ {n} emergencia{'s' if n != 1 else ''} sin atender")
        self.btn_emergencias.setVisible(n > 0)

    def on_tab_changed(self, index):
        self.intrusos.reiniciar()
        if index == 0:
            self.worker.set_mode(MODO_BIOMETRIA)
        elif index == 1:
            self.worker.set_mode(MODO_EPP)
            self.seguimiento_epp.reiniciar()
            self.alerta_epp.reiniciar()
            self._ultimas_epp = None
            self.actualizar_vista_epp()
        else:
            self.worker.set_mode(MODO_MONITOREO)
            self.panel_monitoreo.al_entrar()

    # ---------------- cámaras ----------------
    def recargar_camaras(self):
        self.combo_camara.blockSignals(True)
        self.combo_camara.clear()
        activa = camaras.activa()
        for cam in camaras.listar():
            self.combo_camara.addItem(cam.nombre, cam.id)
        self.combo_camara.setCurrentIndex(max(0, self.combo_camara.findData(activa.id)))
        self.combo_camara.blockSignals(False)

    def on_cambiar_camara(self, indice):
        cam_id = self.combo_camara.itemData(indice)
        try:
            camaras.activar(cam_id)
        except KeyError:
            QMessageBox.warning(self, "Cámara", "Esa cámara ya no existe. Se actualizó la lista.")
            self.recargar_camaras()
            return
        self.sincronizar_camara_activa()

    def sincronizar_camara_activa(self):
        """Hace que el worker use la cámara que indica el registro (cambio, giro o ninguna novedad)."""
        activa = camaras.activa()
        anterior = self._cam_en_worker
        if (activa.id, activa.valor) != (anterior.id, anterior.valor):
            self.intrusos.reiniciar()
            self.panel_monitoreo.camara_cambiada(activa, girada=False)
            self.worker.cambiar_camara(activa)
        elif activa.rotacion != anterior.rotacion:
            self.panel_monitoreo.camara_cambiada(activa, girada=True)   # borra sus zonas ANTES de girar
            self.worker.rotar(activa.rotacion)
        self._cam_en_worker = activa

    def abrir_gestion_camaras(self):
        dialogo = CamarasDialog(self)
        dialogo.cambiado.connect(self._tras_gestion_camaras)
        dialogo.exec()
        self._tras_gestion_camaras()

    def _tras_gestion_camaras(self):
        self.recargar_camaras()
        self.sincronizar_camara_activa()

    def on_girar_camara(self):
        cam = camaras.activa()
        camaras.rotar(cam.id, (cam.rotacion + 90) % 360)
        self.sincronizar_camara_activa()

    def on_estado_camara(self, estado, mensaje):
        self._estado_cam = estado
        self._mensaje_cam = mensaje
        if estado != "OK":
            self._fps_cam = None
        color = {"OK": "#00e676", "CONECTANDO": "#ffb74d", "RECONECTANDO": "#ff9100"}.get(estado, "#888")
        self.lbl_cam_estado.setStyleSheet(f"color: {color}; font-size: 12px; padding: 0 8px;")
        self._pintar_estado_cam()
        if estado != "OK":
            for lbl in (self.video_label_bio, self.video_label_epp, self.video_label_accidents):
                lbl.clear()
                lbl.setText(mensaje)

    def _pintar_estado_cam(self):
        extra = f" · {self._fps_cam[0]:.0f} fps (pantalla {self._fps_cam[1]:.0f})" if self._fps_cam else ""
        self.lbl_cam_estado.setText(f"● {self._mensaje_cam}{extra}")

    def on_fps_camara(self, lectura, pantalla):
        if self._estado_cam == "OK":
            self._fps_cam = (lectura, pantalla)
            self._pintar_estado_cam()

    def on_error_modelos(self, texto):
        QMessageBox.warning(self, "Modelos de IA", f"{texto}\n\nLa cámara funciona, pero sin reconocimiento facial.")

    # ---------------- reconocimiento ----------------
    def recargar_empleados(self):
        self.empleados = {e["id"]: e for e in get_todos_empleados()}

    def _aplicar_resultado(self, res):
        """Actualiza el panel de biometría según el análisis (solo la persona principal)."""
        self.resultado = res
        self.has_face = bool(res and res.hay_rostro)

        if self.intrusos.procesar(res):
            self._alertar_rostro_desconocido(res)

        if res is not None and res.emp_id is not None:
            emp = self.empleados.get(res.emp_id)
            if emp is None:
                self.recargar_empleados()
                emp = self.empleados.get(res.emp_id)
            if emp is not None:
                self._mostrar_empleado(emp, res)
                return

        self.current_emp = None
        if res is None or not res.hay_rostro:
            self._poner_estado("Estado: Buscando rostro en la cámara...", "color: #aaa;")
        elif not res.calidad_ok:
            self._poner_estado(f" {res.aviso}", "color: #ffb74d; font-weight: bold;")
        elif res.desconocido:
            if self.intrusos.pendiente:
                aviso = " Persona no registrada: elige una opción en la tarjeta roja."
            else:
                aviso = " Persona no registrada: capturando su rostro…"
            cercano = self.empleados.get(res.cercano_id) if res.cercano_id is not None else None
            if cercano is not None:
                aviso += f"\n (el más parecido: {cercano['nombre']} {res.similitud:.2f}; se reconoce desde {config.FACE_UMBRAL_COSENO:.2f})"
            self._poner_estado(aviso, "color: #ffb74d; font-weight: bold;")
        else:
            self._poner_estado(" Analizando rostro...", "color: #aaa;")

    def _mostrar_empleado(self, emp, res):
        self.current_emp = emp
        self._poner_estado(f" Empleado reconocido: {emp['nombre']}  (similitud {res.similitud:.2f})", "color: #00e676; font-weight: bold;")

        if not res.hay_rostro:
            return  # la identidad se conserva unos instantes, pero no se marca sin rostro presente
        ahora = time.monotonic()
        if ahora - self._marcado_en.get(emp["id"], -1e9) > config.MARCACION_COOLDOWN_S:
            tipo = "ENTRADA" if self.radio_entrada.isChecked() else "SALIDA"
            try:
                timestamp = registrar_marcacion(emp["id"], tipo)
            except Exception:
                logger.exception("No se pudo guardar la marcación")
                self.lbl_marcacion_info.setText(" No se pudo guardar la marcación (ver logs).")
                return
            self._marcado_en[emp["id"]] = ahora
            self.lbl_marcacion_info.setText(f" Marcación de {tipo} registrada:\n{timestamp}")
            self._mostrar_ingreso(emp, tipo, timestamp, res.recorte)

    def _poner_estado(self, texto: str, estilo: str):
        """Escribe el estado biométrico SOLO si cambió (menos repintados, panel estable)."""
        if (texto, estilo) == self._estado_bio:
            return
        self._estado_bio = (texto, estilo)
        self.lbl_bio_status.setText(texto)
        self.lbl_bio_status.setStyleSheet(estilo)

    def _guardar_empleado(self, nombre, edad, area, vectores, recortes) -> bool:
        """Guarda al empleado con sus vectores y su primera captura como foto; registra su marcación."""
        config.ensure_dirs()
        filename = os.path.join(str(config.FOTOS_DIR), f"emp_{int(time.time() * 1000)}.jpg")
        # cv2.imwrite falla en silencio con rutas con tildes/ñ en Windows;
        # imencode + tofile funciona siempre.
        ok, buffer = cv2.imencode(".jpg", recortes[0])
        if not ok:
            QMessageBox.warning(self, "Error", "No se pudo codificar la foto del rostro.")
            return False
        buffer.tofile(filename)

        try:
            emp_id = face_store.registrar_empleado(nombre, edad, area, filename, vectores)
            tipo = "ENTRADA" if self.radio_entrada.isChecked() else "SALIDA"
            timestamp = registrar_marcacion(emp_id, tipo)
        except Exception:
            logger.exception("No se pudo registrar al empleado")
            try:
                os.remove(filename)
            except OSError:
                pass
            QMessageBox.critical(self, "Error", "No se pudo guardar el registro. Revisa los logs.")
            return False

        self._marcado_en[emp_id] = time.monotonic()
        self.recargar_empleados()
        self.worker.recargar_galeria()
        self._mostrar_ingreso(self.empleados.get(emp_id, {"nombre": nombre, "edad": edad, "area": area}), tipo, timestamp, recortes[0])
        self.lbl_marcacion_info.setText(f" Última Marcación: {tipo} ({timestamp})")
        QMessageBox.information(self, "Éxito", f"¡Empleado '{nombre}' registrado con {len(vectores)} muestras!\nMarcación de {tipo} guardada a las {timestamp}.")
        return True

    def _mostrar_ingreso(self, emp, tipo, timestamp, captura):
        """Tarjeta con la captura del rostro, los datos del trabajador y la hora de ingreso/salida."""
        entrada = tipo == "ENTRADA"
        color = "#00e676" if entrada else "#40c4ff"
        self.card_ingreso.setStyleSheet(
            f"QFrame#cardIngreso {{ background-color: {'#1b2a22' if entrada else '#1b2530'}; border: 2px solid {color}; border-radius: 8px; }}")
        self.lbl_ingreso_titulo.setText("✔ INGRESO REGISTRADO" if entrada else "➜ SALIDA REGISTRADA")
        self.lbl_ingreso_titulo.setStyleSheet(f"color: {color}; border: none;")
        fecha, hora = (timestamp.split(" ") + [""])[:2]
        self.lbl_ingreso_datos.setText(
            f"<span style='font-size:15px;color:#fff'><b>{emp['nombre']}</b></span><br>"
            f"Edad: {emp['edad']} años<br>"
            f"Área: {emp['area']}<br>"
            f"<b style='color:{color}'>{'Hora de ingreso' if entrada else 'Hora de salida'}: {hora}</b><br>"
            f"<span style='color:#999'>{fecha}</span>")
        if captura is not None and getattr(captura, "size", 0):
            img = np.ascontiguousarray(captura)
            h, w = img.shape[:2]
            qimg = QImage(img.data, w, h, 3 * w, QImage.Format.Format_BGR888)
            self.lbl_ingreso_foto.setPixmap(QPixmap.fromImage(qimg).scaled(
                110, 110, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        else:
            self.lbl_ingreso_foto.clear()
        self.card_ingreso.show()

    # ---------------- rostro desconocido / posible intruso ----------------
    def _alertar_rostro_desconocido(self, res):
        """Foto + tarjeta + mensaje + alarma sonora. Se llama una vez por cada desconocido (ver DetectorIntrusos)."""
        config.ensure_dirs()
        ahora = time.strftime("%Y-%m-%d %H:%M:%S")
        foto = os.path.join(str(config.INTRUSOS_DIR), f"desconocido_{int(time.time() * 1000)}.jpg")
        ok, buffer = cv2.imencode(".jpg", res.recorte)
        if ok:
            buffer.tofile(foto)
        else:
            foto = ""
        camara = camaras.activa().nombre
        self._intruso = {"foto": foto, "hora": ahora, "camara": camara}

        AlertManager.trigger_alert(
            "ROSTRO_DESCONOCIDO", camara,
            f"Rostro desconocido frente a la cámara durante {config.INTRUSO_SEGUNDOS:.0f} s. Foto: {foto or 'no disponible'}",
            sonido=True,
        )

        recorte = np.ascontiguousarray(res.recorte)
        h, w = recorte.shape[:2]
        imagen = QImage(recorte.data, w, h, 3 * w, QImage.Format.Format_BGR888)
        self.lbl_intruso_foto.setPixmap(QPixmap.fromImage(imagen).scaled(
            120, 120, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.lbl_intruso_info.setText(f"Persona no registrada · {ahora[11:]}\n¿Es un nuevo empleado o ajena a la empresa?")
        self.lbl_intruso_info.setStyleSheet("color: #ddd; font-size: 12px; border: none;")
        self.card_intruso.show()

    def _cerrar_intruso(self, borrar_foto: bool = False):
        datos, self._intruso = self._intruso, None
        self.card_intruso.hide()
        self.lbl_intruso_foto.clear()
        self.intrusos.resolver()
        if borrar_foto and datos and datos["foto"]:
            try:
                os.remove(datos["foto"])
            except OSError:
                pass

    def agregar_intruso(self):
        """Opción 1: es un nuevo empleado. Se usan las capturas ya tomadas: la persona NO tiene que seguir en cámara."""
        if self._intruso is None:
            return
        listas = registro.preparar(self.intrusos.muestras, face_store.cargar_galeria())
        if listas.duplicado is not None:
            emp_id, similitud = listas.duplicado
            self.recargar_empleados()
            otro = self.empleados.get(emp_id, {}).get("nombre", f"empleado {emp_id}")
            self._cerrar_intruso(borrar_foto=True)
            QMessageBox.warning(self, "Persona ya registrada",
                                f"Este rostro ya pertenece a «{otro}» (similitud {similitud:.2f}). No se registró de nuevo.")
            return
        if listas.error:
            self.lbl_intruso_info.setText(listas.error)
            self.lbl_intruso_info.setStyleSheet("color: #ffb74d; font-size: 12px; border: none;")
            return
        dialogo = DialogoNuevoEmpleado(self, foto=listas.recortes[0])
        if dialogo.exec() != QDialog.DialogCode.Accepted:
            return
        nombre, edad, area = dialogo.datos()
        if self._guardar_empleado(nombre, edad, area, listas.vectores, listas.recortes):
            self._cerrar_intruso(borrar_foto=True)   # ya es empleado: no se conserva su foto de "desconocido"

    def reportar_intruso(self):
        """Opción 2: no pertenece a la empresa → se registra el reporte con su foto."""
        if self._intruso is None:
            return
        datos = self._intruso
        AlertManager.trigger_alert(
            "INTRUSO_REPORTADO", datos["camara"],
            f"Persona ajena a la empresa reportada por el operador (detectada {datos['hora']}). Foto: {datos['foto'] or 'no disponible'}",
            sonido=False,
        )
        self._cerrar_intruso(borrar_foto=False)
        self.lbl_marcacion_info.setText(f" Reporte registrado: persona ajena a la empresa ({time.strftime('%H:%M:%S')}).")

    def on_error_epp(self, texto):
        self._error_epp = texto
        self.actualizar_vista_epp()

    def _aplicar_epp(self, detecciones):
        """Cuenta cada análisis YOLO una sola vez (el worker repite el último en cada frame)."""
        if detecciones is not None and detecciones is not self._ultimas_epp:
            self._ultimas_epp = detecciones
            self.seguimiento_epp.agregar(detecciones)
            self.actualizar_vista_epp()

    def recargar_areas_epp(self):
        try:
            areas = get_areas()
        except Exception:
            logger.exception("No se pudieron leer las áreas de trabajo")
            areas = []
        self.combo_area_epp.blockSignals(True)
        self.combo_area_epp.clear()
        self.combo_area_epp.addItems(areas)
        self.combo_area_epp.blockSignals(False)

    def _mostrar_area_epp(self, area):
        """Pone el selector en el área del empleado sin disparar un cambio."""
        self.combo_area_epp.blockSignals(True)
        self.combo_area_epp.setCurrentIndex(self.combo_area_epp.findText(area) if area else -1)
        self.combo_area_epp.blockSignals(False)

    def on_cambiar_area_epp(self, indice):
        """El supervisor asigna otra área al empleado: se guarda y se reevalúa su EPP."""
        area = self.combo_area_epp.itemText(indice)
        if not self.current_emp or not area or area == self.current_emp["area"]:
            return
        try:
            set_area_empleado(self.current_emp["id"], area)
        except Exception:
            logger.exception("No se pudo cambiar el área del empleado")
            self._mostrar_area_epp(self.current_emp["area"])
            return
        self.current_emp["area"] = area
        if self.current_emp["id"] in self.empleados:
            self.empleados[self.current_emp["id"]]["area"] = area
        self.alerta_epp.reiniciar()
        self._ocultar_alerta_epp()
        self.actualizar_vista_epp()

    def on_prendas_epp(self, prendas):
        self._prendas_modelo = set(prendas)
        self.actualizar_vista_epp()

    def actualizar_vista_epp(self):
        self.lbl_epp_detalle.setText("")
        self._epp_requeridos = None
        self._epp_no_detectables = []
        if self.current_emp is None or self.current_emp["id"] != self._epp_alerta_emp:
            self._ocultar_alerta_epp()   # la tarjeta roja es solo del empleado que la provocó
        if not self.current_emp:
            self.lbl_epp_emp_name.setText("Empleado: Ninguno Identificado")
            self._mostrar_area_epp(None)
            self.combo_area_epp.setEnabled(False)
            self.lbl_status_epp.setText("ESTADO EPP: Pase por Biometría Facial primero")
            self.lbl_status_epp.setStyleSheet("color: #ff1744; font-weight: bold;")
            return

        self.lbl_epp_emp_name.setText(f"Empleado: {self.current_emp['nombre']} ({self.current_emp['edad']} años)")
        self.combo_area_epp.setEnabled(True)
        self._mostrar_area_epp(self.current_emp["area"])
        if self._error_epp:
            self.lbl_status_epp.setText(f"ESTADO EPP: modelo no disponible\n{self._error_epp}")
            self.lbl_status_epp.setStyleSheet("color: #ffb74d; font-weight: bold;")
            return
        requeridos = get_epp_requeridos(self.current_emp["area"])
        self._epp_requeridos = list(requeridos) if requeridos else None
        if not requeridos:
            self.lbl_status_epp.setText("ESTADO EPP: el área no tiene EPP configurado.\nSeleccione el área de trabajo.")
            self.lbl_status_epp.setStyleSheet("color: #ffb74d; font-weight: bold;")
            return
        if self._ultimas_epp is None:
            self.lbl_status_epp.setText("ESTADO EPP: Analizando con YOLO...")
            self.lbl_status_epp.setStyleSheet("color: #aaa; font-weight: bold;")
            return

        ev = self.seguimiento_epp.evaluar(requeridos, self._prendas_modelo)
        self._epp_no_detectables = list(ev.no_detectables)
        lineas = []
        for p in requeridos:
            if p in ev.presentes:
                lineas.append(f"<span style='color:#00e676'>✔ {p}</span>")
            elif p in ev.faltantes:
                lineas.append(f"<span style='color:#ff1744'>✘ {p}</span>")
            else:
                lineas.append(f"<span style='color:#888'>– {p} (no detectable aún)</span>")
        self.lbl_epp_detalle.setText("<br>".join(lineas))
        if not ev.requeridos:
            self.lbl_status_epp.setText("ESTADO EPP: el modelo aún no detecta ninguna prenda de esta área")
            self.lbl_status_epp.setStyleSheet("color: #ffb74d; font-weight: bold;")
        elif ev.aprobado:
            self.lbl_status_epp.setText(f"ESTADO EPP: COMPLETO ({ev.porcentaje:.0f} %)")
            self.lbl_status_epp.setStyleSheet("color: #00e676; font-weight: bold;")
            self._ocultar_alerta_epp()
        else:
            self.lbl_status_epp.setText(f"ESTADO EPP: INCOMPLETO ({ev.porcentaje:.0f} %)\nFalta: {', '.join(ev.faltantes)}")
            self.lbl_status_epp.setStyleSheet("color: #ff1744; font-weight: bold;")
        if self.alerta_epp.procesar(self.current_emp["id"], ev, self.seguimiento_epp.cantidad):
            self._alertar_epp_incompleto(self.current_emp, ev)

    def _alertar_epp_incompleto(self, emp, ev):
        """Alerta en BD + log + alarma sonora (la misma de Biometría) y tarjeta roja en el panel."""
        faltan = ", ".join(ev.faltantes)
        AlertManager.trigger_alert(
            "EPP_INCOMPLETO", emp["area"],
            f"{emp['nombre']} no cumple el protocolo de EPP ({ev.porcentaje:.0f} %). Falta: {faltan}",
            sonido=True,
        )
        self._epp_alerta_emp = emp["id"]
        self.lbl_epp_alerta.setText(f"{emp['nombre']} · {time.strftime('%H:%M:%S')}\nFalta: {faltan}")
        self.card_epp_alerta.show()

    def _ocultar_alerta_epp(self):
        self._epp_alerta_emp = None
        self.card_epp_alerta.hide()

    def on_frame(self, frame, resultado):
        try:
            idx = self.tabs.currentIndex()
            # Al cambiar de pestaña aún pueden llegar frames en cola con el resultado del modo
            # anterior: cada pestaña solo acepta el tipo de resultado de su modo.
            modo = (MODO_BIOMETRIA, MODO_EPP, MODO_MONITOREO)[min(idx, 2)]
            if not resultado_valido(modo, resultado):
                resultado = None
            if idx == 0:
                self._aplicar_resultado(resultado)
            elif idx == 1:
                self._aplicar_epp(resultado)
            else:
                self.panel_monitoreo.mostrar(frame, resultado)
                return
            self.update_video_frame(frame, resultado)
        finally:
            self.worker.frame_mostrado()   # permite que el worker envíe el siguiente

    def _dibujar_rostro(self, pixmap, frame_w, res):
        """Caja y nombre sobre la imagen ya escalada. Se dibuja con Qt: cv2.putText no soporta tildes."""
        escala = pixmap.width() / frame_w
        x, y, w, h = (int(v * escala) for v in res.rostro.caja)
        emp = self.empleados.get(res.emp_id) if res.emp_id is not None else None
        if emp is not None:
            color, texto = QColor("#00e676"), f"{emp['nombre']}  {res.similitud:.2f}"
        elif not res.calidad_ok:
            color, texto = QColor("#ffb74d"), res.aviso
        elif res.desconocido:
            color, texto = QColor("#ff9100"), "Rostro desconocido"
        else:
            color, texto = QColor("#bbbbbb"), "Analizando..."

        pintor = QPainter(pixmap)
        pintor.setRenderHint(QPainter.RenderHint.Antialiasing)
        pintor.setPen(QPen(color, 3))
        pintor.drawRect(x, y, w, h)
        pintor.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        ancho_texto = pintor.fontMetrics().horizontalAdvance(texto) + 12
        y_texto = max(0, y - 26)
        pintor.fillRect(x, y_texto, ancho_texto, 24, QColor(0, 0, 0, 170))
        pintor.setPen(color)
        pintor.drawText(x + 6, y_texto + 18, texto)
        pintor.end()

    def update_video_frame(self, frame, resultado=None):
        idx = self.tabs.currentIndex()
        target_label = (self.video_label_bio, self.video_label_epp, self.video_label_accidents)[min(idx, 2)]

        lbl_w, lbl_h = target_label.width(), target_label.height()
        if lbl_w <= 0 or lbl_h <= 0 or frame is None:
            return
        frame = np.ascontiguousarray(frame)
        h, w, ch = frame.shape
        qt_img = QImage(frame.data, w, h, ch * w, QImage.Format.Format_BGR888)
        # Escalar manteniendo la proporción exacta (evita zoom y distorsión)
        pixmap = QPixmap.fromImage(qt_img).scaled(
            lbl_w, lbl_h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation
        )
        if idx == 1 and isinstance(resultado, list):
            self._dibujar_epp(pixmap, w, resultado)
        elif idx == 0 and resultado is not None and resultado.hay_rostro:
            self._dibujar_rostro(pixmap, w, resultado)
        target_label.setPixmap(pixmap)

    def _cajas_epp_visibles(self, detecciones):
        """[(detección, tipo, texto)] de las cajas que corresponden al área del empleado."""
        visibles = []
        for d in detecciones:
            caja = clasificar_caja(d.clase, d.presente, self._epp_requeridos)
            if caja is not None:
                visibles.append((d, caja[0], caja[1]))
        return visibles

    def _dibujar_epp(self, pixmap, frame_w, detecciones):
        """Cajas de YOLO sobre la imagen escalada, SOLO de las prendas que exige el área del empleado:
        verde = puesta, rojo = falta, ámbar = prenda que no cumple (mascarilla donde se exige respirador),
        gris = persona. Sin empleado identificado se muestran todas."""
        colores = {OK: QColor("#00e676"), FALTA: QColor("#ff1744"), NO_VALIDA: QColor("#ffb74d")}
        escala = pixmap.width() / frame_w
        pintor = QPainter(pixmap)
        pintor.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        for d, tipo, nombre in self._cajas_epp_visibles(detecciones):
            color = colores.get(tipo, QColor("#bbbbbb"))
            texto = f"{nombre} {d.confianza:.2f}"
            x, y, w, h = (int(v * escala) for v in d.caja)
            pintor.setPen(QPen(color, 2))
            pintor.drawRect(x, y, w, h)
            ancho_texto = pintor.fontMetrics().horizontalAdvance(texto) + 10
            y_texto = max(0, y - 22)
            pintor.fillRect(x, y_texto, ancho_texto, 20, QColor(0, 0, 0, 170))
            pintor.setPen(color)
            pintor.drawText(x + 5, y_texto + 15, texto)
        if self._epp_no_detectables:
            texto = "No detectable aún: " + ", ".join(self._epp_no_detectables)
            pintor.setFont(QFont("Arial", 10, QFont.Weight.Bold))
            ancho = pintor.fontMetrics().horizontalAdvance(texto) + 16
            pintor.fillRect(8, 8, ancho, 22, QColor(0, 0, 0, 170))
            pintor.setPen(QColor("#bbbbbb"))
            pintor.drawText(16, 24, texto)
        pintor.end()

    def closeEvent(self, event):
        self.worker.stop()


# ------------------- CONTENEDOR PRINCIPAL DE LA APLICACIÓN -------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.showFullScreen()

        self.stack = QStackedWidget()

        # Instanciar Pantallas
        self.welcome_screen = WelcomeScreen(
            on_enter_callback=self.go_to_system,
            on_privacy_callback=self.go_to_privacy
        )
        self.privacy_screen = PrivacyWidget(on_back_callback=self.go_to_welcome)
        self.system_widget = MainSystemWidget(window_control_parent=self)

        self.stack.addWidget(self.welcome_screen)  # Índice 0
        self.stack.addWidget(self.privacy_screen)  # Índice 1
        self.stack.addWidget(self.system_widget)   # Índice 2

        self.setCentralWidget(self.stack)

    def go_to_system(self):
        self.stack.setCurrentIndex(2)
        self.system_widget.start_camera()

    def go_to_privacy(self):
        self.stack.setCurrentIndex(1)

    def go_to_welcome(self):
        self.stack.setCurrentIndex(0)

    def closeEvent(self, event):
        self.system_widget.closeEvent(event)
        event.accept()


def _excepcion_no_capturada(tipo, valor, tb):
    """Registra el error y deja la app abierta (por defecto PyQt6 cierra todo ante cualquier error)."""
    if issubclass(tipo, KeyboardInterrupt):
        sys.__excepthook__(tipo, valor, tb)
        return
    logger.critical("Error no controlado", exc_info=(tipo, valor, tb))


def _configurar_logs():
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(config.LOGS_DIR / "app.log", encoding="utf-8")],
    )


if __name__ == "__main__":
    _configurar_logs()
    sys.excepthook = _excepcion_no_capturada
    init_db()
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())