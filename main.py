import sys
import cv2
import time
from datetime import datetime
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QTabWidget, 
                             QComboBox, QRadioButton, QMessageBox, QLineEdit, 
                             QFormLayout, QFrame, QTextEdit, QStackedWidget, QSizePolicy)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap, QFont

from database import init_db, add_empleado, registrar_marcacion, get_todos_empleados
from camera_worker import CameraWorker

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
            <h3 style='color:#00e676;'>2. Almacenamiento y Seguridad</h3>
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

        self.last_face_crop = None
        self.has_face = False
        self.current_emp = None
        self.last_registered_time = 0

        self.worker = CameraWorker()
        self.worker.frame_processed.connect(self.update_video_frame)
        self.worker.face_detected_signal.connect(self.on_face_detected)

        self.init_ui()

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

        top_layout.addWidget(app_title)
        top_layout.addStretch()
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

        # 3. Monitoreo SOS (65% / 35%)
        self.tab_accidents = QWidget()
        self.setup_accidents_tab()

        self.tabs.addTab(self.tab_biometria, " Biometría Facial y Registro")
        self.tabs.addTab(self.tab_epp, "🛡️ Detección de EPP")
        self.tabs.addTab(self.tab_accidents, " Monitoreo de Riesgos y SOS")
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

        form = QFormLayout()
        self.input_nombre = QLineEdit()
        self.input_edad = QLineEdit()
        self.combo_bio_area = QComboBox()
        self.combo_bio_area.addItems(["Área Química", "Área Maquinaria"])

        for w in [self.input_nombre, self.input_edad, self.combo_bio_area]:
            w.setStyleSheet("padding: 8px; background-color: #2b2b3b; color: white; border-radius: 4px;")

        form.addRow("Nombre Completo:", self.input_nombre)
        form.addRow("Edad:", self.input_edad)
        form.addRow("Área de Trabajo:", self.combo_bio_area)

        self.btn_registrar = QPushButton("REGISTRAR NUEVO ROSTRO")
        self.btn_registrar.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        self.btn_registrar.setStyleSheet("""
            QPushButton { background-color: #00e676; color: #000; padding: 12px; border-radius: 6px; }
            QPushButton:hover { background-color: #00c853; }
        """)
        self.btn_registrar.clicked.connect(self.guardar_rostro)

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
        panel.addLayout(form)
        panel.addSpacing(15)
        panel.addWidget(self.btn_registrar)
        panel.addSpacing(10)
        panel.addWidget(self.lbl_marcacion_info)
        panel.addStretch()

        layout.addWidget(self.video_label_bio, stretch=3)
        layout.addLayout(panel, stretch=2)
        self.tab_biometria.setLayout(layout)

    def setup_epp_tab(self):
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 65% LADO IZQUIERDO
        self.video_label_epp = QLabel("Cargando Cámara EPP...")
        self.video_label_epp.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
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

        self.lbl_epp_area = QLabel("Área Asignada: N/A")
        self.lbl_epp_area.setStyleSheet("color: #aaa; font-size: 13px;")

        self.lbl_status_epp = QLabel("ESTADO EPP: ESPERANDO BIOMETRÍA")
        self.lbl_status_epp.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        self.lbl_status_epp.setStyleSheet("color: #ffb74d;")

        panel_epp.addWidget(title_epp)
        panel_epp.addSpacing(15)
        panel_epp.addWidget(self.lbl_epp_emp_name)
        panel_epp.addWidget(self.lbl_epp_area)
        panel_epp.addSpacing(15)
        panel_epp.addWidget(self.lbl_status_epp)
        panel_epp.addStretch()

        layout.addWidget(self.video_label_epp, stretch=65)
        layout.addWidget(panel_widget, stretch=35)
        self.tab_epp.setLayout(layout)

    def setup_accidents_tab(self):
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 65% LADO IZQUIERDO
        self.video_label_accidents = QLabel("Cargando Cámara SOS...")
        self.video_label_accidents.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_label_accidents.setStyleSheet("background-color: #000;")
        self.video_label_accidents.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # 35% LADO DERECHO
        panel_widget = QWidget()
        panel_widget.setStyleSheet("background-color: #181825; border-left: 1px solid #2b2b3b;")
        panel_sos = QVBoxLayout(panel_widget)
        panel_sos.setContentsMargins(20, 20, 20, 20)

        title_sos = QLabel("MONITOREO DE SOS")
        title_sos.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title_sos.setStyleSheet("color: #ff1744;")

        lbl_desc_sos = QLabel("Detección de emergencias en tiempo real mediante visión por computadora.")
        lbl_desc_sos.setWordWrap(True)
        lbl_desc_sos.setStyleSheet("color: #aaa; font-size: 12px;")

        self.lbl_sos_status = QLabel("Estado del Sistema: Monitoreando Activamente")
        self.lbl_sos_status.setStyleSheet("color: #00e676; font-weight: bold;")

        panel_sos.addWidget(title_sos)
        panel_sos.addSpacing(10)
        panel_sos.addWidget(lbl_desc_sos)
        panel_sos.addSpacing(20)
        panel_sos.addWidget(self.lbl_sos_status)
        panel_sos.addStretch()

        layout.addWidget(self.video_label_accidents, stretch=65)
        layout.addWidget(panel_widget, stretch=35)
        self.tab_accidents.setLayout(layout)

    def on_tab_changed(self, index):
        if index == 0:
            self.worker.set_mode("BIOMETRIA")
        elif index == 1:
            self.worker.set_mode("EPP")
            self.actualizar_vista_epp()
        else:
            self.worker.set_mode("ACCIDENTES")

    def on_face_detected(self, has_face, face_crop, emp_encontrado):
        self.has_face = has_face
        if has_face:
            self.last_face_crop = face_crop
            if emp_encontrado:
                self.current_emp = emp_encontrado
                self.input_nombre.setText(emp_encontrado["nombre"])
                self.input_edad.setText(str(emp_encontrado["edad"]))
                index = self.combo_bio_area.findText(emp_encontrado["area"])
                if index >= 0:
                    self.combo_bio_area.setCurrentIndex(index)

                self.btn_registrar.setEnabled(False)
                self.lbl_bio_status.setText(f" Empleado Reconocido: {emp_encontrado['nombre']}")
                self.lbl_bio_status.setStyleSheet("color: #00e676; font-weight: bold;")

                now = time.time()
                if now - self.last_registered_time > 10:
                    tipo = "ENTRADA" if self.radio_entrada.isChecked() else "SALIDA"
                    timestamp = registrar_marcacion(emp_encontrado["id"], tipo)
                    self.last_registered_time = now
                    self.lbl_marcacion_info.setText(f" Marcación de {tipo} registrada:\n{timestamp}")
            else:
                self.current_emp = None
                self.btn_registrar.setEnabled(True)
                self.lbl_bio_status.setText(" Rostro Nuevo: Complete el formulario para registrar.")
                self.lbl_bio_status.setStyleSheet("color: #ffb74d; font-weight: bold;")
        else:
            self.lbl_bio_status.setText("Estado: Buscando rostro en la cámara...")
            self.lbl_bio_status.setStyleSheet("color: #aaa;")

    def guardar_rostro(self):
        if not self.has_face or self.last_face_crop is None or self.last_face_crop.size == 0:
            QMessageBox.warning(self, "Atención", "No se detecta ningún rostro en la cámara para registrar.")
            return

        nombre = self.input_nombre.text().strip()
        edad_text = self.input_edad.text().strip()
        area = self.combo_bio_area.currentText()

        if not nombre or not edad_text:
            QMessageBox.warning(self, "Campos Incompletos", "Por favor ingrese el Nombre y la Edad.")
            return

        try:
            edad = int(edad_text)
        except ValueError:
            QMessageBox.warning(self, "Error", "La edad debe ser un número entero.")
            return

        filename = f"fotos_empleados/emp_{int(time.time())}.jpg"
        cv2.imwrite(filename, self.last_face_crop)

        emp_id = add_empleado(nombre, edad, area, filename)
        
        tipo = "ENTRADA" if self.radio_entrada.isChecked() else "SALIDA"
        timestamp = registrar_marcacion(emp_id, tipo)

        self.worker.actualizar_cache_empleados()

        QMessageBox.information(self, "Éxito", f"¡Empleado '{nombre}' registrado correctamente!\nMarcación de {tipo} guardada a las {timestamp}.")
        
        self.lbl_marcacion_info.setText(f" Última Marcación: {tipo} ({timestamp})")
        self.input_nombre.clear()
        self.input_edad.clear()

    def actualizar_vista_epp(self):
        if self.current_emp:
            self.lbl_epp_emp_name.setText(f"Empleado: {self.current_emp['nombre']} ({self.current_emp['edad']} años)")
            self.lbl_epp_area.setText(f"Área Asignada: {self.current_emp['area']}")
            self.lbl_status_epp.setText("ESTADO EPP: Listo para evaluación de modelo")
            self.lbl_status_epp.setStyleSheet("color: #00e676; font-weight: bold;")
        else:
            self.lbl_epp_emp_name.setText("Empleado: Ninguno Identificado")
            self.lbl_epp_area.setText("Área Asignada: N/A")
            self.lbl_status_epp.setText("ESTADO EPP: Pase por Biometría Facial primero")
            self.lbl_status_epp.setStyleSheet("color: #ff1744; font-weight: bold;")

    def update_video_frame(self, frame):
        idx = self.tabs.currentIndex()
        target_label = None

        if idx == 0:
            target_label = self.video_label_bio
        elif idx == 1:
            target_label = self.video_label_epp
        else:
            target_label = self.video_label_accidents

        if target_label:
            lbl_w, lbl_h = target_label.width(), target_label.height()
            if lbl_w > 0 and lbl_h > 0:
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                
                # Convertir directamente el frame BGR a QImage
                qt_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
                
                # Escalar manteniendo la proporción exacta (evita zoom y distorsión)
                pixmap = QPixmap.fromImage(qt_img).scaled(
                    lbl_w, 
                    lbl_h, 
                    Qt.AspectRatioMode.KeepAspectRatio, 
                    Qt.TransformationMode.SmoothTransformation
                )
                
                target_label.setPixmap(pixmap)

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


if __name__ == "__main__":
    init_db()
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())