"""Ventana para completar los datos de un empleado nuevo (nombre, edad, área) junto a su captura."""
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout, QLabel,
                             QLineEdit, QVBoxLayout)

EDAD_MIN, EDAD_MAX = 14, 99


class DialogoNuevoEmpleado(QDialog):
    def __init__(self, parent=None, foto=None, areas=None):
        """``areas``: lista para el selector; por defecto, las áreas de trabajo de la base de datos."""
        super().__init__(parent)
        self.setWindowTitle("Agregar a la base de datos")
        self.setMinimumWidth(400)
        self.setStyleSheet("QDialog { background-color: #1e1e2d; } QLabel { color: #ddd; }"
                           "QLineEdit, QComboBox { padding: 8px; background-color: #2b2b3b; color: white; border-radius: 4px; }")
        self.nombre = QLineEdit()
        self.edad = QLineEdit()
        self.area = QComboBox()
        if areas is None:
            from core.database import get_areas
            areas = get_areas()
        self.area.addItems(list(areas))
        self.lbl_error = QLabel("")
        self.lbl_error.setStyleSheet("color: #ff5252; font-weight: bold;")

        form = QFormLayout()
        form.addRow("Nombre completo:", self.nombre)
        form.addRow("Edad:", self.edad)
        form.addRow("Área de trabajo:", self.area)

        botones = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        botones.button(QDialogButtonBox.StandardButton.Ok).setText("Guardar empleado")
        botones.button(QDialogButtonBox.StandardButton.Cancel).setText("Cancelar")
        botones.accepted.connect(self._validar)
        botones.rejected.connect(self.reject)

        self.lbl_foto = QLabel()
        self.lbl_foto.setFixedSize(110, 110)
        self.lbl_foto.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_foto.setStyleSheet("background-color: #000;")
        if foto is not None and getattr(foto, "size", 0):
            img = np.ascontiguousarray(foto)
            h, w = img.shape[:2]
            self.lbl_foto.setPixmap(QPixmap.fromImage(QImage(img.data, w, h, 3 * w, QImage.Format.Format_BGR888)).scaled(
                110, 110, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

        arriba = QHBoxLayout()
        arriba.addWidget(self.lbl_foto)
        arriba.addLayout(form, 1)
        capa = QVBoxLayout(self)
        capa.addLayout(arriba)
        capa.addWidget(self.lbl_error)
        capa.addWidget(botones)

    def _validar(self):
        if not self.nombre.text().strip() or not self.edad.text().strip():
            self.lbl_error.setText("Escribe el nombre y la edad.")
            return
        if not self.area.currentText():
            self.lbl_error.setText("Selecciona el área de trabajo.")
            return
        try:
            if not EDAD_MIN <= int(self.edad.text().strip()) <= EDAD_MAX:
                raise ValueError
        except ValueError:
            self.lbl_error.setText(f"La edad debe ser un número entero entre {EDAD_MIN} y {EDAD_MAX}.")
            return
        self.accept()

    def datos(self):
        """(nombre, edad, área) ya validados; solo tiene sentido tras aceptar el diálogo."""
        return self.nombre.text().strip(), int(self.edad.text().strip()), self.area.currentText()
