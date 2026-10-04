import cv2
import time
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from database import get_todos_empleados

class CameraWorker(QThread):
    frame_processed = pyqtSignal(np.ndarray)
    face_detected_signal = pyqtSignal(bool, np.ndarray, object)  # (hay_rostro, recorte, emp_encontrado)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = True
        self.mode = "BIOMETRIA"
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Historial de empleados reconocidos localmente
        self.empleados_db = []
        self.actualizar_cache_empleados()

    def actualizar_cache_empleados(self):
        """Carga en memoria los empleados registrados de la base de datos."""
        self.empleados_db = get_todos_empleados()

    def set_mode(self, mode):
        self.mode = mode

    def stop(self):
        self.running = False
        self.wait()

    def _comparar_rostros(self, face_crop):
        """Comparación preliminar de histograma cromático para simulación/reconocimiento ligero."""
        if not self.empleados_db or face_crop is None or face_crop.size == 0:
            return None

        hist_crop = cv2.calcHist([cv2.cvtColor(face_crop, cv2.COLOR_BGR2HSV)], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(hist_crop, hist_crop, 0, 1, cv2.NORM_MINMAX)

        mejor_coincidencia = None
        mayor_score = 0.0

        for emp in self.empleados_db:
            if not emp.get("foto_path"):
                continue
            img_guardada = cv2.imread(emp["foto_path"])
            if img_guardada is None:
                continue

            hist_guardado = cv2.calcHist([cv2.cvtColor(img_guardada, cv2.COLOR_BGR2HSV)], [0, 1], None, [180, 256], [0, 180, 0, 256])
            cv2.normalize(hist_guardado, hist_guardado, 0, 1, cv2.NORM_MINMAX)

            score = cv2.compareHist(hist_crop, hist_guardado, cv2.HISTCMP_CORREL)
            if score > mayor_score:
                mayor_score = score
                mejor_coincidencia = emp

        # Umbral de coincidencia
        if mayor_score > 0.65:
            return mejor_coincidencia
        return None

    def run(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if cv2.os.name == 'nt' else cv2.VideoCapture(0)
        
        # Fijar resolución nativa estable para evitar distorsiones o cambios de zoom
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            print("Error: No se pudo conectar a la cámara.")
            return

        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(120, 120)
            )

            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                color = (0, 255, 0)
                
                # Extraer recorte del rostro
                face_crop = frame[y:y+h, x:x+w].copy()
                emp_identificado = self._comparar_rostros(face_crop)

                etiqueta = emp_identificado["nombre"] if emp_identificado else "ROSTRO DESCONOCIDO"
                if emp_identificado is None:
                    color = (0, 165, 255)  # Naranja para desconocidos

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, etiqueta, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                self.face_detected_signal.emit(True, face_crop, emp_identificado)
            else:
                self.face_detected_signal.emit(False, np.array([]), None)

            self.frame_processed.emit(frame)
            time.sleep(0.03)

        cap.release()