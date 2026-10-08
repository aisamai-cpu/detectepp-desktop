"""Configuración central de DETECTEPP.

Todas las rutas y umbrales viven aquí. El resto del código debe leerlos como
``config.NOMBRE`` (no ``from config import NOMBRE``) para que las pruebas puedan
reemplazarlos.
"""
import sys
from pathlib import Path

# Carpeta raíz del proyecto (o del .exe si se empaqueta con PyInstaller).
if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).resolve().parent
else:
    BASE_DIR = Path(__file__).resolve().parent

# --- Datos locales (NO se suben a git) ---------------------------------------
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "detectepp.db"
FOTOS_DIR = DATA_DIR / "fotos_empleados"
LOGS_DIR = DATA_DIR / "logs"
ALERT_LOG_PATH = LOGS_DIR / "alert_emails.log"

# --- Cámaras (Fase 1.4) -------------------------------------------------------
CAMARAS_PATH = DATA_DIR / "camaras.json"   # lista de cámaras registradas (USB, IP, archivo)
CAMARA_SIN_FRAMES_S = 3.0                  # segundos sin imagen antes de reconectar

# --- Modelos de IA -----------------------------------------------------------
MODELS_DIR = BASE_DIR / "models"
YUNET_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"      # detector de rostros
SFACE_PATH = MODELS_DIR / "face_recognition_sface_2021dec.onnx"    # embeddings faciales

# --- Reconocimiento facial (Fase 1) ------------------------------------------
FACE_SCORE_MIN = 0.9            # confianza mínima del detector para aceptar un rostro
FACE_UMBRAL_COSENO = 0.40       # similitud mínima para considerar "misma persona"
                                # (0.363 es el valor de referencia de OpenCV; 0.40 es
                                #  algo más estricto para evitar suplantaciones)
FACE_TAM_MIN = 80               # ancho mínimo (px) del rostro para registrar/reconocer bien
FACE_NITIDEZ_MIN = 30.0         # varianza del Laplaciano mínima (menos = imagen borrosa)
FACE_INFERENCIA_MAX_LADO = 640  # el frame se reduce a este lado máximo para la IA

# --- Política de EPP (se usará en la Fase 2) ---------------------------------
# Porcentaje mínimo de EPP del área para autorizar la marcación.
# 100 = cualquier prenda faltante genera alerta.
EPP_UMBRAL_APROBACION = 100.0

# --- Sonido de alertas -------------------------------------------------------
SONIDO_ALERTA_ACTIVO = True
SONIDO_REPETICIONES = 3


def ensure_dirs():
    """Crea las carpetas de datos si no existen."""
    for carpeta in (DATA_DIR, FOTOS_DIR, LOGS_DIR, MODELS_DIR):
        carpeta.mkdir(parents=True, exist_ok=True)
