"""Configuración central de DETECTEPP.

Todas las rutas y umbrales viven aquí. El resto del código debe leerlos como
``config.NOMBRE`` (no ``from config import NOMBRE``) para que las pruebas puedan
reemplazarlos.
"""
import os
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
INTRUSOS_DIR = DATA_DIR / "intrusos"        # fotos de rostros desconocidos detectados
LOGS_DIR = DATA_DIR / "logs"
ALERT_LOG_PATH = LOGS_DIR / "alert_emails.log"

# --- Cámaras (Fase 1.4) -------------------------------------------------------
CAMARAS_PATH = DATA_DIR / "camaras.json"   # lista de cámaras registradas (USB, IP, archivo)
CAMARA_SIN_FRAMES_S = 3.0                  # segundos sin imagen antes de reconectar
CAMARA_MAX_LADO = 1280                     # el frame se reduce a este lado máximo al leerlo (0 = no reducir)
CAMARA_RECONEXION_MIN = 0                  # reconectar sola cada N minutos para vaciar el retraso acumulado (0 = nunca)

# --- Modelos de IA -----------------------------------------------------------
# Núcleos de CPU para los modelos YOLO (EPP y pose). La mitad deja aire a la cámara y a la ventana.
IA_HILOS_CPU = max(1, (os.cpu_count() or 2) // 2)
MODELS_DIR = BASE_DIR / "models"
YUNET_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"      # detector de rostros
SFACE_PATH = MODELS_DIR / "face_recognition_sface_2021dec.onnx"    # embeddings faciales
EPP_MODELO_PATH = MODELS_DIR / "epp_yolo.onnx"                     # YOLO de EPP exportado a ONNX (sin torch)
                                                                   # si no existe se usa epp_yolo.pt (ultralytics + torch, más lento al cargar)

# --- Reconocimiento facial (Fase 1) ------------------------------------------
FACE_SCORE_MIN = 0.9            # confianza mínima del detector para aceptar un rostro
FACE_UMBRAL_COSENO = 0.50       # similitud mínima para considerar "misma persona".
                                # 0.363 es la referencia de OpenCV; 0.40 dejaba pasar a un familiar parecido.
                                # Más alto = más estricto (menos falsos positivos, pero puede pedir mejor luz/pose).
                                # Con la similitud que muestra la pantalla puedes afinarlo: sube si un familiar
                                # se reconoce como otro (prueba 0.55), baja si a ti no te reconoce con otra luz.
FACE_MUESTRAS_CONSENSO = 2      # el empleado debe parecerse en sus N mejores muestras (promedio), no solo en una
FACE_TAM_MIN = 80               # ancho mínimo (px) del rostro para registrar/reconocer bien
FACE_NITIDEZ_MIN = 30.0         # varianza del Laplaciano mínima (menos = imagen borrosa)
FACE_INFERENCIA_MAX_LADO = 640  # el frame se reduce a este lado máximo para la IA
FACE_INTERVALO_S = 0.12         # segundos mínimos entre dos análisis de rostro (≈8 por segundo)
FACE_CONFIRMACIONES = 3         # análisis seguidos con el mismo resultado para dar una identidad por buena
FACE_PERDIDA_INFERENCIAS = 4    # análisis seguidos SIN rostro para olvidar a la persona actual
MARCACION_COOLDOWN_S = 10.0     # segundos mínimos entre dos marcaciones del MISMO empleado

# --- Rostros desconocidos / posibles intrusos (1.6) --------------------------
INTRUSO_SEGUNDOS = 2.0          # un rostro desconocido visible este tiempo seguido dispara la alerta
INTRUSO_ENFRIAMIENTO_S = 30.0   # tras resolver una alerta, no se vuelve a disparar durante este tiempo

# --- Registro de rostros nuevos (desde la tarjeta roja, sin pedir movimientos) ---------------------------------------
REGISTRO_COHERENCIA = 0.40      # parecido mínimo entre las muestras de UNA misma persona
REGISTRO_MUESTRAS = 5           # muestras que se guardan del desconocido mientras está frente a la cámara
REGISTRO_MUESTRAS_MIN = 3       # mínimo para poder registrarlo
REGISTRO_SEPARACION_S = 0.15    # tiempo mínimo entre una muestra y la siguiente

# --- Detección de EPP con YOLO (Fase 2) ---------------------------------------
# Porcentaje mínimo de EPP del área para autorizar la marcación.
# 100 = cualquier prenda faltante genera alerta.
EPP_UMBRAL_APROBACION = 100.0
EPP_CONF_MIN = 0.45             # confianza mínima de YOLO para aceptar una detección
EPP_IMGSZ = 640                 # tamaño de entrada del modelo (el mismo con que se entrenó)
EPP_INTERVALO_S = 0.30          # segundos mínimos entre dos análisis de EPP (YOLO en CPU es más lento)
EPP_VENTANA = 5                 # cada prenda se decide por mayoría en los últimos N análisis
EPP_ALERTA_SEGUNDOS = 3.0       # segundos seguidos con EPP INCOMPLETO antes de lanzar la alerta
EPP_ALERTA_ENFRIAMIENTO_S = 60.0  # tras una alerta, el mismo empleado no vuelve a alertar durante este tiempo
                                  # (salvo que complete su EPP y luego se quite algo)
# Nombre de clase del modelo -> nombre de EPP en la base de datos (areas_epp).
# Mayúsculas, "-" y "_" se ignoran; las clases "NO-xxx" del modelo cuentan como ausencia.
EPP_ALIAS = {
    "hardhat": "casco", "helmet": "casco", "hard hat": "casco",
    "safety vest": "chaleco", "vest": "chaleco",
    "mask": "mascarilla", "face mask": "mascarilla",
    "goggles": "lentes", "glasses": "lentes", "safety glasses": "lentes",
    "gloves": "guantes", "glove": "guantes",
    "boots": "botas", "safety boots": "botas", "shoes": "botas",
    "lab coat": "bata", "coat": "bata",
    "respirator": "respirador", "half mask respirator": "respirador",
    "face shield": "careta", "faceshield": "careta",
    "apron": "mandil",
    "coverall": "traje", "coveralls": "traje", "protective suit": "traje", "hazmat suit": "traje",
    "hairnet": "cofia", "hair net": "cofia",
    "ear muffs": "orejeras", "earmuffs": "orejeras", "ear protection": "orejeras",
    "harness": "arnés", "safety harness": "arnés", "arnes": "arnés",
    "gafas": "lentes", "overol": "traje", "delantal": "mandil", "gorro": "cofia",
    "person": "persona",
}

# --- Sonido de alertas -------------------------------------------------------
SONIDO_ALERTA_ACTIVO = True
SONIDO_REPETICIONES = 3


def ensure_dirs():
    """Crea las carpetas de datos si no existen."""
    for carpeta in (DATA_DIR, FOTOS_DIR, INTRUSOS_DIR, LOGS_DIR, MODELS_DIR):
        carpeta.mkdir(parents=True, exist_ok=True)
