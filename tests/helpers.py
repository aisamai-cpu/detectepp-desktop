"""Utilidades compartidas por las pruebas."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config


class BaseConCarpetaTemporal(unittest.TestCase):
    """Redirige todas las rutas de config a una carpeta temporal y silencia el sonido."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        self._parches = [
            mock.patch.object(config, "DATA_DIR", base / "data"),
            mock.patch.object(config, "DB_PATH", base / "data" / "detectepp.db"),
            mock.patch.object(config, "FOTOS_DIR", base / "data" / "fotos"),
            mock.patch.object(config, "INTRUSOS_DIR", base / "data" / "intrusos"),
            mock.patch.object(config, "LOGS_DIR", base / "data" / "logs"),
            mock.patch.object(config, "ALERT_LOG_PATH", base / "data" / "logs" / "alert.log"),
            mock.patch.object(config, "CAMARAS_PATH", base / "data" / "camaras.json"),
            mock.patch.object(config, "MODELS_DIR", base / "models"),
            mock.patch.object(config, "EVENTOS_DIR", base / "data" / "eventos"),
            mock.patch.object(config, "REPORTES_DIR", base / "data" / "reportes"),
            mock.patch.object(config, "ZONAS_PATH", base / "data" / "zonas.json"),
            mock.patch.object(config, "POSE_MODELO_PATH", base / "models" / "pose.onnx"),
            # Las pruebas nunca deben cargar los modelos reales de models/ (si existen, cambian el resultado)
            mock.patch.object(config, "EPP_MODELO_PATH", base / "models" / "epp_yolo.onnx"),
            mock.patch.object(config, "SONIDO_ALERTA_ACTIVO", False),
        ]
        for p in self._parches:
            p.start()

    def tearDown(self):
        for p in self._parches:
            p.stop()
        self._tmp.cleanup()


def esperar(condicion, timeout=8.0, paso=0.01):
    """Espera (sin bloquear para siempre) a que ``condicion()`` sea verdadera."""
    import time
    limite = time.monotonic() + timeout
    while time.monotonic() < limite:
        if condicion():
            return True
        time.sleep(paso)
    return bool(condicion())
