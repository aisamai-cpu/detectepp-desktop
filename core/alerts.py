"""Gestor de alertas de seguridad: base de datos + log + sonido."""
import datetime
import logging

import config
from core.database import registrar_alerta
from core.sound import play_alert

logger = logging.getLogger(__name__)


class AlertManager:
    @staticmethod
    def trigger_alert(tipo_alerta, area, detalle, sonido=True):
        """Registra una alerta. Devuelve su id (o None si no se pudo guardar).

        Cada paso es independiente: si falla la BD igual se escribe el log y
        suena la alarma, y viceversa. Nunca lanza excepciones a la interfaz.
        """
        alerta_id = None

        # 1. Base de datos
        try:
            alerta_id = registrar_alerta(tipo_alerta, area, detalle)
        except Exception:
            logger.exception("No se pudo guardar la alerta en la base de datos")

        # 2. Log local (simulador de correo)
        try:
            config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
            with open(config.ALERT_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(
                    f"[{datetime.datetime.now()}] ALERTA CRÍTICA: {tipo_alerta} "
                    f"en {area} - Detalle: {detalle}\n"
                )
        except OSError:
            logger.exception("No se pudo escribir el log de alertas")

        # 3. Sonido (en hilo aparte, nunca falla)
        if sonido:
            play_alert()

        return alerta_id
