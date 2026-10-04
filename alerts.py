import datetime
import winsound
import threading
from database import get_connection

class AlertManager:
    @staticmethod
    def trigger_alert(tipo_alerta, area, detalle):
        # 1. Registro en Base de Datos
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO alertas (tipo, area, detalle)
        VALUES (?, ?, ?)
        """, (tipo_alerta, area, detalle))
        conn.commit()
        conn.close()

        # 2. Log local de simulador de correos
        with open("alert_emails.log", "a", encoding="utf-8") as f:
            log_entry = f"[{datetime.datetime.now()}] ALERTA CRÍTICA: {tipo_alerta} en {area} - Detalle: {detalle}\n"
            f.write(log_entry)

        # 3. Sonido de alarma en hilo secundario (para no congelar la GUI)
        def play_sound():
            try:
                for _ in range(3):
                    winsound.Beep(1000, 300) # 1000Hz por 300ms
            except Exception:
                pass

        threading.Thread(target=play_sound, daemon=True).start()