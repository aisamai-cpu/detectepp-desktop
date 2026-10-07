"""Sonido de alerta multiplataforma, sin dependencias externas.

- Windows : ``winsound`` (incluido en Python).
- macOS   : ``afplay`` con un sonido del sistema.
- Linux   : ``paplay`` o ``aplay`` si existen; si no, la campana de terminal.

``play_alert()`` nunca lanza excepciones: si el audio falla, la app sigue.
"""
import logging
import shutil
import subprocess
import sys
import threading
import time

import config

logger = logging.getLogger(__name__)

_en_curso = threading.Lock()  # evita que varias alertas se superpongan


def play_alert(repeticiones=None, bloquear=False):
    """Reproduce la alarma.

    Devuelve True si inició el sonido, False si estaba desactivado o ya había
    otro sonando. Por defecto corre en un hilo aparte para no congelar la GUI.
    """
    if not config.SONIDO_ALERTA_ACTIVO:
        return False
    if repeticiones is None:
        repeticiones = config.SONIDO_REPETICIONES
    if not _en_curso.acquire(blocking=False):
        return False

    def _ejecutar():
        try:
            _reproducir(int(repeticiones))
        except Exception:  # el sonido jamás debe tumbar la aplicación
            logger.exception("No se pudo reproducir el sonido de alerta")
        finally:
            _en_curso.release()

    if bloquear:
        _ejecutar()
    else:
        threading.Thread(target=_ejecutar, daemon=True, name="alert-sound").start()
    return True


def _reproducir(repeticiones):
    if sys.platform.startswith("win"):
        _windows(repeticiones)
    elif sys.platform == "darwin":
        _macos(repeticiones)
    else:
        _linux(repeticiones)


def _windows(repeticiones):
    import winsound

    for _ in range(repeticiones):
        try:
            winsound.Beep(1000, 300)
        except RuntimeError:
            winsound.MessageBeep(winsound.MB_ICONHAND)
        time.sleep(0.1)


def _macos(repeticiones):
    sonido = "/System/Library/Sounds/Sosumi.aiff"
    for _ in range(repeticiones):
        _ejecutar_comando(["afplay", sonido])


def _linux(repeticiones):
    comando = _comando_linux()
    for _ in range(repeticiones):
        if comando:
            _ejecutar_comando(comando)
        else:
            sys.stdout.write("\a")
            sys.stdout.flush()
            time.sleep(0.4)


def _comando_linux():
    if shutil.which("paplay"):
        return ["paplay", "/usr/share/sounds/freedesktop/stereo/dialog-warning.oga"]
    if shutil.which("aplay"):
        return ["aplay", "-q", "/usr/share/sounds/alsa/Front_Center.wav"]
    return None


def _ejecutar_comando(comando):
    subprocess.run(
        comando, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5, check=False
    )
