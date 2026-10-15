"""Deja el sistema limpio: sin empleados, rostros, marcaciones, alertas, eventos de Monitoreo ni fotos.

Antes de borrar guarda TODO en ``data/respaldos/<fecha_hora>/`` (copia de la base de datos,
fotos de empleados, fotos de desconocidos, fotos de eventos, reportes y el log de alertas). Si algo
sale mal, se puede recuperar desde ahí. Las áreas de trabajo, las cámaras y las zonas restringidas NO se tocan.

Uso (con la aplicación CERRADA):
    venv\\Scripts\\python -m tools.reiniciar_datos
"""
import shutil
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from core import database

CONFIRMACION = "BORRAR"


def _mover_contenido(origen: Path, destino: Path) -> int:
    """Mueve los archivos de ``origen`` a ``destino``. Devuelve cuántos movió."""
    if not origen.exists():
        return 0
    n = 0
    for archivo in origen.iterdir():
        if archivo.is_file():
            destino.mkdir(parents=True, exist_ok=True)
            shutil.move(str(archivo), str(destino / archivo.name))
            n += 1
    return n


def reiniciar() -> dict:
    """Respalda y borra. Devuelve un resumen con la carpeta del respaldo y lo borrado."""
    database.init_db()
    respaldo = config.DATA_DIR / "respaldos" / time.strftime("%Y%m%d_%H%M%S")
    respaldo.mkdir(parents=True, exist_ok=True)

    if config.DB_PATH.exists():
        shutil.copy2(config.DB_PATH, respaldo / config.DB_PATH.name)
    filas = database.borrar_registros()
    fotos = _mover_contenido(config.FOTOS_DIR, respaldo / "fotos_empleados")
    intrusos = _mover_contenido(config.INTRUSOS_DIR, respaldo / "intrusos")
    for carpeta in (config.EVENTOS_DIR, config.REPORTES_DIR):   # tienen subcarpetas: se mueven enteras
        if carpeta.exists() and any(carpeta.iterdir()):
            shutil.move(str(carpeta), str(respaldo / carpeta.name))
    if config.ALERT_LOG_PATH.exists():
        shutil.move(str(config.ALERT_LOG_PATH), str(respaldo / config.ALERT_LOG_PATH.name))
    return {"respaldo": respaldo, "filas": filas, "fotos": fotos, "intrusos": intrusos}


def main(entrada=input) -> int:
    print("Esto borra TODOS los empleados, sus rostros, marcaciones, alertas, eventos de Monitoreo y fotos.")
    print("Antes se guarda un respaldo en data/respaldos/. Cierre la aplicación antes de seguir.")
    if entrada(f"Escriba {CONFIRMACION} para continuar: ").strip() != CONFIRMACION:
        print("Cancelado. No se borró nada.")
        return 1
    r = reiniciar()
    print(f"Listo. Respaldo en: {r['respaldo']}")
    print(f"  empleados: {r['filas']['empleados']}, vectores faciales: {r['filas']['embeddings_faciales']}, "
          f"marcaciones: {r['filas']['asistencia']}, alertas: {r['filas']['alertas']}, "
          f"eventos: {r['filas']['eventos_riesgo']}")
    print(f"  fotos de empleados: {r['fotos']}, fotos de desconocidos: {r['intrusos']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
