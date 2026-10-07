"""Persistencia SQLite de DETECTEPP con migraciones versionadas.

El número de versión del esquema se guarda en ``PRAGMA user_version``.
Para cambiar el esquema en el futuro NO se edita una migración existente:
se agrega una nueva función ``_migracion_vN`` y se registra en ``MIGRACIONES``.
"""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime

import config

TIPOS_MARCACION = ("ENTRADA", "SALIDA")


# ----------------------------------------------------------------------------
# Conexión
# ----------------------------------------------------------------------------
def get_connection():
    conn = sqlite3.connect(str(config.DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def _db():
    """Abre una conexión, confirma al terminar y la cierra siempre."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ----------------------------------------------------------------------------
# Migraciones
# ----------------------------------------------------------------------------
def _columnas(conn, tabla):
    return {fila["name"] for fila in conn.execute(f"PRAGMA table_info({tabla})")}


def _agregar_columna_si_falta(conn, tabla, columna, definicion):
    if columna not in _columnas(conn, tabla):
        conn.execute(f"ALTER TABLE {tabla} ADD COLUMN {columna} {definicion}")


def _migracion_v1(conn):
    """Esquema base. También repara bases creadas con la versión antigua."""
    conn.execute("""
    CREATE TABLE IF NOT EXISTS empleados (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nombre TEXT NOT NULL,
        edad INTEGER NOT NULL,
        area TEXT NOT NULL,
        foto_path TEXT NOT NULL,
        fecha_registro DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS asistencia (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        empleado_id INTEGER NOT NULL,
        timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        tipo_marcacion TEXT NOT NULL,
        epp_porcentaje REAL NOT NULL DEFAULT 0.0,
        estado TEXT NOT NULL DEFAULT 'PENDIENTE',
        FOREIGN KEY(empleado_id) REFERENCES empleados(id)
    )
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS areas_epp (
        area TEXT PRIMARY KEY,
        epp_requeridos_json TEXT NOT NULL
    )
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS alertas (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tipo TEXT NOT NULL,
        timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        area TEXT NOT NULL,
        detalle TEXT NOT NULL
    )
    """)

    # Bases antiguas no tenían esta columna (SQLite no permite ADD COLUMN con
    # DEFAULT CURRENT_TIMESTAMP, por eso se agrega sin default).
    _agregar_columna_si_falta(conn, "empleados", "fecha_registro", "DATETIME")

    areas_por_defecto = [
        ("Área Química", json.dumps(["mascarilla", "lentes", "bata", "guantes"])),
        ("Área Maquinaria", json.dumps(["casco", "botas", "chaleco", "lentes"])),
    ]
    conn.executemany(
        "INSERT OR IGNORE INTO areas_epp (area, epp_requeridos_json) VALUES (?, ?)",
        areas_por_defecto,
    )


def _migracion_v2(conn):
    """Vectores faciales (embeddings): varias muestras por empleado."""
    conn.execute("""
    CREATE TABLE IF NOT EXISTS embeddings_faciales (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        empleado_id INTEGER NOT NULL,
        vector BLOB NOT NULL,
        creado DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(empleado_id) REFERENCES empleados(id) ON DELETE CASCADE
    )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_embeddings_empleado ON embeddings_faciales(empleado_id)"
    )


# versión -> función. Se ejecutan en orden, una sola vez cada una.
MIGRACIONES = {
    1: _migracion_v1,
    2: _migracion_v2,
}
VERSION_ACTUAL = max(MIGRACIONES)


def run_migrations(conn):
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    for v in sorted(MIGRACIONES):
        if v > version:
            MIGRACIONES[v](conn)
            conn.execute(f"PRAGMA user_version = {int(v)}")
            conn.commit()


def init_db():
    """Crea carpetas y deja la base de datos en la última versión del esquema."""
    config.ensure_dirs()
    with _db() as conn:
        run_migrations(conn)


# ----------------------------------------------------------------------------
# Empleados
# ----------------------------------------------------------------------------
def add_empleado(nombre, edad, area, foto_path):
    with _db() as conn:
        cur = conn.execute(
            "INSERT INTO empleados (nombre, edad, area, foto_path, fecha_registro) "
            "VALUES (?, ?, ?, ?, ?)",
            (nombre, edad, area, str(foto_path), _ahora()),
        )
        return cur.lastrowid


def get_todos_empleados():
    with _db() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM empleados")]


# ----------------------------------------------------------------------------
# Vectores faciales (datos biométricos)
# ----------------------------------------------------------------------------
def add_empleado_con_embeddings(nombre, edad, area, foto_path, vectores_bytes):
    """Registra al empleado y sus vectores en UNA sola transacción.

    Si algo falla a mitad de camino no queda un empleado sin vectores.
    ``vectores_bytes``: lista de BLOBs de 512 bytes (ver ``embedding_a_bytes``).
    """
    if not vectores_bytes:
        raise ValueError("Se necesita al menos un vector facial")
    with _db() as conn:
        cur = conn.execute(
            "INSERT INTO empleados (nombre, edad, area, foto_path, fecha_registro) "
            "VALUES (?, ?, ?, ?, ?)",
            (nombre, edad, area, str(foto_path), _ahora()),
        )
        emp_id = cur.lastrowid
        conn.executemany(
            "INSERT INTO embeddings_faciales (empleado_id, vector, creado) VALUES (?, ?, ?)",
            [(emp_id, sqlite3.Binary(v), _ahora()) for v in vectores_bytes],
        )
        return emp_id


def add_embedding(empleado_id, vector_bytes):
    """Agrega una muestra más a un empleado ya registrado."""
    with _db() as conn:
        cur = conn.execute(
            "INSERT INTO embeddings_faciales (empleado_id, vector, creado) VALUES (?, ?, ?)",
            (empleado_id, sqlite3.Binary(vector_bytes), _ahora()),
        )
        return cur.lastrowid


def get_embeddings():
    """Todos los vectores como lista de ``(empleado_id, bytes)``."""
    with _db() as conn:
        filas = conn.execute("SELECT empleado_id, vector FROM embeddings_faciales ORDER BY id")
        return [(f["empleado_id"], bytes(f["vector"])) for f in filas]


def contar_embeddings(empleado_id):
    with _db() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM embeddings_faciales WHERE empleado_id = ?", (empleado_id,)
        ).fetchone()[0]


def eliminar_embeddings(empleado_id):
    """Borra los datos biométricos de un empleado (derecho de supresión)."""
    with _db() as conn:
        cur = conn.execute("DELETE FROM embeddings_faciales WHERE empleado_id = ?", (empleado_id,))
        return cur.rowcount


# ----------------------------------------------------------------------------
# Asistencia
# ----------------------------------------------------------------------------
def registrar_marcacion(empleado_id, tipo_marcacion, epp_porcentaje=0.0, estado="PENDIENTE"):
    """Guarda una marcación y devuelve su timestamp.

    Por defecto queda ``PENDIENTE`` con 0 %: el EPP real lo calculará la
    verificación con YOLO (Fase 2). Ya no se inventa un 100 % / OK.
    """
    if tipo_marcacion not in TIPOS_MARCACION:
        raise ValueError(f"tipo_marcacion debe ser uno de {TIPOS_MARCACION}")
    ahora = _ahora()
    with _db() as conn:
        conn.execute(
            "INSERT INTO asistencia (empleado_id, timestamp, tipo_marcacion, epp_porcentaje, estado) "
            "VALUES (?, ?, ?, ?, ?)",
            (empleado_id, ahora, tipo_marcacion, epp_porcentaje, estado),
        )
    return ahora


# ----------------------------------------------------------------------------
# Alertas
# ----------------------------------------------------------------------------
def registrar_alerta(tipo, area, detalle):
    with _db() as conn:
        cur = conn.execute(
            "INSERT INTO alertas (tipo, timestamp, area, detalle) VALUES (?, ?, ?, ?)",
            (tipo, _ahora(), area, detalle),
        )
        return cur.lastrowid


def get_alertas(limite=50):
    with _db() as conn:
        filas = conn.execute(
            "SELECT * FROM alertas ORDER BY id DESC LIMIT ?", (int(limite),)
        )
        return [dict(r) for r in filas]


# ----------------------------------------------------------------------------
def _ahora():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    init_db()
    print(f"Base de datos lista en {config.DB_PATH} (esquema v{VERSION_ACTUAL}).")
