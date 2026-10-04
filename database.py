import sqlite3
import json
import os
from datetime import datetime

DB_NAME = "detectepp.db"

def get_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS empleados (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nombre TEXT NOT NULL,
        edad INTEGER NOT NULL,
        area TEXT NOT NULL,
        foto_path TEXT NOT NULL,
        fecha_registro DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS asistencia (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        empleado_id INTEGER NOT NULL,
        timestamp DATETIME NOT NULL,
        tipo_marcacion TEXT NOT NULL,
        epp_porcentaje REAL DEFAULT 0.0,
        estado TEXT DEFAULT 'PENDIENTE',
        FOREIGN KEY(empleado_id) REFERENCES empleados(id)
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS areas_epp (
        area TEXT PRIMARY KEY,
        epp_requeridos_json TEXT NOT NULL
    )
    """)

    default_areas = [
        ("Área Química", json.dumps(["mascarilla", "lentes", "bata", "guantes"])),
        ("Área Maquinaria", json.dumps(["casco", "botas", "chaleco", "lentes"]))
    ]

    cursor.executemany("""
    INSERT OR IGNORE INTO areas_epp (area, epp_requeridos_json)
    VALUES (?, ?)
    """, default_areas)

    conn.commit()
    conn.close()

    if not os.path.exists("fotos_empleados"):
        os.makedirs("fotos_empleados")

def add_empleado(nombre, edad, area, foto_path):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO empleados (nombre, edad, area, foto_path)
    VALUES (?, ?, ?, ?)
    """, (nombre, edad, area, foto_path))
    emp_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return emp_id

def get_todos_empleados():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM empleados")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def registrar_marcacion(empleado_id, tipo_marcacion, epp_porcentaje=100.0, estado="OK"):
    conn = get_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
    INSERT INTO asistencia (empleado_id, timestamp, tipo_marcacion, epp_porcentaje, estado)
    VALUES (?, ?, ?, ?, ?)
    """, (empleado_id, now_str, tipo_marcacion, epp_porcentaje, estado))
    conn.commit()
    conn.close()
    return now_str

if __name__ == "__main__":
    init_db()
    print(" Backend SQLite inicializado correctamente.")