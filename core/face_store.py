"""Puente entre la base de datos y el motor de rostros.

- ``cargar_galeria()``: lee todos los vectores UNA vez y arma la galería en memoria.
- ``registrar_empleado()``: guarda al empleado y sus vectores de forma atómica.
"""
import logging

from core import database
from vision.face_engine import Galeria, bytes_a_embedding, embedding_a_bytes

logger = logging.getLogger(__name__)


def cargar_galeria() -> Galeria:
    """Galería con los vectores de todos los empleados. Ignora BLOBs corruptos."""
    registros = []
    for emp_id, datos in database.get_embeddings():
        try:
            registros.append((emp_id, bytes_a_embedding(datos)))
        except ValueError:
            logger.warning("Vector facial inválido ignorado (empleado %s)", emp_id)
    galeria = Galeria()
    galeria.cargar(registros)
    return galeria


def registrar_empleado(nombre, edad, area, foto_path, vectores) -> int:
    """Registra un empleado con 1 o más vectores (arrays de 128). Devuelve su id."""
    return database.add_empleado_con_embeddings(
        nombre, edad, area, foto_path, [embedding_a_bytes(v) for v in vectores]
    )
