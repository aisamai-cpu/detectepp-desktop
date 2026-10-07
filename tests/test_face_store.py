"""Pruebas de los vectores faciales en la base de datos (Fase 1.3)."""
import sqlite3
from contextlib import closing

import numpy as np

import config
from core import database, face_store
from tests.helpers import BaseConCarpetaTemporal
from vision.face_engine import DIM_EMBEDDING, embedding_a_bytes


def vec(semilla):
    v = np.random.default_rng(semilla).normal(size=DIM_EMBEDDING).astype(np.float32)
    return v / np.linalg.norm(v)


class TestEmbeddingsEnBD(BaseConCarpetaTemporal):
    def setUp(self):
        super().setUp()
        database.init_db()

    def test_registrar_y_reconocer_tras_reiniciar(self):
        ana, luis = vec(1), vec(2)
        id_ana = face_store.registrar_empleado("Ana Pérez", 30, "Área Química", "a.jpg", [ana, vec(11)])
        id_luis = face_store.registrar_empleado("Luis Ñandú", 41, "Área Maquinaria", "l.jpg", [luis])

        galeria = face_store.cargar_galeria()  # como al abrir la app de nuevo
        self.assertEqual(galeria.buscar(ana)[0], id_ana)
        self.assertEqual(galeria.buscar(luis)[0], id_luis)
        self.assertIsNone(galeria.buscar(vec(999))[0])
        self.assertEqual(database.contar_embeddings(id_ana), 2)

    def test_vector_ida_y_vuelta_exacto(self):
        v = vec(3)
        emp = face_store.registrar_empleado("Eva", 25, "Área Química", "e.jpg", [v])
        (_, datos), = database.get_embeddings()
        np.testing.assert_allclose(np.frombuffer(datos, np.float32), v, atol=1e-7)
        self.assertEqual(database.contar_embeddings(emp), 1)

    def test_registro_es_atomico(self):
        # Un vector con tamaño incorrecto debe impedir que se cree el empleado.
        with self.assertRaises(ValueError):
            face_store.registrar_empleado("Mal", 20, "Área Química", "m.jpg", [np.zeros(10)])
        self.assertEqual(database.get_todos_empleados(), [])

        # Si falla DENTRO de la transacción (el empleado ya se insertó y luego el vector
        # es inválido), el empleado también se revierte.
        with self.assertRaises(TypeError):
            database.add_empleado_con_embeddings("Mal2", 20, "Área Química", "m.jpg", [None])
        self.assertEqual(database.get_todos_empleados(), [])

    def test_sin_vectores_no_se_registra(self):
        with self.assertRaises(ValueError):
            database.add_empleado_con_embeddings("X", 20, "Área Química", "x.jpg", [])

    def test_agregar_muestra_extra(self):
        emp = face_store.registrar_empleado("Ana", 30, "Área Química", "a.jpg", [vec(1)])
        database.add_embedding(emp, embedding_a_bytes(vec(5)))
        self.assertEqual(database.contar_embeddings(emp), 2)

    def test_embedding_de_empleado_inexistente_falla(self):
        with self.assertRaises(sqlite3.IntegrityError):
            database.add_embedding(12345, embedding_a_bytes(vec(1)))

    def test_eliminar_datos_biometricos(self):
        emp = face_store.registrar_empleado("Ana", 30, "Área Química", "a.jpg", [vec(1), vec(2)])
        self.assertEqual(database.eliminar_embeddings(emp), 2)
        self.assertIsNone(face_store.cargar_galeria().buscar(vec(1))[0])
        self.assertEqual(len(database.get_todos_empleados()), 1)  # el empleado sigue

    def test_blob_corrupto_se_ignora(self):
        emp = face_store.registrar_empleado("Ana", 30, "Área Química", "a.jpg", [vec(1)])
        database.add_embedding(emp, b"basura")
        galeria = face_store.cargar_galeria()
        self.assertEqual(galeria.buscar(vec(1))[0], emp)


class TestMigracionV2(BaseConCarpetaTemporal):
    def test_base_v1_con_datos_pasa_a_v2(self):
        # Simula una BD creada con el esquema de la Fase 0 (versión 1).
        config.ensure_dirs()
        with closing(database.get_connection()) as conn:
            database._migracion_v1(conn)
            conn.execute("PRAGMA user_version = 1")
            conn.execute("INSERT INTO empleados (nombre, edad, area, foto_path) VALUES ('Luis', 40, 'Área Maquinaria', 'l.jpg')")
            conn.commit()

        database.init_db()

        with closing(database.get_connection()) as conn:
            self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], database.VERSION_ACTUAL)
            tablas = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertIn("embeddings_faciales", tablas)
        self.assertEqual([e["nombre"] for e in database.get_todos_empleados()], ["Luis"])
        self.assertEqual(database.VERSION_ACTUAL, 2)
