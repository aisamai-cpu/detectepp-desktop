"""Registro de cámaras: qué fuentes de video conoce el sistema y cuál está activa.

Se guarda en ``data/camaras.json`` (fuera de git, porque las URL pueden llevar
usuario y contraseña). Soporta tres tipos de fuente:

- ``usb``     : cámara integrada o USB. ``valor`` = índice (0, 1, 2...).
- ``url``     : cámara IP, de seguridad o celular usado como cámara.
                ``valor`` = ``rtsp://...`` o ``http://...`` (MJPEG).
- ``archivo`` : un video grabado (demo de respaldo si falla la cámara).

Este módulo solo administra la lista; abrir y leer el video lo hace ``video_source``.
"""
import json
import logging
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass, fields
from typing import List, Optional

import config

logger = logging.getLogger(__name__)

TIPOS = ("usb", "url", "archivo")
PREFIJOS_URL = ("rtsp://", "rtsps://", "http://", "https://")

_CREDENCIALES = re.compile(r"(://)[^/@\s]+@")


def ocultar_credenciales(texto: str) -> str:
    """``rtsp://admin:1234@192.168.1.5/x`` -> ``rtsp://***@192.168.1.5/x`` (para pantalla y logs)."""
    return _CREDENCIALES.sub(r"\1***@", texto)


@dataclass
class Camara:
    id: str
    nombre: str
    tipo: str
    valor: str
    ancho: Optional[int] = None   # solo para USB: resolución pedida (None = la que traiga la cámara)
    alto: Optional[int] = None

    @property
    def valor_visible(self) -> str:
        """El valor sin contraseñas, seguro para mostrar."""
        return ocultar_credenciales(self.valor)

    def resumen(self) -> str:
        etiqueta = {"usb": "USB", "url": "IP", "archivo": "Archivo"}[self.tipo]
        return f"{self.nombre}  [{etiqueta}: {self.valor_visible}]"


_CAMPOS = {f.name for f in fields(Camara)}


# ----------------------------------------------------------------------------
# Lectura / escritura del archivo
# ----------------------------------------------------------------------------
def _por_defecto() -> dict:
    cam = Camara(id="cam-integrada", nombre="Cámara integrada", tipo="usb", valor="0")
    return {"camaras": [asdict(cam)], "activa": cam.id}


def _escribir(datos: dict):
    config.ensure_dirs()
    ruta = config.CAMARAS_PATH
    parcial = ruta.with_suffix(".json.tmp")
    parcial.write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(parcial, ruta)  # escritura atómica: nunca queda un archivo a medias


def _leer() -> dict:
    ruta = config.CAMARAS_PATH
    if not ruta.exists():
        datos = _por_defecto()
        _escribir(datos)
        return datos
    try:
        datos = json.loads(ruta.read_text(encoding="utf-8"))
        camaras = [Camara(**{k: v for k, v in c.items() if k in _CAMPOS}) for c in datos["camaras"]]
        if not camaras or datos.get("activa") not in {c.id for c in camaras}:
            raise ValueError("lista vacía o cámara activa inexistente")
        return datos
    except (ValueError, KeyError, TypeError) as e:
        respaldo = ruta.with_name(f"camaras.corrupto-{int(time.time())}.json")
        os.replace(ruta, respaldo)
        logger.warning("camaras.json inválido (%s). Se guardó copia en %s y se recreó.", e, respaldo.name)
        datos = _por_defecto()
        _escribir(datos)
        return datos


# ----------------------------------------------------------------------------
# API pública
# ----------------------------------------------------------------------------
def listar() -> List[Camara]:
    return [Camara(**{k: v for k, v in c.items() if k in _CAMPOS}) for c in _leer()["camaras"]]


def obtener(cam_id: str) -> Camara:
    for cam in listar():
        if cam.id == cam_id:
            return cam
    raise KeyError(f"No existe la cámara '{cam_id}'")


def activa() -> Camara:
    return obtener(_leer()["activa"])


def activar(cam_id: str):
    datos = _leer()
    if cam_id not in {c["id"] for c in datos["camaras"]}:
        raise KeyError(f"No existe la cámara '{cam_id}'")
    datos["activa"] = cam_id
    _escribir(datos)


def agregar(nombre: str, tipo: str, valor: str, ancho: Optional[int] = None,
            alto: Optional[int] = None, activar_ahora: bool = False) -> Camara:
    """Registra una cámara nueva. Lanza ValueError con un mensaje claro si los datos no sirven."""
    nombre = (nombre or "").strip()
    valor = (valor or "").strip()
    if not nombre or len(nombre) > 60:
        raise ValueError("El nombre es obligatorio (máximo 60 caracteres)")
    if tipo not in TIPOS:
        raise ValueError(f"Tipo inválido '{tipo}'. Usa: {', '.join(TIPOS)}")
    if tipo == "usb":
        if not valor.isdigit():
            raise ValueError("Una cámara USB se identifica con un número de índice (0, 1, 2...)")
        valor = str(int(valor))
    elif tipo == "url":
        if not valor.lower().startswith(PREFIJOS_URL):
            raise ValueError("La URL debe empezar por rtsp://, http:// o https://")
    elif not valor:
        raise ValueError("Indica la ruta del archivo de video")
    if (ancho is None) != (alto is None):
        raise ValueError("Indica ancho y alto juntos, o ninguno")

    cam = Camara(id=f"cam-{uuid.uuid4().hex[:8]}", nombre=nombre, tipo=tipo, valor=valor, ancho=ancho, alto=alto)
    datos = _leer()
    datos["camaras"].append(asdict(cam))
    if activar_ahora:
        datos["activa"] = cam.id
    _escribir(datos)
    return cam


def eliminar(cam_id: str):
    datos = _leer()
    restantes = [c for c in datos["camaras"] if c["id"] != cam_id]
    if len(restantes) == len(datos["camaras"]):
        raise KeyError(f"No existe la cámara '{cam_id}'")
    if not restantes:
        raise ValueError("Debe quedar al menos una cámara registrada")
    datos["camaras"] = restantes
    if datos["activa"] == cam_id:
        datos["activa"] = restantes[0]["id"]
    _escribir(datos)
