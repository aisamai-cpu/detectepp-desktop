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
ROTACIONES = (0, 90, 180, 270)  # grados en sentido horario
PREFIJOS_URL = ("rtsp://", "rtsps://", "http://", "https://")

_CREDENCIALES = re.compile(r"(://)[^/@\s]+@")

PUERTO_DEFECTO = 8080       # el de la app IP Webcam
RUTA_DEFECTO = "/video"     # transmisión MJPEG de IP Webcam
_IPV4 = re.compile(r"^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$")
_NOMBRE_HOST = re.compile(r"^[A-Za-z0-9]([A-Za-z0-9-]*[A-Za-z0-9])?(\.[A-Za-z0-9]([A-Za-z0-9-]*[A-Za-z0-9])?)*$")
_MSG_URL = ("Escribe la IP del celular (ej. 192.168.0.123 o 192.168.0.123:8080) "
            "o una dirección completa que empiece por http:// o rtsp://")


def normalizar_url(texto: str) -> str:
    """Convierte lo que escribe el usuario en una URL de video completa.

    - ``192.168.0.123``            -> ``http://192.168.0.123:8080/video``  (IP Webcam)
    - ``192.168.0.123:4747``       -> ``http://192.168.0.123:4747/video``
    - ``192.168.0.123/videofeed``  -> ``http://192.168.0.123:8080/videofeed``
    - ``http://192.168.0.123:8080``-> ``http://192.168.0.123:8080/video``
    - ``rtsp://...`` o ``http://.../ruta`` se respetan tal cual.
    Lanza ``ValueError`` con un mensaje claro si no se entiende.
    """
    t = (texto or "").strip()
    if not t or re.search(r"\s", t):
        raise ValueError(_MSG_URL)
    if t.lower().startswith(PREFIJOS_URL):
        sin_ruta = re.match(r"^(https?://[^/?#]+)/?$", t, re.IGNORECASE)
        return sin_ruta.group(1) + RUTA_DEFECTO if sin_ruta else t
    if "://" in t:
        raise ValueError(_MSG_URL)

    direccion, _, resto = t.partition("/")
    host, dos_puntos, puerto = direccion.partition(":")
    if re.fullmatch(r"[\d.]+", host):
        octetos = _IPV4.match(host)
        if not octetos or any(int(o) > 255 for o in octetos.groups()):
            raise ValueError(f"«{host}» no es una IP válida. {_MSG_URL}")
    elif not _NOMBRE_HOST.match(host):
        raise ValueError(_MSG_URL)
    if dos_puntos:
        if not puerto.isdigit() or not 0 < int(puerto) < 65536:
            raise ValueError(f"El puerto «{puerto}» no es válido (use un número entre 1 y 65535)")
    else:
        puerto = str(PUERTO_DEFECTO)
    return f"http://{host}:{int(puerto)}/{resto}" if resto else f"http://{host}:{int(puerto)}{RUTA_DEFECTO}"


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
    rotacion: int = 0             # 0 / 90 / 180 / 270: p. ej. 90 si el celular se usa en vertical

    @property
    def valor_visible(self) -> str:
        """El valor sin contraseñas, seguro para mostrar."""
        return ocultar_credenciales(self.valor)

    def resumen(self) -> str:
        etiqueta = {"usb": "USB", "url": "IP", "archivo": "Archivo"}[self.tipo]
        giro = f", giro {self.rotacion}°" if self.rotacion else ""
        return f"{self.nombre}  [{etiqueta}: {self.valor_visible}{giro}]"


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
            alto: Optional[int] = None, activar_ahora: bool = False, rotacion: int = 0) -> Camara:
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
        valor = normalizar_url(valor)
    elif not valor:
        raise ValueError("Indica la ruta del archivo de video")
    if (ancho is None) != (alto is None):
        raise ValueError("Indica ancho y alto juntos, o ninguno")
    if rotacion not in ROTACIONES:
        raise ValueError(f"La rotación debe ser una de {ROTACIONES} grados")

    cam = Camara(id=f"cam-{uuid.uuid4().hex[:8]}", nombre=nombre, tipo=tipo, valor=valor, ancho=ancho, alto=alto,
                 rotacion=rotacion)
    datos = _leer()
    datos["camaras"].append(asdict(cam))
    if activar_ahora:
        datos["activa"] = cam.id
    _escribir(datos)
    return cam


def rotar(cam_id: str, grados: int):
    """Fija la rotación de una cámara: 0, 90, 180 o 270 grados en sentido horario."""
    if grados not in ROTACIONES:
        raise ValueError(f"La rotación debe ser una de {ROTACIONES} grados")
    datos = _leer()
    for c in datos["camaras"]:
        if c["id"] == cam_id:
            c["rotacion"] = grados
            _escribir(datos)
            return
    raise KeyError(f"No existe la cámara '{cam_id}'")


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
