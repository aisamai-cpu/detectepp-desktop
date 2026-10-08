"""Administra y prueba las cámaras del sistema desde la consola.

    python -m tools.probar_camaras listar
    python -m tools.probar_camaras detectar                     (busca cámaras USB 0..4)
    python -m tools.probar_camaras agregar-usb 1 --nombre "Webcam externa"
    python -m tools.probar_camaras agregar-url "http://192.168.1.50:8080/video" --nombre "Celular Ana"
    python -m tools.probar_camaras agregar-url "rtsp://admin:clave@192.168.1.20:554/stream1" --nombre "Pasillo"
    python -m tools.probar_camaras agregar-archivo "C:\\videos\\demo.mp4" --nombre "Video demo"
    python -m tools.probar_camaras probar [ID]                  (abre una ventana con la cámara; Q para salir)
    python -m tools.probar_camaras activar ID
    python -m tools.probar_camaras eliminar ID

Las contraseñas de las URL nunca se muestran en pantalla; se guardan en data/camaras.json.
"""
import argparse
import sys
import time

import cv2
import numpy as np

from core import camaras
from core.video_source import (CONECTANDO, OK, RECONECTANDO, VideoSource,
                               detectar_usb)

VENTANA = "Prueba de camara - DETECTEPP"


def _poner_texto(img, msg, y, color=(255, 255, 255)):
    cv2.putText(img, msg, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
    cv2.putText(img, msg, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)


def _listar():
    activa_id = camaras.activa().id
    for cam in camaras.listar():
        marca = "*" if cam.id == activa_id else " "
        print(f" {marca} {cam.id:<14} {cam.resumen()}")
    print("\n(* = cámara activa)")


def _detectar():
    print("Buscando cámaras USB (puede tardar unos segundos)...")
    encontradas = detectar_usb()
    if not encontradas:
        print("No se encontró ninguna. ¿Está conectada? ¿Otra aplicación la está usando?")
    for e in encontradas:
        print(f"  Índice {e['indice']}: {e['ancho']}x{e['alto']}")
    if encontradas:
        print('Para registrar una: python -m tools.probar_camaras agregar-usb INDICE --nombre "Nombre"')


def _probar(cam_id):
    cam = camaras.obtener(cam_id) if cam_id else camaras.activa()
    print(f"Probando {cam.resumen()}  (Q o ESC para salir)")
    fuente = VideoSource(cam)
    fuente.iniciar()
    ultimo_n, ultimo_t, fps = 0, time.monotonic(), 0.0
    try:
        while True:
            frame, n = fuente.leer()
            if frame is None:
                imagen = np.zeros((480, 640, 3), np.uint8)
                _poner_texto(imagen, "Conectando..." if fuente.estado == CONECTANDO else "Sin senal, reintentando...", 240, (0, 165, 255))
            else:
                imagen = frame.copy()
                ahora = time.monotonic()
                if ahora - ultimo_t >= 1.0:
                    fps = (n - ultimo_n) / (ahora - ultimo_t)
                    ultimo_n, ultimo_t = n, ahora
                alto, ancho = imagen.shape[:2]
                color = (0, 220, 0) if fuente.estado == OK else (0, 165, 255)
                _poner_texto(imagen, f"{ancho}x{alto}  {fps:.0f} fps  estado: {fuente.estado}", 25, color)
            cv2.imshow(VENTANA, imagen)
            tecla = cv2.waitKey(15) & 0xFF
            if tecla in (ord("q"), 27) or cv2.getWindowProperty(VENTANA, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        fuente.detener()
        cv2.destroyAllWindows()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Administra y prueba las cámaras de DETECTEPP")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("listar")
    sub.add_parser("detectar")
    for nombre, ayuda in (("agregar-usb", "índice (0, 1, 2...)"), ("agregar-url", "rtsp://... o http://..."),
                          ("agregar-archivo", "ruta del video")):
        p = sub.add_parser(nombre)
        p.add_argument("valor", help=ayuda)
        p.add_argument("--nombre", required=True)
        p.add_argument("--activar", action="store_true", help="dejarla como cámara activa")
    for nombre in ("probar", "activar", "eliminar"):
        p = sub.add_parser(nombre)
        p.add_argument("id", nargs="?" if nombre == "probar" else None)

    args = ap.parse_args(argv)
    try:
        if args.cmd == "listar":
            _listar()
        elif args.cmd == "detectar":
            _detectar()
        elif args.cmd.startswith("agregar-"):
            tipo = {"agregar-usb": "usb", "agregar-url": "url", "agregar-archivo": "archivo"}[args.cmd]
            cam = camaras.agregar(args.nombre, tipo, args.valor, activar_ahora=args.activar)
            print(f"Agregada: {cam.id}  {cam.resumen()}")
        elif args.cmd == "activar":
            camaras.activar(args.id)
            print(f"Cámara activa: {camaras.activa().resumen()}")
        elif args.cmd == "eliminar":
            camaras.eliminar(args.id)
            print("Eliminada.")
        elif args.cmd == "probar":
            _probar(args.id)
    except (ValueError, KeyError) as e:
        print(f"Error: {e.args[0] if e.args else e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
