"""Administra y prueba las cámaras del sistema desde la consola.

    python -m tools.probar_camaras listar
    python -m tools.probar_camaras detectar                     (busca cámaras USB 0..4)
    python -m tools.probar_camaras agregar-usb 1 --nombre "Webcam externa"
    python -m tools.probar_camaras agregar-url "http://192.168.1.50:8080/video" --nombre "Celular Ana" --rotar 90
    python -m tools.probar_camaras agregar-url "rtsp://admin:clave@192.168.1.20:554/stream1" --nombre "Pasillo"
    python -m tools.probar_camaras agregar-archivo "C:\\videos\\demo.mp4" --nombre "Video demo"
    python -m tools.probar_camaras probar [ID]                  (ventana en vivo)
    python -m tools.probar_camaras rotar ID 90                  (0, 90, 180 o 270 grados)
    python -m tools.probar_camaras activar ID
    python -m tools.probar_camaras eliminar ID

En la ventana de «probar»:  R = girar 90°   G = guardar ese giro en la cámara   Q / ESC = salir.
Las contraseñas de las URL nunca se muestran en pantalla; se guardan en data/camaras.json.
"""
import argparse
import sys
import time

import cv2
import numpy as np

from core import camaras
from core.video_source import CONECTANDO, OK, VideoSource, detectar_usb

VENTANA = "Prueba de camara - DETECTEPP"
MAX_ANCHO_VENTANA, MAX_ALTO_VENTANA = 1000, 620  # que la ventana quepa en una laptop (también en vertical)


def ajustar_a_pantalla(img: np.ndarray, max_ancho: int = MAX_ANCHO_VENTANA, max_alto: int = MAX_ALTO_VENTANA) -> np.ndarray:
    """Reduce la imagen (sin deformarla ni agrandarla) para que entre completa en la ventana."""
    alto, ancho = img.shape[:2]
    escala = min(1.0, max_ancho / ancho, max_alto / alto)
    if escala >= 1.0:
        return img.copy()
    return cv2.resize(img, (round(ancho * escala), round(alto * escala)), interpolation=cv2.INTER_AREA)


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
    print(f"Probando {cam.resumen()}")
    print("Teclas: R = girar 90 grados | G = guardar el giro | Q o ESC = salir")
    fuente = VideoSource(cam)
    fuente.iniciar()
    ultimo_n = 0
    t_pantalla, n_pantalla, fps_pantalla = time.monotonic(), 0, 0.0
    vista = None
    try:
        while True:
            frame, n = fuente.leer()
            if n != ultimo_n or vista is None:  # solo se redibuja cuando llega una imagen nueva
                ultimo_n = n
                if frame is None:
                    vista = np.zeros((360, 640, 3), np.uint8)
                    _poner_texto(vista, "Conectando..." if fuente.estado == CONECTANDO else "Sin senal, reintentando...",
                                 180, (0, 165, 255))
                else:
                    vista = ajustar_a_pantalla(frame)
                    alto, ancho = frame.shape[:2]
                    color = (0, 220, 0) if fuente.estado == OK else (0, 165, 255)
                    _poner_texto(vista, f"imagen {ancho}x{alto} | giro {fuente.rotacion} | estado {fuente.estado}", 22, color)
                    _poner_texto(vista, f"lectura {fuente.fps:.0f} fps | pantalla {fps_pantalla:.0f} fps", 46, color)
                n_pantalla += 1
                ahora = time.monotonic()
                if ahora - t_pantalla >= 1.0:
                    fps_pantalla = n_pantalla / (ahora - t_pantalla)
                    t_pantalla, n_pantalla = ahora, 0
                cv2.imshow(VENTANA, vista)

            tecla = cv2.waitKey(5) & 0xFF
            if tecla in (ord("q"), 27) or cv2.getWindowProperty(VENTANA, cv2.WND_PROP_VISIBLE) < 1:
                break
            if tecla == ord("r"):
                fuente.rotacion = (fuente.rotacion + 90) % 360
            if tecla == ord("g"):
                camaras.rotar(cam.id, fuente.rotacion)
                print(f"Giro {fuente.rotacion} grados guardado en «{cam.nombre}».")
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
        p.add_argument("--rotar", type=int, choices=camaras.ROTACIONES, default=0, help="giro en grados")
    p = sub.add_parser("rotar")
    p.add_argument("id")
    p.add_argument("grados", type=int, choices=camaras.ROTACIONES)
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
            cam = camaras.agregar(args.nombre, tipo, args.valor, activar_ahora=args.activar, rotacion=args.rotar)
            print(f"Agregada: {cam.id}  {cam.resumen()}")
        elif args.cmd == "rotar":
            camaras.rotar(args.id, args.grados)
            print(f"Giro de «{args.id}»: {args.grados} grados.")
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
