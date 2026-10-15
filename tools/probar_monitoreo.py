"""Calibración del Monitoreo con videos grabados (sin abrir la aplicación).

Pasa un video por el modelo de pose y las reglas de riesgo, y muestra qué detecta y cuándo.
Sirve para ajustar los umbrales MON_* de config.py con videos reales (caída simulada sobre
colchoneta, levantar un balde bien y mal, limpiar en alto, de rodillas, agitar los brazos…).

Uso (desde la carpeta del proyecto):
    venv\\Scripts\\python -m tools.probar_monitoreo videos\\caida1.mp4
    venv\\Scripts\\python -m tools.probar_monitoreo videos\\caida1.mp4 --esperado CAIDA
    venv\\Scripts\\python -m tools.probar_monitoreo videos\\  --esperados videos\\esperado.json
    venv\\Scripts\\python -m tools.probar_monitoreo videos\\caida1.mp4 --salida revisar.mp4   (video con esqueletos)

``esperado.json``: {"caida1.mp4": ["CAIDA"], "trapear.mp4": [], "sos1.mp4": ["SOS"]}
(una lista vacía = el clip NO debe generar ninguna emergencia).
"""
import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

import config
from core.postura import ESQUELETO
from core.riesgos import CRITICO, MonitorRiesgos
from vision.pose_engine import PoseEngine

COLOR = {"verde": (118, 230, 0), "amarillo": (77, 183, 255), "rojo": (68, 23, 255)}
EXTENSIONES = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}


def analizar(ruta: Path, motor: PoseEngine, rotacion: int = 0, salida: Path = None, verbose: bool = True):
    """Devuelve la lista de eventos (core.riesgos.Evento) del video."""
    cap = cv2.VideoCapture(str(ruta))
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir {ruta}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    paso = max(1, int(round(fps * config.MON_INTERVALO_S)))     # se analiza como en vivo: cada MON_INTERVALO_S
    giro = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(rotacion)
    monitor, eventos, escritor = MonitorRiesgos(), [], None
    n, ultimo_seg = 0, -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        if (n - 1) % paso:
            continue
        if giro is not None:
            frame = cv2.rotate(frame, giro)
        t = (n - 1) / fps
        personas = motor.detectar(frame)
        h, w = frame.shape[:2]
        estados, nuevos = monitor.procesar(personas, t, (w, h))
        for ev in nuevos:
            eventos.append(ev)
            if verbose:
                marca = "🚨" if ev.grupo == CRITICO else "⚠"
                print(f"  {t:7.1f} s  {marca} {ev.tipo:20s} {ev.detalle}")
        if verbose and int(t) != ultimo_seg:
            ultimo_seg = int(t)
            resumen = ", ".join(f"{e.id}:{e.semaforo}:{e.etiqueta}" for e in estados) or "sin personas"
            print(f"  {t:7.1f} s  {resumen}")
        if salida is not None:
            if escritor is None:
                escritor = cv2.VideoWriter(str(salida), cv2.VideoWriter_fourcc(*"mp4v"), fps / paso, (w, h))
            vista = frame.copy()
            for e in estados:
                pts = np.asarray(e.puntos).reshape(17, 3)
                for a, b in ESQUELETO:
                    if pts[a, 2] >= config.MON_PUNTO_CONF_MIN and pts[b, 2] >= config.MON_PUNTO_CONF_MIN:
                        cv2.line(vista, tuple(int(v) for v in pts[a, :2]), tuple(int(v) for v in pts[b, :2]),
                                 COLOR[e.semaforo], 2, cv2.LINE_AA)
                x1, y1 = int(e.caja[0]), int(e.caja[1])
                cv2.putText(vista, e.id, (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR[e.semaforo], 2)
            cv2.putText(vista, f"{t:5.1f}s", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            escritor.write(vista)
    cap.release()
    if escritor is not None:
        escritor.release()
    return eventos


def comparar(eventos, esperados):
    """True si los eventos CRÍTICOS coinciden con lo esperado y aparecen los de ergonomía esperados."""
    tipos = [e.tipo for e in eventos]
    criticos = {e.tipo for e in eventos if e.grupo == CRITICO}
    esperados = set(esperados)
    faltan = sorted(esperados - set(tipos))
    sobran = sorted(criticos - esperados)
    return not faltan and not sobran, faltan, sobran


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Prueba las reglas de Monitoreo con videos grabados.")
    ap.add_argument("ruta", type=Path, help="video o carpeta con videos")
    ap.add_argument("--modelo", type=Path, default=None, help="ONNX de pose (por defecto config.POSE_MODELO_PATH)")
    ap.add_argument("--rotacion", type=int, default=0, choices=(0, 90, 180, 270))
    ap.add_argument("--esperado", nargs="*", default=None, help="eventos esperados para UN video (p. ej. CAIDA)")
    ap.add_argument("--esperados", type=Path, default=None, help="JSON {video: [eventos]} para una carpeta")
    ap.add_argument("--salida", type=Path, default=None, help="guarda un video con los esqueletos dibujados")
    ap.add_argument("--silencioso", action="store_true", help="solo el resultado final")
    a = ap.parse_args(argv)

    motor = PoseEngine(a.modelo)
    videos = sorted(p for p in a.ruta.iterdir() if p.suffix.lower() in EXTENSIONES) if a.ruta.is_dir() else [a.ruta]
    esperados = json.loads(a.esperados.read_text(encoding="utf-8")) if a.esperados else {}
    if a.esperado is not None and len(videos) == 1:
        esperados[videos[0].name] = a.esperado

    aciertos = total = 0
    for v in videos:
        print(f"\n▶ {v.name}")
        eventos = analizar(v, motor, a.rotacion, a.salida if len(videos) == 1 else None, not a.silencioso)
        print(f"  eventos: {[e.tipo for e in eventos] or 'ninguno'}")
        if v.name in esperados:
            total += 1
            ok, faltan, sobran = comparar(eventos, esperados[v.name])
            aciertos += ok
            print("  ✔ OK" if ok else f"  ✘ faltan: {faltan or '-'} · emergencias de más: {sobran or '-'}")
    if total:
        print(f"\nResultado: {aciertos}/{total} videos como se esperaba")
        return 0 if aciertos == total else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
