"""Etiquetado automático de fotos de EPP guiado por el esqueleto (ronda 2 de entrenamiento).

Cada carpeta de fotos corresponde a una prenda (Careta, Cofia, Lentes…). El modelo de pose ubica
cabeza, ojos, orejas, boca, hombros y rodillas de la persona principal, y con eso se dibuja la caja
de la prenda de esa carpeta (la cofia sobre la cabeza, el respirador en la boca, el mandil del pecho
a las rodillas…). También se etiqueta "persona" en todas las personas detectadas.

Resultado: un dataset en formato YOLO listo para subir a Roboflow y REVISAR allí (las cajas son una
aproximación: hay que corregir las que queden corridas).

Uso (desde la carpeta del proyecto, en la laptop):
    venv\\Scripts\\python -m tools.etiquetar_epp "C:\\ruta\\EPPfase2"
Crea entrenamiento\\ronda2\\ (imágenes + etiquetas + data.yaml + vistas previas + informe.txt)
y entrenamiento\\ronda2.zip para subir a Roboflow.
"""
import argparse
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

CLASES = ["careta", "cofia", "lentes", "mandil", "orejeras", "respirador", "traje", "persona"]

# Carpeta -> prendas que se etiquetan en la persona principal.
# En Cofia y Respirador la persona lleva también el traje en las fotos revisadas.
REGLAS = {
    "careta": ["careta"],
    "cofia": ["cofia", "traje"],
    "lentes": ["lentes"],
    "mandil": ["mandil"],
    "orejera": ["orejeras"],
    "orejeras": ["orejeras"],
    "respirador": ["respirador", "traje"],
    "traje de protección": ["traje"],
    "traje de proteccion": ["traje"],
    "traje": ["traje"],
}
CONF_PUNTO = 0.5
LADO_MAX = 1280            # las fotos se reducen a este lado mayor (sube rápido y entrena igual)
BLOQUE = 10                # fotos seguidas de una misma ráfaga van juntas al mismo grupo
CADA_VALIDACION = 7        # 1 de cada 7 bloques va a validación (~15 %)
EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _pts(k, idx):
    return [k[i, :2] for i in idx if k[i, 2] >= CONF_PUNTO]


def caja_prenda(clase, k, caja_persona):
    """Caja (x1, y1, x2, y2) de la prenda según los puntos del esqueleto, o None si faltan puntos."""
    ojos, orejas, nariz = _pts(k, (1, 2)), _pts(k, (3, 4)), _pts(k, (0,))
    hombros, rodillas, caderas = _pts(k, (5, 6)), _pts(k, (13, 14)), _pts(k, (11, 12))
    x1, y1, x2, y2 = caja_persona
    if clase == "traje":
        return caja_persona
    if clase == "mandil":
        if not hombros:
            return None
        h = np.array(hombros)
        ancho = float(np.ptp(h[:, 0])) if len(h) == 2 else 0.25 * (x2 - x1)
        abajo = np.array(rodillas)[:, 1].mean() + 0.1 * ancho if rodillas else y2
        return (h[:, 0].mean() - 0.75 * ancho, h[:, 1].mean() - 0.05 * ancho, h[:, 0].mean() + 0.75 * ancho, abajo)
    cabeza = ojos + orejas + nariz
    if not cabeza:
        return None
    c = np.array(cabeza)
    ancho_hombros = float(np.ptp(np.array(hombros)[:, 0])) if len(hombros) == 2 else 0.0
    s = max(float(np.ptp(c[:, 0])), 0.45 * ancho_hombros, 0.06 * (y2 - y1), 15.0)   # tamaño de la cabeza
    ojo_y = np.array(ojos)[:, 1].mean() if ojos else c[:, 1].mean()
    # centro de la cabeza: entre orejas y ojos (de perfil, los ojos quedan adelante)
    cx = np.array(orejas + ojos)[:, 0].mean() if orejas else c[:, 0].mean()
    cx_cara = np.array(ojos + nariz)[:, 0].mean() if (ojos or nariz) else cx
    # Ancho real de la cabeza: distancia entre orejas (de frente). De cerca los hombros salen muy
    # anchos y agrandan "s"; para cofia y orejeras se usa la medida de las orejas cuando se ven ambas.
    oreja_a_oreja = float(np.ptp(np.array(orejas)[:, 0])) if len(orejas) == 2 else 0.0
    if clase == "cofia":
        if oreja_a_oreja > 0.6 * s:
            a = oreja_a_oreja
            return (cx - 0.62 * a, ojo_y - 1.0 * a, cx + 0.62 * a, ojo_y - 0.08 * a)
        return (cx - 0.9 * s, ojo_y - 1.2 * s, cx + 0.9 * s, ojo_y + 0.12 * s)    # de perfil
    if clase == "orejeras":
        if oreja_a_oreja > 0.6 * s:
            a = oreja_a_oreja
            return (cx - 0.72 * a, ojo_y - 0.8 * a, cx + 0.72 * a, ojo_y + 0.5 * a)
        return (cx - 0.9 * s, ojo_y - 0.85 * s, cx + 0.9 * s, ojo_y + 0.55 * s)
    if clase == "respirador":
        return (cx_cara - 0.65 * s, ojo_y + 0.15 * s, cx_cara + 0.65 * s, ojo_y + 1.1 * s)
    if clase == "careta":
        return (cx_cara - 0.9 * s, ojo_y - 1.0 * s, cx_cara + 0.9 * s, ojo_y + 1.25 * s)
    if clase == "lentes":
        return (cx_cara - 0.62 * s, ojo_y - 0.25 * s, cx_cara + 0.62 * s, ojo_y + 0.25 * s)
    return None


def a_yolo(caja, w, h):
    x1, y1, x2, y2 = (max(0.0, min(float(v), lim)) for v, lim in zip(caja, (w, h, w, h)))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return f"{(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"


def leer(ruta: Path):
    datos = np.fromfile(str(ruta), dtype=np.uint8)          # admite tildes en la ruta (Windows)
    return cv2.imdecode(datos, cv2.IMREAD_COLOR)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Etiqueta automáticamente fotos de EPP por carpeta.")
    ap.add_argument("origen", type=Path, help="carpeta con una subcarpeta por prenda (EPPfase2)")
    ap.add_argument("--salida", type=Path, default=Path("entrenamiento") / "ronda2")
    ap.add_argument("--modelo", default="models/yolo11s-pose.pt")
    a = ap.parse_args(argv)

    from ultralytics import YOLO
    modelo = YOLO(a.modelo)
    if a.salida.exists():
        shutil.rmtree(a.salida)
    for split in ("train", "valid"):
        (a.salida / split / "images").mkdir(parents=True)
        (a.salida / split / "labels").mkdir(parents=True)
    (a.salida / "previas").mkdir(parents=True)

    conteo, sin_persona, sin_prenda, por_carpeta = Counter(), [], [], Counter()
    for carpeta in sorted(p for p in a.origen.iterdir() if p.is_dir()):
        clases = REGLAS.get(carpeta.name.lower())
        if clases is None:
            print(f"  (se ignora la carpeta {carpeta.name}: no tiene regla)")
            continue
        fotos = sorted(p for p in carpeta.iterdir() if p.suffix.lower() in EXTENSIONES)
        print(f"▶ {carpeta.name}: {len(fotos)} fotos")
        for i, foto in enumerate(fotos):
            img = leer(foto)
            if img is None:
                continue
            h, w = img.shape[:2]
            esc = min(1.0, LADO_MAX / max(h, w))
            if esc < 1.0:
                img = cv2.resize(img, (int(w * esc), int(h * esc)), interpolation=cv2.INTER_AREA)
                h, w = img.shape[:2]
            r = modelo.predict(img, conf=0.5, verbose=False)[0]
            lineas = []
            if r.boxes is not None and len(r.boxes):
                cajas = r.boxes.xyxy.cpu().numpy()
                puntos = r.keypoints.data.cpu().numpy()
                for caja in cajas:
                    y = a_yolo(caja, w, h)
                    if y:
                        lineas.append(f"{CLASES.index('persona')} {y}")
                principal = int(np.argmax((cajas[:, 2] - cajas[:, 0]) * (cajas[:, 3] - cajas[:, 1])))
                for clase in clases:
                    caja = caja_prenda(clase, puntos[principal], cajas[principal])
                    y = a_yolo(caja, w, h) if caja is not None else None
                    if y:
                        lineas.append(f"{CLASES.index(clase)} {y}")
                        conteo[clase] += 1
                    else:
                        sin_prenda.append(f"{carpeta.name}/{foto.name} ({clase})")
            else:
                sin_persona.append(f"{carpeta.name}/{foto.name}")
                continue          # sin persona no hay forma de ubicar la prenda: se deja fuera
            split = "valid" if (i // BLOQUE) % CADA_VALIDACION == CADA_VALIDACION - 1 else "train"
            nombre = f"{carpeta.name.replace(' ', '_')}_{foto.stem}"
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            buf.tofile(str(a.salida / split / "images" / f"{nombre}.jpg"))
            (a.salida / split / "labels" / f"{nombre}.txt").write_text("\n".join(lineas))
            por_carpeta[(carpeta.name, split)] += 1
            if i % 25 == 0:      # vista previa para revisar rápido
                vista = img.copy()
                for linea in lineas:
                    c, cx, cy, bw, bh = linea.split()
                    cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
                    color = (200, 200, 200) if CLASES[int(c)] == "persona" else (0, 255, 0)
                    cv2.rectangle(vista, (int(cx - bw / 2), int(cy - bh / 2)), (int(cx + bw / 2), int(cy + bh / 2)), color, 3)
                    cv2.putText(vista, CLASES[int(c)], (int(cx - bw / 2), int(cy - bh / 2) - 8), 0, 1.0, color, 2)
                cv2.imencode(".jpg", cv2.resize(vista, None, fx=0.5, fy=0.5))[1].tofile(
                    str(a.salida / "previas" / f"{nombre}.jpg"))

    (a.salida / "data.yaml").write_text(
        "train: train/images\nval: valid/images\n"
        f"nc: {len(CLASES)}\nnames: [{', '.join(CLASES)}]\n", encoding="utf-8")
    informe = ["ETIQUETADO AUTOMÁTICO · RONDA 2", "",
               "Cajas por prenda: " + ", ".join(f"{k} {v}" for k, v in sorted(conteo.items())), "",
               "Fotos por carpeta (train / valid):"]
    for carpeta in sorted({c for c, _ in por_carpeta}):
        informe.append(f"  {carpeta}: {por_carpeta[(carpeta, 'train')]} / {por_carpeta[(carpeta, 'valid')]}")
    informe += ["", f"Fotos SIN persona detectada (quedan fuera; etiquetar a mano si se quieren): {len(sin_persona)}"]
    informe += [f"  {x}" for x in sin_persona]
    informe += ["", f"Prendas que no se pudieron ubicar (faltaban puntos del esqueleto): {len(sin_prenda)}"]
    informe += [f"  {x}" for x in sin_prenda]
    (a.salida / "informe.txt").write_text("\n".join(informe), encoding="utf-8")

    zip_ruta = a.salida.with_suffix(".zip")
    with zipfile.ZipFile(zip_ruta, "w", zipfile.ZIP_STORED) as z:
        for f in a.salida.rglob("*"):
            if f.is_file() and "previas" not in f.parts and f.name != "informe.txt":
                z.write(f, f.relative_to(a.salida))
    print("\n".join(informe[:12]))
    print(f"\nListo: {zip_ruta}  (súbelo a Roboflow)\nVistas previas: {a.salida / 'previas'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
