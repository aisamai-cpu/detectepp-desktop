"""Esqueletos sintéticos (17 puntos COCO) para probar las reglas de Monitoreo sin cámara ni modelo."""
import math

import numpy as np

from core.riesgos import PersonaPose

_PIERNAS = {
    # (rodilla, tobillo) del lado izquierdo en coordenadas locales (cadera en 0,0; y hacia abajo; unidad = torso)
    "rectas": ((-0.12, 0.9), (-0.12, 1.8)),
    "rodillas": ((-0.12, 0.9), (-0.9, 0.95)),     # arrodillado: espinilla horizontal en el piso
    "cuclillas": ((0.5, 0.1), (-0.05, 0.5)),       # cadera cerca de los tobillos
    "sentado": ((0.85, 0.0), (0.85, 0.9)),         # muslo horizontal, espinilla vertical
    "gatas": ((-0.12, 0.85), (-0.9, 0.9)),         # muslos verticales, tronco horizontal
}
_BRAZOS = {
    "abajo": ((0.3, -0.55), (0.3, -0.1)),
    "arriba": ((0.3, -1.3), (0.3, -1.75)),         # codos y muñecas sobre la cabeza (limpiar en alto)
    "cara": ((0.3, -0.55), (0.12, -1.2)),          # muñeca a la altura de la cara, codo abajo
}


def _rot(p, ang):
    a = math.radians(ang)
    return np.array([p[0] * math.cos(a) - p[1] * math.sin(a), p[0] * math.sin(a) + p[1] * math.cos(a)])


def cuerpo(cx=640.0, cy=400.0, T=100.0, tronco=0.0, todo=0.0, piernas="rectas", brazos="abajo",
           cabeza="normal", munecas_dx=0.0, conf=0.9, ocultar=()):
    """Devuelve una PersonaPose. ``tronco``: inclinación del torso (grados, hacia +x).
    ``todo``: giro del cuerpo entero (90 = tendido). ``munecas_dx``: desplazamiento lateral de las muñecas (torsos).
    """
    (rod, tob) = _PIERNAS[piernas]
    (codo, muneca) = _BRAZOS.get(brazos, _BRAZOS["abajo"])
    sup = {  # tronco y cabeza, relativos a la cadera
        5: (-0.25, -1.0), 6: (0.25, -1.0),
        7: (-codo[0], codo[1] - 0.0), 8: (codo[0], codo[1]),
        9: (-muneca[0] + munecas_dx, muneca[1]), 10: (muneca[0] + munecas_dx, muneca[1]),
    }
    if brazos == "piso":   # a gatas: con el tronco girado 90°, brazos colgando hacia el piso
        sup.update({7: (0.45, -1.0), 8: (0.45, -1.0), 9: (0.85, -1.0), 10: (0.85, -1.0)})
    if cabeza == "agachada":
        sup.update({0: (0.45, -1.15), 1: (0.4, -1.22), 2: (0.48, -1.22), 3: (0.3, -1.25), 4: (0.42, -1.25)})
    else:
        sup.update({0: (0.0, -1.35), 1: (-0.05, -1.4), 2: (0.05, -1.4), 3: (-0.1, -1.32), 4: (0.1, -1.32)})
    inf = {11: (-0.15, 0.0), 12: (0.15, 0.0),
           13: rod, 14: (rod[0] + 0.24, rod[1]), 15: tob, 16: (tob[0] + 0.24, tob[1])}
    pts = np.zeros((17, 3), np.float32)
    for k, v in sup.items():
        pts[k, :2] = _rot(np.array(v), tronco)
    for k, v in inf.items():
        pts[k, :2] = v
    for k in range(17):
        pts[k, :2] = _rot(pts[k, :2], todo) * T + np.array([cx, cy])
    pts[:, 2] = conf
    for k in ocultar:
        pts[k, 2] = 0.1
    vis = pts[pts[:, 2] >= 0.5, :2]
    x1, y1 = vis.min(axis=0) - 0.1 * T
    x2, y2 = vis.max(axis=0) + 0.1 * T
    return PersonaPose((float(x1), float(y1), float(x2), float(y2)), 0.9, pts)


class Escena:
    """Reproduce una secuencia de posturas a ``fps`` análisis por segundo sobre un MonitorRiesgos."""

    def __init__(self, monitor, fps=5.0, t0=100.0, tam=(1280, 720)):
        self.m, self.dt, self.t, self.tam = monitor, 1.0 / fps, t0, tam
        self.eventos, self.estados = [], []

    def pasar(self, segundos, generador):
        """``generador(k, fraccion)`` -> lista de PersonaPose para el paso k (fraccion 0..1 del tramo)."""
        n = max(1, int(round(segundos / self.dt)))
        for k in range(n):
            personas = generador(k, k / max(1, n - 1))
            est, ev = self.m.procesar(personas, self.t, self.tam)
            self.estados.append(est)
            self.eventos.extend(ev)
            self.t += self.dt
        return self

    def tipos(self):
        return [e.tipo for e in self.eventos]
