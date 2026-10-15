"""Geometría del esqueleto (17 puntos COCO) para el Monitoreo de riesgos. Funciones puras, sin PyQt.

Coordenadas de imagen: ``y`` crece hacia ABAJO ("más alto" = ``y`` menor).
Las distancias que dependen del tamaño de la persona se expresan en "torsos"
(distancia hombros-caderas) para no depender de qué tan lejos está de la cámara.
"""
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

import config

NARIZ, OJO_I, OJO_D, OREJA_I, OREJA_D = 0, 1, 2, 3, 4
HOMBRO_I, HOMBRO_D, CODO_I, CODO_D, MUNECA_I, MUNECA_D = 5, 6, 7, 8, 9, 10
CADERA_I, CADERA_D, RODILLA_I, RODILLA_D, TOBILLO_I, TOBILLO_D = 11, 12, 13, 14, 15, 16

# Líneas del esqueleto para dibujar
ESQUELETO = (
    (HOMBRO_I, HOMBRO_D), (HOMBRO_I, CODO_I), (CODO_I, MUNECA_I), (HOMBRO_D, CODO_D), (CODO_D, MUNECA_D),
    (HOMBRO_I, CADERA_I), (HOMBRO_D, CADERA_D), (CADERA_I, CADERA_D),
    (CADERA_I, RODILLA_I), (RODILLA_I, TOBILLO_I), (CADERA_D, RODILLA_D), (RODILLA_D, TOBILLO_D),
    (NARIZ, OJO_I), (NARIZ, OJO_D), (OJO_I, OREJA_I), (OJO_D, OREJA_D),
)
# (hombro, codo, muñeca) y (cadera, rodilla, tobillo) de cada lado
BRAZOS = ((HOMBRO_I, CODO_I, MUNECA_I), (HOMBRO_D, CODO_D, MUNECA_D))
PIERNAS = ((CADERA_I, RODILLA_I, TOBILLO_I), (CADERA_D, RODILLA_D, TOBILLO_D))


@dataclass
class Medidas:
    """Lo que se puede medir en UNA imagen, sin historia."""
    caja: Tuple[float, float, float, float]
    hombro_m: Optional[np.ndarray] = None
    cadera_m: Optional[np.ndarray] = None
    torso: Optional[float] = None               # distancia hombros-caderas en esta imagen (px)
    ang_tronco: Optional[float] = None          # 0 = erguido, 90 = horizontal
    cabeza: Optional[np.ndarray] = None         # nariz, o punto medio de orejas/ojos
    ang_cuello: Optional[float] = None          # inclinación de la cabeza respecto al tronco
    rodillas: List[Tuple[float, np.ndarray, Optional[np.ndarray]]] = field(default_factory=list)
    # ^ (ángulo o NaN si falta el tobillo, punto de rodilla, punto de tobillo o None)
    codo_arriba: bool = False
    munecas: List[np.ndarray] = field(default_factory=list)
    tobillos: List[np.ndarray] = field(default_factory=list)
    visibles: np.ndarray = None                 # máscara (17,) de puntos fiables

    @property
    def tiene_tronco(self) -> bool:
        return self.hombro_m is not None and self.cadera_m is not None and bool(self.torso)

    @property
    def tiene_piernas(self) -> bool:
        return any(t is not None for _, _, t in self.rodillas)

    @property
    def cuerpo_completo(self) -> bool:
        return self.tiene_tronco and self.tiene_piernas

    @property
    def ancho_alto(self) -> float:
        x1, y1, x2, y2 = self.caja
        return (x2 - x1) / max(1.0, y2 - y1)


def _medio(p: np.ndarray, vis: np.ndarray, a: int, b: int) -> Optional[np.ndarray]:
    if vis[a] and vis[b]:
        return (p[a, :2] + p[b, :2]) / 2
    if vis[a]:
        return p[a, :2].copy()
    if vis[b]:
        return p[b, :2].copy()
    return None


def angulo(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Ángulo en ``b`` (grados) entre ``b->a`` y ``b->c``."""
    v1, v2 = a - b, c - b
    n = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if n < 1e-6:
        return float("nan")
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(v1, v2)) / n))))


def angulo_vertical(desde: np.ndarray, hacia: np.ndarray) -> float:
    """Ángulo del vector ``desde -> hacia`` respecto a la vertical hacia ARRIBA (0..180)."""
    v = hacia - desde
    return math.degrees(math.atan2(abs(float(v[0])), -float(v[1])))


def medir(puntos: np.ndarray, caja, conf_min: Optional[float] = None) -> Medidas:
    conf_min = config.MON_PUNTO_CONF_MIN if conf_min is None else conf_min
    p = np.asarray(puntos, dtype=np.float32).reshape(17, 3)
    vis = p[:, 2] >= conf_min
    m = Medidas(caja=tuple(float(v) for v in caja), visibles=vis)
    m.hombro_m = _medio(p, vis, HOMBRO_I, HOMBRO_D)
    m.cadera_m = _medio(p, vis, CADERA_I, CADERA_D)
    if m.hombro_m is not None and m.cadera_m is not None:
        m.torso = float(np.linalg.norm(m.hombro_m - m.cadera_m)) or None
        if m.torso:
            m.ang_tronco = angulo_vertical(m.cadera_m, m.hombro_m)

    if vis[NARIZ]:
        m.cabeza = p[NARIZ, :2].copy()
    else:
        m.cabeza = _medio(p, vis, OREJA_I, OREJA_D)
        if m.cabeza is None:
            m.cabeza = _medio(p, vis, OJO_I, OJO_D)
    oreja = _medio(p, vis, OREJA_I, OREJA_D)
    if m.tiene_tronco and (oreja is not None or m.cabeza is not None):
        cabeza = oreja if oreja is not None else m.cabeza
        tronco = m.hombro_m - m.cadera_m
        cuello = cabeza - m.hombro_m
        n = float(np.linalg.norm(tronco) * np.linalg.norm(cuello))
        if n > 1e-6:
            m.ang_cuello = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(tronco, cuello)) / n))))

    for h, c, w in BRAZOS:
        if vis[h] and vis[c] and p[c, 1] < p[h, 1]:
            m.codo_arriba = True
        if vis[w]:
            m.munecas.append(p[w, :2].copy())
    for cad, rod, tob in PIERNAS:
        if not vis[rod]:
            continue
        tobillo = p[tob, :2].copy() if vis[tob] else None
        if tobillo is not None:
            m.tobillos.append(tobillo)
        ang = angulo(p[cad, :2], p[rod, :2], tobillo) if (vis[cad] and tobillo is not None) else float("nan")
        m.rodillas.append((ang, p[rod, :2].copy(), tobillo))
    return m


# ----------------------------------------------------------------------------
# Indicadores que necesitan el torso de referencia (la escala de la persona)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Indicadores:
    horizontal: bool = False
    a_gatas: bool = False
    tronco_amarillo: bool = False
    espalda_roja: bool = False          # espalda doblada > 60° con piernas rectas
    postura_baja: bool = False          # de rodillas o en cuclillas
    cuello_agachado: bool = False
    brazo_elevado: bool = False
    manos_sobre_cabeza: bool = False


def indicadores(m: Medidas, torso_ref: float) -> Indicadores:
    if not m.tiene_tronco or not torso_ref:
        return Indicadores(brazo_elevado=m.codo_arriba)
    ang = m.ang_tronco
    cad_y = float(m.cadera_m[1])

    # Tronco horizontal con una rodilla DOBLADA claramente por debajo de la cadera = a gatas (fregando).
    # (Con la pierna recta es alguien agachado de pie, no a gatas.)
    a_gatas = ang >= 60 and any(
        float(r[1]) - cad_y > 0.5 * torso_ref and (t is None or math.isnan(a) or a < 120)
        for a, r, t in m.rodillas)
    # Tobillos por debajo de la cadera = la persona se apoya en sus pies (agachada, en cuclillas), no tendida.
    # Tendida en el piso, los tobillos quedan a la altura de la cadera.
    # Se mira la DIRECCIÓN cadera->tobillo: hacia abajo (< 45° de la vertical) = piernas que sostienen.
    def _apoya(t):
        dx, dy = float(t[0]) - float(m.cadera_m[0]), float(t[1]) - cad_y
        return dy > 0.2 * torso_ref and abs(dx) < dy
    sobre_sus_pies = any(_apoya(t) for t in m.tobillos)
    horizontal = (ang >= 60 and m.ancho_alto >= config.MON_RELACION_HORIZONTAL
                  and not sobre_sus_pies and not a_gatas)

    angulos = [a for a, _, t in m.rodillas if t is not None and not math.isnan(a)]
    piernas_rectas = bool(angulos) and all(a > config.MON_RODILLA_RECTA for a in angulos)
    espalda_roja = ang > config.MON_TRONCO_ROJO and piernas_rectas and not horizontal

    # De rodillas: espinilla casi horizontal (tobillo a la altura de la rodilla), rodilla bajo la cadera.
    arrodillado = any(t is not None and abs(float(t[1]) - float(r[1])) < 0.35 * torso_ref
                      and float(r[1]) > cad_y for _, r, t in m.rodillas)
    # En cuclillas: cadera cerca de los tobillos y rodillas dobladas.
    cuclillas = (any(float(t[1]) - cad_y < config.MON_CADERA_BAJA_TORSOS * torso_ref for t in m.tobillos)
                 and any(a < config.MON_RODILLA_FLEXION for a in angulos))
    postura_baja = (arrodillado or cuclillas or a_gatas) and not horizontal

    manos_arriba = False
    if m.cabeza is not None and ang < 45:     # tendido o muy inclinado, "arriba" en la imagen no significa nada
        limite = float(m.cabeza[1]) - 0.20 * torso_ref
        manos_arriba = any(float(w[1]) < limite for w in m.munecas)

    return Indicadores(
        horizontal=horizontal,
        a_gatas=a_gatas,
        tronco_amarillo=config.MON_TRONCO_AMARILLO <= ang <= config.MON_TRONCO_ROJO and not horizontal,
        espalda_roja=espalda_roja,
        postura_baja=postura_baja,
        cuello_agachado=(m.ang_cuello is not None and m.ang_cuello > config.MON_CUELLO_ANG
                         and ang < config.MON_TRONCO_ROJO),
        brazo_elevado=m.codo_arriba and ang < 60,
        manos_sobre_cabeza=manos_arriba,
    )


def movimiento(antes: np.ndarray, ahora: np.ndarray, conf_min: float, torso_ref: float) -> Optional[float]:
    """Mediana del desplazamiento de los puntos visibles en ambos análisis, en torsos."""
    a, b = np.asarray(antes).reshape(17, 3), np.asarray(ahora).reshape(17, 3)
    comunes = (a[:, 2] >= conf_min) & (b[:, 2] >= conf_min)
    if comunes.sum() < 3 or not torso_ref:
        return None
    d = np.linalg.norm(a[comunes, :2] - b[comunes, :2], axis=1)
    return float(np.median(d)) / torso_ref
