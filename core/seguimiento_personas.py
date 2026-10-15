"""Seguidor simple de personas entre análisis de pose (ids P1, P2…). Sin PyQt.

Empareja por la distancia entre centros de cadera (o de caja), medida en torsos, y acepta
también una superposición de cajas baja: una persona que pasa de pie a tendida cambia mucho
la forma de su caja (IoU ≈ 0.1-0.25) pero su cadera casi no se mueve.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

import config


@dataclass
class Pista:
    id: str
    centro: np.ndarray
    caja: Tuple[float, float, float, float]
    escala: float                    # torso (px) o, si no hay, un tercio del alto de la caja
    visto_en: float
    horizontal: bool = False
    emergencia: bool = False
    datos: Dict = field(default_factory=dict)   # estado de las reglas (lo usa core.riesgos)


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def centro_y_escala(caja, cadera_m: Optional[np.ndarray], torso: Optional[float]):
    x1, y1, x2, y2 = caja
    centro = np.asarray(cadera_m, dtype=float) if cadera_m is not None else np.array([(x1 + x2) / 2, (y1 + y2) / 2])
    escala = float(torso) if torso else max(1.0, (y2 - y1) / 3)
    return centro, escala


class SeguidorPersonas:
    MAX_DIST_TORSOS = 1.5     # más lejos que esto (y sin superposición) = otra persona
    SALTO_TORSOS = 3.0        # un salto mayor nunca es la misma persona (corte de video, etc.)

    def __init__(self):
        self.pistas: Dict[str, Pista] = {}
        self._siguiente = 1

    def reiniciar(self):
        self.pistas.clear()
        self._siguiente = 1

    def actualizar(self, detecciones: List[Tuple[tuple, Optional[np.ndarray], Optional[float]]], t: float) -> List[str]:
        """``detecciones``: (caja, cadera_m, torso) por persona. Devuelve el id asignado a cada una."""
        candidatos = []
        for i, (caja, cadera, torso) in enumerate(detecciones):
            centro, escala = centro_y_escala(caja, cadera, torso)
            for pid, p in self.pistas.items():
                ref = max(escala, p.escala)
                dist = float(np.linalg.norm(centro - p.centro)) / ref
                if dist > self.SALTO_TORSOS:
                    continue
                solape = iou(caja, p.caja)
                if dist <= self.MAX_DIST_TORSOS or solape >= 0.1:
                    candidatos.append((dist - solape, i, pid))
        candidatos.sort()
        asignados: Dict[int, str] = {}
        usados = set()
        for _, i, pid in candidatos:
            if i in asignados or pid in usados:
                continue
            asignados[i] = pid
            usados.add(pid)

        ids = []
        for i, (caja, cadera, torso) in enumerate(detecciones):
            centro, escala = centro_y_escala(caja, cadera, torso)
            pid = asignados.get(i)
            if pid is None:
                pid = f"P{self._siguiente}"
                self._siguiente += 1
                self.pistas[pid] = Pista(pid, centro, tuple(caja), escala, t)
            p = self.pistas[pid]
            p.centro, p.caja, p.visto_en = centro, tuple(caja), t
            # La escala cambia lento: al tenderse el torso visible se acorta mucho.
            p.escala = escala if not p.escala else 0.7 * p.escala + 0.3 * escala
            ids.append(pid)

        for pid in list(self.pistas):
            p = self.pistas[pid]
            limite = config.MON_PERDIDA_HORIZONTAL_S if (p.horizontal or p.emergencia) else config.MON_PERDIDA_S
            if t - p.visto_en > limite:
                del self.pistas[pid]
        return ids
