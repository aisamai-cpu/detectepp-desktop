"""Reglas de riesgo del Monitoreo (F1-F9): de esqueletos a semáforo por persona y eventos. Sin PyQt.

``MonitorRiesgos.procesar(personas, t, tam_frame)`` se llama una vez por análisis de pose, en el hilo
de análisis, con el instante de captura ``t`` (segundos). Devuelve el estado de cada persona visible y
los eventos NUEVOS. No guarda nada ni hace sonar nada: eso lo hace ``core.eventos``.

Detalle de cada regla: docs/superpowers/specs/2026-10-13-monitoreo-riesgos-design.md (§5).
"""
from collections import deque
from dataclasses import dataclass, field
from statistics import median
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import config
from core import postura
from core.postura import MUNECA_D, MUNECA_I
from core.seguimiento_personas import SeguidorPersonas

CRITICO, ERGONOMIA, ZONA = "CRITICO", "ERGONOMIA", "ZONA"

# tipo -> (grupo, texto corto para la pantalla)
TIPOS = {
    "CAIDA": (CRITICO, "Caída"),
    "PERSONA_INMOVIL": (CRITICO, "Persona inmóvil"),
    "SOS": (CRITICO, "Gesto de auxilio (SOS)"),
    "ESPALDA_INCORRECTA": (ERGONOMIA, "Espalda doblada al levantar"),
    "BRAZOS_ELEVADOS": (ERGONOMIA, "Brazos elevados"),
    "RODILLAS_PROLONGADO": (ERGONOMIA, "De rodillas / cuclillas"),
    "CUELLO_AGACHADO": (ERGONOMIA, "Cuello agachado"),
    "REPETITIVIDAD": (ERGONOMIA, "Movimientos repetitivos"),
    "ZONA_RESTRINGIDA": (ZONA, "Zona restringida"),
}
VERDE, AMARILLO, ROJO = "verde", "amarillo", "rojo"


@dataclass(frozen=True)
class PersonaPose:
    caja: Tuple[float, float, float, float]   # x1, y1, x2, y2 en píxeles del frame
    confianza: float
    puntos: np.ndarray                         # (17, 3): x, y, confianza


@dataclass(frozen=True)
class Evento:
    tipo: str
    grupo: str
    persona: str
    t: float
    detalle: str
    sonido: bool
    caja: Tuple[float, float, float, float]
    zona: Optional[str] = None


@dataclass(frozen=True)
class EstadoPersona:
    id: str
    caja: Tuple[float, float, float, float]
    puntos: np.ndarray
    semaforo: str
    etiqueta: str                 # condición más grave activa, p. ej. "Espalda doblada 1 s"
    emergencia: Optional[str]     # "CAÍDA" / "INMÓVIL" / "SOS" o None
    cuerpo_completo: bool
    seg_amarillo: float
    seg_rojo: float


@dataclass(frozen=True)
class FotoEvento:
    """Frame analizado (copia) y las personas que había, para la foto del evento."""
    frame: np.ndarray
    personas: Tuple[PersonaPose, ...]


@dataclass(frozen=True)
class ResultadoPose:
    """Lo que el worker entrega a la pestaña Monitoreo en cada frame (el último análisis)."""
    estados: Tuple[EstadoPersona, ...]
    frecuencia: float              # análisis de pose por segundo, medidos
    tam_frame: Tuple[int, int]     # ancho, alto del frame analizado


class Sostenido:
    """Cuánto tiempo seguido se cumple una condición, tolerando pausas cortas."""

    def __init__(self):
        self.desde: Optional[float] = None
        self.ultimo: Optional[float] = None

    def actualizar(self, cond: bool, t: float, tolerancia: float = 0.0) -> float:
        if cond:
            if self.desde is None:
                self.desde = t
            self.ultimo = t
            return t - self.desde
        if self.desde is not None and t - self.ultimo <= tolerancia:
            return self.ultimo - self.desde
        self.desde = self.ultimo = None
        return 0.0


class _Agitado:
    """Cuenta cambios de dirección horizontal de una muñeca, con histéresis."""

    def __init__(self):
        self.ext: Optional[float] = None
        self.dir = 0
        self.visto: Optional[float] = None
        self.cambios: deque = deque()

    def reiniciar(self):
        self.ext, self.dir, self.visto = None, 0, None
        self.cambios.clear()

    def agregar(self, x: float, t: float, amplitud: float):
        if self.ext is None:
            self.ext = x
        elif self.dir == 0:
            if abs(x - self.ext) >= amplitud:
                self.dir, self.ext = (1 if x > self.ext else -1), x
        elif self.dir == 1:
            if x > self.ext:
                self.ext = x
            elif self.ext - x >= amplitud:
                self.dir, self.ext = -1, x
                self.cambios.append(t)
        else:
            if x < self.ext:
                self.ext = x
            elif x - self.ext >= amplitud:
                self.dir, self.ext = 1, x
                self.cambios.append(t)
        self.visto = t

    def contar(self, t: float, ventana: float) -> int:
        while self.cambios and t - self.cambios[0] > ventana:
            self.cambios.popleft()
        return len(self.cambios)


@dataclass
class _Reglas:
    """Estado de las reglas de UNA persona seguida."""
    torsos: deque = field(default_factory=deque)       # (t, torso) mientras estaba erguida
    torso_ref: Optional[float] = None
    congelado: bool = False
    hist: deque = field(default_factory=deque)         # (t, hombro_y, ang_tronco)
    bajada_t: Optional[float] = None
    horizontal: Sostenido = field(default_factory=Sostenido)
    caida_activa: bool = False
    puntos_prev: Optional[np.ndarray] = None
    movs: deque = field(default_factory=deque)         # (t, movimiento) mientras está horizontal
    inmovil_t: Optional[float] = None
    agitado: Dict[int, _Agitado] = field(default_factory=lambda: {MUNECA_I: _Agitado(), MUNECA_D: _Agitado()})
    sos_activo: bool = False
    espalda: Sostenido = field(default_factory=Sostenido)
    brazos: Sostenido = field(default_factory=Sostenido)
    rodillas: Sostenido = field(default_factory=Sostenido)
    cuello: Sostenido = field(default_factory=Sostenido)
    disparado: set = field(default_factory=set)        # reglas sostenidas ya avisadas en el episodio actual
    flexionado: bool = False
    flexiones: deque = field(default_factory=deque)
    zonas: Dict[str, Sostenido] = field(default_factory=dict)
    t_prev: Optional[float] = None
    seg_amarillo: float = 0.0
    seg_rojo: float = 0.0


class MonitorRiesgos:
    def __init__(self):
        self.seguidor = SeguidorPersonas()
        self.zonas: List[Tuple[str, float, float, float, float]] = []   # (nombre, x1, y1, x2, y2) en 0..1
        self._ultimo: Dict[tuple, float] = {}    # enfriamiento por (tipo, zona): sobrevive a reiniciar()

    def reiniciar(self):
        """Cambio de cámara, giro, reconexión o vuelta del video: se olvida a todas las personas."""
        self.seguidor.reiniciar()

    # ------------------------------------------------------------------
    def procesar(self, personas: Sequence[PersonaPose], t: float,
                 tam_frame: Tuple[int, int]) -> Tuple[List[EstadoPersona], List[Evento]]:
        medidas = [postura.medir(p.puntos, p.caja) for p in personas]
        ids = self.seguidor.actualizar([(p.caja, m.cadera_m, m.torso) for p, m in zip(personas, medidas)], t)
        estados, eventos = [], []
        for p, m, pid in zip(personas, medidas, ids):
            pista = self.seguidor.pistas[pid]
            r = pista.datos.setdefault("reglas", _Reglas())
            estado, nuevos = self._persona(pid, p, m, r, t, tam_frame)
            pista.horizontal = estado.emergencia is not None or bool(r.horizontal.desde)
            pista.emergencia = estado.emergencia is not None
            estados.append(estado)
            eventos.extend(nuevos)
        return estados, eventos

    # ------------------------------------------------------------------
    def _evento(self, tipo, pid, t, detalle, caja, eventos, zona=None, sonido=True) -> bool:
        grupo = TIPOS[tipo][0]
        if grupo != CRITICO:
            clave = (tipo, zona)
            if t - self._ultimo.get(clave, -1e18) < config.MON_ENFRIAMIENTO_S:
                return False
            self._ultimo[clave] = t
        eventos.append(Evento(tipo, grupo, pid, t, detalle, sonido, tuple(caja), zona))
        return True

    def _torso_ref(self, m: postura.Medidas, r: _Reglas, t: float) -> Optional[float]:
        if m.torso and m.ang_tronco is not None and m.ang_tronco < 30 and not r.congelado:
            r.torsos.append((t, m.torso))
        while r.torsos and t - r.torsos[0][0] > 3.0:
            r.torsos.popleft()
        if r.torsos and not r.congelado:
            r.torso_ref = median(v for _, v in r.torsos)
        elif r.torso_ref is None and m.torso:
            r.torso_ref = m.torso
        return r.torso_ref

    def _sostenida(self, clave, s, umbral, r, cond_sigue) -> bool:
        """True una sola vez por episodio, cuando una postura sostenida alcanza su umbral."""
        if not cond_sigue and s == 0:
            r.disparado.discard(clave)
            return False
        if s >= umbral and clave not in r.disparado:
            r.disparado.add(clave)
            return True
        return False

    def _persona(self, pid, p: PersonaPose, m: postura.Medidas, r: _Reglas, t, tam_frame):
        eventos: List[Evento] = []
        ref = self._torso_ref(m, r, t)
        ind = postura.indicadores(m, ref) if ref else postura.Indicadores(brazo_elevado=m.codo_arriba)
        caja = p.caja

        # ---------------- F1 caída ----------------
        if m.tiene_tronco and ref:
            r.hist.append((t, float(m.hombro_m[1]), m.ang_tronco))
            while r.hist and t - r.hist[0][0] > config.MON_CAIDA_VENTANA_S:
                r.hist.popleft()
            bajo = float(m.hombro_m[1]) - min(h for _, h, _ in r.hist)
            giro = m.ang_tronco > 60 and any(a < 30 for _, _, a in r.hist)
            if r.bajada_t is None and (bajo > config.MON_CAIDA_BAJADA_TORSOS * ref or giro):
                r.bajada_t = t
                r.congelado = True          # la escala no debe encogerse durante la caída
        seg_horizontal = r.horizontal.actualizar(ind.horizontal, t, tolerancia=0.4)
        if r.bajada_t is not None:
            inicio = r.horizontal.desde
            if (seg_horizontal >= config.MON_CAIDA_HORIZONTAL_S and inicio is not None
                    and inicio - r.bajada_t <= config.MON_CAIDA_CONFIRMAR_S):
                r.caida_activa = True
                r.bajada_t = None
                self._evento("CAIDA", pid, t, f"Caída detectada ({pid})", caja, eventos)
            elif t - r.bajada_t > config.MON_CAIDA_CONFIRMAR_S + config.MON_CAIDA_HORIZONTAL_S + 0.5:
                r.bajada_t = None           # no quedó tendida: no era caída (se agachó, se sentó…)
        if seg_horizontal == 0:
            r.caida_activa = False
            r.inmovil_t = None
            r.movs.clear()
            if r.bajada_t is None and m.ang_tronco is not None and m.ang_tronco < 30:
                r.congelado = False

        # ---------------- F2 tendido e inmóvil ----------------
        if ind.horizontal and r.puntos_prev is not None and ref:
            mov = postura.movimiento(r.puntos_prev, p.puntos, config.MON_PUNTO_CONF_MIN, ref)
            if mov is not None:
                r.movs.append((t, mov))
        while r.movs and t - r.movs[0][0] > config.MON_INMOVIL_S:
            r.movs.popleft()
        if (seg_horizontal >= config.MON_INMOVIL_S and len(r.movs) >= 3
                and float(np.mean([v for _, v in r.movs])) < config.MON_INMOVIL_MOV_TORSOS
                and (r.inmovil_t is None or t - r.inmovil_t >= config.MON_ENFRIAMIENTO_S)):
            r.inmovil_t = t
            extra = ", tras una caída" if r.caida_activa else ""
            self._evento("PERSONA_INMOVIL", pid, t,
                         f"Persona tendida e inmóvil {seg_horizontal:.0f} s ({pid}){extra}", caja, eventos)
        r.puntos_prev = np.asarray(p.puntos, dtype=np.float32).copy()

        # ---------------- F3 SOS ----------------
        conteo = 0
        if ref and m.cabeza is not None:
            limite = float(m.cabeza[1]) - 0.20 * ref
            for k, ag in r.agitado.items():
                punto = p.puntos[k]
                if punto[2] >= config.MON_PUNTO_CONF_MIN and float(punto[1]) < limite:
                    ag.agregar(float(punto[0]), t, config.MON_SOS_AMPLITUD_TORSOS * ref)
                elif ag.visto is not None and t - ag.visto > 0.6:   # se tolera una muestra perdida
                    ag.reiniciar()
                conteo = max(conteo, ag.contar(t, config.MON_SOS_S))
        if not ind.manos_sobre_cabeza and r.sos_activo:
            r.sos_activo = False
        if conteo >= config.MON_SOS_CAMBIOS and not r.sos_activo:
            r.sos_activo = True
            for ag in r.agitado.values():
                ag.reiniciar()
            r.brazos = Sostenido()          # gana el SOS sobre "brazos elevados"
            self._evento("SOS", pid, t, f"Gesto de auxilio ({pid})", caja, eventos)

        # ---------------- F4-F7 posturas sostenidas ----------------
        tol = config.MON_TOLERANCIA_S
        s_espalda = r.espalda.actualizar(ind.espalda_roja, t, tolerancia=0.3)
        if self._sostenida("espalda", s_espalda, config.MON_ESPALDA_S, r, ind.espalda_roja):
            self._evento("ESPALDA_INCORRECTA", pid, t,
                         f"Levantamiento con espalda doblada y piernas rectas ({pid})", caja, eventos)
        s_brazos = r.brazos.actualizar(ind.brazo_elevado and not r.sos_activo, t, tol)
        if self._sostenida("brazos", s_brazos, config.MON_BRAZOS_S, r, ind.brazo_elevado):
            self._evento("BRAZOS_ELEVADOS", pid, t, f"Brazos sobre los hombros {s_brazos:.0f} s ({pid})", caja, eventos)
        s_rodillas = r.rodillas.actualizar(ind.postura_baja, t, tol)
        if self._sostenida("rodillas", s_rodillas, config.MON_RODILLAS_S, r, ind.postura_baja):
            self._evento("RODILLAS_PROLONGADO", pid, t, f"De rodillas / cuclillas {s_rodillas:.0f} s ({pid})",
                         caja, eventos)
        s_cuello = r.cuello.actualizar(ind.cuello_agachado, t, tol)
        if self._sostenida("cuello", s_cuello, config.MON_CUELLO_S, r, ind.cuello_agachado):
            self._evento("CUELLO_AGACHADO", pid, t, f"Cuello agachado {s_cuello:.0f} s ({pid})", caja, eventos,
                         sonido=config.MON_CUELLO_SONIDO)

        # ---------------- F8 repetitividad ----------------
        if m.ang_tronco is not None and not ind.horizontal:
            if m.ang_tronco > 45:
                r.flexionado = True
            elif m.ang_tronco < 20 and r.flexionado:
                r.flexionado = False
                r.flexiones.append(t)
        ventana = config.MON_REPETICION_VENTANA_MIN * 60
        while r.flexiones and t - r.flexiones[0] > ventana:
            r.flexiones.popleft()
        if len(r.flexiones) >= config.MON_REPETICIONES_MIN * config.MON_REPETICION_VENTANA_MIN:
            n = len(r.flexiones)
            r.flexiones.clear()
            self._evento("REPETITIVIDAD", pid, t,
                         f"{n} flexiones de tronco en {config.MON_REPETICION_VENTANA_MIN:g} min ({pid})", caja, eventos)

        # ---------------- F9 zonas ----------------
        if self.zonas and m.tobillos:
            w, h = tam_frame
            pie = np.mean(m.tobillos, axis=0)
            px, py = float(pie[0]) / max(1, w), float(pie[1]) / max(1, h)
            for nombre, x1, y1, x2, y2 in self.zonas:
                dentro = x1 <= px <= x2 and y1 <= py <= y2
                s = r.zonas.setdefault(nombre, Sostenido()).actualizar(dentro, t, tolerancia=0.4)
                if self._sostenida(("zona", nombre), s, config.MON_ZONA_S, r, dentro):
                    self._evento("ZONA_RESTRINGIDA", pid, t, f"{pid} entró a {nombre}", caja, eventos, zona=nombre)

        # ---------------- semáforo y etiqueta ----------------
        emergencia = "SOS" if r.sos_activo else ("INMÓVIL" if r.inmovil_t is not None else
                                                 ("CAÍDA" if r.caida_activa else None))
        if emergencia or ind.espalda_roja:
            semaforo = ROJO
        elif ind.tronco_amarillo or ind.brazo_elevado or ind.postura_baja or ind.cuello_agachado:
            semaforo = AMARILLO
        else:
            semaforo = VERDE
        if r.t_prev is not None:
            dt = min(1.0, t - r.t_prev)
            if semaforo == AMARILLO:
                r.seg_amarillo += dt
            elif semaforo == ROJO:
                r.seg_rojo += dt
        r.t_prev = t

        if emergencia:
            etiqueta = emergencia
        elif ind.espalda_roja:
            etiqueta = f"Espalda doblada {s_espalda:.0f} s"
        elif ind.brazo_elevado:
            etiqueta = f"Brazos arriba {s_brazos:.0f} s"
        elif ind.postura_baja:
            etiqueta = f"De rodillas {s_rodillas:.0f} s"
        elif ind.cuello_agachado:
            etiqueta = f"Cuello agachado {s_cuello:.0f} s"
        elif ind.tronco_amarillo:
            etiqueta = "Tronco inclinado"
        elif not m.cuerpo_completo:
            etiqueta = "cuerpo incompleto"
        else:
            etiqueta = "OK"
        estado = EstadoPersona(pid, tuple(caja), p.puntos, semaforo, etiqueta, emergencia, m.cuerpo_completo,
                               r.seg_amarillo, r.seg_rojo)
        return estado, eventos
