"""Reporte diario en HTML (asistencia, EPP, emergencias, ergonomía). Sin PyQt ni dependencias nuevas.

Se abre en el navegador y desde ahí se puede imprimir o guardar como PDF.
Monitoreo no identifica personas: sus eventos se agrupan por tipo y cámara, nunca por trabajador.
"""
import datetime
import html
from collections import Counter
from pathlib import Path
from typing import Optional

import config
from core import database
from core.riesgos import CRITICO, TIPOS


def _h(v) -> str:
    return html.escape("" if v is None else str(v))


def _segundos(a: str, b: Optional[str]) -> Optional[float]:
    if not b:
        return None
    f = "%Y-%m-%d %H:%M:%S"
    return (datetime.datetime.strptime(b, f) - datetime.datetime.strptime(a, f)).total_seconds()


def _duracion(seg: Optional[float]) -> str:
    if seg is None:
        return "—"
    seg = int(round(seg))
    return f"{seg // 60} min {seg % 60} s" if seg >= 60 else f"{seg} s"


def _epp(detalle: str):
    """'Ana Ruiz no cumple el protocolo de EPP (50 %). Falta: guantes, lentes' -> (nombre, faltantes)."""
    nombre = detalle.split(" no cumple")[0] if " no cumple" in detalle else ""
    faltan = detalle.split("Falta:", 1)[1].strip() if "Falta:" in detalle else ""
    return nombre, faltan


def _tabla(cabeceras, filas, vacio="Sin registros."):
    if not filas:
        return f"<p class='vacio'>{_h(vacio)}</p>"
    th = "".join(f"<th>{_h(c)}</th>" for c in cabeceras)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in f) + "</tr>" for f in filas)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table>"


def generar(fecha: Optional[str] = None) -> Path:
    fecha = fecha or datetime.date.today().strftime("%Y-%m-%d")
    asistencia = database.get_asistencia_dia(fecha)
    alertas_epp = database.get_alertas_dia(fecha, "EPP_INCOMPLETO")
    eventos = database.eventos_dia(fecha)
    criticos = [e for e in eventos if e["grupo"] == CRITICO]
    otros = [e for e in eventos if e["grupo"] != CRITICO]

    # Indicadores
    con_entrada = {a["nombre"] for a in asistencia if a["tipo_marcacion"] == "ENTRADA"}
    con_alerta = {_epp(a["detalle"])[0] for a in alertas_epp}
    cumplen = len(con_entrada - con_alerta)
    respuestas = [s for s in (_segundos(e["timestamp"], e["atendido_en"]) for e in criticos) if s is not None]
    sin_atender = sum(1 for e in criticos if not e["atendido_en"])
    pct = f"{100 * cumplen / len(con_entrada):.0f} %" if con_entrada else "—"
    media = _duracion(sum(respuestas) / len(respuestas)) if respuestas else "—"

    sec_asistencia = _tabla(
        ["Hora", "Empleado", "Área", "Marcación"],
        [(_h(a["timestamp"][11:]), _h(a["nombre"]), _h(a["area"]), _h(a["tipo_marcacion"])) for a in asistencia])
    sec_epp = _tabla(
        ["Hora", "Empleado", "Área", "Faltaba"],
        [(_h(a["timestamp"][11:]), _h(_epp(a["detalle"])[0]), _h(a["area"]), _h(_epp(a["detalle"])[1]))
         for a in alertas_epp], "Ninguna alerta de EPP.")
    sec_criticos = _tabla(
        ["Hora", "Tipo", "Cámara", "Detalle", "Respuesta"],
        [(_h(e["timestamp"][11:]), _h(TIPOS.get(e["tipo"], ("", e["tipo"]))[1]), _h(e["camara"]), _h(e["detalle"]),
          (_h(_duracion(_segundos(e["timestamp"], e["atendido_en"]))) if e["atendido_en"]
           else "<span class='pend'>Sin atender</span>")) for e in criticos], "Ninguna emergencia.")
    conteo = Counter((e["tipo"], e["camara"]) for e in otros)
    sec_otros = _tabla(
        ["Tipo", "Cámara", "Eventos"],
        [(_h(TIPOS.get(t, ("", t))[1]), _h(c), str(n)) for (t, c), n in sorted(conteo.items())],
        "Ningún evento de ergonomía ni de zonas.")

    html_txt = f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><title>DETECTEPP · Reporte {fecha}</title>
<style>
 body {{ font-family: Arial, sans-serif; margin: 32px; color: #222; }}
 h1 {{ margin: 0 0 4px; }} .sub {{ color: #666; margin-bottom: 24px; }}
 h2 {{ border-bottom: 2px solid #00a65a; padding-bottom: 4px; margin-top: 32px; }}
 table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
 th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; }}
 th {{ background: #f3f3f3; }} .vacio {{ color: #888; }} .pend {{ color: #c62828; font-weight: bold; }}
 .kpis {{ display: flex; gap: 16px; flex-wrap: wrap; }}
 .kpi {{ border: 1px solid #ddd; border-radius: 8px; padding: 12px 16px; min-width: 180px; }}
 .kpi b {{ display: block; font-size: 26px; }}
 .nota {{ color: #666; font-size: 12px; margin-top: 32px; }}
</style></head><body>
<h1>Reporte diario de seguridad</h1>
<div class="sub">DETECTEPP · {fecha} · generado {datetime.datetime.now().strftime("%H:%M")}</div>
<h2>Indicadores</h2>
<div class="kpis">
 <div class="kpi"><b>{len(con_entrada)}</b>empleados con entrada</div>
 <div class="kpi"><b>{pct}</b>sin alerta de EPP</div>
 <div class="kpi"><b>{len(criticos)}</b>emergencias</div>
 <div class="kpi"><b>{sin_atender}</b>emergencias sin atender</div>
 <div class="kpi"><b>{media}</b>respuesta media a emergencias</div>
</div>
<h2>Asistencia</h2>{sec_asistencia}
<h2>Equipo de protección (EPP)</h2>{sec_epp}
<h2>Emergencias (caídas, persona inmóvil, SOS)</h2>{sec_criticos}
<h2>Ergonomía y zonas restringidas</h2>{sec_otros}
<p class="nota">El monitoreo de posturas no identifica a las personas: sus eventos se agrupan por tipo y cámara.
Los umbrales de postura son orientativos y no reemplazan una evaluación ergonómica profesional.</p>
</body></html>"""
    config.REPORTES_DIR.mkdir(parents=True, exist_ok=True)
    ruta = config.REPORTES_DIR / f"reporte_{fecha}.html"
    ruta.write_text(html_txt, encoding="utf-8")
    return ruta
