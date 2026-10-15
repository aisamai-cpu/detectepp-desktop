"""Reglas de Monitoreo (F1-F9) con esqueletos sintéticos y reloj simulado."""
import unittest
from unittest import mock

import config
from core import postura
from core.riesgos import MonitorRiesgos, ROJO, AMARILLO, VERDE
from core.seguimiento_personas import SeguidorPersonas
from tests.esqueletos import Escena, cuerpo


def de_pie(**kw):
    return lambda k, f: [cuerpo(**kw)]


def caer(cx=640, cy_pie=400, cy_piso=620, T=100):
    """De pie -> tendido en el piso en el tramo (el cuerpo gira 90° y la cadera baja)."""
    return lambda k, f: [cuerpo(cx=cx + 60 * f, cy=cy_pie + (cy_piso - cy_pie) * f, T=T, todo=90 * f)]


def tendido(cx=700, cy=620, T=100, temblor=0.0):
    return lambda k, f: [cuerpo(cx=cx + (temblor * T if k % 2 else 0), cy=cy, T=T, todo=90)]


class BaseMonitor(unittest.TestCase):
    def setUp(self):
        self.m = MonitorRiesgos()
        self.e = Escena(self.m)


class TestIndicadores(unittest.TestCase):
    def ind(self, **kw):
        p = cuerpo(**kw)
        return postura.indicadores(postura.medir(p.puntos, p.caja), 100)

    def test_posturas_basicas(self):
        self.assertFalse(any(vars(self.ind()).values()))
        self.assertTrue(self.ind(todo=90).horizontal)
        i = self.ind(tronco=80)
        self.assertTrue(i.espalda_roja)
        self.assertFalse(i.horizontal)            # agachado sobre sus pies no es "tendido"
        self.assertFalse(i.a_gatas)

    def test_a_gatas_no_es_horizontal_y_si_es_postura_baja(self):
        i = self.ind(tronco=90, piernas="gatas", brazos="piso")
        self.assertTrue(i.a_gatas)
        self.assertFalse(i.horizontal)
        self.assertTrue(i.postura_baja)

    def test_sentado_en_silla_no_es_postura_baja(self):
        self.assertFalse(self.ind(piernas="sentado").postura_baja)
        self.assertTrue(self.ind(piernas="rodillas").postura_baja)
        self.assertTrue(self.ind(piernas="cuclillas").postura_baja)

    def test_codo_y_no_muneca_define_brazo_elevado(self):
        self.assertTrue(self.ind(brazos="arriba").brazo_elevado)
        self.assertFalse(self.ind(brazos="cara").brazo_elevado)

    def test_doblar_rodillas_al_levantar_no_es_espalda_roja(self):
        i = self.ind(tronco=40, piernas="cuclillas")
        self.assertFalse(i.espalda_roja)
        self.assertTrue(i.tronco_amarillo)

    def test_cuello_agachado(self):
        self.assertTrue(self.ind(cabeza="agachada").cuello_agachado)


class TestCaida(BaseMonitor):
    def test_caida_de_lado_genera_un_evento_critico(self):
        self.e.pasar(2, de_pie()).pasar(0.6, caer()).pasar(3, tendido())
        self.assertEqual(self.e.tipos(), ["CAIDA"])
        ev = self.e.eventos[0]
        self.assertEqual((ev.grupo, ev.persona, ev.sonido), ("CRITICO", "P1", True))
        ultimo = self.e.estados[-1][0]
        self.assertEqual((ultimo.semaforo, ultimo.emergencia), (ROJO, "CAÍDA"))

    def test_sentarse_no_es_caida(self):
        sentarse = lambda k, f: [cuerpo(cy=400 + 80 * f, piernas="sentado" if f > 0.5 else "rectas")]
        self.e.pasar(2, de_pie()).pasar(0.6, sentarse).pasar(4, de_pie(cy=480, piernas="sentado"))
        self.assertNotIn("CAIDA", self.e.tipos())

    def test_ponerse_a_gatas_a_fregar_no_es_caida(self):
        agacharse = lambda k, f: [cuerpo(cy=400 + 80 * f, tronco=90 * f, piernas="gatas" if f > 0.3 else "rectas",
                                         brazos="piso" if f > 0.3 else "abajo")]
        self.e.pasar(2, de_pie()).pasar(0.6, agacharse).pasar(15, de_pie(cy=480, tronco=90, piernas="gatas",
                                                                          brazos="piso"))
        self.assertNotIn("CAIDA", self.e.tipos())
        self.assertNotIn("PERSONA_INMOVIL", self.e.tipos())

    def test_agacharse_con_piernas_rectas_no_es_caida(self):
        doblarse = lambda k, f: [cuerpo(tronco=85 * f)]
        self.e.pasar(2, de_pie()).pasar(0.5, doblarse).pasar(3, de_pie(tronco=85))
        self.assertNotIn("CAIDA", self.e.tipos())
        self.assertIn("ESPALDA_INCORRECTA", self.e.tipos())

    def test_lejos_y_pequena_tambien_se_detecta(self):
        self.e.pasar(2, de_pie(T=30, cy=300)).pasar(0.6, caer(cy_pie=300, cy_piso=360, T=30)) \
              .pasar(3, tendido(cy=360, T=30))
        self.assertIn("CAIDA", self.e.tipos())


class TestInmovil(BaseMonitor):
    def test_tendido_inmovil_tras_caida(self):
        self.e.pasar(2, de_pie()).pasar(0.6, caer()).pasar(config.MON_INMOVIL_S + 2, tendido())
        self.assertEqual(self.e.tipos(), ["CAIDA", "PERSONA_INMOVIL"])
        self.assertIn("tras una caída", self.e.eventos[1].detalle)

    def test_desmayo_lento_sin_caida_brusca(self):
        lento = lambda k, f: [cuerpo(cy=400 + 220 * f, todo=90 * f)]
        self.e.pasar(2, de_pie()).pasar(4, lento).pasar(config.MON_INMOVIL_S + 2, tendido())
        self.assertIn("PERSONA_INMOVIL", self.e.tipos())

    def test_tendido_pero_moviendose_no_es_inmovil(self):
        self.e.pasar(config.MON_INMOVIL_S + 3, tendido(temblor=0.4))
        self.assertNotIn("PERSONA_INMOVIL", self.e.tipos())


class TestSOS(BaseMonitor):
    def agitar(self, hz=1.0, amplitud=0.5, saltar=()):
        def gen(k, f):
            if k in saltar:
                return []
            dx = amplitud * (1 if int(self.e.t * hz * 2) % 2 else -1)
            return [cuerpo(brazos="arriba", munecas_dx=dx)]
        return gen

    def test_agitar_brazos_sobre_la_cabeza(self):
        self.e.pasar(1, de_pie()).pasar(4, self.agitar())
        self.assertEqual(self.e.tipos(), ["SOS"])
        self.assertEqual(self.e.estados[-1][0].emergencia, "SOS")

    def test_tolera_una_muestra_perdida(self):
        self.e.pasar(1, de_pie()).pasar(4, self.agitar(saltar=(7,)))
        self.assertIn("SOS", self.e.tipos())

    def test_limpiar_en_alto_sin_agitar_no_es_sos(self):
        self.e.pasar(1, de_pie()).pasar(10, de_pie(brazos="arriba"))
        self.assertNotIn("SOS", self.e.tipos())

    def test_movimiento_pequeno_no_cuenta(self):
        self.e.pasar(1, de_pie()).pasar(5, self.agitar(amplitud=0.1))
        self.assertNotIn("SOS", self.e.tipos())


class TestErgonomia(BaseMonitor):
    def test_brazos_elevados_sostenidos(self):
        self.e.pasar(config.MON_BRAZOS_S + 1, de_pie(brazos="arriba"))
        self.assertEqual(self.e.tipos(), ["BRAZOS_ELEVADOS"])
        self.assertEqual(self.e.estados[-1][0].semaforo, AMARILLO)

    def test_pausa_corta_no_reinicia_y_larga_si(self):
        pausa = lambda dur: (lambda k, f: [cuerpo(brazos="abajo")])
        self.e.pasar(10, de_pie(brazos="arriba")).pasar(0.6, pausa(0.6)).pasar(config.MON_BRAZOS_S - 9,
                                                                                de_pie(brazos="arriba"))
        self.assertIn("BRAZOS_ELEVADOS", self.e.tipos())
        e2 = Escena(MonitorRiesgos())
        e2.pasar(10, de_pie(brazos="arriba")).pasar(3, pausa(3)).pasar(config.MON_BRAZOS_S - 9, de_pie(brazos="arriba"))
        self.assertNotIn("BRAZOS_ELEVADOS", e2.tipos())

    def test_de_rodillas_prolongado(self):
        self.e.pasar(config.MON_RODILLAS_S + 1, de_pie(piernas="rodillas"))
        self.assertIn("RODILLAS_PROLONGADO", self.e.tipos())

    def test_sentado_en_silla_no_alerta_rodillas(self):
        self.e.pasar(config.MON_RODILLAS_S + 5, de_pie(piernas="sentado"))
        self.assertEqual(self.e.tipos(), [])
        self.assertEqual(self.e.estados[-1][0].semaforo, VERDE)

    def test_cuello_sin_sonido_por_defecto(self):
        self.e.pasar(config.MON_CUELLO_S + 1, de_pie(cabeza="agachada"))
        self.assertEqual(self.e.tipos(), ["CUELLO_AGACHADO"])
        self.assertFalse(self.e.eventos[0].sonido)

    def test_flexiones_repetidas(self):
        def flexionar(k, f):
            return [cuerpo(tronco=70 if (k // 5) % 2 else 0, piernas="cuclillas" if (k // 5) % 2 else "rectas")]
        self.e.pasar(60, flexionar)       # 1 flexión cada 2 s = 30 por minuto
        self.assertIn("REPETITIVIDAD", self.e.tipos())

    def test_enfriamiento_por_tipo_aunque_cambie_el_id(self):
        self.e.pasar(config.MON_BRAZOS_S + 1, de_pie(brazos="arriba"))
        self.e.pasar(config.MON_PERDIDA_S + 1, lambda k, f: [])          # sale de cuadro: id nuevo al volver
        self.e.pasar(config.MON_BRAZOS_S + 1, de_pie(brazos="arriba"))
        self.assertEqual(self.e.tipos(), ["BRAZOS_ELEVADOS"])


class TestZonas(BaseMonitor):
    def test_entrar_a_zona_restringida(self):
        self.m.zonas = [("Zona 1", 0.0, 0.5, 0.3, 1.0)]
        self.e.pasar(2, de_pie(cx=900)).pasar(2, de_pie(cx=150))
        self.assertEqual(self.e.tipos(), ["ZONA_RESTRINGIDA"])
        self.assertEqual(self.e.eventos[0].zona, "Zona 1")

    def test_sin_tobillos_no_se_evalua(self):
        self.m.zonas = [("Zona 1", 0.0, 0.0, 1.0, 1.0)]
        self.e.pasar(3, de_pie(ocultar=(15, 16)))
        self.assertEqual(self.e.tipos(), [])


class TestVariasPersonas(BaseMonitor):
    def test_ids_estables_y_eventos_por_persona(self):
        def escena(k, f):
            return [cuerpo(cx=300), cuerpo(cx=1000, brazos="arriba")]
        self.e.pasar(config.MON_BRAZOS_S + 1, escena)
        ids = [{s.id for s in est} for est in self.e.estados]
        self.assertTrue(all(i == {"P1", "P2"} for i in ids))
        self.assertEqual([(e.tipo, e.persona) for e in self.e.eventos], [("BRAZOS_ELEVADOS", "P2")])

    def test_una_cae_y_la_otra_sigue_de_pie(self):
        otra = lambda: cuerpo(cx=200, T=60, cy=350)
        self.e.pasar(2, lambda k, f: [otra(), cuerpo()])
        self.e.pasar(0.6, lambda k, f: [otra()] + caer()(k, f))
        self.e.pasar(3, lambda k, f: [otra()] + tendido()(k, f))
        self.assertEqual([(e.tipo, e.persona) for e in self.e.eventos], [("CAIDA", "P2")])

    def test_seguidor_no_pierde_a_quien_cae(self):
        s = SeguidorPersonas()
        de_pie_ = cuerpo()
        tirado = cuerpo(cx=700, cy=520, todo=90)
        m1, m2 = (postura.medir(p.puntos, p.caja) for p in (de_pie_, tirado))
        self.assertEqual(s.actualizar([(de_pie_.caja, m1.cadera_m, m1.torso)], 0.0), ["P1"])
        self.assertEqual(s.actualizar([(tirado.caja, m2.cadera_m, m2.torso)], 0.5), ["P1"])

    def test_salto_grande_es_otra_persona(self):
        self.e.pasar(1, de_pie(cx=200)).pasar(1, de_pie(cx=1100))
        self.assertEqual({s.id for s in self.e.estados[-1]}, {"P2"})

    def test_reiniciar_olvida_personas_pero_no_enfriamientos(self):
        self.e.pasar(config.MON_BRAZOS_S + 1, de_pie(brazos="arriba"))
        self.m.reiniciar()
        self.e.pasar(config.MON_BRAZOS_S + 1, de_pie(brazos="arriba"))
        self.assertEqual(self.e.tipos(), ["BRAZOS_ELEVADOS"])
        self.assertEqual(self.e.estados[-1][0].id, "P1")


if __name__ == "__main__":
    unittest.main()


class TestFalsasCaidas(BaseMonitor):
    def test_en_cuclillas_inclinado_hacia_adelante_no_es_caida(self):
        bajar = lambda k, f: [cuerpo(cy=400 + 60 * f, tronco=75 * f, piernas="cuclillas" if f > 0.3 else "rectas")]
        self.e.pasar(2, de_pie()).pasar(0.6, bajar).pasar(4, de_pie(cy=460, tronco=75, piernas="cuclillas"))
        self.assertNotIn("CAIDA", self.e.tipos())
