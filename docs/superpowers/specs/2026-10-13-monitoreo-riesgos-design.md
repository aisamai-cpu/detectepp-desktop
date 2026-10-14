# Diseño · Pestaña "Monitoreo de Riesgos y SOS" (DETECTEPP)

Fecha: 2026-10-13 · Estado: **para aprobación del usuario** (revisado por un constructor; ver §14)

---

## 1. Intención

DETECTEPP es una demo de sistema de seguridad para una empresa de **servicios de limpieza y fabricación de
productos de limpieza**. La usa un **supervisor**. Hoy tiene dos pestañas funcionando:

- **Biometría**: reconoce al empleado y registra su hora de entrada/salida.
- **EPP**: comprueba con YOLO que lleve el equipo de protección que exige su área de trabajo y lanza alerta.

La tercera pestaña, **Monitoreo**, debe vigilar **accidentes, posturas y movimientos** de los trabajadores
mientras trabajan, y avisar al supervisor con **la misma alarma sonora** que ya usan Biometría y EPP.

Por qué importa en este sector:

- **Caídas** por pisos mojados: el accidente más común en limpieza.
- **Desmayos** por inhalar mezclas de químicos (cloro + amoníaco, etc.): no siempre empiezan con una caída brusca.
- **Lesiones musculoesqueléticas** (espalda, hombros, rodillas) por levantar baldes y galones, limpiar en alto,
  trabajar de rodillas y repetir el mismo gesto horas seguidas.

**Decisión del usuario:** la pestaña Monitoreo **no** detecta EPP. Solo posturas, movimientos y accidentes.

---

## 2. Objetivos, no-objetivos y anti-objetivos

### Objetivos (las 13 funciones)

| # | Función | Grupo | Alarma sonora |
|---|---|---|---|
| F1 | Caída | Emergencia | Sí, repetida hasta "Atendido" |
| F2 | Persona tendida / inmóvil | Emergencia | Sí, repetida hasta "Atendido" |
| F3 | Gesto de auxilio (SOS) | Emergencia | Sí, repetida hasta "Atendido" |
| F4 | Levantamiento con la espalda doblada y piernas rectas | Ergonomía | Sí, una vez |
| F5 | Brazos sobre los hombros prolongado | Ergonomía | Sí, una vez |
| F6 | De rodillas / en cuclillas prolongado | Ergonomía | Sí, una vez |
| F7 | Cuello agachado prolongado | Ergonomía | **No** por defecto (poco fiable, ver §5.8) |
| F8 | Movimientos repetitivos de tronco | Ergonomía (movimiento) | Sí, una vez |
| F9 | Zona restringida | Contexto | Sí, una vez |
| F10 | Registro de eventos con foto | Gestión | — |
| F11 | Botón "Atendido" y tiempo de respuesta | Gestión | — |
| F12 | Reporte diario | Gestión | — |
| F13 | Privacidad (sin identificar, rostros difuminados, retención) | Gestión / legal | — |

> En la lista conversada, el punto 8 era "EPP durante el trabajo". El usuario lo descartó para esta pestaña.
> En su lugar entra **F8 · Movimientos repetitivos**, que cubre el "movimientos" que pidió. *(Decisión de Claude.)*

### No-objetivos

- Detectar EPP en Monitoreo.
- Identificar a la persona (no se usa reconocimiento facial en esta pestaña).
- Fuego, humo, derrames o piso mojado (requieren entrenar otros modelos: "trabajo futuro").
- Torsión de tronco y peso de la carga (una cámara 2D no los mide de forma fiable).
- Varias cámaras analizadas a la vez (la app trabaja con **una cámara activa**, como hoy).
- Evaluación ergonómica formal (REBA/RULA completos). Es una **aproximación** inspirada en ellos.

### Anti-objetivos (sería un fracaso aunque "funcione")

- **Fatiga de alarmas**: que suene a cada rato por falsos positivos. Todo evento exige que la condición se
  **mantenga** un tiempo y tiene **enfriamiento** por tipo y cámara.
- **Congelar la ventana** (ya pasó con torch en EPP). La IA corre en el hilo de análisis con ONNX.
- Guardar imágenes identificables sin necesidad.
- Que un fallo en Monitoreo cierre la app.

### Criterio de aceptación (para dar por calibrado)

Con los videos de prueba que graba el usuario (§11):

- **≥ 4 de 5** caídas simuladas detectadas (cámara de lado o en diagonal).
- **≥ 4 de 5** gestos SOS detectados.
- **0 emergencias falsas** en 10 minutos de trabajo normal grabado (fregar de rodillas, trapear, agacharse,
  sentarse, limpiar en alto).
- Eventos de ergonomía: cada clip "mal hecho" lo detecta; cada clip "bien hecho" no.

---

## 3. Restricciones

- **Stack actual:** Python 3.12, PyQt6, OpenCV 4.14, numpy, onnxruntime; Windows; **solo CPU**.
- **Reglas del repo** (CLAUDE.md): `core/` y `vision/` no importan PyQt; configuración como `config.NOMBRE`;
  correr `python -m unittest discover -s tests -t .` tras cada cambio; texto con tildes sobre imagen con Qt;
  guardar imágenes con `cv2.imencode(...)[1].tofile(ruta)`.
- **Modelo de pose:** `models/yolo11n-pose.onnx` (ya exportado y en la carpeta; 17 puntos COCO), ejecutado con
  `vision/yolo_onnx.ModeloYoloOnnx` (ya existe; ≈ 70–100 ms por imagen con onnxruntime en un servidor; **hay que
  medirlo en la laptop**).
- **Cámaras:** la activa (USB, celular por IP, archivo de video o RTSP). Giro 0/90/180/270 ya existe.
  Las cámaras USB se muestran en espejo: no se etiqueta nada como "izquierda/derecha".
- **Alarma:** `core.alerts.AlertManager.trigger_alert(tipo, area, detalle, sonido=True)` (BD + log + sonido).
  `core.sound.play_alert()` no superpone sonidos.

### Restricciones de cámara (se muestran como ayuda en la pestaña)

- Caída, inmóvil, espalda, rodillas y repetitividad necesitan ver **el cuerpo completo** y la cámara
  **de lado o en diagonal**. La webcam de la laptop solo ve medio cuerpo. En la demo, usar el **celular como
  cámara IP a 2–4 m, de lado o en diagonal**, o una webcam USB lejos.
- **Limitación conocida:** una caída **hacia la cámara o alejándose de ella** casi no cambia la silueta en 2D y
  puede no detectarse.
- **Limitación conocida:** una persona ya tendida que agita los brazos desde el piso no cumple el gesto SOS
  (las manos no quedan "sobre la cabeza" en la imagen); ese caso lo cubre F2 (inmóvil) o F1.

---

## 4. Arquitectura

```
Cámara ─► CameraWorker (hilo cámara) ──frame + último ResultadoPose──► UI (video, esqueletos, panel)
                │
                └─► hilo de análisis (cada MON_INTERVALO_S):
                      PoseEngine.detectar(frame)          → personas (caja + 17 puntos)
                      MonitorRiesgos.procesar(personas, t) → estados + eventos nuevos
                      ResultadoPose(estados, frecuencia, generación)
                      por cada evento nuevo: señal evento_riesgo(evento, copia del frame analizado)
                                                     │  (señal aparte: nunca se pierde por control de flujo)
                                                     ▼
                                        UI → GestorEventos: foto difuminada + BD + AlertManager
```

**Dónde corren las reglas:** dentro del **hilo de análisis** del worker, con el tiempo de captura de cada
análisis. Así cada análisis se procesa una sola vez (no se repite ni se pierde con los frames descartados) y la
foto del evento es **el frame analizado**, no el mostrado.

### Módulos nuevos

| Archivo | Responsabilidad |
|---|---|
| `vision/pose_engine.py` | Carga `config.POSE_MODELO_PATH` con `ModeloYoloOnnx`. Verifica `tarea == "pose"` y `kpt_shape == (17, 3)` (error claro si no). `detectar(frame) -> List[PersonaPose]` (caja xyxy, confianza, puntos 17×3 en píxeles del frame). Error claro si falta el modelo (como `EppEngine`). |
| `core/postura.py` | Geometría pura: puntos medios, ángulos, brazo elevado, escala de referencia, horizontalidad. Funciones puras. |
| `core/seguimiento_personas.py` | Seguidor: empareja por **distancia del centro de cadera (en torsos) + IoU baja**; conserva una persona perdida `MON_PERDIDA_S` (3 s; 5 s si estaba horizontal o en emergencia). Ids P1, P2… |
| `core/riesgos.py` | `MonitorRiesgos.procesar(personas, t) -> (estados, eventos)`. Máquinas de estado por regla y persona. Sin PyQt ni BD. `reiniciar()` borra todo el estado. |
| `core/zonas.py` | Guardar/leer zonas por cámara en `data/zonas.json` (coordenadas normalizadas 0–1). Punto-en-rectángulo. |
| `core/eventos.py` | `GestorEventos`: foto con cabezas difuminadas, registro en BD, `AlertManager`, "Atendido", eventos abiertos al iniciar, limpieza por retención. |
| `core/reporte.py` | Reporte diario en HTML. |
| `tools/probar_monitoreo.py` | Pasa un video por pose + monitor y muestra eventos y medidas por segundo; compara con un archivo de eventos esperados. |

### Cambios en archivos existentes

- `camera_worker.py`
  - Nuevo `MODO_MONITOREO = "MONITOREO"` (hoy la pestaña usa el texto `"ACCIDENTES"` en `main.py` y
    `tests/test_epp.py`; se reemplaza en ambos).
  - El filtro de tipo de resultado pasa a un **mapa modo → tipo esperado** (BIOMETRÍA → `Resultado`,
    EPP → `list`, MONITOREO → `ResultadoPose`) en `_paso` y en `main.on_frame`.
  - Intervalo `MON_INTERVALO_S`, `_puede_analizar` y carga perezosa del `PoseEngine` (señal `error_pose`).
  - **Girar la cámara** sube `_generacion` y reinicia el monitor (hoy solo el cambio de cámara lo hace).
  - Cambio de generación (cambio de cámara, giro, reconexión, **vuelta del video en bucle**) → `MonitorRiesgos.reiniciar()`.
    `VideoSource` expone un contador de vueltas del archivo para detectarlo.
  - Nueva señal `evento_riesgo(object, object)`.
- `main.py`: UI de la pestaña (§6), sonido repetido, borrar zonas **antes** de `worker.rotar` en `on_girar_camara`.
- `core/database.py`: migración **v4** (§8); `borrar_registros` borra `eventos_riesgo` **antes** que `alertas`.
- `tools/reiniciar_datos.py`: respalda la carpeta `data/eventos/` completa (con subcarpetas) y `data/zonas.json` no se toca.
- `config.py`: umbrales `MON_*` (§5.13).
- Texto de `PrivacyWidget` (§9).

### Rendimiento

- Pose cada `MON_INTERVALO_S = 0.20` s (objetivo ≈ 5 análisis/s). La **frecuencia real** se mide y viaja en
  `ResultadoPose`; se muestra en el panel.
- Cada regla exige un **mínimo de muestras** en su ventana. Si la frecuencia baja de `MON_FRECUENCIA_MIN`
  (3 análisis/s), el panel avisa "Análisis lento: SOS y caídas pueden fallar".
- Si la laptop no llega, se exporta un segundo modelo `yolo11n-pose-480.onnx` y se cambia `POSE_MODELO_PATH`
  (sin tocar código).

---

## 5. Cómo se detecta cada función

### 5.0 Medidas base (por persona y por análisis)

Puntos COCO: 0 nariz, 1–2 ojos, 3–4 orejas, 5–6 hombros, 7–8 codos, 9–10 muñecas, 11–12 caderas,
13–14 rodillas, 15–16 tobillos. Un punto **cuenta** solo si su confianza ≥ `MON_PUNTO_CONF_MIN` (0.5).

| Medida | Cálculo |
|---|---|
| `hombro_m`, `cadera_m` | punto medio de hombros / caderas (si falta uno, el visible) |
| `torso_ref` (escala) | distancia `hombro_m`–`cadera_m` **de referencia**: la mediana de los últimos 3 s mientras la persona estaba erguida (`ang_tronco < 30°`). Se **congela** al detectar una bajada rápida (F1) o al quedar horizontal. Si nunca se la vio erguida: el torso actual. Todas las distancias van en `torso_ref`. |
| `ang_tronco` | ángulo de `cadera_m → hombro_m` respecto a la vertical (0° erguido, 90° horizontal) |
| `ang_rodilla` | ángulo cadera–rodilla–tobillo de cada rodilla **visible** (180° = recta) |
| `codo_arriba` | el codo está más alto que su hombro (brazo elevado más de ≈ 90°) |
| `manos_sobre_cabeza` | **alguna** muñeca más alta que la nariz (u orejas) por más de `0.20 × torso_ref` |
| `caderas_bajas` | distancia vertical cadera–tobillo (o cadera–rodilla si no hay tobillo) < `MON_CADERA_BAJA_TORSOS` (0.5) |
| `horizontal` | `ang_tronco ≥ 60°` **y** caja ancho/alto ≥ `MON_RELACION_HORIZONTAL` (1.0) **y** no hay tobillos visibles muy por debajo de la cadera (distancia vertical > `0.8 × torso_ref`). Así *agacharse con piernas rectas* no es "horizontal". |
| `a_gatas` | `ang_tronco ≥ 60°` **y** alguna rodilla < 120° **y** rodilla(s) a la altura del piso (debajo de la cadera). Fregar el piso de rodillas/a gatas: **no** es caída. |
| `movimiento` | **mediana** del desplazamiento de los puntos visibles respecto al análisis anterior, en `torso_ref`, suavizada sobre 1 s |
| `pie` | punto medio de los tobillos visibles (sin tobillos: no se evalúa zona) |

#### Puntos que necesita cada regla

| Regla | Puntos mínimos | Si faltan |
|---|---|---|
| F1 caída, F2 inmóvil | hombros + caderas | no se evalúa |
| F3 SOS | nariz u orejas + una muñeca | no se evalúa |
| F4 espalda | hombros + caderas + ≥ 1 rodilla + su tobillo | no se evalúa |
| F5 brazos | hombro + codo del mismo lado | no se evalúa ese brazo |
| F6 rodillas | caderas + ≥ 1 rodilla (+ tobillo o caderas bajas) | no se evalúa |
| F7 cuello | hombros + oreja o nariz + caderas | no se evalúa |
| F8 repetitivo | hombros + caderas | no se evalúa |
| F9 zona | ≥ 1 tobillo | no se evalúa |

El panel muestra "P2 · cuerpo incompleto: no se evalúan piernas" cuando faltan rodillas/tobillos.

### 5.1 F1 · Caída

1. **Bajada rápida:** en la ventana de `MON_CAIDA_VENTANA_S` (1.0 s), `hombro_m` baja más de
   `MON_CAIDA_BAJADA_TORSOS` (1.0 `torso_ref`) **o** `ang_tronco` pasa de < 30° a > 60°.
2. **Confirmación:** dentro de los `MON_CAIDA_CONFIRMAR_S` (2.0 s) siguientes, la persona queda `horizontal`
   (y **no** `a_gatas`) durante `MON_CAIDA_HORIZONTAL_S` (1.0 s) seguidos.
3. → Evento **CAÍDA** (crítico).

No son caída: sentarse (no queda horizontal), agacharse con piernas rectas (tobillos muy por debajo de la
cadera), ponerse a gatas a fregar (`a_gatas`).

### 5.2 F2 · Persona tendida / inmóvil

- Persona `horizontal` y no `a_gatas` durante `MON_INMOVIL_S` (10 s demo; recomendado 30 s) con `movimiento`
  < `MON_INMOVIL_MOV_TORSOS` (0.10; se calibra con la herramienta) → evento **PERSONA_INMOVIL** (crítico).
- Si hay una **CAÍDA abierta** (sin "Atendido") de la misma persona: se crea la fila de PERSONA_INMOVIL
  enlazada (`relacionado_id`) y **sin sonido extra** (ya está sonando). Si la caída ya fue atendida y la
  persona sigue tendida e inmóvil: es un **evento crítico nuevo**.
- Cubre el desmayo lento (sin bajada brusca). Tras el enfriamiento vuelve a dispararse si sigue igual.

### 5.3 F3 · Gesto de auxilio (SOS)

- `manos_sobre_cabeza` y la muñeca (cualquiera de las dos) **se agita de lado a lado**: se cuentan cambios de
  dirección horizontal **con histéresis** (solo cuenta un cambio si el recorrido desde el anterior es
  ≥ `MON_SOS_AMPLITUD_TORSOS` = 0.25), tolerando **una** muestra perdida.
- ≥ `MON_SOS_CAMBIOS` (4) cambios dentro de `MON_SOS_S` (4 s) → evento **SOS** (crítico).
- Por qué agitar y no solo "manos arriba": limpiar vidrios o techos también levanta los brazos (eso es F5).
- Requiere ≥ 3 análisis/s; si la frecuencia es menor, el panel avisa (§4).

### 5.4 F4 · Levantamiento con la espalda doblada y piernas rectas

- **Rojo:** `ang_tronco > MON_TRONCO_ROJO` (60°) **y** todas las rodillas **visibles** > `MON_RODILLA_RECTA`
  (150°) **y** no `horizontal`.
- **Amarillo:** `ang_tronco` entre `MON_TRONCO_AMARILLO` (20°) y 60°.
- Evento **ESPALDA_INCORRECTA** si el rojo se mantiene `MON_ESPALDA_S` (**1.5 s**): un levantamiento mal hecho
  dura 1–2 s.
- Doblar las rodillas (técnica correcta) no genera evento.
- Necesita cámara de lado o diagonal: de frente el tronco inclinado casi no cambia de ángulo en 2D.

### 5.5 F5 · Brazos sobre los hombros prolongado

- Condición: `codo_arriba` en **al menos un** brazo (el codo, no la muñeca: sostener algo a la altura de la
  cara no cuenta).
- Amarillo apenas ocurre; evento **BRAZOS_ELEVADOS** si se mantiene `MON_BRAZOS_S` (20 s demo; recomendado
  60 s). Tolerancia: puede bajar el brazo hasta `MON_TOLERANCIA_S` (1 s) sin reiniciar el conteo.
- Si se cumple F3 (SOS), gana F3 y F5 se reinicia.

### 5.6 F6 · De rodillas / en cuclillas prolongado

- Condición: alguna rodilla < `MON_RODILLA_FLEXION` (**115°**) **y** `caderas_bajas` **y** no `horizontal`.
  `caderas_bajas` excluye a quien está **sentado en una silla** (sentado, la cadera queda a la altura de la
  rodilla, lejos del piso).
- Con el tobillo oculto (pasa al arrodillarse) basta cadera + rodilla.
- Amarillo apenas ocurre; evento **RODILLAS_PROLONGADO** a los `MON_RODILLAS_S` (30 s demo; recomendado 120 s),
  con tolerancia de 1 s. Fregar a gatas cuenta aquí (es justo el riesgo de rodillas).

### 5.7 F7 · Cuello agachado prolongado

- Ángulo entre `hombro_m → oreja_m` (o nariz) y la línea del tronco > `MON_CUELLO_ANG` (30°), sostenido
  `MON_CUELLO_S` (30 s demo; recomendado 60 s).
- **Limitación:** solo es fiable con la cámara **de lado**. Por eso: semáforo amarillo + evento registrado,
  **sin sonido por defecto** (`MON_CUELLO_SONIDO = False`).

### 5.8 F8 · Movimientos repetitivos de tronco

- Una **flexión** = `ang_tronco` sube de < 20° a > 45° y vuelve a < 20°.
- **Ventana deslizante** de `MON_REPETICION_VENTANA_MIN` minutos (1 demo; 5 recomendado): evento
  **REPETITIVIDAD** si hay ≥ `MON_REPETICIONES_MIN` (10) × minutos de la ventana flexiones dentro de ella.
  Tras el evento, el conteo vuelve a cero.
- Se cuenta por persona seguida; si el seguidor la pierde (sale de cuadro), su conteo se pierde. Con una sola
  persona en cámara (caso de la demo) no hay problema.

### 5.9 F9 · Zona restringida

- El supervisor dibuja **zonas rectangulares** sobre el video (botón "Dibujar zona": clic y arrastre; Esc
  cancela). Las coordenadas del ratón se convierten a coordenadas del frame **descontando las franjas negras**
  del video escalado (`KeepAspectRatio`).
- Se guardan por cámara en `data/zonas.json`, normalizadas (0–1) sobre la imagen ya girada.
- Al **girar la cámara**, la UI borra sus zonas **antes** de girar y avisa ("Las zonas se borraron porque giraste
  la cámara").
- Evento **ZONA_RESTRINGIDA** cuando el `pie` de una persona está dentro de una zona `MON_ZONA_S` (1 s).

### 5.10 Semáforo por persona

- **Rojo:** F4 rojo, o una emergencia F1–F3 activa.
- **Amarillo:** tronco 20–60°, brazo elevado, rodillas flexionadas, cuello agachado.
- **Verde:** ninguna.
- Por persona: segundos acumulados en amarillo y en rojo desde que apareció.

### 5.11 Reglas comunes a todos los eventos

- **Enfriamiento** por **(tipo, cámara)** — y por zona en F9 —, no por persona: así un id nuevo tras perder de
  vista a alguien no repite la alarma. `MON_ENFRIAMIENTO_S` (60 s demo; 300 s recomendado).
  Excepción: un evento crítico nuevo (caída de otra persona) siempre se registra.
- Emergencias (F1–F3): el sonido se **repite cada** `MON_REPETIR_SONIDO_S` (15 s) hasta "Atendido".
  Ergonomía y zona: suenan **una vez**.
- Un salto del centro de cadera > 3 `torso_ref` entre dos análisis se trata como **otra persona**, no como
  movimiento (evita caídas falsas por cortes de video).

### 5.12 Qué pasa al cambiar de pestaña, cámara o reiniciar

Ver la pregunta abierta en §14 (si el monitoreo sigue analizando fuera de su pestaña). En todos los casos:

- Las **emergencias abiertas** (sin "Atendido") siguen visibles y sonando en cualquier pestaña: tarjeta roja
  en el panel de Monitoreo y **aviso en la barra superior** ("⚠ 1 emergencia sin atender") en todas las pestañas.
- Al iniciar la app, las emergencias abiertas de las últimas 24 h se recargan desde la BD (sin volver a sonar;
  aviso en la barra).
- Cambio de cámara, giro, reconexión o vuelta del video → `MonitorRiesgos.reiniciar()`.

### 5.13 Tabla de umbrales (`config.py`)

| Nombre | Demo | Recomendado real | Uso |
|---|---|---|---|
| `POSE_MODELO_PATH` | `models/yolo11n-pose.onnx` | — | modelo |
| `MON_INTERVALO_S` | 0.20 | 0.20 | segundos entre análisis |
| `MON_FRECUENCIA_MIN` | 3.0 | 3.0 | análisis/s mínimos antes de avisar |
| `MON_CONF_PERSONA` | 0.40 | 0.40 | confianza mínima de persona |
| `MON_PUNTO_CONF_MIN` | 0.50 | 0.50 | confianza mínima de un punto |
| `MON_PERDIDA_S` / `MON_PERDIDA_HORIZONTAL_S` | 3 / 5 | 3 / 5 | olvidar persona no vista |
| `MON_RELACION_HORIZONTAL` | 1.0 | 1.0 | ancho/alto de caja para "horizontal" |
| `MON_CADERA_BAJA_TORSOS` | 0.5 | 0.5 | `caderas_bajas` |
| `MON_CAIDA_VENTANA_S` | 1.0 | 1.0 | F1 |
| `MON_CAIDA_BAJADA_TORSOS` | 1.0 | 1.0 | F1 |
| `MON_CAIDA_CONFIRMAR_S` | 2.0 | 2.0 | F1 |
| `MON_CAIDA_HORIZONTAL_S` | 1.0 | 1.0 | F1 |
| `MON_INMOVIL_S` | 10 | 30 | F2 |
| `MON_INMOVIL_MOV_TORSOS` | 0.10 | calibrar | F2 |
| `MON_SOS_S` / `MON_SOS_CAMBIOS` / `MON_SOS_AMPLITUD_TORSOS` | 4 s / 4 / 0.25 | igual | F3 |
| `MON_TRONCO_AMARILLO` / `_ROJO` | 20° / 60° | 20° / 60° | F4, semáforo |
| `MON_RODILLA_RECTA` | 150° | 150° | F4 |
| `MON_ESPALDA_S` | 1.5 | 1.5 | F4 |
| `MON_BRAZOS_S` | 20 | 60 | F5 |
| `MON_RODILLA_FLEXION` | 115° | 115° | F6 |
| `MON_RODILLAS_S` | 30 | 120 | F6 |
| `MON_CUELLO_ANG` / `MON_CUELLO_S` | 30° / 30 s | 30° / 60 s | F7 |
| `MON_CUELLO_SONIDO` | False | False | F7 |
| `MON_REPETICIONES_MIN` | 10 | 10 | F8 (por minuto) |
| `MON_REPETICION_VENTANA_MIN` | 1 | 5 | F8 |
| `MON_ZONA_S` | 1.0 | 1.0 | F9 |
| `MON_TOLERANCIA_S` | 1.0 | 1.0 | F5, F6 |
| `MON_ENFRIAMIENTO_S` | 60 | 300 | todos |
| `MON_REPETIR_SONIDO_S` | 15 | 15 | F1–F3 |
| `MON_RETENCION_DIAS` | 30 | 30 | F13 |
| `MON_DIFUMINAR_ROSTROS` | True | True | F13 |

Los valores "Demo" quedan en `config.py`; los "recomendados" como comentario. Los recomendados son
**orientativos** (inspirados en criterios de posturas forzadas como OWAS/REBA e ISO 11226), no una evaluación
ergonómica formal.

---

## 6. Pantalla

Mismo esquema que EPP: video 65 % a la izquierda, panel 35 % a la derecha, mismos colores oscuros.

### Video (izquierda)

- Esqueleto de cada persona **del color de su semáforo** (verde `#00e676`, amarillo `#ffb74d`, rojo `#ff1744`),
  dibujado con Qt.
- Etiqueta sobre la caja: `P1 · Espalda 1 s` (la condición más grave activa y su tiempo).
- Persona en emergencia: caja roja gruesa parpadeante + `CAÍDA` / `INMÓVIL` / `SOS`.
- Zonas: rectángulos rojos semitransparentes con "Zona restringida".
- Modo "Dibujar zona": cursor en cruz; clic y arrastre; Esc cancela.

### Panel (derecha), de arriba abajo

1. **Título** "MONITOREO DE RIESGOS Y SOS" y estado: `● Activo · 2 personas · 5 análisis/s`, o el aviso que
   corresponda (modelo no disponible · análisis lento · sin personas).
2. **Tarjeta de emergencia** (si hay alguna abierta): fondo rojo, `CAÍDA – P1`, hora, segundos transcurridos,
   miniatura de la foto y botón **"Atendido"**. Varias: se apilan (la más antigua arriba).
3. **Personas en cámara:** una fila por persona: punto de color, `P1`, condición activa, tiempo acumulado
   (`amarillo 1 min 20 s · rojo 12 s`), o `cuerpo incompleto: no se evalúan piernas`.
4. **Eventos recientes:** últimos 20 del día: hora, tipo, descripción corta, estado
   (Atendido en 35 s / Pendiente / Informativo). Doble clic abre la foto; si fue borrada por retención:
   "foto no disponible".
5. **Botones:** `Dibujar zona` · `Borrar zonas` · `Reporte del día`.
6. **Ayuda de cámara** (texto pequeño): "Para caídas y posturas: cámara a 2–4 m, de lado o en diagonal, con el
   cuerpo completo a la vista."

Antes de construir la pantalla se muestra un **mockup** al usuario para aprobarlo (etapa 1).

---

## 7. Alertas

| Tipo (`alertas.tipo`) | Grupo | Sonido | Texto de detalle (ejemplo) |
|---|---|---|---|
| `CAIDA` | crítico | repetido hasta Atendido | "Caída detectada (P1) en Cámara celular" |
| `PERSONA_INMOVIL` | crítico | repetido (sin extra si hay caída abierta) | "Persona tendida e inmóvil 10 s (P1), tras una caída" |
| `SOS` | crítico | repetido | "Gesto de auxilio (P2)" |
| `ESPALDA_INCORRECTA` | ergonomía | una vez | "Levantamiento con espalda doblada y piernas rectas (P1)" |
| `BRAZOS_ELEVADOS` | ergonomía | una vez | "Brazos sobre los hombros 20 s (P1)" |
| `RODILLAS_PROLONGADO` | ergonomía | una vez | "De rodillas / cuclillas 30 s (P1)" |
| `CUELLO_AGACHADO` | ergonomía | no (por defecto) | "Cuello agachado 30 s (P1)" |
| `REPETITIVIDAD` | ergonomía | una vez | "12 flexiones de tronco en 1 min (P1)" |
| `ZONA_RESTRINGIDA` | zona | una vez | "P1 entró a zona restringida 1" |

- Todas pasan por `AlertManager.trigger_alert(tipo, area=<nombre de la cámara>, detalle, sonido=...)`.
  En Monitoreo `alertas.area` guarda el **nombre de la cámara** (en EPP guarda el área de trabajo); el reporte
  lo distingue por tipo.
- La repetición del sonido de emergencias: un `QTimer` de la ventana principal que llama a `play_alert()`
  mientras haya emergencias abiertas, en cualquier pestaña.

---

## 8. Datos

### Tabla nueva (migración v4)

```sql
CREATE TABLE IF NOT EXISTS eventos_riesgo (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tipo TEXT NOT NULL,             -- CAIDA, SOS, ESPALDA_INCORRECTA, ...
    grupo TEXT NOT NULL,            -- CRITICO / ERGONOMIA / ZONA
    timestamp DATETIME NOT NULL,
    camara TEXT NOT NULL,
    persona TEXT NOT NULL,          -- id local de seguimiento (P1, P2…); NO identifica a nadie
    detalle TEXT NOT NULL,
    foto_path TEXT,                 -- NULL si no se pudo guardar o si se borró por retención
    alerta_id INTEGER,              -- fila en la tabla alertas
    relacionado_id INTEGER,         -- p. ej. INMÓVIL -> la CAÍDA que lo precedió
    atendido_en DATETIME,           -- NULL = pendiente (solo críticos)
    FOREIGN KEY(alerta_id) REFERENCES alertas(id),
    FOREIGN KEY(relacionado_id) REFERENCES eventos_riesgo(id)
);
CREATE INDEX IF NOT EXISTS idx_eventos_fecha ON eventos_riesgo(timestamp);
```

- Tiempo de respuesta = `atendido_en − timestamp` (se calcula al mostrar).
- Fotos: `data/eventos/AAAA-MM-DD/<tipo>_<hhmmss>_<P>.jpg` (frame analizado, esqueleto dibujado con líneas cv2,
  sin texto; cabezas difuminadas).
- `borrar_registros()` borra `eventos_riesgo` **antes** que `alertas` (claves foráneas activas).
- `tools/reiniciar_datos.py` mueve la carpeta `data/eventos/` completa al respaldo.

---

## 9. F13 · Privacidad

- Monitoreo **no identifica**: nunca cruza con la galería de rostros; las personas son P1, P2…
- Fotos de eventos: se difumina la **cabeza de todas las personas detectadas** (región alrededor de
  nariz/ojos/orejas, radio ≈ 0.6 `torso_ref`). Si una persona no tiene puntos de cabeza fiables, se difumina el
  **30 % superior de su caja**. Además se pasa el detector de rostros YuNet (ya existe) y se difumina cualquier
  rostro que encuentre, para cubrir a personas que la pose no detectó.
- Solo se guardan fotos **de eventos**, nunca video continuo.
- Al iniciar la app se borran las fotos con más de `MON_RETENCION_DIAS` (30) días y su `foto_path` pasa a NULL.
- Texto de la pantalla de privacidad: se analiza la postura para prevenir accidentes, sin identificar a la
  persona; las fotos de eventos se guardan difuminadas por 30 días. (Marco: LOPDP de Ecuador.)

---

## 10. F12 · Reporte diario

- Botón "Reporte del día" → `data/reportes/reporte_AAAA-MM-DD.html`, abierto en el navegador
  (`QDesktopServices.openUrl`). HTML autocontenido, imprimible a PDF desde el navegador. Sin dependencias nuevas.
- Secciones:
  1. **Asistencia:** entradas/salidas del día por empleado (tabla `asistencia`), con nombre.
  2. **EPP:** alertas `EPP_INCOMPLETO` del día (empleado y prendas faltantes, tomados del texto de `detalle`,
     que tiene formato fijo), y área.
  3. **Emergencias:** cada evento crítico: hora, tipo, cámara, tiempo de respuesta, estado.
  4. **Ergonomía y zonas:** conteo por tipo y por cámara (no por trabajador: Monitoreo no identifica).
  5. **Indicadores:** empleados con entrada registrada y **sin** alerta de EPP / total con entrada; tiempo medio
     de respuesta a emergencias; número de emergencias sin atender.
- El reporte muestra nombres de empleados **solo** en Asistencia y EPP (datos que ya identifican en sus pestañas).
  *(Decisión de Claude.)*

---

## 11. Plan de construcción por etapas

Cada etapa termina con pruebas en verde y algo que el usuario puede probar.

1. **Mockup + pose en pantalla:** mockup aprobado; `PoseEngine`, modo `MONITOREO`, filtro por tipo, esqueletos,
   estado "Activo · N personas · X análisis/s", ayuda de cámara.
2. **Medidas + seguidor + semáforo** y lista de personas.
3. **Emergencias F1–F3** + eventos en BD + foto difuminada + tarjeta roja + "Atendido" + sonido repetido +
   aviso en barra superior + recarga al iniciar (F10, F11).
4. **Ergonomía F4–F8.**
5. **Zonas F9.**
6. **Reporte F12** y resto de **privacidad F13** (retención, texto).
7. **`tools/probar_monitoreo.py`** y **calibración** con los videos del usuario hasta cumplir el criterio de
   aceptación (§2).

### Pruebas

- Reglas con **secuencias sintéticas de puntos** y reloj falso: caída de lado, sentarse, agacharse con piernas
  rectas, a gatas fregando, tendido inmóvil, agitar brazos rápido/lento/con una muestra perdida, brazo con codo
  arriba vs. solo muñeca, sentado en silla vs. en cuclillas, flexiones repetidas, salto de posición por corte de
  video, persona que sale y vuelve a entrar (enfriamiento).
- `PoseEngine` con un ONNX diminuto falso (como `tests/test_yolo_onnx.py`), incluido el error por modelo que no
  es de pose.
- Worker: filtro por modo, reinicio al girar y al cambiar de generación, señal `evento_riesgo` no se pierde.
- BD: migración v3 → v4 sobre una base existente; orden de borrado; respaldo de carpetas; retención.
- Privacidad: difuminado con y sin puntos de cabeza.
- UI (`QT_QPA_PLATFORM=offscreen`): tarjeta, Atendido, aviso en barra en otras pestañas, lista, sin cierres al
  cambiar de pestaña.
- Calibración real: el usuario graba con el celular clips cortos (caída simulada sobre colchoneta, levantar un
  balde bien y mal, limpiar en alto, de rodillas, a gatas fregando, sentarse, agitar brazos, 10 min de trabajo
  normal). Para cada clip se anota qué eventos se esperan (`tests/videos/esperado.json`, los videos no se suben
  a git) y `tools/probar_monitoreo.py` compara.

---

## 12. Decisiones tomadas por Claude (para que el usuario las revise)

1. F8 "Movimientos repetitivos" reemplaza a "EPP durante el trabajo".
2. SOS = agitar los brazos sobre la cabeza (no solo levantarlos).
3. Cuello (F7) sin sonido por defecto.
4. Zonas rectangulares; se borran si se gira la cámara.
5. Emergencias repiten el sonido cada 15 s hasta "Atendido", en cualquier pestaña; ergonomía y zona suenan una vez.
6. Umbrales cortos "de demo" en `config.py`, con los recomendados como comentario.
7. Reporte en HTML (imprimible a PDF); ergonomía por cámara, no por trabajador; nombres solo en Asistencia y EPP.
8. Inmóvil tras caída abierta: fila enlazada sin sonido extra.
9. F4 mide el **levantamiento** (1.5 s), no estar agachado mucho rato.
10. F5 usa el **codo** sobre el hombro, no la muñeca.
11. F6: rodilla < 115° con caderas bajas (excluye estar sentado en una silla).
12. Enfriamiento por tipo y cámara, no por persona.
13. Limitaciones aceptadas: caídas hacia/desde la cámara y SOS desde el piso pueden no detectarse.

## 13. Lo que queda a criterio del constructor

- Nombres internos y reparto exacto en archivos (respetando §4).
- El seguidor puede ser simple (distancia de cadera + IoU); no hace falta un tracker sofisticado.
- Detalles visuales menores siguiendo el estilo de `main.py`.

## 14. Pregunta abierta (la responde el usuario)

**¿El monitoreo debe analizar solo mientras la pestaña Monitoreo está abierta, o siempre (también cuando el
supervisor está en Biometría o EPP)?**

- *Solo en su pestaña* (como funcionan EPP y Biometría hoy): más simple y liviano para la CPU. Si el supervisor
  está registrando entradas, no se detectan caídas en ese momento.
- *Siempre*: más seguro, pero hay que correr dos modelos a la vez (pose + el de la pestaña activa) sobre la misma
  cámara; en la laptop puede bajar la fluidez, y solo tiene sentido si la misma cámara ve la zona de trabajo.
