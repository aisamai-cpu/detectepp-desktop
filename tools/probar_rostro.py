"""Prueba interactiva del motor de rostros con tu cámara.

    python -m tools.probar_rostro            (cámara 0)
    python -m tools.probar_rostro --camara 1 (otra cámara, p. ej. la USB externa)

Teclas (con la ventana de video seleccionada):
    G  = guardar el rostro actual como REFERENCIA
    R  = borrar la referencia
    Q o ESC = salir

Sirve para dos cosas:
  1. Comprobar que se detecta tu rostro y cuánto tarda (ms por cuadro, en TU CPU).
  2. Calibrar el umbral: guarda tu rostro (G), mira la similitud que da contigo
     mismo con otra luz/ángulo, y luego pide a OTRA persona que se ponga frente a
     la cámara. La similitud con tu propio rostro debe quedar claramente por
     encima del umbral, y con otra persona claramente por debajo.

(Los textos en pantalla van sin tildes porque cv2.putText no las dibuja.)
"""
import argparse
import os
import sys
import time

import cv2

import config
from vision.face_engine import FaceEngine, ModeloNoEncontrado

VERDE, ROJO, NARANJA, BLANCO = (0, 220, 0), (0, 0, 255), (0, 165, 255), (255, 255, 255)


def texto(img, msg, y, color=BLANCO):
    cv2.putText(img, msg, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
    cv2.putText(img, msg, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camara", type=int, default=0)
    args = ap.parse_args()

    try:
        motor = FaceEngine()
    except ModeloNoEncontrado as e:
        print(e)
        return 1

    cap = cv2.VideoCapture(args.camara, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(args.camara)
    if not cap.isOpened():
        print(f"No se pudo abrir la camara {args.camara}. Prueba con --camara 1")
        return 1

    referencia = None
    print(__doc__.split("Sirve para")[0])
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        frame = cv2.flip(frame, 1)

        t0 = time.perf_counter()
        rostro = motor.principal(frame)
        vector = motor.embedding(frame, rostro) if rostro else None
        ms = (time.perf_counter() - t0) * 1000

        if rostro:
            calidad_ok, aviso = motor.evaluar_calidad(frame, rostro)
            color = VERDE if calidad_ok else NARANJA
            x, y, w, h = rostro.caja
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            texto(frame, f"rostro {w}x{h}px  confianza {rostro.score:.2f}  {ms:.0f} ms", 25)
            if not calidad_ok:
                texto(frame, aviso, 50, NARANJA)
            if referencia is not None:
                sim = float(referencia @ vector)
                misma = sim >= config.FACE_UMBRAL_COSENO
                texto(frame, f"similitud con referencia: {sim:+.2f} (umbral {config.FACE_UMBRAL_COSENO:.2f})", 75)
                texto(frame, "MISMA PERSONA" if misma else "PERSONA DIFERENTE", 100, VERDE if misma else ROJO)
        else:
            texto(frame, "Sin rostro", 25, NARANJA)

        texto(frame, "G=guardar referencia  R=borrar  Q=salir" + ("   [referencia guardada]" if referencia is not None else ""),
              frame.shape[0] - 12)
        cv2.imshow("Prueba de rostro - DETECTEPP", frame)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla in (ord("q"), 27):
            break
        if tecla == ord("g"):
            if vector is not None:
                referencia = vector
                print("Referencia guardada.")
            else:
                print("No hay rostro para guardar.")
        if tecla == ord("r"):
            referencia = None
            print("Referencia borrada.")
        if cv2.getWindowProperty("Prueba de rostro - DETECTEPP", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
