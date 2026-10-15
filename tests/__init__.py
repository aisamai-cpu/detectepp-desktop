import os

# Las pruebas de interfaz usan Qt sin ventana ("offscreen"); en Windows ese modo no encuentra las fuentes
# por sí solo y muestra un aviso: se le indican las de Windows.
if os.name == "nt":
    os.environ.setdefault("QT_QPA_FONTDIR", os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"))
