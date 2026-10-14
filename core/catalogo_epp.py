"""Catálogo de áreas de trabajo (limpieza) y el EPP que exige cada una.

Basado en práctica habitual de seguridad industrial para servicios de limpieza y
fabricación de productos de limpieza. Antes de usarlo en una empresa real, contrastarlo
con las hojas de seguridad (SDS) de los químicos y el reglamento de SST vigente.

Se copia a la tabla ``areas_epp`` en la migración v3; desde ahí se puede editar sin tocar código.
No importa PyQt.
"""

# Nombres canónicos de las prendas (los que usa la base de datos y la interfaz).
PRENDAS = (
    "casco", "chaleco", "lentes", "careta", "mascarilla", "respirador", "guantes",
    "botas", "mandil", "bata", "traje", "cofia", "orejeras", "arnés",
)

# Prenda exigida -> prendas que también la cumplen (protegen igual o más).
# La careta NO reemplaza a los lentes: en químicos se usan ambos.
EQUIVALENCIAS = {
    "mascarilla": ("respirador",),
    "bata": ("traje",),
    "mandil": ("traje",),
}

# (área, EPP exigido), en el orden en que se muestran.
AREAS = (
    # --- Servicios de limpieza ---
    ("Limpieza general / superficial", ["guantes", "mascarilla"]),
    ("Limpieza de baños y sanitarios", ["guantes", "mascarilla", "lentes", "mandil", "botas"]),
    ("Desinfección con químicos", ["guantes", "respirador", "lentes", "mandil", "botas"]),
    ("Desinfección de área tóxica / riesgo biológico", ["traje", "respirador", "careta", "guantes", "botas", "cofia"]),
    ("Limpieza en áreas de maquinaria", ["casco", "lentes", "guantes", "botas", "chaleco", "orejeras"]),
    ("Limpieza en altura", ["casco", "arnés", "guantes", "lentes", "botas"]),
    ("Limpieza de exteriores / parqueaderos", ["chaleco", "guantes", "botas"]),
    ("Recolección y manejo de residuos", ["guantes", "mascarilla", "chaleco", "botas", "lentes"]),
    # --- Fábrica de productos de limpieza ---
    ("Mezcla y dosificación de químicos", ["respirador", "careta", "guantes", "mandil", "botas", "cofia"]),
    ("Envasado / llenado", ["guantes", "mascarilla", "cofia", "bata", "lentes"]),
    ("Laboratorio de control de calidad", ["bata", "lentes", "guantes", "mascarilla"]),
    ("Bodega y despacho", ["casco", "chaleco", "botas", "guantes"]),
    ("Mantenimiento de planta", ["casco", "lentes", "guantes", "botas", "orejeras"]),
)

# Áreas de prueba de las primeras versiones: se eliminan en la migración v3.
AREAS_DE_PRUEBA = ("Área Química", "Área Maquinaria")
