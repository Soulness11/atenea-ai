"""
═══════════════════════════════════════════════════════════════════════════════
GENERADOR DE DATOS SINTÉTICOS — INSTRUMENTO PSICOEMOCIONAL (165 ítems)
Tesis Doctoral: Modelos Predictivos ML para la Deserción Escolar (UAQ-DTE)
═══════════════════════════════════════════════════════════════════════════════
Ejecutar en Google Colab: https://colab.research.google.com
O localmente con: pip install numpy pandas scikit-learn

Genera datos sintéticos realistas basados en las 6 dimensiones del
instrumento psicométrico de 165 ítems desarrollado en la tesis.
Reemplazar con datos reales cuando estén disponibles.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import os

np.random.seed(42)

# ─── Configuración del dataset ────────────────────────────────────────────────
N_ESTUDIANTES = 500       # Muestra objetivo (aumentar con datos reales)
TASA_DESERCION = 0.28     # ~28% de deserción (referencia UAQ DTE)

# ─── Dimensiones del instrumento (6 dimensiones, 165 ítems en total) ─────────
DIMENSIONES = {
    "bienestar_emocional":      {"items": list(range(1, 29)),    "n": 28, "alpha": 0.87},
    "autopercepcion_academica": {"items": list(range(29, 57)),   "n": 28, "alpha": 0.85},
    "motivacion_compromiso":    {"items": list(range(57, 85)),   "n": 28, "alpha": 0.89},
    "resiliencia_academica":    {"items": list(range(85, 113)),  "n": 28, "alpha": 0.83},
    "relaciones_interpersonales": {"items": list(range(113, 141)), "n": 28, "alpha": 0.81},
    "afiliacion_institucional": {"items": list(range(141, 166)), "n": 25, "alpha": 0.86},
}

def simular_dimension(n, media_alta, media_baja, sd=0.8, desercion=None):
    """
    Simula puntuaciones de una dimensión psicoemocional.
    Estudiantes desertores tienden a puntuar más bajo en todas las dimensiones.
    """
    puntuaciones = np.where(
        desercion == 1,
        np.random.normal(media_baja, sd, n),
        np.random.normal(media_alta, sd, n)
    )
    return np.clip(puntuaciones, 1.0, 5.0).round(2)


def generar_dataset():
    # ── Variable dependiente ────────────────────────────────────────────────
    desercion = np.random.binomial(1, TASA_DESERCION, N_ESTUDIANTES)

    # ── Variables demográficas ───────────────────────────────────────────────
    sexo = np.random.choice([0, 1], N_ESTUDIANTES, p=[0.47, 0.53])  # 0=M, 1=F
    edad = np.random.randint(17, 30, N_ESTUDIANTES)
    trabaja = np.random.choice([0, 1], N_ESTUDIANTES, p=[0.61, 0.39])
    beca = np.random.choice([0, 1], N_ESTUDIANTES, p=[0.55, 0.45])
    nivel_educativo_padres = np.random.choice([1, 2, 3, 4], N_ESTUDIANTES, p=[0.22, 0.35, 0.28, 0.15])
    area = np.random.choice(['ingenieria', 'sociales', 'salud', 'humanidades', 'exactas'],
                            N_ESTUDIANTES, p=[0.28, 0.24, 0.18, 0.15, 0.15])
    semestre = np.random.choice(range(1, 11), N_ESTUDIANTES)

    # ── Variables académicas (correlacionadas con deserción) ────────────────
    promedio_base = np.where(desercion == 1,
                             np.random.normal(6.8, 1.2, N_ESTUDIANTES),
                             np.random.normal(8.3, 0.9, N_ESTUDIANTES))
    promedio = np.clip(promedio_base, 0, 10).round(1)

    materias_reprobadas = np.where(desercion == 1,
                                   np.random.poisson(2.8, N_ESTUDIANTES),
                                   np.random.poisson(0.4, N_ESTUDIANTES))
    materias_reprobadas = np.clip(materias_reprobadas, 0, 12)

    carga_actual = np.where(desercion == 1,
                            np.random.choice([2, 3, 4, 5], N_ESTUDIANTES, p=[0.2, 0.4, 0.3, 0.1]),
                            np.random.choice([4, 5, 6, 7], N_ESTUDIANTES, p=[0.1, 0.3, 0.4, 0.2]))

    # ── Dimensiones psicoemocionales (medias alta vs. baja) ─────────────────
    # Medias: [media_en_no_desertor, media_en_desertor]
    config_dimensiones = {
        "bienestar_emocional":        (3.9, 2.5),
        "autopercepcion_academica":   (4.0, 2.7),
        "motivacion_compromiso":      (4.1, 2.6),
        "resiliencia_academica":      (3.8, 2.4),
        "relaciones_interpersonales": (3.7, 2.9),  # Menor diferencia
        "afiliacion_institucional":   (3.6, 2.3),
    }

    datos_psicoemocionales = {}
    for nombre, (media_alta, media_baja) in config_dimensiones.items():
        datos_psicoemocionales[f"dim_{nombre}"] = simular_dimension(
            N_ESTUDIANTES, media_alta, media_baja, sd=0.75, desercion=desercion
        )

    # ── Construir DataFrame ──────────────────────────────────────────────────
    df = pd.DataFrame({
        "id_participante": [f"UAQ{str(i+1).zfill(4)}" for i in range(N_ESTUDIANTES)],
        "sexo": sexo,
        "edad": edad,
        "semestre": semestre,
        "area": area,
        "trabaja": trabaja,
        "beca": beca,
        "nivel_edu_padres": nivel_educativo_padres,
        "promedio_acumulado": promedio,
        "materias_reprobadas": materias_reprobadas,
        "carga_academica": carga_actual,
        **datos_psicoemocionales,
        "desercion": desercion
    })

    return df


if __name__ == "__main__":
    print("=" * 65)
    print("GENERANDO DATASET SINTÉTICO — INSTRUMENTO PSICOEMOCIONAL UAQ")
    print("=" * 65)

    df = generar_dataset()

    # ── Estadísticas descriptivas básicas ───────────────────────────────────
    print(f"\n📊 Dimensiones del dataset: {df.shape[0]} estudiantes × {df.shape[1]} variables")
    print(f"   · Desertores: {df['desercion'].sum()} ({df['desercion'].mean()*100:.1f}%)")
    print(f"   · No desertores: {(df['desercion']==0).sum()} ({(1-df['desercion'].mean())*100:.1f}%)")
    print(f"\n📐 Estadísticas dimensiones psicoemocionales:")

    dims = [c for c in df.columns if c.startswith("dim_")]
    for d in dims:
        nombre_limpio = d.replace("dim_", "").replace("_", " ").title()
        media_nd = df.loc[df['desercion']==0, d].mean()
        media_d  = df.loc[df['desercion']==1, d].mean()
        print(f"   · {nombre_limpio:<30} No desertor: {media_nd:.2f}  |  Desertor: {media_d:.2f}")

    # ── Guardar dataset ──────────────────────────────────────────────────────
    ruta = "dataset_psicoemocional_uaq.csv"
    df.to_csv(ruta, index=False)
    print(f"\n✅ Dataset guardado en: {ruta}")
    print(f"   Listo para análisis con Random Forest, XGBoost, GB y Stacking")

    # ── Guardar versión codificada (one-hot) para ML ─────────────────────────
    df_ml = pd.get_dummies(df.drop(columns=["id_participante"]), columns=["area"], drop_first=True)
    df_ml.to_csv("dataset_ml_encoded.csv", index=False)
    print(f"   Dataset codificado guardado en: dataset_ml_encoded.csv")
