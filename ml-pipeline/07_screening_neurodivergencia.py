"""
═══════════════════════════════════════════════════════════════════════════════
MÓDULO 07 — SCREENING DE NEURODIVERGENCIAS
Sistema Atenea AI v2.0 | Tesis DTE-UAQ | Luis Francisco Pardo Perea
═══════════════════════════════════════════════════════════════════════════════

⚠️  AVISO LEGAL IMPORTANTE ⚠️
Este módulo genera INDICADORES DE RIESGO, NO DIAGNÓSTICOS CLÍNICOS.
Los resultados de este módulo:
  - NO reemplazan la evaluación de un profesional de salud mental
  - NO constituyen diagnóstico de ningún tipo
  - SON indicadores orientativos para considerar una evaluación especializada
  - DEBEN comunicarse con lenguaje empático y no estigmatizante

Marco ético: Potentia vs. Potestas (Espinoza, vía Deleuze) — el screening
amplía las posibilidades del estudiante (potentia), no lo etiqueta (potestas).

Indicadores implementados:
  1. TDAH — Trastorno por Déficit de Atención e Hiperactividad
     Proxy: patrones de motivación variable + dificultades de resiliencia
     Escala de referencia: ASRS-v1.1 (Kessler et al., 2005)

  2. TEA — Trastorno del Espectro Autista (niveles de apoyo 1-2)
     Proxy: patrones atípicos en relaciones interpersonales + afiliación
     Escala de referencia: AQ-10 (Allison et al., 2012)

  3. Dislexia / Dificultades de Procesamiento
     Proxy: brecha entre autopercepción académica y desempeño real
     Escala de referencia: Cuestionario de Dificultades de Aprendizaje (CDA)

  4. Ansiedad Académica
     Proxy: bajo bienestar emocional + baja resiliencia ante el entorno académico
     Escala de referencia: GAD-7 / AMAS (Zeidner, 1998)

Uso:
    python 07_screening_neurodivergencia.py                   # analiza el dataset
    from 07_screening_neurodivergencia import screening_estudiante  # API backend

Referencias:
    Kessler et al. (2005). The World Health Organization Adult ADHD Self-Report Scale.
    Allison et al. (2012). The Q-CHAT. Journal of Autism and Developmental Disorders.
    Armstrong, T. (2012). Neurodiversity in the Classroom. ASCD.
    Pardo Perea, L.F. (2026). §5.8 Posicionalidad del investigador. Tesis DTE-UAQ.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import json

# ─── Tipos de indicadores ─────────────────────────────────────────────────────

INDICADORES = {
    "tdah": {
        "nombre": "Indicadores TDAH",
        "nombre_completo": "Trastorno por Déficit de Atención e Hiperactividad",
        "icono": "⚡",
        "descripcion_bajo": "No se observan indicadores significativos relacionados con TDAH.",
        "descripcion_medio": "Se observan algunos patrones que pueden relacionarse con dificultades atencionales. Considera explorar estrategias de organización.",
        "descripcion_alto": "Se observan múltiples indicadores asociados al TDAH. Te recomendamos consultar con el servicio de apoyo estudiantil para una evaluación especializada.",
        "recursos": ["Servicio de Psicología UAQ (CAPSI) — ext. 1147",
                     "Coaching académico para TDAH — solicitar en tu coordinación",
                     "App Forest / Focusmate para manejo del tiempo y atención"],
        "estrategias_apoyo": [
            "Divide tareas grandes en pasos muy pequeños con fechas intermedias",
            "Usa temporizadores visuales (reloj de arena, app) para cada sesión de estudio",
            "Habla con tus profesores sobre posibles ajustes razonables (tiempo extra en exámenes)",
            "Estudia en el mismo lugar y horario todos los días para crear rutina",
        ],
    },
    "tea": {
        "nombre": "Indicadores Espectro Autista",
        "nombre_completo": "Trastorno del Espectro Autista (rasgos de alta funcionalidad)",
        "icono": "🧩",
        "descripcion_bajo": "No se observan indicadores significativos relacionados con TEA.",
        "descripcion_medio": "Se observan algunas características que pueden relacionarse con el espectro autista (rasgos de alta funcionalidad). Esto no es diagnóstico.",
        "descripcion_alto": "Se observan indicadores consistentes con rasgos del espectro autista. Muchas personas con estas características tienen éxito académico con apoyos adecuados. Te recomendamos explorar una evaluación especializada.",
        "recursos": ["Unidad de Atención a Estudiantes con Discapacidad UAQ",
                     "Grupo de estudiantes neurodivergentes UAQ (si existe)",
                     "Orientación sobre ajustes razonables en evaluaciones"],
        "estrategias_apoyo": [
            "Solicita a tus profesores el programa del curso con anticipación (reduces incertidumbre)",
            "Identifica un 'compañero de referencia' en cada materia para dudas de organización social",
            "Usa auriculares y busca espacios tranquilos para estudiar si el ruido es molesto",
            "Escribe explícitamente las 'reglas no escritas' de cada clase/grupo para clarificarlas",
        ],
    },
    "dislexia": {
        "nombre": "Indicadores de Dificultades de Procesamiento",
        "nombre_completo": "Dislexia / Dificultades de procesamiento lector-escritor",
        "icono": "📝",
        "descripcion_bajo": "No se observan indicadores significativos de dificultades de procesamiento.",
        "descripcion_medio": "Se observa una posible brecha entre tu potencial y tu desempeño lector/escritor. Esto puede tener múltiples causas, incluyendo dislexia no diagnosticada.",
        "descripcion_alto": "Se observan indicadores de posibles dificultades de procesamiento lector-escritor. Una evaluación especializada puede abrirte acceso a apoyos académicos importantes.",
        "recursos": ["Departamento de Educación Especial / Psicopedagogía UAQ",
                     "Herramientas de texto a voz: NaturalReader, Microsoft Immersive Reader",
                     "Extensión de tiempo en exámenes — solicitar con documentación de apoyo"],
        "estrategias_apoyo": [
            "Usa texto a voz para leer documentos largos — reduce la carga cognitiva",
            "Prefiere tomar apuntes en mapas mentales o diagramas, no en prosa lineal",
            "Pide al profesor permiso para grabar la clase y escucharla después",
            "Usa correctores ortográficos agresivos — no es trampa, es tecnología de apoyo",
        ],
    },
    "ansiedad_academica": {
        "nombre": "Indicadores de Ansiedad Académica",
        "nombre_completo": "Ansiedad relacionada con el contexto académico",
        "icono": "🌊",
        "descripcion_bajo": "No se observan indicadores significativos de ansiedad académica.",
        "descripcion_medio": "Se observan indicadores de ansiedad académica moderada. Es importante atenderla antes de que afecte tu desempeño.",
        "descripcion_alto": "Se observan indicadores importantes de ansiedad académica. Buscar apoyo no es debilidad — es inteligencia emocional. El CAPSI UAQ puede ayudarte.",
        "recursos": ["Centro de Atención Psicológica Integral (CAPSI) UAQ — ext. 1147",
                     "Taller de Manejo de Ansiedad — Bienestar Estudiantil UAQ",
                     "App Headspace / Calm (primera semana gratis con correo UAQ)"],
        "estrategias_apoyo": [
            "Practica respiración diafragmática 5 minutos antes de cada examen",
            "Haz simulacros de examen en condiciones reales — reduce la ansiedad anticipatoria",
            "Habla con tu profesor antes del examen sobre dudas de formato — la incertidumbre amplifica la ansiedad",
            "Llevar una lista de 'lo que sí sé' antes del examen (no solo lo que no sé)",
        ],
    },
}

# ─── Algoritmos de scoring ────────────────────────────────────────────────────

def score_tdah(
    dim_motivacion: float,
    dim_resiliencia: float,
    dim_autopercepcion: float,
    promedio: Optional[float] = None,
    materias_rep: Optional[int] = None,
) -> float:
    """
    Score TDAH (0-100).
    Indicadores principales:
    - Variabilidad motivacional alta: motivación baja pero autopercepción no tan baja
      (el estudiante sabe que puede pero no puede mantener el esfuerzo)
    - Baja resiliencia ante el fracaso académico
    - Brecha entre potencial percibido y desempeño real
    """
    score = 0.0

    # Indicador 1: Motivación baja con autopercepción relativa (no completamente derrumbada)
    # TDAH: "sé que puedo pero no puedo arrancar / sostener"
    brecha_motivacion = max(0, dim_autopercepcion - dim_motivacion)
    score += brecha_motivacion * 15  # máx 60 si brecha = 4

    # Indicador 2: Baja resiliencia académica
    if dim_resiliencia < 2.5:
        score += (2.5 - dim_resiliencia) * 12

    # Indicador 3: Brecha rendimiento-promedio si disponible
    if promedio is not None:
        # Si el promedio es bajo pero la autopercepción no tan baja → posible TDAH
        rendimiento_norm = promedio / 10  # normalizar 0-10 → 0-1
        potencial_norm = (dim_autopercepcion - 1) / 4  # normalizar 1-5 → 0-1
        brecha_rendimiento = max(0, potencial_norm - rendimiento_norm)
        score += brecha_rendimiento * 20

    # Indicador 4: Materias reprobadas pese a buena autopercepción
    if materias_rep is not None and dim_autopercepcion > 3.0:
        score += min(materias_rep * 3, 15)

    return min(100.0, max(0.0, round(score, 1)))


def score_tea(
    dim_relaciones: float,
    dim_afiliacion: float,
    dim_bienestar: float,
    dim_autopercepcion: float,
) -> float:
    """
    Score TEA rasgos (0-100).
    Indicadores principales:
    - Dificultades en relaciones interpersonales pese a bienestar general aceptable
    - Baja afiliación institucional (dificultad para "pertenecer" a la comunidad)
    - Autopercepción académica puede ser alta (los rasgos TEA no reducen capacidad cognitiva)
    - Patrón: competencia intelectual alta + dificultad social
    """
    score = 0.0

    # Indicador 1: Relaciones interpersonales bajas
    if dim_relaciones < 3.0:
        score += (3.0 - dim_relaciones) * 18  # máx 36

    # Indicador 2: Baja afiliación (sentido de pertenencia bajo)
    if dim_afiliacion < 3.0:
        score += (3.0 - dim_afiliacion) * 12  # máx 24

    # Indicador 3: Patrón "competencia alta, socialización baja"
    # Alta autopercepción académica con bajas relaciones es indicativo
    competencia_vs_social = max(0, dim_autopercepcion - dim_relaciones)
    score += competencia_vs_social * 8  # máx 32

    # Indicador 4: Bienestar bajo a pesar de capacidad (el esfuerzo social es agotador)
    if dim_bienestar < 2.5 and dim_relaciones < 2.5:
        score += 10  # combinación específica

    return min(100.0, max(0.0, round(score, 1)))


def score_dislexia(
    dim_autopercepcion: float,
    promedio: Optional[float] = None,
    materias_rep: Optional[int] = None,
) -> float:
    """
    Score dislexia/dificultades de procesamiento (0-100).
    Indicadores principales:
    - Brecha notable entre autopercepción (siente que puede) y rendimiento real
    - Puede tener buena comprensión oral pero dificultades escritas
    """
    score = 0.0

    if promedio is not None:
        rendimiento_norm = promedio / 10
        potencial_norm = (dim_autopercepcion - 1) / 4
        brecha = max(0, potencial_norm - rendimiento_norm - 0.1)  # threshold pequeño
        score += brecha * 50  # máx ~45 si brecha muy grande

    if materias_rep is not None:
        # Muchas reprobadas con buena autopercepción = posible dificultad específica
        if dim_autopercepcion > 3.0 and materias_rep > 2:
            score += min(materias_rep * 5, 30)

    # Sin datos académicos: señal baja (no podemos calcularlo bien)
    if promedio is None:
        score = max(0, score)  # solo hay score si hay datos académicos

    return min(100.0, max(0.0, round(score, 1)))


def score_ansiedad(
    dim_bienestar: float,
    dim_resiliencia: float,
    dim_motivacion: float,
) -> float:
    """
    Score ansiedad académica (0-100).
    Indicadores principales:
    - Bajo bienestar emocional
    - Baja resiliencia ante el fracaso
    - Motivación baja por evitación (no por falta de interés)
    """
    score = 0.0

    # Indicador 1: Bajo bienestar emocional (componente central de la ansiedad)
    if dim_bienestar < 3.0:
        score += (3.0 - dim_bienestar) * 20  # máx 40

    # Indicador 2: Baja resiliencia (catastrofismo ante el fracaso)
    if dim_resiliencia < 3.0:
        score += (3.0 - dim_resiliencia) * 15  # máx 30

    # Indicador 3: Baja motivación combinada con bajo bienestar
    # (Ansiedad puede bloquear la motivación)
    if dim_motivacion < 2.5 and dim_bienestar < 2.5:
        score += 20  # combinación específica de alto riesgo

    return min(100.0, max(0.0, round(score, 1)))


# ─── Clasificación de nivel ───────────────────────────────────────────────────

def clasificar_nivel(score: float) -> Tuple[str, str]:
    """Devuelve (nivel, color) según el score."""
    if score < 30:
        return "bajo", "verde"
    elif score < 60:
        return "medio", "amarillo"
    else:
        return "alto", "naranja"


# ─── Dataclass resultado ──────────────────────────────────────────────────────

@dataclass
class ResultadoScreening:
    """Resultado del screening de neurodivergencias para un estudiante."""
    student_id: str
    scores: Dict[str, float]
    niveles: Dict[str, str]
    indicadores_activos: List[str]  # indicadores con nivel "medio" o "alto"
    mensaje_general: str
    recomendaciones: List[str]
    recursos: List[str]
    estrategias_apoyo: List[str]
    aviso_legal: str = (
        "AVISO IMPORTANTE: Estos resultados son indicadores orientativos, "
        "NO constituyen diagnóstico clínico. Si algún indicador es relevante "
        "para ti, te recomendamos consultar con un profesional de salud mental. "
        "La neurodivergencia es una variante de la diversidad humana, no un déficit."
    )

    def to_dict(self) -> dict:
        return {
            "student_id": self.student_id,
            "scores": self.scores,
            "niveles": self.niveles,
            "indicadores_activos": self.indicadores_activos,
            "mensaje_general": self.mensaje_general,
            "recomendaciones": self.recomendaciones,
            "recursos": self.recursos,
            "estrategias_apoyo": self.estrategias_apoyo,
            "aviso_legal": self.aviso_legal,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


# ─── API principal ────────────────────────────────────────────────────────────

def screening_estudiante(
    student_id: str,
    dim_bienestar: float,
    dim_autopercepcion: float,
    dim_motivacion: float,
    dim_resiliencia: float,
    dim_relaciones: float,
    dim_afiliacion: float,
    promedio: Optional[float] = None,
    materias_rep: Optional[int] = None,
) -> ResultadoScreening:
    """
    Realiza el screening de neurodivergencias para un estudiante.

    Args:
        student_id: ID del estudiante
        dim_*: Puntuaciones 1-5 de cada dimensión SIP
        promedio: Promedio académico 0-10 (mejora precisión del screening)
        materias_rep: Número de materias reprobadas

    Returns:
        ResultadoScreening con scores, niveles y recomendaciones
    """
    # Calcular scores
    scores = {
        "tdah": score_tdah(dim_motivacion, dim_resiliencia, dim_autopercepcion,
                           promedio, materias_rep),
        "tea":  score_tea(dim_relaciones, dim_afiliacion, dim_bienestar, dim_autopercepcion),
        "dislexia": score_dislexia(dim_autopercepcion, promedio, materias_rep),
        "ansiedad_academica": score_ansiedad(dim_bienestar, dim_resiliencia, dim_motivacion),
    }

    # Clasificar niveles
    niveles = {ind: clasificar_nivel(s)[0] for ind, s in scores.items()}

    # Identificar indicadores activos
    activos = [ind for ind, nivel in niveles.items() if nivel in ("medio", "alto")]

    # Generar recomendaciones específicas
    recomendaciones = []
    recursos_todos = []
    estrategias_todas = []

    for ind in activos:
        nivel = niveles[ind]
        info = INDICADORES[ind]
        desc_key = f"descripcion_{nivel}"
        recomendaciones.append(f"{info['icono']} {info['nombre']}: {info[desc_key]}")
        recursos_todos.extend(info['recursos'][:2])
        estrategias_todas.extend(info['estrategias_apoyo'][:2])

    # Mensaje general
    if not activos:
        mensaje = (
            "No se observaron indicadores relevantes en ninguna de las áreas exploradas. "
            "Recuerda que este screening tiene limitaciones y no agota todas las posibilidades."
        )
    elif len(activos) == 1:
        ind = activos[0]
        mensaje = (
            f"Se observaron indicadores en el área de {INDICADORES[ind]['nombre_completo']}. "
            "Esto no significa que tengas esta condición, pero puede valer la pena explorar "
            "si algunos apoyos específicos te serían útiles."
        )
    else:
        nombres = " y ".join([INDICADORES[a]['nombre_completo'] for a in activos])
        mensaje = (
            f"Se observaron indicadores en múltiples áreas ({nombres}). "
            "La neurodivergencia frecuentemente se presenta de forma combinada. "
            "Una evaluación especializada puede darte claridad y acceso a apoyos concretos."
        )

    # Eliminar duplicados en recursos y estrategias
    recursos_finales = list(dict.fromkeys(recursos_todos))[:4]
    estrategias_finales = list(dict.fromkeys(estrategias_todas))[:4]

    return ResultadoScreening(
        student_id=student_id,
        scores=scores,
        niveles=niveles,
        indicadores_activos=activos,
        mensaje_general=mensaje,
        recomendaciones=recomendaciones,
        recursos=recursos_finales,
        estrategias_apoyo=estrategias_finales,
    )


# ─── Análisis sobre el dataset completo ──────────────────────────────────────

def analizar_dataset_screening(csv_path: str = "dataset_real_encoded.csv") -> pd.DataFrame:
    """Aplica el screening a todos los estudiantes del dataset."""
    import os
    if not os.path.exists(csv_path):
        csv_path = "dataset_ml_encoded.csv"
        print(f"⚠️  Usando dataset sintético: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"\nAplicando screening a {len(df)} estudiantes...")

    resultados = []
    for i, row in df.iterrows():
        res = screening_estudiante(
            student_id=f"EST{str(i+1).zfill(5)}",
            dim_bienestar=row.get('dim_bienestar_emocional', 3.0),
            dim_autopercepcion=row.get('dim_autopercepcion_academica', 3.0),
            dim_motivacion=row.get('dim_motivacion_compromiso', 3.0),
            dim_resiliencia=row.get('dim_resiliencia_academica', 3.0),
            dim_relaciones=row.get('dim_relaciones_interpersonales', 3.0),
            dim_afiliacion=row.get('dim_afiliacion_institucional', 3.0),
            promedio=row.get('promedio_acumulado'),
            materias_rep=row.get('materias_reprobadas'),
        )
        resultados.append({
            "id": res.student_id,
            "score_tdah": res.scores["tdah"],
            "score_tea": res.scores["tea"],
            "score_dislexia": res.scores["dislexia"],
            "score_ansiedad": res.scores["ansiedad_academica"],
            "nivel_tdah": res.niveles["tdah"],
            "nivel_tea": res.niveles["tea"],
            "nivel_dislexia": res.niveles["dislexia"],
            "nivel_ansiedad": res.niveles["ansiedad_academica"],
            "n_activos": len(res.indicadores_activos),
            "desercion": row.get('desercion', -1),
        })

    df_nd = pd.DataFrame(resultados)

    # Reporte
    print("\n📊 DISTRIBUCIÓN DE INDICADORES DE NEURODIVERGENCIA:")
    for ind in ["tdah", "tea", "dislexia", "ansiedad"]:
        col = f"nivel_{ind}"
        dist = df_nd[col].value_counts()
        total = len(df_nd)
        medio_alto = dist.get("medio", 0) + dist.get("alto", 0)
        print(f"  {ind.upper():<20} Indicadores medio/alto: {medio_alto} ({medio_alto/total:.1%})")

    # Relación con deserción
    print("\n🔗 INDICADORES ACTIVOS vs DESERCIÓN:")
    if df_nd['desercion'].max() > 0:
        for n in [0, 1, 2, 3, 4]:
            mask = df_nd['n_activos'] == n
            if mask.sum() > 5:
                tasa = df_nd.loc[mask, 'desercion'].mean()
                print(f"  {n} indicadores activos: n={mask.sum():>4}, deserción={tasa:.1%}")

    df_nd.to_csv("screening_neurodivergencia.csv", index=False)
    print(f"\n✅ Screening guardado: screening_neurodivergencia.csv")

    # Combinar con dataset principal para análisis integrado
    df_ml = pd.read_csv(csv_path)
    df_ml['score_tdah'] = df_nd['score_tdah'].values
    df_ml['score_tea'] = df_nd['score_tea'].values
    df_ml['score_dislexia'] = df_nd['score_dislexia'].values
    df_ml['score_ansiedad'] = df_nd['score_ansiedad'].values
    df_ml.to_csv("dataset_con_neurodivergencia.csv", index=False)
    print(f"✅ Dataset enriquecido con scores ND: dataset_con_neurodivergencia.csv")
    print("   (Estos 4 scores pueden añadirse como features adicionales al modelo ML)")

    return df_nd


# ─── Demo ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("SCREENING DE NEURODIVERGENCIAS — Sistema Atenea AI v2.0")
    print("⚠️  INDICADORES ORIENTATIVOS, NO DIAGNÓSTICO CLÍNICO")
    print("=" * 70)

    # Demo 1: Perfil con posibles rasgos TDAH
    print("\n📋 DEMO 1 — Estudiante con posibles indicadores TDAH:")
    print("   (Alta autopercepción, baja motivación sostenida, baja resiliencia)")
    resultado1 = screening_estudiante(
        student_id="UAQ_DEMO_001",
        dim_bienestar=3.0,
        dim_autopercepcion=4.1,    # siente que puede
        dim_motivacion=2.0,        # pero no puede sostenerlo
        dim_resiliencia=2.3,       # fácilmente se desanima ante el fracaso
        dim_relaciones=3.5,
        dim_afiliacion=3.2,
        promedio=6.5,              # promedio bajo para su autopercepción
        materias_rep=3,
    )
    print(resultado1.to_json())

    print("\n" + "─" * 70)

    # Demo 2: Perfil con posibles rasgos TEA + ansiedad
    print("📋 DEMO 2 — Estudiante con posibles indicadores espectro + ansiedad:")
    resultado2 = screening_estudiante(
        student_id="UAQ_DEMO_002",
        dim_bienestar=2.2,
        dim_autopercepcion=4.3,    # alta competencia intelectual
        dim_motivacion=3.1,
        dim_resiliencia=2.8,
        dim_relaciones=1.9,        # dificultades sociales marcadas
        dim_afiliacion=2.1,        # no se siente parte de la institución
        promedio=8.5,              # excelente promedio
        materias_rep=0,
    )
    print(resultado2.to_json())

    # Analizar dataset completo
    print("\n" + "=" * 70)
    analizar_dataset_screening()
