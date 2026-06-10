"""
═══════════════════════════════════════════════════════════════════════════════
MÓDULO 06 — PERFIL DE APRENDIZAJE (Motor VARK adaptado al SIP)
Sistema Atenea AI v2.0 | Tesis DTE-UAQ | Luis Francisco Pardo Perea
═══════════════════════════════════════════════════════════════════════════════

Genera el perfil de estilo de aprendizaje de cada estudiante a partir de
las puntuaciones en las 6 dimensiones psicoemocionales del instrumento SIP.

Modelo base: VARK (Fleming, 2001) + Kolb (1984) + adaptaciones SIP.

Los estilos de aprendizaje NO son diagnóstico — son orientaciones preferentes
que guían las recomendaciones pedagógicas. Un estudiante puede tener
características de múltiples estilos (perfil mixto).

Uso:
    python 06_perfil_aprendizaje.py                        # analiza el dataset
    from 06_perfil_aprendizaje import generar_perfil_estudiante  # API para backend

Referencias:
    Fleming, N.D. (2001). Teaching and Learning Styles: VARK Strategies.
    Kolb, D.A. (1984). Experiential Learning. Prentice Hall.
    Felder, R.M. & Silverman, L.K. (1988). Learning and teaching styles in engineering.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import json

# ─── Tipos de aprendizaje ─────────────────────────────────────────────────────

ESTILOS = {
    "visual": {
        "nombre": "Visual-Espacial",
        "icono": "👁️",
        "descripcion": "Aprendes mejor con diagramas, mapas conceptuales, colores y representaciones gráficas.",
        "fortalezas": ["Organización visual de información", "Memorización por imágenes",
                       "Comprensión de relaciones espaciales"],
    },
    "auditivo": {
        "nombre": "Auditivo-Verbal",
        "icono": "🎧",
        "descripcion": "Aprendes mejor escuchando explicaciones, debates y discusiones en voz alta.",
        "fortalezas": ["Retención de explicaciones orales", "Aprendizaje en grupo",
                       "Procesamiento secuencial de información"],
    },
    "lectoescritor": {
        "nombre": "Lectoescritor",
        "icono": "📖",
        "descripcion": "Aprendes mejor leyendo, tomando notas y organizando información en texto.",
        "fortalezas": ["Análisis de textos", "Toma de notas estructuradas",
                       "Síntesis escrita de conceptos"],
    },
    "kinestesico": {
        "nombre": "Kinestésico-Práctico",
        "icono": "🤲",
        "descripcion": "Aprendes mejor haciendo, experimentando y con actividades prácticas.",
        "fortalezas": ["Aprendizaje basado en proyectos", "Experimentación directa",
                       "Resolución práctica de problemas"],
    },
    "reflexivo": {
        "nombre": "Reflexivo-Analítico",
        "icono": "🧠",
        "descripcion": "Prefieres tiempo para observar, analizar y conectar ideas antes de actuar.",
        "fortalezas": ["Análisis profundo", "Pensamiento crítico",
                       "Conexión entre conceptos abstractos"],
    },
    "social": {
        "nombre": "Social-Colaborativo",
        "icono": "🤝",
        "descripcion": "Aprendes mejor en equipo, discutiendo ideas y trabajando con otros.",
        "fortalezas": ["Trabajo colaborativo", "Aprendizaje entre pares",
                       "Comunicación de ideas"],
    },
}

# ─── Mapa de dimensiones SIP → estilos ───────────────────────────────────────
# Cada dimensión contribuye a ciertos estilos con cierto peso
# Valores > 3.5 = fortaleza; < 2.5 = área de mejora

MAPA_DIMENSIONES = {
    # (dimension, umbral_alto, umbral_bajo) → contribución a cada estilo
    "dim_bienestar_emocional": {
        "reflexivo": 0.25,    # mayor bienestar → más capacidad de reflexión
        "social":    0.15,    # bienestar permite apertura social
    },
    "dim_autopercepcion_academica": {
        "lectoescritor": 0.30,  # autoconcepto académico fuerte → preferencia por lectura/escritura
        "reflexivo":     0.20,  # metacognición → reflexión
        "visual":        0.10,
    },
    "dim_motivacion_compromiso": {
        "kinestesico": 0.35,  # alta motivación → prefer. por hacer, experimentar
        "auditivo":    0.15,  # motivación incluye participación activa en clase
        "visual":      0.10,
    },
    "dim_resiliencia_academica": {
        "kinestesico": 0.20,  # resiliencia → práctica, intentar de nuevo
        "reflexivo":   0.20,  # resiliencia → aprender del error con reflexión
        "auditivo":    0.10,
    },
    "dim_relaciones_interpersonales": {
        "social":    0.45,    # alta relación interpersonal → aprendizaje colaborativo
        "auditivo":  0.20,    # social incluye comunicación oral
        "kinestesico": 0.10,
    },
    "dim_afiliacion_institucional": {
        "social":    0.20,    # afiliación → participación en la comunidad educativa
        "lectoescritor": 0.10,
        "visual":    0.10,
    },
}


@dataclass
class PerfilAprendizaje:
    """Resultado del análisis de perfil de aprendizaje de un estudiante."""
    student_id: str
    puntuaciones: Dict[str, float]            # {estilo: puntuación 0-100}
    estilo_primario: str
    estilo_secundario: str
    es_perfil_mixto: bool
    descripcion_perfil: str
    estrategias: List[str]                    # 5 estrategias concretas
    recursos_uaq: List[str]                   # recursos institucionales sugeridos
    compatibilidad_virtual: float             # 0-1: qué tan bien se adapta a e-learning
    dimension_scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "student_id": self.student_id,
            "puntuaciones": self.puntuaciones,
            "estilo_primario": {
                "id": self.estilo_primario,
                "nombre": ESTILOS[self.estilo_primario]["nombre"],
                "icono": ESTILOS[self.estilo_primario]["icono"],
                "descripcion": ESTILOS[self.estilo_primario]["descripcion"],
            },
            "estilo_secundario": {
                "id": self.estilo_secundario,
                "nombre": ESTILOS[self.estilo_secundario]["nombre"],
            },
            "es_perfil_mixto": self.es_perfil_mixto,
            "descripcion_perfil": self.descripcion_perfil,
            "estrategias": self.estrategias,
            "recursos_uaq": self.recursos_uaq,
            "compatibilidad_virtual": round(self.compatibilidad_virtual, 2),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


def calcular_puntuaciones_estilo(dim_scores: Dict[str, float]) -> Dict[str, float]:
    """
    Calcula la puntuación (0-100) para cada estilo de aprendizaje
    a partir de las puntuaciones de las 6 dimensiones SIP.
    """
    puntuaciones = {estilo: 0.0 for estilo in ESTILOS}

    for dim_name, pesos_estilos in MAPA_DIMENSIONES.items():
        dim_val = dim_scores.get(dim_name, 3.0)  # default = neutro
        # Normalizar dimensión al rango 0-1 (original es 1-5)
        dim_norm = (dim_val - 1) / 4.0

        for estilo, peso in pesos_estilos.items():
            puntuaciones[estilo] += dim_norm * peso

    # Normalizar al rango 0-100 y añadir línea base (ningún estilo en 0)
    max_posible = max(sum(v for v in pesos.values())
                      for pesos in MAPA_DIMENSIONES.values()
                      if pesos)
    for estilo in puntuaciones:
        raw = puntuaciones[estilo]
        puntuaciones[estilo] = round(20 + 80 * (raw / (max_posible + 1e-8)), 1)

    return puntuaciones


def generar_estrategias(estilo_primario: str,
                         estilo_secundario: str,
                         dim_scores: Dict[str, float]) -> List[str]:
    """Genera 5 estrategias de estudio personalizadas."""

    banco_estrategias = {
        "visual": [
            "Crea mapas mentales o diagramas para cada tema nuevo",
            "Usa colores para organizar notas: verde=conceptos clave, amarillo=ejemplos, rojo=dudas",
            "Convierte listas de texto en tablas o esquemas visuales antes de estudiar",
            "Usa herramientas como Canva, MindMeister o papel cuadriculado para sintetizar",
            "Antes del examen, dibuja de memoria el mapa del tema completo",
        ],
        "auditivo": [
            "Graba tus propias explicaciones del tema y escúchalas al día siguiente",
            "Forma un grupo de estudio donde cada quien explique un subtema en voz alta",
            "Lee los textos difíciles en voz alta — activa mejor la comprensión",
            "Usa podcasts académicos o videos de YouTube para reforzar conceptos abstractos",
            "Antes del examen, explícale el tema a alguien (o a un espejo)",
        ],
        "lectoescritor": [
            "Elabora resúmenes escritos propios de cada lectura — con tus propias palabras",
            "Mantén un cuaderno de apuntes con secciones: conceptos, ejemplos, preguntas propias",
            "Convierte diagramas y gráficas en descripciones escritas detalladas",
            "Usa la técnica Cornell para tomar notas (columna izquierda: preguntas; derecha: notas)",
            "Escribe una 'ficha de concepto' por cada término nuevo del curso",
        ],
        "kinestesico": [
            "Busca prácticas, laboratorios o proyectos que apliquen los conceptos teóricos",
            "Estudia en sesiones cortas de 25 min con 5 min de movimiento (Técnica Pomodoro)",
            "Usa simuladores, maquetas o roleplay para practicar antes del examen",
            "Relaciona cada concepto abstracto con algo que hayas hecho o experimentado",
            "Propón al profesor casos prácticos o proyectos aplicados para acreditar",
        ],
        "reflexivo": [
            "Dedica 10 minutos al final de cada clase para escribir: ¿qué aprendí? ¿qué preguntas me quedan?",
            "Antes de resolver un ejercicio, analiza el problema desde 3 ángulos distintos",
            "Lee con tiempo — no tomes notas durante la primera lectura, solo comprende",
            "Lleva un diario de aprendizaje semanal: avances, dificultades y conexiones",
            "Conecta explícitamente cada tema nuevo con algo que ya sabías antes",
        ],
        "social": [
            "Organiza o únete a grupos de estudio — explica los temas a tus compañeros",
            "Participa activamente en foros, discusiones y actividades en clase",
            "Busca proyectos colaborativos donde puedas asumir un rol de liderazgo compartido",
            "Contacta al profesor para sesiones de tutoría grupal — es más efectivo para ti que el autoestudio",
            "Usa la red de monitores y tutores pares que ofrece la UAQ",
        ],
    }

    estrategias = []
    # 3 estrategias del estilo primario
    estrategias.extend(banco_estrategias[estilo_primario][:3])
    # 2 estrategias del estilo secundario (sin repetir)
    for e in banco_estrategias[estilo_secundario]:
        if e not in estrategias:
            estrategias.append(e)
        if len(estrategias) == 5:
            break

    # Estrategia adaptativa según dimensiones bajas
    if dim_scores.get("dim_motivacion_compromiso", 3) < 2.5:
        estrategias[-1] = "⚡ Establece una meta pequeña y alcanzable para esta semana — el logro pequeño reconstruye la motivación"
    elif dim_scores.get("dim_bienestar_emocional", 3) < 2.5:
        estrategias[-1] = "🌱 Prioriza tu bienestar: estudiar 30 minutos tranquilo vale más que 3 horas con ansiedad"

    return estrategias[:5]


def generar_recursos_uaq(estilo: str, dim_scores: Dict[str, float]) -> List[str]:
    """Sugiere recursos institucionales UAQ según el perfil."""
    recursos = []

    recursos_base = {
        "visual":        ["Centro de Cómputo UAQ — software de mapas mentales (FreeMind, XMind)",
                          "Biblioteca Digital UAQ — videos educativos e infografías"],
        "auditivo":      ["Tutorías grupales — Coordinación Académica UAQ",
                          "Club de Debate y Oratoria UAQ"],
        "lectoescritor": ["Biblioteca Central UAQ — salas de estudio individual",
                          "Taller de Escritura Académica — Centro de Lenguas UAQ"],
        "kinestesico":   ["Laboratorios y talleres de tu facultad — solicitar acceso extra",
                          "Programa de Prácticas Profesionales Tempranas UAQ"],
        "reflexivo":     ["Tutorías individuales — solicitar en tu coordinación académica",
                          "Centro de Atención Psicológica UAQ (CAPSI) — si el autoexamen genera ansiedad"],
        "social":        ["Programa de Monitores y Tutores Pares UAQ",
                          "Actividades de Extensión y Vinculación — Dirección de Extensión UAQ"],
    }

    recursos.extend(recursos_base.get(estilo, []))

    # Recursos de apoyo según dimensiones críticas
    if dim_scores.get("dim_bienestar_emocional", 3) < 2.5:
        recursos.append("🏥 Centro de Atención Psicológica Integral (CAPSI) UAQ — ext. 1147")
    if dim_scores.get("dim_afiliacion_institucional", 3) < 2.5:
        recursos.append("🤝 Programa de Inducción y Bienvenida para Estudiantes — Rectoría UAQ")
    if dim_scores.get("dim_resiliencia_academica", 3) < 2.5:
        recursos.append("💪 Programa Tutoral UAQ — apoyo académico personalizado")

    return list(dict.fromkeys(recursos))[:4]  # máx 4, sin duplicados


def generar_perfil_estudiante(
    student_id: str,
    dim_bienestar: float,
    dim_autopercepcion: float,
    dim_motivacion: float,
    dim_resiliencia: float,
    dim_relaciones: float,
    dim_afiliacion: float,
) -> PerfilAprendizaje:
    """
    API principal del módulo. Genera el perfil completo de aprendizaje
    para un estudiante dado sus puntuaciones en las 6 dimensiones SIP.

    Args:
        student_id: Identificador del estudiante
        dim_*: Puntuaciones en escala 1-5 para cada dimensión

    Returns:
        PerfilAprendizaje con estilo primario, estrategias y recursos
    """
    dim_scores = {
        "dim_bienestar_emocional":        dim_bienestar,
        "dim_autopercepcion_academica":   dim_autopercepcion,
        "dim_motivacion_compromiso":      dim_motivacion,
        "dim_resiliencia_academica":      dim_resiliencia,
        "dim_relaciones_interpersonales": dim_relaciones,
        "dim_afiliacion_institucional":   dim_afiliacion,
    }

    # Calcular puntuaciones por estilo
    puntuaciones = calcular_puntuaciones_estilo(dim_scores)

    # Ordenar estilos por puntuación
    estilos_ordenados = sorted(puntuaciones.items(), key=lambda x: x[1], reverse=True)
    estilo_primario   = estilos_ordenados[0][0]
    estilo_secundario = estilos_ordenados[1][0]

    # ¿Perfil mixto? Si la diferencia entre primario y secundario < 10 puntos
    es_mixto = (puntuaciones[estilo_primario] - puntuaciones[estilo_secundario]) < 10.0

    # Descripción del perfil
    p_nombre = ESTILOS[estilo_primario]["nombre"]
    s_nombre = ESTILOS[estilo_secundario]["nombre"]
    if es_mixto:
        descripcion = (
            f"Tienes un perfil mixto {p_nombre}/{s_nombre}. "
            f"Combinas estrategias de ambos estilos de forma flexible, "
            f"lo que te da ventaja en contextos de aprendizaje variados."
        )
    else:
        descripcion = (
            f"Tu estilo predominante es {p_nombre}. "
            f"{ESTILOS[estilo_primario]['descripcion']} "
            f"Tu estilo secundario es {s_nombre}, que complementa tu forma de aprender."
        )

    # Estrategias y recursos
    estrategias = generar_estrategias(estilo_primario, estilo_secundario, dim_scores)
    recursos    = generar_recursos_uaq(estilo_primario, dim_scores)

    # Compatibilidad con educación virtual (relevante para e-learning)
    compat_virtual = (
        0.3 * (puntuaciones["visual"] / 100) +
        0.25 * (puntuaciones["lectoescritor"] / 100) +
        0.2 * (puntuaciones["reflexivo"] / 100) +
        0.15 * (puntuaciones["auditivo"] / 100) +
        0.1 * (puntuaciones["kinestesico"] / 100)
    )

    return PerfilAprendizaje(
        student_id=student_id,
        puntuaciones=puntuaciones,
        estilo_primario=estilo_primario,
        estilo_secundario=estilo_secundario,
        es_perfil_mixto=es_mixto,
        descripcion_perfil=descripcion,
        estrategias=estrategias,
        recursos_uaq=recursos,
        compatibilidad_virtual=compat_virtual,
        dimension_scores=dim_scores,
    )


# ─── Análisis sobre el dataset completo ──────────────────────────────────────

def analizar_dataset(csv_path: str = "dataset_real_encoded.csv") -> pd.DataFrame:
    """Genera perfiles de aprendizaje para todos los estudiantes del dataset."""
    import os
    if not os.path.exists(csv_path):
        csv_path = "dataset_ml_encoded.csv"  # fallback al sintético
        print(f"⚠️  Usando dataset sintético: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"\nAnalizando {len(df)} estudiantes...")

    resultados = []
    for i, row in df.iterrows():
        perfil = generar_perfil_estudiante(
            student_id=f"EST{str(i+1).zfill(5)}",
            dim_bienestar=row.get('dim_bienestar_emocional', 3.0),
            dim_autopercepcion=row.get('dim_autopercepcion_academica', 3.0),
            dim_motivacion=row.get('dim_motivacion_compromiso', 3.0),
            dim_resiliencia=row.get('dim_resiliencia_academica', 3.0),
            dim_relaciones=row.get('dim_relaciones_interpersonales', 3.0),
            dim_afiliacion=row.get('dim_afiliacion_institucional', 3.0),
        )
        resultados.append({
            "id": perfil.student_id,
            "estilo_primario": perfil.estilo_primario,
            "estilo_secundario": perfil.estilo_secundario,
            "es_mixto": perfil.es_perfil_mixto,
            **{f"score_{k}": v for k, v in perfil.puntuaciones.items()},
            "compatibilidad_virtual": perfil.compatibilidad_virtual,
            "desercion": row.get('desercion', -1),
        })

    df_perfiles = pd.DataFrame(resultados)

    # Reporte de distribución
    print("\n📊 DISTRIBUCIÓN DE ESTILOS DE APRENDIZAJE:")
    print(df_perfiles['estilo_primario'].value_counts().to_string())
    print(f"\nPerfil mixto: {df_perfiles['es_mixto'].mean():.1%} de los estudiantes")

    # Relación estilo-deserción
    print("\n🔗 TASA DE DESERCIÓN POR ESTILO DE APRENDIZAJE:")
    if 'desercion' in df_perfiles.columns and df_perfiles['desercion'].max() > 0:
        for estilo in ESTILOS:
            mask = df_perfiles['estilo_primario'] == estilo
            n = mask.sum()
            if n > 0:
                tasa = df_perfiles.loc[mask, 'desercion'].mean()
                print(f"  {ESTILOS[estilo]['nombre']:<25} n={n:>4}  deserción={tasa:.1%}")

    df_perfiles.to_csv("perfiles_aprendizaje.csv", index=False)
    print(f"\n✅ Perfiles guardados: perfiles_aprendizaje.csv")
    return df_perfiles


# ─── Demo ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("MOTOR DE PERFIL DE APRENDIZAJE — Sistema Atenea AI v2.0")
    print("=" * 70)

    # Demo: estudiante con perfil específico
    print("\n📋 DEMO — Estudiante con baja motivación y alta relacionalidad:")
    perfil = generar_perfil_estudiante(
        student_id="UAQ_DEMO_001",
        dim_bienestar=2.8,
        dim_autopercepcion=3.2,
        dim_motivacion=2.1,
        dim_resiliencia=3.0,
        dim_relaciones=4.2,
        dim_afiliacion=3.5,
    )
    print(perfil.to_json())

    print("\n" + "─"*70)
    print("📋 DEMO — Estudiante con alto bienestar y alta autopercepción:")
    perfil2 = generar_perfil_estudiante(
        student_id="UAQ_DEMO_002",
        dim_bienestar=4.3,
        dim_autopercepcion=4.1,
        dim_motivacion=3.8,
        dim_resiliencia=4.0,
        dim_relaciones=2.9,
        dim_afiliacion=3.2,
    )
    print(perfil2.to_json())

    # Analizar el dataset completo
    print("\n" + "="*70)
    analizar_dataset()
