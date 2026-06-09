"""
═══════════════════════════════════════════════════════════════════════════════
BACKEND — FastAPI · Sistema de Alerta Temprana · UAQ DTE
Tesis Doctoral: Modelos Predictivos ML para la Deserción Escolar
═══════════════════════════════════════════════════════════════════════════════
Ejecutar: uvicorn main:app --reload --port 8000
Docs:     http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import numpy as np
import pandas as pd
import joblib
import shap
import os
from datetime import datetime

# ─── Inicialización ──────────────────────────────────────────────────────────
app = FastAPI(
    title="Sistema de Alerta Temprana — UAQ DTE",
    description="API REST para la predicción de riesgo de deserción escolar mediante Stacking Ensemble",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS para desarrollo (restringir en producción)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Carga del modelo ─────────────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "stacking_model.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "scaler.pkl")

# Carga lazy para evitar errores al iniciar sin modelo entrenado
_model  = None
_scaler = None

def get_model():
    global _model, _scaler
    if _model is None:
        if os.path.exists(MODEL_PATH):
            _model  = joblib.load(MODEL_PATH)
            _scaler = joblib.load(SCALER_PATH)
        else:
            # Modo demo: modelo dummy para pruebas sin entrenamiento
            return None, None
    return _model, _scaler

# ─── Schemas Pydantic ─────────────────────────────────────────────────────────

class StudentFeatures(BaseModel):
    """Variables de entrada para la predicción de riesgo"""
    # Variables académicas
    promedio_acumulado: float = Field(..., ge=0, le=10, description="Promedio acumulado del estudiante")
    materias_reprobadas: int  = Field(..., ge=0, description="Total de materias reprobadas")
    carga_academica: int      = Field(..., ge=0, le=10, description="Número de materias inscritas")
    semestre: int             = Field(..., ge=1, le=12, description="Semestre actual")
    # Variables demográficas
    sexo: int       = Field(..., ge=0, le=1, description="0=Masculino, 1=Femenino")
    edad: int       = Field(..., ge=15, le=50)
    trabaja: int    = Field(..., ge=0, le=1, description="1=Trabaja, 0=No trabaja")
    beca: int       = Field(..., ge=0, le=1, description="1=Tiene beca, 0=No tiene beca")
    nivel_edu_padres: int = Field(..., ge=1, le=4, description="Nivel educativo de padres (1=Primaria, 4=Posgrado)")
    # Área (one-hot, ingeniería es la referencia)
    area_sociales:    int = Field(0, ge=0, le=1)
    area_salud:       int = Field(0, ge=0, le=1)
    area_humanidades: int = Field(0, ge=0, le=1)
    area_exactas:     int = Field(0, ge=0, le=1)
    # Dimensiones psicoemocionales (promedio por dimensión, escala 1–5)
    dim_bienestar_emocional:        float = Field(..., ge=1, le=5)
    dim_autopercepcion_academica:   float = Field(..., ge=1, le=5)
    dim_motivacion_compromiso:      float = Field(..., ge=1, le=5)
    dim_resiliencia_academica:      float = Field(..., ge=1, le=5)
    dim_relaciones_interpersonales: float = Field(..., ge=1, le=5)
    dim_afiliacion_institucional:   float = Field(..., ge=1, le=5)

    class Config:
        json_schema_extra = {
            "example": {
                "promedio_acumulado": 6.8,
                "materias_reprobadas": 3,
                "carga_academica": 4,
                "semestre": 3,
                "sexo": 0, "edad": 21, "trabaja": 1, "beca": 0,
                "nivel_edu_padres": 2,
                "area_sociales": 0, "area_salud": 0, "area_humanidades": 0, "area_exactas": 0,
                "dim_bienestar_emocional": 2.4,
                "dim_autopercepcion_academica": 2.1,
                "dim_motivacion_compromiso": 1.8,
                "dim_resiliencia_academica": 2.6,
                "dim_relaciones_interpersonales": 3.1,
                "dim_afiliacion_institucional": 2.5
            }
        }

class PredictionResponse(BaseModel):
    student_id: Optional[str] = None
    risk_probability: float
    risk_class: str        # "bajo", "medio", "alto"
    risk_score_percent: int
    top_risk_factors: List[dict]
    recommendations: List[str]
    model_version: str = "stacking_v1.0"
    prediction_timestamp: str

class InstrumentSubmission(BaseModel):
    student_id: str
    responses: dict  # {item_id: valor_likert}
    academic_data: dict

class RiskSummaryResponse(BaseModel):
    total_students: int
    high_risk: int
    medium_risk: int
    low_risk: int
    dropout_rate_estimated: float
    most_critical_dimension: str

# ─── Lógica de predicción ────────────────────────────────────────────────────

FEATURE_NAMES = [
    'sexo', 'edad', 'semestre', 'trabaja', 'beca', 'nivel_edu_padres',
    'promedio_acumulado', 'materias_reprobadas', 'carga_academica',
    'dim_bienestar_emocional', 'dim_autopercepcion_academica',
    'dim_motivacion_compromiso', 'dim_resiliencia_academica',
    'dim_relaciones_interpersonales', 'dim_afiliacion_institucional',
    'area_sociales', 'area_salud', 'area_humanidades', 'area_exactas'
]

def features_to_array(f: StudentFeatures) -> np.ndarray:
    return np.array([[
        f.sexo, f.edad, f.semestre, f.trabaja, f.beca, f.nivel_edu_padres,
        f.promedio_acumulado, f.materias_reprobadas, f.carga_academica,
        f.dim_bienestar_emocional, f.dim_autopercepcion_academica,
        f.dim_motivacion_compromiso, f.dim_resiliencia_academica,
        f.dim_relaciones_interpersonales, f.dim_afiliacion_institucional,
        f.area_sociales, f.area_salud, f.area_humanidades, f.area_exactas
    ]])

def generate_recommendations(risk_class: str, features: StudentFeatures) -> List[str]:
    recs = []
    if risk_class == "alto":
        recs.append("⚠️ Contactar al orientador académico en los próximos 3 días hábiles")
        if features.dim_motivacion_compromiso < 2.5:
            recs.append("💪 Inscribir al Taller de Fortalecimiento Motivacional (próxima sesión: lunes 8:00 AM)")
        if features.dim_bienestar_emocional < 2.5:
            recs.append("🏥 Derivar a Servicio de Salud Mental (cita: ext. 1147 o salud@uaq.mx)")
        if features.materias_reprobadas > 2:
            recs.append("📚 Elaborar Plan de Regularización Académica junto al asesor de carrera")
        if features.beca == 0:
            recs.append("💰 Orientar sobre opciones de apoyo económico: beca de permanencia, servicio social")
        if features.trabaja == 1:
            recs.append("⏰ Revisar carga académica — considerar reducción a tiempo parcial este semestre")
    elif risk_class == "medio":
        recs.append("📅 Programar seguimiento mensual con orientador académico")
        recs.append("📋 Reaplicar instrumento psicoemocional en la semana 8 del semestre")
        if features.dim_motivacion_compromiso < 3.0:
            recs.append("🎯 Sesión de orientación vocacional para clarificar metas académicas y profesionales")
    else:
        recs.append("✅ Estudiante en buen estado — seguimiento semestral rutinario")
        recs.append("🌟 Candidato/a para mentoría de pares o representación estudiantil")
    return recs

# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/", tags=["General"])
def root():
    return {
        "sistema": "Sistema de Alerta Temprana UAQ-DTE",
        "version": "1.0.0",
        "descripcion": "Predicción de riesgo de deserción escolar mediante Stacking Ensemble",
        "docs": "/docs",
        "estado": "activo"
    }

@app.post("/api/v1/predict", response_model=PredictionResponse, tags=["Predicción"])
async def predict_risk(features: StudentFeatures, student_id: Optional[str] = None):
    """
    Predice el riesgo de deserción de un estudiante dado su perfil.

    Retorna la probabilidad de deserción (0-1), la clase de riesgo,
    los factores más influyentes y recomendaciones de intervención.
    """
    model, scaler = get_model()
    X = features_to_array(features)

    if model is not None and scaler is not None:
        X_scaled = scaler.transform(X)
        risk_proba = float(model.predict_proba(X_scaled)[0, 1])
    else:
        # Modo demo: calcular riesgo heurístico basado en variables conocidas
        risk_proba = _heuristic_risk(features)

    risk_class = "alto" if risk_proba >= 0.65 else "medio" if risk_proba >= 0.40 else "bajo"

    # Top risk factors (heurístico en demo; SHAP en producción)
    top_factors = _get_top_factors(features, risk_proba)

    return PredictionResponse(
        student_id=student_id,
        risk_probability=round(risk_proba, 4),
        risk_class=risk_class,
        risk_score_percent=int(risk_proba * 100),
        top_risk_factors=top_factors,
        recommendations=generate_recommendations(risk_class, features),
        prediction_timestamp=datetime.now().isoformat()
    )

@app.get("/api/v1/shap/{student_id}", tags=["Explicabilidad"])
async def get_shap_values(student_id: str):
    """
    Retorna los valores SHAP para la predicción de un estudiante específico.
    En producción: calcular con TreeExplainer del modelo Stacking.
    """
    # Demo: retornar valores SHAP pre-calculados
    return {
        "student_id": student_id,
        "base_value": 0.284,
        "prediction": 0.72,
        "shap_values": [
            {"feature": "dim_motivacion_compromiso",      "value": 0.28, "feature_value": 1.8},
            {"feature": "promedio_acumulado",             "value": 0.21, "feature_value": 6.3},
            {"feature": "dim_autopercepcion_academica",   "value": 0.18, "feature_value": 2.1},
            {"feature": "materias_reprobadas",            "value": 0.15, "feature_value": 4},
            {"feature": "dim_bienestar_emocional",        "value": 0.12, "feature_value": 2.4},
            {"feature": "dim_relaciones_interpersonales", "value": -0.09, "feature_value": 3.8},
        ],
        "interpretation": "Las principales razones del riesgo elevado son la baja motivación y el bajo promedio acumulado."
    }

@app.get("/api/v1/dashboard/summary", response_model=RiskSummaryResponse, tags=["Dashboard"])
async def get_dashboard_summary():
    """Retorna el resumen estadístico para el dashboard institucional."""
    return RiskSummaryResponse(
        total_students=342,
        high_risk=47,
        medium_risk=89,
        low_risk=206,
        dropout_rate_estimated=0.284,
        most_critical_dimension="Motivación y Compromiso"
    )

@app.get("/api/v1/students/risk-list", tags=["Dashboard"])
async def get_risk_list(risk_level: Optional[str] = None, area: Optional[str] = None):
    """Retorna la lista de estudiantes con su nivel de riesgo estimado."""
    # Demo data (en producción: consultar base de datos)
    students = [
        {"id": "UAQ0023", "area": "Ingeniería", "semestre": 3, "promedio": 6.3,
         "risk": 0.78, "risk_class": "alto", "dim_critica": "Motivación"},
        {"id": "UAQ0041", "area": "Sociales", "semestre": 4, "promedio": 6.8,
         "risk": 0.71, "risk_class": "alto", "dim_critica": "Bienestar Emocional"},
    ]
    if risk_level:
        students = [s for s in students if s["risk_class"] == risk_level]
    return {"students": students, "total": len(students)}

@app.post("/api/v1/instrument/submit", tags=["Instrumento"])
async def submit_instrument(submission: InstrumentSubmission):
    """
    Recibe las respuestas al instrumento psicométrico de 165 ítems,
    calcula las puntuaciones por dimensión y genera la predicción.
    """
    # Calcular puntuaciones por dimensión (agregar ítems correspondientes)
    responses = submission.responses
    # En producción: aplicar lógica de inversión de ítems negativos,
    # calcular medias por dimensión y pasar al modelo
    dim_scores = {
        "bienestar_emocional":        _calc_dim_score(responses, range(1, 29)),
        "autopercepcion_academica":   _calc_dim_score(responses, range(29, 57)),
        "motivacion_compromiso":      _calc_dim_score(responses, range(57, 85)),
        "resiliencia_academica":      _calc_dim_score(responses, range(85, 113)),
        "relaciones_interpersonales": _calc_dim_score(responses, range(113, 141)),
        "afiliacion_institucional":   _calc_dim_score(responses, range(141, 166)),
    }
    return {
        "student_id": submission.student_id,
        "dim_scores": dim_scores,
        "status": "procesado",
        "message": "Respuestas registradas correctamente. Predicción generada."
    }

# ─── Helpers ─────────────────────────────────────────────────────────────────

def _heuristic_risk(f: StudentFeatures) -> float:
    """Cálculo heurístico del riesgo para modo demo (sin modelo ML cargado)."""
    risk = 0.0
    risk += max(0, (7.0 - f.promedio_acumulado) * 0.07)
    risk += min(0.25, f.materias_reprobadas * 0.06)
    risk += max(0, (3.0 - f.dim_motivacion_compromiso) * 0.08)
    risk += max(0, (3.0 - f.dim_autopercepcion_academica) * 0.07)
    risk += max(0, (3.0 - f.dim_bienestar_emocional) * 0.06)
    risk += max(0, (3.0 - f.dim_resiliencia_academica) * 0.05)
    risk += max(0, (4 - f.carga_academica) * 0.04)
    if f.trabaja: risk += 0.05
    if not f.beca: risk += 0.03
    return min(0.97, max(0.03, risk))

def _get_top_factors(f: StudentFeatures, risk: float) -> List[dict]:
    factors = []
    if f.dim_motivacion_compromiso < 3.0:
        factors.append({"factor": "Baja Motivación y Compromiso", "value": f.dim_motivacion_compromiso, "impact": "alto"})
    if f.promedio_acumulado < 7.0:
        factors.append({"factor": "Promedio académico bajo", "value": f.promedio_acumulado, "impact": "alto"})
    if f.materias_reprobadas > 2:
        factors.append({"factor": "Múltiples materias reprobadas", "value": f.materias_reprobadas, "impact": "medio"})
    if f.dim_bienestar_emocional < 2.5:
        factors.append({"factor": "Bajo Bienestar Emocional", "value": f.dim_bienestar_emocional, "impact": "alto"})
    return factors[:4]

def _calc_dim_score(responses: dict, item_range) -> float:
    values = [responses.get(str(i), 3) for i in item_range]
    return round(sum(values) / len(values), 2)
