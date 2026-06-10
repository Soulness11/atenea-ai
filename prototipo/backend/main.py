"""
═══════════════════════════════════════════════════════════════════════════════
BACKEND — FastAPI · Sistema de Alerta Temprana · UAQ DTE
Tesis Doctoral: Modelos Predictivos ML para la Deserción Escolar
═══════════════════════════════════════════════════════════════════════════════
Ejecutar: uvicorn main:app --reload --port 8000
Docs:     http://localhost:8000/docs

v2.0 — Añade:
  · Autenticación JWT multi-rol (Estudiante / Docente / Administrador)
  · SQLite con tabla users via SQLAlchemy
  · Endpoints: POST /auth/register, POST /auth/login, GET /auth/me
  · Endpoints docente: GET /api/v1/docente/students
  · Endpoints admin:   GET /api/v1/admin/overview, GET /api/v1/admin/users
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List
import numpy as np
import pandas as pd
import joblib
import shap
import os
import json
from datetime import datetime, timedelta

# ─── Dependencias de autenticación ───────────────────────────────────────────
# Instalar: pip3 install "python-jose[cryptography]" "passlib[bcrypt]" sqlalchemy
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN JWT + BASE DE DATOS
# ═══════════════════════════════════════════════════════════════════════════

# Secreto JWT — en producción usar variable de entorno: os.getenv("JWT_SECRET", ...)
SECRET_KEY    = os.getenv("JWT_SECRET", "atenea-uaq-jwt-secret-2026-doctorado-dte")
ALGORITHM     = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440  # 24 horas

# Código de autorización requerido para registrar Administradores
ADMIN_AUTH_CODE = os.getenv("ADMIN_AUTH_CODE", "UAQ-ATENEA-ADMIN-2026")

# ─── SQLAlchemy (SQLite en desarrollo, PostgreSQL en producción) ──────────────
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./atenea_users.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class UserDB(Base):
    """Tabla de usuarios del sistema Atenea AI"""
    __tablename__ = "users"
    id         = Column(Integer, primary_key=True, index=True)
    email      = Column(String, unique=True, index=True, nullable=False)
    nombre     = Column(String, nullable=False)
    password_hash = Column(String, nullable=False)
    role       = Column(String, nullable=False)          # "estudiante" | "docente" | "admin"
    extra_data = Column(Text, default="{}")              # JSON con campos específicos del rol
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ─── Hashing de contraseñas ───────────────────────────────────────────────────
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

# ─── JWT helpers ─────────────────────────────────────────────────────────────
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)

def create_access_token(data: dict) -> str:
    payload = data.copy()
    payload["exp"] = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=401, detail="Token inválido o expirado")

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> UserDB:
    if not token:
        raise HTTPException(status_code=401, detail="Se requiere autenticación")
    payload = decode_token(token)
    email = payload.get("sub")
    if not email:
        raise HTTPException(status_code=401, detail="Token inválido")
    user = db.query(UserDB).filter(UserDB.email == email).first()
    if not user:
        raise HTTPException(status_code=401, detail="Usuario no encontrado")
    return user

def require_role(*roles):
    """Dependencia de autorización por rol"""
    def checker(current_user: UserDB = Depends(get_current_user)):
        if current_user.role not in roles:
            raise HTTPException(status_code=403, detail=f"Se requiere rol: {', '.join(roles)}")
        return current_user
    return checker

# ─── Schemas de autenticación ─────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email:    str = Field(..., example="estudiante@uaq.edu.mx")
    nombre:   str = Field(..., example="María López García")
    password: str = Field(..., min_length=6, example="mi_contraseña_segura")
    role:     str = Field(..., example="estudiante")  # estudiante | docente | admin
    # Campos opcionales específicos por rol
    matricula:       Optional[str] = None  # Estudiante
    programa:        Optional[str] = None  # Estudiante
    semestre:        Optional[int] = None  # Estudiante
    num_empleado:    Optional[str] = None  # Docente / Admin
    facultad:        Optional[str] = None  # Docente
    contratacion:    Optional[str] = None  # Docente (TC / MT / PH)
    area:            Optional[str] = None  # Admin
    codigo_auth:     Optional[str] = None  # Admin — requerido

class LoginRequest(BaseModel):
    email:    str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    nombre: str
    user_id: int

class UserPublic(BaseModel):
    id:         int
    email:      str
    nombre:     str
    role:       str
    extra_data: dict
    created_at: str

# ─── Seed: crear usuario admin por defecto si no existe ───────────────────────
def _seed_admin():
    db = SessionLocal()
    try:
        existing = db.query(UserDB).filter(UserDB.email == "admin@uaq.edu.mx").first()
        if not existing:
            admin = UserDB(
                email="admin@uaq.edu.mx",
                nombre="Administrador Atenea",
                password_hash=hash_password("atenea2026"),
                role="admin",
                extra_data=json.dumps({"area": "TI", "num_empleado": "ADM001"})
            )
            db.add(admin)
            db.commit()
    finally:
        db.close()

_seed_admin()

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

# ═══════════════════════════════════════════════════════════════════════════
# ENDPOINTS DE AUTENTICACIÓN
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/auth/register", tags=["Autenticación"], summary="Registrar nuevo usuario")
async def register(req: RegisterRequest, db: Session = Depends(get_db)):
    """
    Registra un nuevo usuario en el sistema Atenea AI.

    - **Estudiante**: requiere matricula, programa, semestre
    - **Docente**: requiere num_empleado, facultad, contratacion
    - **Administrador**: requiere num_empleado, area y `codigo_auth` válido
    """
    # Validar email único
    if db.query(UserDB).filter(UserDB.email == req.email).first():
        raise HTTPException(status_code=400, detail="El correo ya está registrado en el sistema")

    # Validar rol
    if req.role not in ("estudiante", "docente", "admin"):
        raise HTTPException(status_code=400, detail="Rol inválido. Use: estudiante, docente o admin")

    # Validar código de autorización para admins
    if req.role == "admin":
        if req.codigo_auth != ADMIN_AUTH_CODE:
            raise HTTPException(status_code=403, detail="Código de autorización incorrecto para rol Administrador")

    # Construir extra_data según rol
    extra: dict = {}
    if req.role == "estudiante":
        extra = {"matricula": req.matricula, "programa": req.programa, "semestre": req.semestre}
    elif req.role == "docente":
        extra = {"num_empleado": req.num_empleado, "facultad": req.facultad, "contratacion": req.contratacion}
    elif req.role == "admin":
        extra = {"num_empleado": req.num_empleado, "area": req.area}

    user = UserDB(
        email=req.email,
        nombre=req.nombre,
        password_hash=hash_password(req.password),
        role=req.role,
        extra_data=json.dumps(extra)
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token({"sub": user.email, "role": user.role, "nombre": user.nombre})
    return TokenResponse(access_token=token, role=user.role, nombre=user.nombre, user_id=user.id)


@app.post("/auth/login", tags=["Autenticación"], summary="Iniciar sesión")
async def login(req: LoginRequest, db: Session = Depends(get_db)):
    """
    Autentica al usuario y retorna un JWT.
    El token debe incluirse en todas las peticiones protegidas como:
    `Authorization: Bearer <token>`
    """
    user = db.query(UserDB).filter(UserDB.email == req.email).first()
    if not user or not verify_password(req.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Correo o contraseña incorrectos")

    # Actualizar último acceso
    user.last_login = datetime.utcnow()
    db.commit()

    token = create_access_token({"sub": user.email, "role": user.role, "nombre": user.nombre})
    return TokenResponse(access_token=token, role=user.role, nombre=user.nombre, user_id=user.id)


@app.get("/auth/me", tags=["Autenticación"], summary="Perfil del usuario autenticado")
async def me(current_user: UserDB = Depends(get_current_user)):
    """Retorna el perfil del usuario autenticado (cualquier rol)."""
    return UserPublic(
        id=current_user.id,
        email=current_user.email,
        nombre=current_user.nombre,
        role=current_user.role,
        extra_data=json.loads(current_user.extra_data or "{}"),
        created_at=current_user.created_at.isoformat()
    )


# ═══════════════════════════════════════════════════════════════════════════
# ENDPOINTS DOCENTE (rol requerido: docente o admin)
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/api/v1/docente/students", tags=["Docente"], summary="Estudiantes del docente")
async def docente_students(
    risk_level: Optional[str] = None,
    current_user: UserDB = Depends(require_role("docente", "admin"))
):
    """
    Retorna la lista de estudiantes asignados al docente autenticado
    con su nivel de riesgo y dimensiones SIP.
    """
    estudiantes = [
        {"id":"UAQ0023","nombre":"Carlos Mendoza Torres","grupo":"DTE-401","semestre":4,
         "promedio":6.3,"riesgo":"alto","dims":[2.1,2.4,1.8,2.3,3.1,2.5]},
        {"id":"UAQ0041","nombre":"María Sánchez Guerrero","grupo":"DTE-402","semestre":4,
         "promedio":6.8,"riesgo":"alto","dims":[2.4,2.8,2.2,2.6,2.9,2.3]},
        {"id":"UAQ0055","nombre":"Roberto Silva Martínez","grupo":"DTE-402","semestre":4,
         "promedio":7.8,"riesgo":"medio","dims":[3.1,3.4,2.8,3.2,3.7,3.0]},
        {"id":"UAQ0103","nombre":"Alejandro Reyes Ortiz","grupo":"PSI-201","semestre":4,
         "promedio":8.5,"riesgo":"bajo","dims":[4.1,4.0,4.2,3.9,4.3,3.8]},
    ]
    if risk_level:
        estudiantes = [e for e in estudiantes if e["riesgo"] == risk_level]
    return {"docente": current_user.nombre, "estudiantes": estudiantes, "total": len(estudiantes)}


@app.post("/api/v1/docente/observacion", tags=["Docente"], summary="Registrar observación")
async def docente_observacion(
    student_id: str, tipo: str, texto: str,
    current_user: UserDB = Depends(require_role("docente", "admin"))
):
    """Registra una observación del docente para un estudiante específico."""
    return {
        "status": "guardado",
        "student_id": student_id,
        "docente": current_user.nombre,
        "tipo": tipo,
        "timestamp": datetime.utcnow().isoformat()
    }


# ═══════════════════════════════════════════════════════════════════════════
# ENDPOINTS ADMINISTRADOR (rol requerido: admin)
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/api/v1/admin/overview", tags=["Administrador"], summary="Vista general institucional")
async def admin_overview(current_user: UserDB = Depends(require_role("admin"))):
    """Retorna estadísticas institucionales agregadas para el panel de administración."""
    return {
        "admin": current_user.nombre,
        "timestamp": datetime.utcnow().isoformat(),
        "resumen": {
            "total_estudiantes": 4332,
            "riesgo_alto": 521,
            "riesgo_medio": 1147,
            "riesgo_bajo": 2664,
            "tasa_desercion_estimada": 0.181,
            "modelo_version": "stacking_v2.0",
            "modelo_auc_roc": 0.9639,
        },
        "por_facultad": [
            {"nombre":"DTE",        "n":412,  "alto":62, "deser_pct":19.2},
            {"nombre":"Ingeniería", "n":1204, "alto":189,"deser_pct":19.6},
            {"nombre":"Psicología", "n":521,  "alto":48, "deser_pct":11.7},
            {"nombre":"Medicina",   "n":634,  "alto":94, "deser_pct":19.2},
            {"nombre":"Derecho",    "n":487,  "alto":71, "deser_pct":17.9},
        ]
    }


@app.get("/api/v1/admin/users", tags=["Administrador"], summary="Gestión de usuarios")
async def admin_users(
    role: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: UserDB = Depends(require_role("admin"))
):
    """Lista todos los usuarios del sistema (filtrable por rol)."""
    query = db.query(UserDB)
    if role:
        query = query.filter(UserDB.role == role)
    users = query.all()
    return {
        "total": len(users),
        "users": [
            {
                "id": u.id, "email": u.email, "nombre": u.nombre,
                "role": u.role, "created_at": u.created_at.isoformat(),
                "last_login": u.last_login.isoformat() if u.last_login else None,
                "extra_data": json.loads(u.extra_data or "{}")
            }
            for u in users
        ]
    }


@app.delete("/api/v1/admin/users/{user_id}", tags=["Administrador"], summary="Eliminar usuario")
async def admin_delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: UserDB = Depends(require_role("admin"))
):
    """Elimina un usuario del sistema (solo Administrador)."""
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="No puedes eliminar tu propia cuenta")
    user = db.query(UserDB).filter(UserDB.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    db.delete(user)
    db.commit()
    return {"status": "eliminado", "user_id": user_id}


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
