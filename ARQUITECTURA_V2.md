# Sistema Atenea AI — Arquitectura v2.0
> Tesis Doctoral DTE-UAQ | Luis Francisco Pardo Perea | 2026

## Evolución del sistema

| Versión | Usuarios | Módulos | Datos |
|---------|----------|---------|-------|
| v1.0 (MVP) | Orientadores | Predicción + SHAP + Ficha | N=500 sintético |
| **v2.0** | **Orientadores + Estudiantes** | **+ Portal Estudiante + Estilos + Neurodiv. + Longitudinal** | **Real (Kaggle + UAQ)** |

---

## Diagrama de capas

```
╔══════════════════════════════════════════════════════════════════════╗
║                      SISTEMA ATENEA AI v2.0                         ║
╠═════════════════════════════╦════════════════════════════════════════╣
║   PORTAL ESTUDIANTE          ║   PORTAL ORIENTADOR / INSTITUCIÓN      ║
║                              ║                                        ║
║  ┌─────────────────────┐    ║   ┌──────────────────────────────────┐ ║
║  │ Mi Dashboard        │    ║   │ Lista de riesgo por grupo        │ ║
║  │ · Radar 6 dims      │    ║   │ Ficha Atenea AI individual       │ ║
║  │ · Historial temporal│    ║   │ SHAP waterfall por estudiante    │ ║
║  │ · Alertas propias   │    ║   │ Resumen estadístico UAQ          │ ║
║  └─────────────────────┘    ║   └──────────────────────────────────┘ ║
║  ┌─────────────────────┐    ║                                        ║
║  │ Mi Perfil Aprendizaje│   ║                                        ║
║  │ · Tipo VARK         │    ║                                        ║
║  │ · Estrategias recom.│    ║                                        ║
║  │ · Recursos UAQ      │    ║                                        ║
║  └─────────────────────┘    ║                                        ║
║  ┌─────────────────────┐    ║                                        ║
║  │ Screening Neurodiv. │    ║                                        ║
║  │ · Indicadores riesgo│    ║                                        ║
║  │ · No diagnóstico    │    ║                                        ║
║  │ · Derivación clínica│    ║                                        ║
║  └─────────────────────┘    ║                                        ║
╠═════════════════════════════╩════════════════════════════════════════╣
║                          FastAPI Backend (main.py)                   ║
║                                                                      ║
║  ┌──────────┐  ┌───────────────┐  ┌─────────────┐  ┌────────────┐  ║
║  │ Auth     │  │ Instrumento   │  │  Predicción │  │  SHAP/XAI  │  ║
║  │ JWT      │  │ 80 ítems      │  │  Stacking   │  │  Engine    │  ║
║  │ /token   │  │ /instrument   │  │  /predict   │  │  /shap     │  ║
║  └──────────┘  └───────────────┘  └─────────────┘  └────────────┘  ║
║  ┌──────────┐  ┌───────────────┐  ┌─────────────┐  ┌────────────┐  ║
║  │ Perfil   │  │  Screening    │  │  Monitoreo  │  │  Dashboard │  ║
║  │ Aprend.  │  │  Neurodiv.    │  │  Temporal   │  │  Instituc. │  ║
║  │/learning │  │ /screening    │  │  /history   │  │ /dashboard │  ║
║  └──────────┘  └───────────────┘  └─────────────┘  └────────────┘  ║
╠══════════════════════════════════════════════════════════════════════╣
║                        ML Pipeline                                   ║
║                                                                      ║
║  00_generar_datos_sinteticos.py  → dataset sintético (base)          ║
║  00b_integrar_datos_kaggle.py    → dataset real (UCI + Kaggle) ← NEW ║
║  01_random_forest.py                                                  ║
║  02_xgboost.py                                                        ║
║  03_gradient_boosting.py                                              ║
║  04_red_neuronal_fnn.py                                               ║
║  05_stacking_ensemble.py         → modelo final exportado             ║
║  06_perfil_aprendizaje.py        → motor estilos VARK          ← NEW ║
║  07_screening_neurodivergencia.py→ indicadores neurodiv.       ← NEW ║
║  08_monitoreo_longitudinal.py    → análisis temporal           ← NEW ║
╠══════════════════════════════════════════════════════════════════════╣
║                     Base de Datos                                    ║
║                                                                      ║
║  SQLite (desarrollo) / PostgreSQL (producción)                       ║
║  Tablas:                                                             ║
║    estudiantes       — perfil base + credenciales                    ║
║    aplicaciones      — cada vez que se aplica el instrumento         ║
║    respuestas_items  — respuestas individuales (ítem x ítem)         ║
║    predicciones      — historial de predicciones ML                  ║
║    perfiles_aprend.  — perfil VARK y estrategias                     ║
║    screening_nd      — indicadores de neurodivergencia               ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## Nuevos módulos — descripción

### 1. Portal Estudiante
El estudiante puede:
- Completar el instrumento de forma autónoma (no requiere orientador)
- Ver su radar psicoemocional con las 6 dimensiones
- Ver su historial longitudinal (evolución por semestre)
- Ver su perfil de aprendizaje (tipo VARK + estrategias)
- Ver indicadores de screening (si aplica) con mensaje de derivación clínica

### 2. Motor de Estilos de Aprendizaje (`06_perfil_aprendizaje.py`)
- Basado en modelo **VARK** (Visual, Auditory, Read/Write, Kinesthetic)
- Mapeo desde dimensiones SIP:
  - D2 Autopercepción → metacognición → perfil reflexivo/activo
  - D3 Motivación → preferencias de reto → convergente/divergente
  - D5 Relaciones → aprendizaje social → collaborative/individual
- Output: perfil VARK + 5 estrategias de estudio personalizadas + recursos UAQ

### 3. Screening de Neurodivergencias (`07_screening_neurodivergencia.py`)
> **IMPORTANTE: No diagnóstico clínico. Solo indicadores de riesgo para derivación.**

Indicadores rastreados:
| Tipo | Instrumento proxy | Ítems SIP relevantes |
|------|-------------------|----------------------|
| TDAH | Patrón de motivación + resiliencia | D3 ítems 15, 22, 28; D4 ítems 8, 14 |
| TEA (espectro) | Patrón de relaciones + afiliación | D5 ítems 3, 9, 17, 24; D6 ítems 7, 12 |
| Dislexia/procesar. | Autopercepción académica baja + promedio | D2 ítems 4, 11, 19 |

Output por estudiante:
- Score 0-100 por tipo (no diagnóstico, solo indicador)
- Si score > 70: mensaje "Considera consultar con servicio de apoyo UAQ"
- Nunca se etiqueta al estudiante directamente

### 4. Monitoreo Longitudinal (`08_monitoreo_longitudinal.py`)
- Comparar dimensiones entre aplicaciones (t1, t2, t3)
- Detectar **tendencia de declive** (alertas tempranas más precisas que snapshot)
- Generar curva de trayectoria emocional
- Identificar eventos críticos (caída brusca en dimensión específica)

---

## Roadmap de implementación

| Fase | Script | Prioridad | Semana |
|------|--------|-----------|--------|
| 1 | `00b_integrar_datos_kaggle.py` | 🔴 CRÍTICO | 1 |
| 1 | Reentrenamiento con datos reales (01-05) | 🔴 CRÍTICO | 1-2 |
| 2 | `06_perfil_aprendizaje.py` | 🟠 ALTA | 2 |
| 2 | `07_screening_neurodivergencia.py` | 🟠 ALTA | 2 |
| 3 | `08_monitoreo_longitudinal.py` | 🟡 MEDIA | 3 |
| 3 | Backend: endpoints student portal | 🟡 MEDIA | 3 |
| 4 | Frontend: portal_estudiante.html | 🟢 LARGA | 4 |
| 5 | Piloto UAQ (80-120 estudiantes reales) | 🔵 PILOTO | Mes 2-3 |
