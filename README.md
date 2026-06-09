# Atenea AI — Sistema Instrumental Psicotécnico para la Detección Temprana del Abandono Escolar

**Autor:** Luis Francisco Pardo Pera  
**Institución:** Universidad Autónoma de Querétaro (UAQ) — Doctorado en Tecnología Educativa  
**Estado:** MVP con datos sintéticos (N=500) | Piloto UAQ en diseño  

---

## ¿Qué es Atenea AI?

Atenea AI es la instancia computacional del **SIP (Sistema Instrumental Psicotécnico)**: un sistema de detección temprana del riesgo de abandono escolar en educación superior que combina:

- **165 ítems psicométricos** en 6 dimensiones psicoemocionales
- **Stacking Ensemble** (RF + XGBoost + Gradient Boosting → FNN meta-learner)
- **XAI/SHAP** en dos niveles (clasificadores base + meta-learner)
- **Ciclo CIR** (Comprensión → Intervención → Retroalimentación) como esquema de mediación pedagógica

El sistema no solo predice — **media** la intervención del orientador mediante la Ficha Atenea AI.

---

## Métricas MVP (datos sintéticos N=500)

| Métrica | Valor |
|---|---|
| AUC-ROC | 98.5% |
| Accuracy | 94% |
| F1-score | 75% |
| Recall (clase riesgo) | 69% |

---

## Estructura del repositorio

```
atenea-ai-github/
├── README.md                         ← Este archivo
│
├── ml-pipeline/                      ← Pipeline completo de ML
│   ├── 00_generar_datos_sinteticos.py
│   ├── 01_random_forest.py
│   ├── 02_xgboost.py
│   ├── 03_gradient_boosting.py
│   ├── 04_red_neuronal_fnn.py
│   ├── 05_stacking_ensemble.py
│   └── INSTRUCCIONES_COLAB.md        ← Cómo ejecutar en Google Colab
│
├── prototipo/                        ← Prototipo funcional (Docker)
│   ├── Dockerfile
│   ├── app.html                      ← Interfaz web
│   └── backend/
│       ├── main.py                   ← API FastAPI
│       └── requirements.txt
│
├── visualizaciones/                  ← Diagramas interactivos HTML
│   ├── Diagrama_Stacking_Ensemble.html
│   ├── SIP_Modelo_Cibernetico.html
│   ├── PRISMA_Diagrama_CartografiaSIP.html
│   └── Dashboard_Tesis.html
│
├── tesis/                            ← Documento de tesis
│   └── Tesis DTE Luis Pardo MEJORADA.docx
│
└── docs/                             ← Documentación técnica
    ├── ML_Desercion_Variables_Psicoemocionales.md
    ├── Referencias_ML_Desercion_Escolar.md
    └── Literatura Marco Teórico - Referencias.md
```

---

## Pipeline ML — Cómo ejecutar

### Opción 1: Google Colab (recomendado)
Ver `ml-pipeline/INSTRUCCIONES_COLAB.md`

### Opción 2: Local

```bash
pip install scikit-learn xgboost tensorflow shap pandas numpy matplotlib
python ml-pipeline/00_generar_datos_sinteticos.py
python ml-pipeline/05_stacking_ensemble.py
```

### Opción 3: Docker (prototipo completo)

```bash
cd prototipo/
docker build -t atenea-ai .
docker run -p 8000:8000 atenea-ai
# Abrir app.html en el navegador
```

---

## Marco teórico

El SIP se fundamenta en:

- **Rabardel & Béguin (2000)** — Génesis instrumental: artefacto + esquemas de utilización = instrumento
- **Engeström (2001)** — Teoría de la Actividad (4a generación): sistemas de actividad en red
- **Braidotti (2019)** — Posthumanismo crítico: potentia vs. potestas
- **Tobón (2012)** — Cartografía conceptual (8 ejes)
- **Pardo Pera (2023)** — Modelo Ecológico de Autoconciencia (tesis de maestría)

---

## Contacto

**Luis Francisco Pardo Pera**  
Doctorado en Tecnología Educativa — UAQ  
soulness356@gmail.com

---

> *"El SIP no predice el abandono — activa la capacidad del orientador de comprenderlo y actuar sobre él."*  
> — Pardo Pera (2026)
