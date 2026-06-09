# ─── Build Stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && rm -rf /var/lib/apt/lists/*

# Instalar dependencias Python
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código backend
COPY backend/ .

# Copiar modelo entrenado (generado por 05_stacking_ensemble.py)
# COPY ML_Analysis/stacking_model.pkl .
# COPY ML_Analysis/scaler.pkl .

# ─── Frontend (archivos estáticos) ────────────────────────────────────────────
COPY app.html /app/static/index.html

# Servir frontend desde FastAPI
RUN pip install --no-cache-dir aiofiles==23.2.1

# Montar archivos estáticos en FastAPI (agregar al main.py):
# from fastapi.staticfiles import StaticFiles
# app.mount("/", StaticFiles(directory="static", html=True), name="static")

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
