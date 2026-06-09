# Instrucciones para ejecutar el análisis ML en Google Colab

## Requisitos previos
Google Colab ya tiene preinstalado: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `xgboost`, `shap`.
Solo necesitas instalar `imbalanced-learn`:

```python
!pip install imbalanced-learn -q
```

## Orden de ejecución

1. **`00_generar_datos_sinteticos.py`** — Genera el dataset de 500 estudiantes sintéticos
2. **`01_random_forest.py`** — Análisis individual Random Forest
3. **`02_xgboost.py`** — Análisis individual XGBoost
4. **`03_gradient_boosting.py`** — Análisis individual Gradient Boosting
5. **`04_red_neuronal_fnn.py`** — Análisis individual FNN
6. **`05_stacking_ensemble.py`** — Integración final del Stacking Ensemble

## Cómo subir los archivos a Colab

```python
# Opción 1: Subir desde tu computadora
from google.colab import files
uploaded = files.upload()

# Opción 2: Montar Google Drive
from google.colab import drive
drive.mount('/content/drive')
```

## Estructura de resultados generados

```
resultados/
├── RF/
│   ├── RF_resultados.png        # Curva ROC + Matriz confusión + Importancias
│   ├── RF_shap_summary.png      # SHAP Summary Plot
│   └── RF_metricas.csv          # Tabla de métricas (media ± std)
├── XGBoost/
│   ├── XGB_resultados.png
│   ├── XGB_shap_summary.png
│   └── XGB_metricas.csv
├── GB/
│   ├── GB_resultados.png
│   ├── GB_shap_summary.png
│   └── GB_metricas.csv
├── FNN/
│   ├── FNN_resultados.png
│   └── FNN_metricas.csv
└── Stacking/
    ├── Stacking_resultados_finales.png   # Figura principal para tesis
    ├── Stacking_shap_summary.png         # Interpretabilidad global
    ├── Stacking_metricas.csv
    └── tabla_comparativa_modelos.csv     # Tabla comparativa final
```

## Métricas reportadas por modelo

| Métrica | Descripción |
|---------|-------------|
| Accuracy | Proporción de predicciones correctas |
| Precision | De los identificados como desertores, ¿cuántos lo son? |
| **Recall** | De los desertores reales, ¿cuántos detecta el modelo? *(métrica principal)* |
| F1-Score | Media armónica de Precision y Recall |
| **AUC-ROC** | Área bajo la curva ROC *(métrica de comparación principal)* |

## Nota sobre datos reales

Cuando se disponga de datos reales:
1. Exportar la base de datos del instrumento a CSV
2. Asegurarse de que las columnas coincidan con las del dataset sintético
3. Reemplazar la llamada a `pd.read_csv("dataset_ml_encoded.csv")` con el CSV real
4. Ejecutar el análisis desde el script `01_random_forest.py`

## Referencia APA 6ª edición (para citar este trabajo)

Pardo [Luis], [Apellido]. ([Año]). Modelos predictivos basados en aprendizaje 
automático para la predicción de deserción escolar universitaria: Un enfoque 
de Stacking Ensemble con variables psicoemocionales [Tesis doctoral]. 
Universidad Autónoma de Querétaro.
