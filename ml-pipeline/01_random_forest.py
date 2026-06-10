"""
═══════════════════════════════════════════════════════════════════════════════
ANÁLISIS 1/4 — RANDOM FOREST
Tesis Doctoral: Modelos Predictivos ML para la Deserción Escolar (UAQ-DTE)
═══════════════════════════════════════════════════════════════════════════════
Ejecutar DESPUÉS de: 00_generar_datos_sinteticos.py

Referencia principal:
  Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5–32.
  https://doi.org/10.1023/A:1010933404324
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os, warnings
warnings.filterwarnings('ignore')

# ── Importaciones ML ─────────────────────────────────────────────────────────
# Google Colab: ya instaladas. Local: pip install scikit-learn shap imbalanced-learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import shap

# ─── Configuración visual ────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi': 150
})
COLORES = {'no_desertor': '#2ecc71', 'desertor': '#e74c3c', 'rf': '#3498db', 'grid': '#ecf0f1'}
os.makedirs("resultados/RF", exist_ok=True)

# ─── 1. CARGA DE DATOS ───────────────────────────────────────────────────────
print("=" * 65)
print("ANÁLISIS 1/4 — RANDOM FOREST")
print("Tesis DTE-UAQ | Predicción de Deserción Escolar")
print("=" * 65)

df = pd.read_csv("dataset_real_encoded.csv")
X = df.drop(columns=["desercion"])
y = df["desercion"]

print(f"\n📊 Dataset: {X.shape[0]} estudiantes, {X.shape[1]} variables")
print(f"   Distribución clases: {dict(y.value_counts())}")

# ─── 2. PREPROCESAMIENTO ────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Balanceo con SMOTE
smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X_scaled, y)
print(f"\n🔄 Post-SMOTE: {dict(pd.Series(y_bal).value_counts())}")

# ─── 3. BÚSQUEDA DE HIPERPARÁMETROS ─────────────────────────────────────────
print("\n⚙️  Ajuste de hiperparámetros (GridSearchCV)...")
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'class_weight': ['balanced']
}
rf_base = RandomForestClassifier(random_state=42)
cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(rf_base, param_grid, cv=cv_inner, scoring='roc_auc', n_jobs=-1, verbose=0)
grid_search.fit(X_bal, y_bal)

best_params = grid_search.best_params_
print(f"   Mejores parámetros: {best_params}")
print(f"   AUC-ROC CV interno: {grid_search.best_score_:.4f}")

# ─── 4. VALIDACIÓN CRUZADA 10-FOLD ──────────────────────────────────────────
print("\n🔁 Validación cruzada 10-fold...")
rf_final = RandomForestClassifier(**best_params, random_state=42)
cv_outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
cv_results = cross_validate(rf_final, X_bal, y_bal, cv=cv_outer, scoring=scoring, return_train_score=False)

print("\n📈 RESULTADOS — RANDOM FOREST (Media ± Desviación Estándar, 10-fold CV)")
print("-" * 55)
metricas = {
    'Accuracy':  ('test_accuracy',  cv_results['test_accuracy']),
    'Precision': ('test_precision', cv_results['test_precision']),
    'Recall':    ('test_recall',    cv_results['test_recall']),
    'F1-Score':  ('test_f1',        cv_results['test_f1']),
    'AUC-ROC':   ('test_roc_auc',   cv_results['test_roc_auc']),
}
resultados_df = {}
for nombre, (key, vals) in metricas.items():
    media = vals.mean()
    std   = vals.std()
    print(f"   {nombre:<12}: {media:.4f} ± {std:.4f}")
    resultados_df[nombre] = {'Media': round(media, 4), 'Std': round(std, 4)}

# ─── 5. ENTRENAMIENTO FINAL Y MATRIZ DE CONFUSIÓN ────────────────────────────
X_train, X_test, y_train, y_test = (
    X_bal[:int(len(X_bal)*0.8)], X_bal[int(len(X_bal)*0.8):],
    y_bal[:int(len(y_bal)*0.8)], y_bal[int(len(y_bal)*0.8):]
)
rf_final.fit(X_train, y_train)
y_pred = rf_final.predict(X_test)
y_proba = rf_final.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, y_pred)
print(f"\n📊 Matriz de Confusión (conjunto de prueba):")
print(f"   TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")

# ─── 6. GRÁFICAS ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Random Forest — Predicción de Deserción Escolar\nUAQ DTE | Tesis Doctoral',
             fontsize=13, fontweight='bold', y=1.02)

# 6a. Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_proba)
auc_val = roc_auc_score(y_test, y_proba)
axes[0].plot(fpr, tpr, color=COLORES['rf'], lw=2, label=f'RF (AUC = {auc_val:.3f})')
axes[0].plot([0,1],[0,1], 'k--', lw=1)
axes[0].fill_between(fpr, tpr, alpha=0.15, color=COLORES['rf'])
axes[0].set_title('Curva ROC')
axes[0].set_xlabel('Tasa de Falsos Positivos')
axes[0].set_ylabel('Tasa de Verdaderos Positivos')
axes[0].legend()
axes[0].set_xlim([0, 1]); axes[0].set_ylim([0, 1.02])

# 6b. Matriz de confusión
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
            xticklabels=['No Deserta', 'Deserta'],
            yticklabels=['No Deserta', 'Deserta'],
            annot_kws={'size': 14, 'weight': 'bold'})
axes[1].set_title('Matriz de Confusión')
axes[1].set_ylabel('Etiqueta Real')
axes[1].set_xlabel('Etiqueta Predicha')

# 6c. Importancia de características (top 10)
importancias = rf_final.feature_importances_
feat_names = X.columns.tolist()
top_idx = np.argsort(importancias)[::-1][:10]
top_feats = [feat_names[i] for i in top_idx]
top_imp   = importancias[top_idx]
axes[2].barh(range(10), top_imp[::-1], color=COLORES['rf'], edgecolor='white')
axes[2].set_yticks(range(10))
axes[2].set_yticklabels([f.replace('dim_', '').replace('_', ' ').title() for f in top_feats[::-1]], fontsize=9)
axes[2].set_title('Top 10 Variables Importantes')
axes[2].set_xlabel('Importancia (Gini)')

plt.tight_layout()
plt.savefig('resultados/RF/RF_resultados.png', bbox_inches='tight', dpi=150)
print("\n📊 Figura guardada en: resultados/RF/RF_resultados.png")

# ─── 7. ANÁLISIS SHAP ───────────────────────────────────────────────────────
print("\n🔍 Calculando valores SHAP...")
explainer = shap.TreeExplainer(rf_final)
shap_raw = explainer.shap_values(X_test[:100])  # Muestra de 100 para velocidad
# Compatibilidad shap >= 0.40: puede devolver ndarray 3D (n, features, classes)
# o lista [class0, class1]. Extraer valores para la clase positiva (deserción=1)
if isinstance(shap_raw, np.ndarray) and shap_raw.ndim == 3:
    shap_values = shap_raw[:, :, 1]
elif isinstance(shap_raw, list):
    shap_values = shap_raw[1]
else:
    shap_values = shap_raw

fig_shap, ax_shap = plt.subplots(1, 1, figsize=(10, 7))
shap.summary_plot(shap_values, X_test[:100], feature_names=feat_names,
                  show=False, max_display=12)
plt.title('SHAP Summary Plot — Random Forest\nContribución de cada variable a la predicción de deserción',
          fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('resultados/RF/RF_shap_summary.png', bbox_inches='tight', dpi=150)
print("📊 SHAP Summary guardado en: resultados/RF/RF_shap_summary.png")

# ─── 8. GUARDAR RESULTADOS ──────────────────────────────────────────────────
pd.DataFrame(resultados_df).T.to_csv('resultados/RF/RF_metricas.csv')
print("\n✅ Análisis Random Forest completado.")
print(f"   AUC-ROC: {cv_results['test_roc_auc'].mean():.4f}")
print(f"   F1-Score: {cv_results['test_f1'].mean():.4f}")
print(f"   Recall (sensibilidad): {cv_results['test_recall'].mean():.4f}")
print("\n➡️  Siguiente análisis: 02_xgboost.py")
