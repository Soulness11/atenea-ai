"""
═══════════════════════════════════════════════════════════════════════════════
ANÁLISIS 2/4 — XGBOOST
Tesis Doctoral: Modelos Predictivos ML para la Deserción Escolar (UAQ-DTE)
═══════════════════════════════════════════════════════════════════════════════
Ejecutar DESPUÉS de: 01_random_forest.py

Referencia principal:
  Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system.
  ACM SIGKDD. https://doi.org/10.1145/2939672.2939785
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os, warnings
warnings.filterwarnings('ignore')

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import shap

plt.rcParams.update({'font.family': 'serif', 'font.size': 11, 'figure.dpi': 150})
COLORES = {'xgb': '#e67e22'}
os.makedirs("resultados/XGBoost", exist_ok=True)

# ─── 1. CARGA ────────────────────────────────────────────────────────────────
print("=" * 65)
print("ANÁLISIS 2/4 — XGBOOST")
print("Tesis DTE-UAQ | Predicción de Deserción Escolar")
print("=" * 65)

df = pd.read_csv("dataset_ml_encoded.csv")
X = df.drop(columns=["desercion"])
y = df["desercion"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X_scaled, y)

# ─── 2. BÚSQUEDA DE HIPERPARÁMETROS ─────────────────────────────────────────
print("\n⚙️  Ajuste de hiperparámetros XGBoost...")
scale_pos_weight = (y_bal == 0).sum() / (y_bal == 1).sum()

param_grid_xgb = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
}
xgb_base = XGBClassifier(
    use_label_encoder=False, eval_metric='logloss',
    scale_pos_weight=scale_pos_weight, random_state=42
)
cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(xgb_base, param_grid_xgb, cv=cv_inner, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_bal, y_bal)

best_params = grid_search.best_params_
print(f"   Mejores parámetros: {best_params}")
print(f"   AUC-ROC CV interno: {grid_search.best_score_:.4f}")

# ─── 3. VALIDACIÓN CRUZADA 10-FOLD ──────────────────────────────────────────
print("\n🔁 Validación cruzada 10-fold...")
xgb_final = XGBClassifier(
    **best_params, use_label_encoder=False, eval_metric='logloss',
    scale_pos_weight=scale_pos_weight, random_state=42
)
cv_outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_results = cross_validate(
    xgb_final, X_bal, y_bal, cv=cv_outer,
    scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
)

print("\n📈 RESULTADOS — XGBOOST (10-fold CV)")
print("-" * 50)
metricas_nombres = {
    'Accuracy': 'test_accuracy', 'Precision': 'test_precision',
    'Recall': 'test_recall', 'F1-Score': 'test_f1', 'AUC-ROC': 'test_roc_auc'
}
resultados_df = {}
for nombre, key in metricas_nombres.items():
    vals = cv_results[key]
    print(f"   {nombre:<12}: {vals.mean():.4f} ± {vals.std():.4f}")
    resultados_df[nombre] = {'Media': round(vals.mean(), 4), 'Std': round(vals.std(), 4)}

# ─── 4. ENTRENAMIENTO FINAL ──────────────────────────────────────────────────
split = int(len(X_bal) * 0.8)
X_train, X_test = X_bal[:split], X_bal[split:]
y_train, y_test = y_bal[:split], y_bal[split:]

xgb_final.fit(X_train, y_train)
y_pred = xgb_final.predict(X_test)
y_proba = xgb_final.predict_proba(X_test)[:, 1]
cm = confusion_matrix(y_test, y_pred)

# ─── 5. GRÁFICAS ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('XGBoost — Predicción de Deserción Escolar\nUAQ DTE | Tesis Doctoral', fontsize=13, fontweight='bold')

# Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_proba)
auc_val = roc_auc_score(y_test, y_proba)
axes[0].plot(fpr, tpr, color=COLORES['xgb'], lw=2, label=f'XGBoost (AUC = {auc_val:.3f})')
axes[0].plot([0,1],[0,1], 'k--', lw=1)
axes[0].fill_between(fpr, tpr, alpha=0.15, color=COLORES['xgb'])
axes[0].set_title('Curva ROC'); axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR')
axes[0].legend()

# Matriz confusión
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=axes[1],
            xticklabels=['No Deserta', 'Deserta'],
            yticklabels=['No Deserta', 'Deserta'],
            annot_kws={'size': 14, 'weight': 'bold'})
axes[1].set_title('Matriz de Confusión')

# Importancia XGBoost (gain)
feat_names = X.columns.tolist()
importancias = xgb_final.feature_importances_
top_idx = np.argsort(importancias)[::-1][:10]
top_feats = [feat_names[i] for i in top_idx]
axes[2].barh(range(10), importancias[top_idx][::-1], color=COLORES['xgb'], edgecolor='white')
axes[2].set_yticks(range(10))
axes[2].set_yticklabels([f.replace('dim_', '').replace('_', ' ').title() for f in top_feats[::-1]], fontsize=9)
axes[2].set_title('Top 10 Variables — XGBoost')
axes[2].set_xlabel('Importancia (F-score)')

plt.tight_layout()
plt.savefig('resultados/XGBoost/XGB_resultados.png', bbox_inches='tight', dpi=150)

# ─── 6. SHAP ─────────────────────────────────────────────────────────────────
print("\n🔍 Calculando valores SHAP (TreeExplainer)...")
explainer = shap.TreeExplainer(xgb_final)
shap_values = explainer.shap_values(X_test[:100])

fig2, ax2 = plt.subplots(figsize=(10, 7))
shap.summary_plot(shap_values, X_test[:100], feature_names=feat_names, show=False, max_display=12)
plt.title('SHAP Summary Plot — XGBoost', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('resultados/XGBoost/XGB_shap_summary.png', bbox_inches='tight', dpi=150)

pd.DataFrame(resultados_df).T.to_csv('resultados/XGBoost/XGB_metricas.csv')

print("\n✅ Análisis XGBoost completado.")
print(f"   AUC-ROC: {cv_results['test_roc_auc'].mean():.4f}")
print(f"   F1-Score: {cv_results['test_f1'].mean():.4f}")
print("\n➡️  Siguiente análisis: 03_gradient_boosting.py")
