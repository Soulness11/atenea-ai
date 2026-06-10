"""
═══════════════════════════════════════════════════════════════════════════════
ANÁLISIS 3/4 — GRADIENT BOOSTING (sklearn GBM)
Tesis Doctoral: Modelos Predictivos ML para la Deserción Escolar (UAQ-DTE)
═══════════════════════════════════════════════════════════════════════════════
Referencia principal:
  Friedman, J. H. (2001). Greedy function approximation: A gradient boosting
  machine. Annals of Statistics, 29(5), 1189–1232.
  https://doi.org/10.1214/aos/1013203451
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os, warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import shap

plt.rcParams.update({'font.family': 'serif', 'font.size': 11, 'figure.dpi': 150})
os.makedirs("resultados/GB", exist_ok=True)

print("=" * 65)
print("ANÁLISIS 3/4 — GRADIENT BOOSTING")
print("Tesis DTE-UAQ | Predicción de Deserción Escolar")
print("=" * 65)

df = pd.read_csv("dataset_real_encoded.csv")
X = df.drop(columns=["desercion"])
y = df["desercion"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X_scaled, y)

# ─── Hiperparámetros ──────────────────────────────────────────────────────────
print("\n⚙️  Ajuste de hiperparámetros GB...")
param_grid_gb = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 4, 5],
    'subsample': [0.8, 1.0],
    'min_samples_split': [2, 5]
}
gb_base = GradientBoostingClassifier(random_state=42)
cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(gb_base, param_grid_gb, cv=cv_inner, scoring='roc_auc', n_jobs=-1)
grid.fit(X_bal, y_bal)
best_params = grid.best_params_
print(f"   Mejores parámetros: {best_params}")

# ─── Validación cruzada 10-fold ───────────────────────────────────────────────
print("\n🔁 Validación cruzada 10-fold...")
gb_final = GradientBoostingClassifier(**best_params, random_state=42)
cv_outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_results = cross_validate(
    gb_final, X_bal, y_bal, cv=cv_outer,
    scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
)

print("\n📈 RESULTADOS — GRADIENT BOOSTING (10-fold CV)")
print("-" * 50)
nombres = {'Accuracy': 'test_accuracy', 'Precision': 'test_precision',
           'Recall': 'test_recall', 'F1-Score': 'test_f1', 'AUC-ROC': 'test_roc_auc'}
resultados_df = {}
for nombre, key in nombres.items():
    vals = cv_results[key]
    print(f"   {nombre:<12}: {vals.mean():.4f} ± {vals.std():.4f}")
    resultados_df[nombre] = {'Media': round(vals.mean(), 4), 'Std': round(vals.std(), 4)}

# ─── Entrenamiento final ──────────────────────────────────────────────────────
split = int(len(X_bal) * 0.8)
X_train, X_test = X_bal[:split], X_bal[split:]
y_train, y_test = y_bal[:split], y_bal[split:]
gb_final.fit(X_train, y_train)
y_pred  = gb_final.predict(X_test)
y_proba = gb_final.predict_proba(X_test)[:, 1]
cm = confusion_matrix(y_test, y_pred)

# ─── Gráficas ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Gradient Boosting — Predicción de Deserción Escolar\nUAQ DTE | Tesis Doctoral',
             fontsize=13, fontweight='bold')

fpr, tpr, _ = roc_curve(y_test, y_proba)
auc_val = roc_auc_score(y_test, y_proba)
axes[0].plot(fpr, tpr, color='#9b59b6', lw=2, label=f'GB (AUC = {auc_val:.3f})')
axes[0].plot([0,1],[0,1],'k--',lw=1)
axes[0].fill_between(fpr, tpr, alpha=0.15, color='#9b59b6')
axes[0].set_title('Curva ROC'); axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR')
axes[0].legend()

sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=axes[1],
            xticklabels=['No Deserta','Deserta'], yticklabels=['No Deserta','Deserta'],
            annot_kws={'size': 14, 'weight': 'bold'})
axes[1].set_title('Matriz de Confusión')

feat_names = X.columns.tolist()
importancias = gb_final.feature_importances_
top_idx = np.argsort(importancias)[::-1][:10]
axes[2].barh(range(10), importancias[top_idx][::-1], color='#9b59b6', edgecolor='white')
axes[2].set_yticks(range(10))
axes[2].set_yticklabels([feat_names[i].replace('dim_','').replace('_',' ').title()
                         for i in top_idx[::-1]], fontsize=9)
axes[2].set_title('Top 10 Variables — GB'); axes[2].set_xlabel('Importancia')
plt.tight_layout()
plt.savefig('resultados/GB/GB_resultados.png', bbox_inches='tight', dpi=150)

# ─── SHAP ─────────────────────────────────────────────────────────────────────
print("\n🔍 Calculando valores SHAP...")
explainer = shap.TreeExplainer(gb_final)
shap_values = explainer.shap_values(X_test[:100])
fig2, _ = plt.subplots(figsize=(10, 7))
shap.summary_plot(shap_values, X_test[:100], feature_names=feat_names, show=False, max_display=12)
plt.title('SHAP Summary Plot — Gradient Boosting', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('resultados/GB/GB_shap_summary.png', bbox_inches='tight', dpi=150)

pd.DataFrame(resultados_df).T.to_csv('resultados/GB/GB_metricas.csv')
print("\n✅ Análisis Gradient Boosting completado.")
print(f"   AUC-ROC: {cv_results['test_roc_auc'].mean():.4f}")
print("\n➡️  Siguiente análisis: 04_red_neuronal.py")
