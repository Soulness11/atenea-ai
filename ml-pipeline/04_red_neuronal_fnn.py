"""
═══════════════════════════════════════════════════════════════════════════════
ANÁLISIS 4/4 — RED NEURONAL FEEDFORWARD (FNN) como meta-aprendiz
Tesis Doctoral: Modelos Predictivos ML para la Deserción Escolar (UAQ-DTE)
═══════════════════════════════════════════════════════════════════════════════
Referencias principales:
  Rumelhart, D.E., Hinton, G.E., & Williams, R.J. (1986). Nature, 323, 533–536.
  Srivastava, N. et al. (2014). Dropout. JMLR, 15(1), 1929–1958.
  Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.

NOTA: Este script usa sklearn MLPClassifier (red neuronal sin framework pesado).
Para la fase de Stacking, la FNN actúa como meta-aprendiz sobre las predicciones
de RF, XGBoost y GB.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os, warnings
warnings.filterwarnings('ignore')

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

plt.rcParams.update({'font.family': 'serif', 'font.size': 11, 'figure.dpi': 150})
os.makedirs("resultados/FNN", exist_ok=True)

print("=" * 65)
print("ANÁLISIS 4/4 — RED NEURONAL FEEDFORWARD (FNN)")
print("Arquitectura: entrada → 128 → 64 → 32 → salida (sigmoid)")
print("Tesis DTE-UAQ | Predicción de Deserción Escolar")
print("=" * 65)

df = pd.read_csv("dataset_ml_encoded.csv")
X = df.drop(columns=["desercion"])
y = df["desercion"]
feat_names = X.columns.tolist()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X_scaled, y)

# ─── Hiperparámetros ──────────────────────────────────────────────────────────
print("\n⚙️  Ajuste de hiperparámetros FNN...")
# Arquitecturas a probar: inspiradas en la sección 3.17 del Marco Teórico
param_grid_fnn = {
    'hidden_layer_sizes': [(128, 64, 32), (64, 32), (128, 64)],
    'alpha': [0.0001, 0.001, 0.01],   # L2 regularización (equivalente a dropout conceptualmente)
    'learning_rate_init': [0.001, 0.0005],
    'max_iter': [300]
}
fnn_base = MLPClassifier(
    activation='relu',
    solver='adam',
    early_stopping=True,
    validation_fraction=0.15,
    random_state=42
)
cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(fnn_base, param_grid_fnn, cv=cv_inner, scoring='roc_auc', n_jobs=-1)
grid.fit(X_bal, y_bal)
best_params = grid.best_params_
print(f"   Mejores parámetros: {best_params}")
print(f"   AUC-ROC CV interno: {grid.best_score_:.4f}")

# ─── Validación cruzada 10-fold ───────────────────────────────────────────────
print("\n🔁 Validación cruzada 10-fold...")
fnn_final = MLPClassifier(
    **best_params, activation='relu', solver='adam',
    early_stopping=True, validation_fraction=0.15, random_state=42
)
cv_outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_results = cross_validate(
    fnn_final, X_bal, y_bal, cv=cv_outer,
    scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
)

print("\n📈 RESULTADOS — FNN (10-fold CV)")
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
fnn_final.fit(X_train, y_train)
y_pred  = fnn_final.predict(X_test)
y_proba = fnn_final.predict_proba(X_test)[:, 1]
cm = confusion_matrix(y_test, y_pred)

# ─── Curva de pérdida durante el entrenamiento ────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Red Neuronal FNN — Predicción de Deserción Escolar\nUAQ DTE | Tesis Doctoral',
             fontsize=13, fontweight='bold')

# Curva de loss
if hasattr(fnn_final, 'loss_curve_'):
    axes[0].plot(fnn_final.loss_curve_, color='#1abc9c', lw=2, label='Pérdida entrenamiento')
    if hasattr(fnn_final, 'validation_scores_'):
        axes[0].plot(fnn_final.validation_scores_, color='#e74c3c', lw=2, ls='--', label='AUC validación')
    axes[0].set_title('Curva de Aprendizaje')
    axes[0].set_xlabel('Iteraciones (épocas)')
    axes[0].set_ylabel('Pérdida (cross-entropy)')
    axes[0].legend()
else:
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_val = roc_auc_score(y_test, y_proba)
    axes[0].plot(fpr, tpr, color='#1abc9c', lw=2, label=f'FNN (AUC = {auc_val:.3f})')
    axes[0].plot([0,1],[0,1],'k--',lw=1)
    axes[0].fill_between(fpr, tpr, alpha=0.15, color='#1abc9c')
    axes[0].set_title('Curva ROC'); axes[0].legend()

# Matriz de confusión
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=['No Deserta', 'Deserta'],
            yticklabels=['No Deserta', 'Deserta'],
            annot_kws={'size': 14, 'weight': 'bold'})
axes[1].set_title('Matriz de Confusión')

# Arquitectura esquemática
axes[2].axis('off')
n_input  = len(feat_names)
capas = [n_input, 128, 64, 32, 1]
nombres_capas = ['Entrada\n(variables\npsicoemocionales\n+ académicas)',
                 'Capa 1\nReLU\n128 neuronas',
                 'Capa 2\nReLU\n64 neuronas',
                 'Capa 3\nReLU\n32 neuronas',
                 'Salida\nSigmoid\n(riesgo 0–1)']
x_pos = np.linspace(0.05, 0.95, len(capas))
for i, (x, n, nombre) in enumerate(zip(x_pos, capas, nombres_capas)):
    n_show = min(n, 6)
    y_positions = np.linspace(0.15, 0.85, n_show)
    for y in y_positions:
        color = '#1abc9c' if i == 0 else ('#e74c3c' if i == len(capas)-1 else '#3498db')
        circle = plt.Circle((x, y), 0.025, color=color, fill=True, alpha=0.8)
        axes[2].add_patch(circle)
    if n > 6:
        axes[2].text(x, 0.08, f'...{n}...', ha='center', fontsize=7, color='gray')
    axes[2].text(x, 0.00, nombre, ha='center', fontsize=7, wrap=True)
    if i < len(capas) - 1:
        axes[2].annotate('', xy=(x_pos[i+1]-0.03, 0.5), xytext=(x+0.03, 0.5),
                         arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
axes[2].set_xlim(0, 1); axes[2].set_ylim(-0.1, 1.0)
axes[2].set_title('Arquitectura FNN (meta-aprendiz)', fontsize=10)

plt.tight_layout()
plt.savefig('resultados/FNN/FNN_resultados.png', bbox_inches='tight', dpi=150)

# ─── Guardar predicciones probabilísticas para Stacking ──────────────────────
# Estas predicciones son el "input" que la FNN meta-aprendiz recibirá en el stacking
pd.DataFrame({
    'y_true': y_test,
    'fnn_proba': y_proba,
    'fnn_pred': y_pred
}).to_csv('resultados/FNN/FNN_predicciones.csv', index=False)

pd.DataFrame(resultados_df).T.to_csv('resultados/FNN/FNN_metricas.csv')

print("\n✅ Análisis FNN completado.")
print(f"   AUC-ROC: {cv_results['test_roc_auc'].mean():.4f}")
print(f"   F1-Score: {cv_results['test_f1'].mean():.4f}")
print("\n➡️  Siguiente análisis: 05_stacking_ensemble.py")
print("   (Integración final: RF + XGBoost + GB → FNN meta-aprendiz)")
