"""
═══════════════════════════════════════════════════════════════════════════════
ANÁLISIS 5/5 — STACKING ENSEMBLE COMPLETO
Arquitectura: RF + XGBoost + Gradient Boosting → FNN (meta-aprendiz)
Tesis Doctoral: Modelos Predictivos ML para la Deserción Escolar (UAQ-DTE)
═══════════════════════════════════════════════════════════════════════════════
Referencias:
  Wolpert, D.H. (1992). Stacked generalization. Neural Networks, 5(2), 241–259.
  Talamás-Carvajal & Ceballos (2023). Education and Information Technologies.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os, warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, roc_curve,
    accuracy_score, f1_score, recall_score, precision_score, classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import shap

plt.rcParams.update({'font.family': 'serif', 'font.size': 11, 'figure.dpi': 150})
os.makedirs("resultados/Stacking", exist_ok=True)

print("=" * 70)
print("STACKING ENSEMBLE FINAL")
print("Capa 1: Random Forest + XGBoost + Gradient Boosting")
print("Capa 2: FNN Meta-aprendiz (MLPClassifier)")
print("Tesis DTE-UAQ | Predicción de Deserción Escolar")
print("=" * 70)

# ─── 1. CARGA Y PREPROCESAMIENTO ─────────────────────────────────────────────
df = pd.read_csv("dataset_ml_encoded.csv")
X = df.drop(columns=["desercion"])
y = df["desercion"]
feat_names = X.columns.tolist()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X_scaled, y)

print(f"\n📊 Dataset balanceado: {X_bal.shape[0]} muestras × {X_bal.shape[1]} variables")

# ─── 2. DEFINIR MODELOS BASE (Capa 1) ────────────────────────────────────────
# Parámetros optimizados en los análisis individuales (pasos 01-03)
estimadores_capa1 = [
    ('random_forest', RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_split=2,
        class_weight='balanced', random_state=42
    )),
    ('xgboost', XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric='logloss',
        scale_pos_weight=(y_bal==0).sum()/(y_bal==1).sum(),
        random_state=42
    )),
    ('gradient_boosting', GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=4,
        subsample=0.8, min_samples_split=2, random_state=42
    )),
]

# ─── 3. META-APRENDIZ — FNN (Capa 2) ─────────────────────────────────────────
# La FNN recibe las probabilidades predichas por RF, XGBoost y GB
meta_aprendiz = MLPClassifier(
    hidden_layers=(64, 32),
    activation='relu',
    solver='adam',
    alpha=0.001,              # Regularización L2
    learning_rate_init=0.001,
    max_iter=300,
    early_stopping=True,
    validation_fraction=0.15,
    random_state=42
)

# ─── 4. STACKING CLASSIFIER ──────────────────────────────────────────────────
stacking = StackingClassifier(
    estimators=estimadores_capa1,
    final_estimator=meta_aprendiz,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    stack_method='predict_proba',
    passthrough=False,        # La FNN solo ve probabilidades de los modelos base
    n_jobs=-1
)

# ─── 5. VALIDACIÓN CRUZADA 10-FOLD ───────────────────────────────────────────
print("\n🔁 Validación cruzada 10-fold del Stacking Ensemble...")
print("   (Puede tardar varios minutos — es computacionalmente intensivo)")
cv_outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_results = cross_validate(
    stacking, X_bal, y_bal, cv=cv_outer,
    scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
)

print("\n📈 RESULTADOS FINALES — STACKING ENSEMBLE (10-fold CV)")
print("=" * 60)
nombres = {'Accuracy': 'test_accuracy', 'Precision': 'test_precision',
           'Recall': 'test_recall', 'F1-Score': 'test_f1', 'AUC-ROC': 'test_roc_auc'}
resultados_stacking = {}
for nombre, key in nombres.items():
    vals = cv_results[key]
    print(f"   {nombre:<12}: {vals.mean():.4f} ± {vals.std():.4f}")
    resultados_stacking[nombre] = {'Media': round(vals.mean(), 4), 'Std': round(vals.std(), 4)}

# ─── 6. COMPARACIÓN DE TODOS LOS MODELOS ──────────────────────────────────────
# Cargar métricas de análisis anteriores
def cargar_metricas(ruta):
    if os.path.exists(ruta):
        return pd.read_csv(ruta, index_col=0)
    return None

rf_metrics  = cargar_metricas('resultados/RF/RF_metricas.csv')
xgb_metrics = cargar_metricas('resultados/XGBoost/XGB_metricas.csv')
gb_metrics  = cargar_metricas('resultados/GB/GB_metricas.csv')
fnn_metrics = cargar_metricas('resultados/FNN/FNN_metricas.csv')

modelos_nombres = ['Random Forest', 'XGBoost', 'Gradient\nBoosting', 'FNN\nIndep.', 'Stacking\nEnsemble']
colores_modelos = ['#3498db', '#e67e22', '#9b59b6', '#1abc9c', '#e74c3c']

# ─── 7. ENTRENAMIENTO FINAL ──────────────────────────────────────────────────
print("\n🏋️  Entrenando modelo final sobre todos los datos disponibles...")
split = int(len(X_bal) * 0.85)
X_train, X_test = X_bal[:split], X_bal[split:]
y_train, y_test = y_bal[:split], y_bal[split:]
stacking.fit(X_train, y_train)
y_pred  = stacking.predict(X_test)
y_proba = stacking.predict_proba(X_test)[:, 1]
cm = confusion_matrix(y_test, y_pred)
auc_final = roc_auc_score(y_test, y_proba)

print(f"\n   AUC-ROC prueba final: {auc_final:.4f}")
print(f"   Recall (sensibilidad): {recall_score(y_test, y_pred):.4f}")
print(f"   Precisión: {precision_score(y_test, y_pred):.4f}")
print(f"   F1-Score: {f1_score(y_test, y_pred):.4f}")
print("\n📋 Reporte de clasificación completo:")
print(classification_report(y_test, y_pred, target_names=['No Deserta', 'Deserta']))

# ─── 8. GRÁFICAS FINALES ─────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 10))
fig.suptitle('STACKING ENSEMBLE — Resultados Finales\nPredicción de Deserción Escolar | UAQ DTE | Tesis Doctoral',
             fontsize=14, fontweight='bold', y=1.02)

# 8a. Curva ROC Final
ax1 = fig.add_subplot(2, 3, 1)
fpr, tpr, _ = roc_curve(y_test, y_proba)
ax1.plot(fpr, tpr, color='#e74c3c', lw=2.5, label=f'Stacking (AUC = {auc_final:.3f})')
ax1.plot([0,1],[0,1],'k--',lw=1); ax1.fill_between(fpr, tpr, alpha=0.1, color='#e74c3c')
ax1.set_title('Curva ROC — Stacking Ensemble'); ax1.set_xlabel('FPR'); ax1.set_ylabel('TPR')
ax1.legend(); ax1.set_xlim([0,1]); ax1.set_ylim([0,1.02])

# 8b. Matriz de confusión
ax2 = fig.add_subplot(2, 3, 2)
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=ax2,
            xticklabels=['No Deserta','Deserta'], yticklabels=['No Deserta','Deserta'],
            annot_kws={'size': 16, 'weight': 'bold'})
ax2.set_title('Matriz de Confusión — Stacking')
# Añadir métricas derivadas
tn, fp, fn, tp = cm.ravel()
ax2.text(0.5, -0.25, f'Sensibilidad: {tp/(tp+fn):.3f}  |  Especificidad: {tn/(tn+fp):.3f}',
         transform=ax2.transAxes, ha='center', fontsize=9, color='#555')

# 8c. Comparación AUC-ROC entre modelos
ax3 = fig.add_subplot(2, 3, 3)
# Datos ejemplo (reemplazar con valores reales de los archivos de métricas)
aucs = [0.0, 0.0, 0.0, 0.0, cv_results['test_roc_auc'].mean()]
if rf_metrics is not None:
    aucs[0] = rf_metrics.loc['AUC-ROC', 'Media']
if xgb_metrics is not None:
    aucs[1] = xgb_metrics.loc['AUC-ROC', 'Media']
if gb_metrics is not None:
    aucs[2] = gb_metrics.loc['AUC-ROC', 'Media']
if fnn_metrics is not None:
    aucs[3] = fnn_metrics.loc['AUC-ROC', 'Media']

if all(a > 0 for a in aucs):
    bars = ax3.bar(modelos_nombres, aucs, color=colores_modelos, edgecolor='white', width=0.6)
    for bar, val in zip(bars, aucs):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax3.set_ylim([min(aucs)*0.95, 1.0])
    ax3.set_ylabel('AUC-ROC')
    ax3.set_title('Comparación AUC-ROC — Todos los Modelos')
    ax3.axhline(y=0.9, color='gray', ls='--', lw=1, alpha=0.7, label='AUC = 0.90')
    ax3.legend(fontsize=8)

# 8d. Comparación F1-Score
ax4 = fig.add_subplot(2, 3, 4)
f1s = [0.0, 0.0, 0.0, 0.0, cv_results['test_f1'].mean()]
if rf_metrics is not None: f1s[0] = rf_metrics.loc['F1-Score', 'Media']
if xgb_metrics is not None: f1s[1] = xgb_metrics.loc['F1-Score', 'Media']
if gb_metrics is not None: f1s[2] = gb_metrics.loc['F1-Score', 'Media']
if fnn_metrics is not None: f1s[3] = fnn_metrics.loc['F1-Score', 'Media']

if all(f > 0 for f in f1s):
    bars2 = ax4.bar(modelos_nombres, f1s, color=colores_modelos, edgecolor='white', width=0.6)
    for bar, val in zip(bars2, f1s):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax4.set_ylim([min(f1s)*0.95, 1.0])
    ax4.set_ylabel('F1-Score')
    ax4.set_title('Comparación F1-Score — Todos los Modelos')

# 8e. Diagrama arquitectura Stacking
ax5 = fig.add_subplot(2, 3, (5, 6))
ax5.axis('off')
ax5.set_xlim(0, 10); ax5.set_ylim(0, 6)

# Capa de entrada
ax5.text(1.0, 5.5, 'DATOS DE ENTRADA', ha='center', fontsize=9, fontweight='bold', color='#2c3e50')
ax5.add_patch(plt.FancyBboxPatch((0.1, 4.2), 1.8, 0.9, boxstyle="round,pad=0.1",
              facecolor='#ecf0f1', edgecolor='#bdc3c7', lw=1.5))
ax5.text(1.0, 4.65, 'Variables\nPsicoemocionales\n+ Académicas', ha='center', fontsize=7.5, color='#2c3e50')

# Capa 1 — modelos base
for i, (nombre, color) in enumerate(zip(['Random Forest', 'XGBoost', 'Gradient\nBoosting'],
                                         ['#3498db', '#e67e22', '#9b59b6'])):
    x = 3.5 + i * 1.8
    ax5.add_patch(plt.FancyBboxPatch((x - 0.7, 3.2), 1.4, 1.2,
                  boxstyle="round,pad=0.1", facecolor=color, edgecolor='white', lw=1.5, alpha=0.85))
    ax5.text(x, 3.8, nombre, ha='center', fontsize=8, fontweight='bold', color='white')
    ax5.annotate('', xy=(x, 3.2), xytext=(1.9, 4.65),
                 arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, connectionstyle='arc3,rad=0'))

ax5.text(5.3, 5.5, 'CAPA 1 — Modelos Base', ha='center', fontsize=9, fontweight='bold', color='#2c3e50')
ax5.text(5.3, 2.95, 'Probabilidades P̂(deserción)', ha='center', fontsize=8, color='gray', style='italic')

# Flecha hacia capa 2
for x in [3.5, 5.3, 7.1]:
    ax5.annotate('', xy=(5.3, 2.0), xytext=(x, 3.2),
                 arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5))

# Meta-aprendiz FNN
ax5.add_patch(plt.FancyBboxPatch((3.8, 1.0), 3.0, 1.0,
              boxstyle="round,pad=0.15", facecolor='#e74c3c', edgecolor='white', lw=2, alpha=0.9))
ax5.text(5.3, 1.55, 'META-APRENDIZ: FNN', ha='center', fontsize=9, fontweight='bold', color='white')
ax5.text(5.3, 1.15, '128 → 64 → 32 neuronas | ReLU + Sigmoid', ha='center', fontsize=7.5, color='#ffeaa7')
ax5.text(5.3, 0.6, 'CAPA 2 — Meta-aprendiz', ha='center', fontsize=9, fontweight='bold', color='#2c3e50')

ax5.set_title('Arquitectura del Stacking Ensemble', fontsize=11, fontweight='bold', pad=10)

plt.tight_layout()
plt.savefig('resultados/Stacking/Stacking_resultados_finales.png', bbox_inches='tight', dpi=150)
print("\n📊 Figura final guardada en: resultados/Stacking/Stacking_resultados_finales.png")

# ─── 9. ANÁLISIS SHAP DEL STACKING ──────────────────────────────────────────
print("\n🔍 Análisis SHAP del Stacking (via KernelExplainer - más lento)...")
# Para el stacking usamos KernelExplainer (más general que TreeExplainer)
background = shap.sample(X_test, 50, random_state=42)
explainer = shap.KernelExplainer(lambda x: stacking.predict_proba(x)[:, 1], background)
shap_values = explainer.shap_values(X_test[:30], nsamples=100)

fig_shap, _ = plt.subplots(figsize=(10, 7))
shap.summary_plot(shap_values, X_test[:30], feature_names=feat_names, show=False, max_display=12)
plt.title('SHAP Summary Plot — Stacking Ensemble\nContribución global de variables al riesgo de deserción',
          fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('resultados/Stacking/Stacking_shap_summary.png', bbox_inches='tight', dpi=150)

# ─── 10. GUARDAR TODO ────────────────────────────────────────────────────────
pd.DataFrame(resultados_stacking).T.to_csv('resultados/Stacking/Stacking_metricas.csv')
pd.DataFrame({'y_true': y_test, 'stacking_proba': y_proba, 'stacking_pred': y_pred}).to_csv(
    'resultados/Stacking/Stacking_predicciones.csv', index=False)

# Tabla comparativa final
tabla = {
    'Modelo': modelos_nombres,
    'AUC-ROC': aucs,
    'F1-Score': f1s,
    'Recall': [
        rf_metrics.loc['Recall','Media'] if rf_metrics is not None else None,
        xgb_metrics.loc['Recall','Media'] if xgb_metrics is not None else None,
        gb_metrics.loc['Recall','Media'] if gb_metrics is not None else None,
        fnn_metrics.loc['Recall','Media'] if fnn_metrics is not None else None,
        cv_results['test_recall'].mean()
    ]
}
pd.DataFrame(tabla).to_csv('resultados/Stacking/tabla_comparativa_modelos.csv', index=False)
print("📊 Tabla comparativa guardada en: resultados/Stacking/tabla_comparativa_modelos.csv")

print("\n" + "=" * 70)
print("✅ ANÁLISIS COMPLETO — STACKING ENSEMBLE FINALIZADO")
print(f"   AUC-ROC final: {cv_results['test_roc_auc'].mean():.4f} ± {cv_results['test_roc_auc'].std():.4f}")
print(f"   F1-Score final: {cv_results['test_f1'].mean():.4f}")
print(f"   Recall final:  {cv_results['test_recall'].mean():.4f}")
print("=" * 70)
print("\n🎓 Modelo listo para integración en el prototipo web (FastAPI)")
print("   Guardar modelo: import joblib; joblib.dump(stacking, 'stacking_model.pkl')")
