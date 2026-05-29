import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── CHARGEMENT ───────────────────────────────────────────────────
df = pd.read_csv('wti_features_v3.csv', parse_dates=['Datetime'])
df = df.sort_values('Datetime').reset_index(drop=True)  # TRI CHRONOLOGIQUE OBLIGATOIRE

# ── TARGET ───────────────────────────────────────────────────────
df['target'] = (df['WTI_Close'].shift(-1) > df['WTI_Close']).astype(int)
df = df.dropna()

exclude = ['Datetime', 'target', 'WTI_Close']
X = df[[c for c in df.columns if c not in exclude]]
y = df['target']

# ── TIME SERIES SPLIT ─────────────────────────────────────────────
scaler = StandardScaler()
tscv = TimeSeriesSplit(n_splits=5)
results = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Normalisation FIT sur train uniquement
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)

    y_pred       = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='weighted')
    auc = roc_auc_score(y_test, y_pred_proba)

    results.append({'fold': fold+1, 'accuracy': acc, 'f1_score': f1, 'auc_roc': auc})
    print(f"Fold {fold+1} | Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

# ── RÉSUMÉ ───────────────────────────────────────────────────────
df_results = pd.DataFrame(results)
print("\n===== RÉSUMÉ =====")
print(f"Accuracy moyenne : {df_results['accuracy'].mean():.4f} ± {df_results['accuracy'].std():.4f}")
print(f"F1 moyenne       : {df_results['f1_score'].mean():.4f} ± {df_results['f1_score'].std():.4f}")
print(f"AUC-ROC moyenne  : {df_results['auc_roc'].mean():.4f} ± {df_results['auc_roc'].std():.4f}")

# ── GRAPHIQUE ────────────────────────────────────────────────────
plt.figure(figsize=(8, 4))
plt.plot(df_results['fold'], df_results['accuracy'], 'o-', label='Accuracy')
plt.plot(df_results['fold'], df_results['f1_score'],  's-', label='F1-score')
plt.plot(df_results['fold'], df_results['auc_roc'],   '^-', label='AUC-ROC')
plt.xlabel('Fold')
plt.ylabel('Score')
plt.title('Performance par fold (TimeSeriesSplit)')
plt.legend()
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ts_split_performance.png', dpi=150)
plt.show()
print("✅ Graphique sauvegardé : ts_split_performance.png")