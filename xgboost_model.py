import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── CHARGEMENT ───────────────────────────────────────────────────
df = pd.read_csv('wti_features_1min.csv', parse_dates=['Datetime'])
df = df.sort_values('Datetime').reset_index(drop=True)

# ── TARGET ───────────────────────────────────────────────────────
df['target'] = (df['WTI_Close'].shift(-1) > df['WTI_Close']).astype(int)
df = df.dropna()

exclude = ['Datetime', 'target', 'WTI_Close']
X = df[[c for c in df.columns if c not in exclude]]
y = df['target']

print(f"✅ {len(df)} lignes | {X.shape[1]} features")
print(f"   Features : {list(X.columns)}")

# ── TIME SERIES SPLIT + XGBOOST ──────────────────────────────────
scaler  = StandardScaler()
tscv    = TimeSeriesSplit(n_splits=5)
results = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    model = XGBClassifier(
        n_estimators     = 200,
        max_depth        = 4,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        random_state     = 42,
        eval_metric      = 'logloss',
        verbosity        = 0
    )
    model.fit(X_train_scaled, y_train)

    y_pred        = model.predict(X_test_scaled)
    y_pred_proba  = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='weighted')
    auc = roc_auc_score(y_test, y_pred_proba)

    results.append({'fold': fold+1, 'accuracy': acc, 'f1_score': f1, 'auc_roc': auc})
    print(f"Fold {fold+1} | Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

# ── RÉSUMÉ ───────────────────────────────────────────────────────
df_results = pd.DataFrame(results)
print("\n===== RÉSUMÉ XGBOOST (données 1 minute) =====")
print(f"Accuracy moyenne : {df_results['accuracy'].mean():.4f} ± {df_results['accuracy'].std():.4f}")
print(f"F1 moyenne       : {df_results['f1_score'].mean():.4f} ± {df_results['f1_score'].std():.4f}")
print(f"AUC-ROC moyenne  : {df_results['auc_roc'].mean():.4f} ± {df_results['auc_roc'].std():.4f}")

# ── COMPARAISON RANDOM FOREST VS XGBOOST ────────────────────────
rf_scores  = [0.4995, 0.4936, 0.5000, 0.4995, 0.5100]
xgb_scores = df_results['accuracy'].tolist()

plt.figure(figsize=(8, 4))
plt.plot(range(1, 6), rf_scores,  'o-', label='Random Forest (horaire)', color='steelblue')
plt.plot(range(1, 6), xgb_scores, 's-', label='XGBoost (1 minute)',      color='orange')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Random Forest vs XGBoost — données 1 minute (TimeSeriesSplit)')
plt.legend()
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('rf_vs_xgboost_1min.png', dpi=150)
plt.show()
print("✅ Graphique sauvegardé : rf_vs_xgboost_1min.png")