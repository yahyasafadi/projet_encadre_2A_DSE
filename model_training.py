"""
WTI Oil Price Prediction - ML Pipeline
========================================
Modèles  : Logistic Regression, Random Forest, XGBoost
Targets  : 1) Signal BUY/SELL (HOLD exclu)
           2) Volatilité High/Low
Anti-leakage : TimeSeriesSplit (walk-forward)
"""

import os
import copy
import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
import xgboost as xgb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────
# 0.  CONFIGURATION  –  adapte ces chemins !
# ─────────────────────────────────────────────
FEATURES_PATH = r"C:\Users\moham\Desktop\EA_project\projet_encadre_2A_DSE\wti_features_v3.csv"
MERGED_PATH   = r"C:\Users\moham\Desktop\EA_project\projet_encadre_2A_DSE\data_merged.csv"
OUTPUT_DIR    = r"C:\Users\moham\Desktop\EA_project\projet_encadre_2A_DSE\outputs"

HORIZON        = 6      # horizon de prédiction en heures
N_SPLITS       = 5      # folds walk-forward
TEST_SIZE      = 500    # lignes par fenêtre de test

BUY_THRESHOLD  =  0.003  # return futur >= +0.3 % → BUY
SELL_THRESHOLD = -0.003  # return futur <= -0.3 % → SELL
VOL_QUANTILE   =  0.75   # au-dessus du 75e percentile → High Volatility

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1.  CHARGEMENT & MERGE
# ─────────────────────────────────────────────
print("=" * 60)
print("  WTI ML PIPELINE  –  BUY/SELL  +  VOLATILITÉ")
print("=" * 60)

features = pd.read_csv(FEATURES_PATH, parse_dates=["Datetime"])
merged   = pd.read_csv(MERGED_PATH,   parse_dates=["Datetime"])

close_col   = "WTI_('Close', 'CL=F')"
merged_slim = merged[["Datetime", close_col]].rename(columns={close_col: "Close"})

df = features.merge(merged_slim, on="Datetime", how="inner")
df = df.sort_values("Datetime").reset_index(drop=True)

print(f"\n[DATA]  Lignes brutes : {len(df):,}")
print(f"[DATA]  Période : {df['Datetime'].min().date()} → {df['Datetime'].max().date()}")

# ─────────────────────────────────────────────
# 2.  CONSTRUCTION DES TARGETS (shift → pas de leakage)
# ─────────────────────────────────────────────
df["future_return"] = df["Close"].shift(-HORIZON) / df["Close"] - 1

# Target volatilité calculée sur le df complet AVANT filtrage HOLD
df["future_vol"] = (
    df["Close"].pct_change()
    .rolling(HORIZON).std()
    .shift(-HORIZON)
)
vol_thresh = df["future_vol"].quantile(VOL_QUANTILE)
df["target_vol"] = (df["future_vol"] >= vol_thresh).astype(float)

# Target BUY/SELL — HOLD → NaN puis supprimé
df["target_signal"] = np.where(
    df["future_return"] >= BUY_THRESHOLD,  1,
    np.where(
    df["future_return"] <= SELL_THRESHOLD, 0,
    np.nan
))

# DataFrame BUY/SELL (sans HOLD)
df_signal = df.dropna(subset=["target_signal", "future_vol"]).copy().reset_index(drop=True)
df_signal["target_signal"] = df_signal["target_signal"].astype(int)

# DataFrame Volatilité (toutes les lignes utiles)
df_vol = df.dropna(subset=["future_vol"]).copy().reset_index(drop=True)
df_vol["target_vol"] = df_vol["target_vol"].astype(int)

print(f"\n[TARGET BUY/SELL]  {len(df_signal):,} lignes  |  "
      f"SELL={( df_signal['target_signal']==0).sum():,}  BUY={(df_signal['target_signal']==1).sum():,}")
print(f"[TARGET VOL]       {len(df_vol):,} lignes  |  "
      f"Low={(df_vol['target_vol']==0).sum():,}  High={(df_vol['target_vol']==1).sum():,}")

# ─────────────────────────────────────────────
# 3.  FEATURES
# ─────────────────────────────────────────────
FEATURE_COLS = [
    "Close_SMA20_ratio", "EMA_ratio", "RSI_14",
    "return_12h", "Spread_Brent_WTI", "Corr_Brent_USD_30",
    "ADX_proxy", "session_europe", "MACD_slope",
    "volatility_ratio", "hour_sin", "hour_cos",
    "volume_ratio", "USD_return", "sentiment_score",
    "sentiment_available",
]

X_signal = df_signal[FEATURE_COLS].values
y_signal = df_signal["target_signal"].values

X_vol = df_vol[FEATURE_COLS].values
y_vol = df_vol["target_vol"].values

print(f"\n[FEATURES]  {len(FEATURE_COLS)} features")

# ─────────────────────────────────────────────
# 4.  MODÈLES
# ─────────────────────────────────────────────
def make_models():
    return {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000, C=0.1,
                class_weight="balanced", solver="lbfgs"
            )),
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=8,
            min_samples_leaf=20, class_weight="balanced",
            random_state=42, n_jobs=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300, max_depth=5,
            learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, eval_metric="logloss",
            random_state=42, verbosity=0
        ),
    }

# ─────────────────────────────────────────────
# 5.  WALK-FORWARD EVALUATION (fonction réutilisable)
# ─────────────────────────────────────────────
def walk_forward(X, y, target_label):
    tscv    = TimeSeriesSplit(n_splits=N_SPLITS, test_size=TEST_SIZE)
    models  = make_models()
    results = {name: {"acc_folds": [], "y_true": [], "y_pred": []} for name in models}

    print(f"\n{'─'*60}")
    print(f"  TARGET : {target_label}")
    print(f"{'─'*60}")

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        for name, model in models.items():
            m = copy.deepcopy(model)
            m.fit(X_tr, y_tr)
            preds = m.predict(X_te)
            results[name]["acc_folds"].append(accuracy_score(y_te, preds))
            results[name]["y_true"].extend(y_te.tolist())
            results[name]["y_pred"].extend(preds.tolist())

        print(f"  Fold {fold}/{N_SPLITS}  train={len(train_idx):,}  test={len(test_idx):,}  ✓")

    # Consolider
    for name in models:
        folds = results[name]["acc_folds"]
        yt    = np.array(results[name]["y_true"])
        yp    = np.array(results[name]["y_pred"])
        results[name]["mean_acc"]   = np.mean(folds)
        results[name]["std_acc"]    = np.std(folds)
        results[name]["global_acc"] = accuracy_score(yt, yp)
        results[name]["y_true"]     = yt
        results[name]["y_pred"]     = yp

    return results, models

# ─────────────────────────────────────────────
# 6.  ENTRAÎNEMENTS
# ─────────────────────────────────────────────
res_signal, mdl_signal = walk_forward(X_signal, y_signal, "BUY / SELL")
res_vol,    mdl_vol    = walk_forward(X_vol,    y_vol,    "VOLATILITÉ  (High / Low)")

# ─────────────────────────────────────────────
# 7.  RÉSUMÉ CONSOLE
# ─────────────────────────────────────────────
def print_results(results, target_names, label):
    print(f"\n{'='*60}")
    print(f"  RÉSULTATS  –  {label}")
    print(f"{'='*60}")
    rows = []
    for name, res in results.items():
        print(f"\n  [{name}]")
        print(f"    Acc/fold   : {res['mean_acc']:.4f} ± {res['std_acc']:.4f}")
        print(f"    Acc globale: {res['global_acc']:.4f}")
        print(classification_report(res["y_true"], res["y_pred"],
                                    target_names=target_names,
                                    zero_division=0, digits=3))
        rows.append({
            "Target": label, "Modèle": name,
            "Acc Mean": f"{res['mean_acc']:.4f}",
            "Acc Std":  f"{res['std_acc']:.4f}",
            "Acc Global": f"{res['global_acc']:.4f}",
        })
    return rows

rows1 = print_results(res_signal, ["SELL", "BUY"],          "BUY / SELL")
rows2 = print_results(res_vol,    ["Low Vol", "High Vol"],  "VOLATILITÉ")

# ─────────────────────────────────────────────
# 8.  VISUALISATIONS
# ─────────────────────────────────────────────
colors = ["#4C72B0", "#DD8452", "#55A868"]

def plot_report(results, title, display_labels, filename):
    model_names = list(results.keys())
    fig = plt.figure(figsize=(22, 18))
    fig.suptitle(f"WTI ML Pipeline – {title}\n(Walk-Forward, {N_SPLITS} folds, horizon={HORIZON}h)",
                 fontsize=16, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.35)

    # Row 0 : accuracy par fold
    for i, name in enumerate(model_names):
        ax = fig.add_subplot(gs[0, i])
        folds = results[name]["acc_folds"]
        mean  = results[name]["mean_acc"]
        ax.bar(range(1, N_SPLITS + 1), folds, color=colors[i], alpha=0.75)
        ax.axhline(mean, color="black", linestyle="--", lw=1.5, label=f"Mean={mean:.3f}")
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Fold"); ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    # Row 1 : matrices de confusion
    for i, name in enumerate(model_names):
        ax = fig.add_subplot(gs[1, i])
        cm   = confusion_matrix(results[name]["y_true"], results[name]["y_pred"])
        disp = ConfusionMatrixDisplay(cm, display_labels=display_labels)
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{name}\nConfusion Matrix", fontsize=10)

    # Row 2 : comparaison globale
    ax = fig.add_subplot(gs[2, :])
    global_accs = [results[n]["global_acc"] for n in model_names]
    std_accs    = [results[n]["std_acc"]    for n in model_names]
    bars = ax.bar(model_names, global_accs, color=colors, alpha=0.8, yerr=std_accs, capsize=10)
    ax.set_title("Comparaison Accuracy Globale", fontsize=13, fontweight="bold")
    ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1); ax.grid(axis="y", alpha=0.3)
    for bar, acc in zip(bars, global_accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{acc:.3f}", ha="center", fontsize=13, fontweight="bold")

    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[PLOT] → {path}")

def plot_feature_importance(X, y, models_dict, title, filename):
    split_idx = len(X) - TEST_SIZE
    rf = copy.deepcopy(models_dict["RandomForest"])
    rf.fit(X[:split_idx], y[:split_idx])
    feat_imp = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values()
    fig, ax = plt.subplots(figsize=(10, 7))
    feat_imp.tail(12).plot(kind="barh", ax=ax, color="#4C72B0", alpha=0.8)
    ax.set_title(f"Feature Importance – Random Forest\n{title}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance"); ax.grid(axis="x", alpha=0.3)
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] → {path}")

# Rapport BUY/SELL
plot_report(res_signal, "BUY / SELL", ["SELL", "BUY"], "report_buysell.png")
plot_feature_importance(X_signal, y_signal, mdl_signal,
                        "BUY/SELL", "feat_importance_buysell.png")

# Rapport Volatilité
plot_report(res_vol, "VOLATILITÉ  High / Low", ["Low Vol", "High Vol"], "report_volatility.png")
plot_feature_importance(X_vol, y_vol, mdl_vol,
                        "Volatilité", "feat_importance_volatility.png")

# ─────────────────────────────────────────────
# 9.  CSV RÉCAPITULATIF GLOBAL
# ─────────────────────────────────────────────
summary_df = pd.DataFrame(rows1 + rows2)
print(f"\n{'='*60}")
print(summary_df.to_string(index=False))
csv_path = os.path.join(OUTPUT_DIR, "summary_all_targets.csv")
summary_df.to_csv(csv_path, index=False)
print(f"\n[CSV] → {csv_path}")
print("\n✅ Pipeline terminé.")