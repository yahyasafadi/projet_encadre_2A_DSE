import pandas as pd
import numpy as np

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv("wti_features_v3.csv", parse_dates=["Datetime"])
df = df.sort_values("Datetime").reset_index(drop=True)

df_price = pd.read_csv("data_merged.csv", parse_dates=["Datetime"])
df_price = df_price.rename(columns={"WTI_('Close', 'CL=F')": "WTI_Close"})

df = df.merge(df_price[["Datetime", "WTI_Close"]], on="Datetime", how="left")

ret_12h = df["WTI_Close"].shift(-48) / df["WTI_Close"] - 1
df["target"] = (ret_12h > 0).astype(int)

df = df.dropna()

FEATURES = [
    'Close_SMA20_ratio', 'EMA_ratio', 'RSI_14', 'return_12h',
    'Spread_Brent_WTI', 'Corr_Brent_USD_30', 'ADX_proxy',
    'session_europe', 'MACD_slope', 'volatility_ratio',
    'hour_sin', 'hour_cos', 'volume_ratio',
    'USD_return', 'sentiment_score', 'sentiment_available'
]

X = df[FEATURES]
y = df["target"]



models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced",
        max_iter=2000
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=30,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    ),

    "LightGBM": LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42
    )
}


tscv = TimeSeriesSplit(n_splits=5)

results = []

for name, model in models.items():

    accs, f1s, aucs = [], [], []

    print(f"\n===== {name} =====")

    for train_idx, test_idx in tscv.split(X):

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        accs.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred))
        aucs.append(roc_auc_score(y_test, y_prob))

    results.append({
        "Model": name,
        "Accuracy": np.mean(accs),
        "F1": np.mean(f1s),
        "AUC": np.mean(aucs)
    })


results_df = pd.DataFrame(results)
results_df = results_df.sort_values("AUC", ascending=False)

print("\n===== FINAL COMPARISON =====")
print(results_df.round(4))