"""
WTI ML Dashboard – Streamlit
Run: streamlit run dashboard.py
"""

import os, copy, warnings
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import xgboost as xgb

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
FEATURES_PATH = r"C:\Users\moham\Desktop\EA_project\projet_encadre_2A_DSE\wti_features_v3.csv"
MERGED_PATH   = r"C:\Users\moham\Desktop\EA_project\projet_encadre_2A_DSE\data_merged.csv"

HORIZON        = 6
N_SPLITS       = 5
TEST_SIZE      = 500
BUY_THRESHOLD  =  0.003
SELL_THRESHOLD = -0.003
VOL_QUANTILE   =  0.75

FEATURE_COLS = [
    "Close_SMA20_ratio", "EMA_ratio", "RSI_14",
    "return_12h", "Spread_Brent_WTI", "Corr_Brent_USD_30",
    "ADX_proxy", "session_europe", "MACD_slope",
    "volatility_ratio", "hour_sin", "hour_cos",
    "volume_ratio", "USD_return", "sentiment_score", "sentiment_available",
]

st.set_page_config(page_title="WTI ML Dashboard", layout="wide", page_icon="🛢️")

# ──────────────────────────────────────────────
# LOAD & PREPARE DATA
# ──────────────────────────────────────────────
@st.cache_data
def load_data():
    feat   = pd.read_csv(FEATURES_PATH, parse_dates=["Datetime"])
    merged = pd.read_csv(MERGED_PATH,   parse_dates=["Datetime"])

    close_col   = "WTI_('Close', 'CL=F')"
    high_col    = "WTI_('High', 'CL=F')"
    low_col     = "WTI_('Low', 'CL=F')"
    open_col    = "WTI_('Open', 'CL=F')"
    vol_col     = "WTI_('Volume', 'CL=F')"
    brent_col   = "BRENT_('Close', 'BZ=F')"
    usd_col     = "USD_INDEX_('Close', 'DX-Y.NYB')"

    price_df = merged[["Datetime", open_col, high_col, low_col, close_col, vol_col, brent_col, usd_col]].rename(columns={
        open_col:  "Open",
        high_col:  "High",
        low_col:   "Low",
        close_col: "Close",
        vol_col:   "Volume",
        brent_col: "Brent",
        usd_col:   "USD_Index",
    })

    df = feat.merge(price_df, on="Datetime", how="inner")
    df = df.sort_values("Datetime").reset_index(drop=True)

    # Targets
    df["future_return"] = df["Close"].shift(-HORIZON) / df["Close"] - 1
    df["future_vol"]    = df["Close"].pct_change().rolling(HORIZON).std().shift(-HORIZON)
    vol_thresh          = df["future_vol"].quantile(VOL_QUANTILE)
    df["target_vol"]    = (df["future_vol"] >= vol_thresh).astype(float)
    df["target_signal"] = np.where(
        df["future_return"] >= BUY_THRESHOLD,  1,
        np.where(df["future_return"] <= SELL_THRESHOLD, 0, np.nan)
    )

    df_signal = df.dropna(subset=["target_signal", "future_vol"]).copy().reset_index(drop=True)
    df_signal["target_signal"] = df_signal["target_signal"].astype(int)
    df_vol    = df.dropna(subset=["future_vol"]).copy().reset_index(drop=True)
    df_vol["target_vol"] = df_vol["target_vol"].astype(int)

    return df, df_signal, df_vol

@st.cache_data
def train_models(df_signal, df_vol):
    def make_models():
        return {
            "Logistic Regression": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, C=0.1,
                                           class_weight="balanced", solver="lbfgs")),
            ]),
            "Random Forest": RandomForestClassifier(
                n_estimators=200, max_depth=8, min_samples_leaf=20,
                class_weight="balanced", random_state=42, n_jobs=-1),
            "XGBoost": xgb.XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                random_state=42, verbosity=0),
        }

    def walk_forward(X, y):
        tscv    = TimeSeriesSplit(n_splits=N_SPLITS, test_size=TEST_SIZE)
        models  = make_models()
        results = {n: {"acc_folds": [], "y_true": [], "y_pred": [], "dates": []} for n in models}
        for fold, (tr, te) in enumerate(tscv.split(X), 1):
            for name, model in models.items():
                m = copy.deepcopy(model)
                m.fit(X[tr], y[tr])
                preds = m.predict(X[te])
                results[name]["acc_folds"].append(accuracy_score(y[te], preds))
                results[name]["y_true"].extend(y[te].tolist())
                results[name]["y_pred"].extend(preds.tolist())
        for name in models:
            yt = np.array(results[name]["y_true"])
            yp = np.array(results[name]["y_pred"])
            results[name]["mean_acc"]   = np.mean(results[name]["acc_folds"])
            results[name]["std_acc"]    = np.std(results[name]["acc_folds"])
            results[name]["global_acc"] = accuracy_score(yt, yp)
            results[name]["y_true"]     = yt
            results[name]["y_pred"]     = yp
            results[name]["cm"]         = confusion_matrix(yt, yp)
        return results

    X_s = df_signal[FEATURE_COLS].values
    y_s = df_signal["target_signal"].values
    X_v = df_vol[FEATURE_COLS].values
    y_v = df_vol["target_vol"].values

    res_signal = walk_forward(X_s, y_s)
    res_vol    = walk_forward(X_v, y_v)

    # Feature importance (RF on last split)
    split_s = len(X_s) - TEST_SIZE
    split_v = len(X_v) - TEST_SIZE
    rf_s = copy.deepcopy(make_models()["Random Forest"])
    rf_v = copy.deepcopy(make_models()["Random Forest"])
    rf_s.fit(X_s[:split_s], y_s[:split_s])
    rf_v.fit(X_v[:split_v], y_v[:split_v])

    fi_signal = pd.Series(rf_s.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    fi_vol    = pd.Series(rf_v.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)

    return res_signal, res_vol, fi_signal, fi_vol

@st.cache_data
def run_backtest(df_signal, res_signal, model_name,
                 capital_init=10_000.0, cost_pct=0.0005,
                 stop_loss_pct=None, take_profit_pct=None,
                 vol_filter=False, df_vol=None, res_vol=None, vol_model=None):
    """
    Simule une stratégie long/short sur les signaux BUY/SELL prédits
    par walk-forward (pour éviter tout data leakage).

    Règles :
      - Signal BUY  → position LONG  (on achète 1 unité de WTI)
      - Signal SELL → position SHORT (on vend à découvert 1 unité)
      - Filtre volatilité optionnel : on ne trade que si le modèle
        vol prédit Low Vol (marché calme = signaux plus fiables)
      - Stop-loss / take-profit optionnels par trade
      - Coût de transaction cost_pct appliqué à chaque changement de position
    """
    yt = res_signal[model_name]["y_true"]   # 0=SELL, 1=BUY
    yp = res_signal[model_name]["y_pred"]

    n = len(yt)
    dates_arr = df_signal["Datetime"].values[-n:]
    close_arr = df_signal["Close"].values[-n:]

    # Filtre vol (optionnel) : masque True = on trade
    if vol_filter and df_vol is not None and res_vol is not None and vol_model is not None:
        yp_vol = res_vol[vol_model]["y_pred"]
        n_vol  = len(yp_vol)
        # Aligne sur les dernières n lignes
        if n_vol >= n:
            vol_mask = (yp_vol[-n:] == 0)   # 0 = Low Vol → on trade
        else:
            vol_mask = np.ones(n, dtype=bool)
    else:
        vol_mask = np.ones(n, dtype=bool)

    equity       = [capital_init]
    cash         = capital_init
    position     = 0          # +1 = long, -1 = short, 0 = flat
    entry_price  = None
    entry_idx    = None
    trades       = []          # liste de dicts par trade terminé
    returns_list = []          # rendement de chaque barre
    positions_history = []     # historique des positions

    for i in range(n - 1):
        price_now  = close_arr[i]
        price_next = close_arr[i + 1]
        signal     = int(yp[i])   # 1=BUY, 0=SELL
        target_pos = 1 if (signal == 1 and vol_mask[i]) else (-1 if (signal == 0 and vol_mask[i]) else 0)
        
        # Enregistrer la position actuelle
        positions_history.append(position)

        # ── Gestion stop-loss / take-profit en cours de position ──
        if position != 0 and entry_price is not None:
            pnl_pct = position * (price_now - entry_price) / entry_price
            hit_sl = stop_loss_pct   is not None and pnl_pct <= -stop_loss_pct
            hit_tp = take_profit_pct is not None and pnl_pct >=  take_profit_pct
            if hit_sl or hit_tp:
                # Clôture forcée
                trade_ret = position * (price_now - entry_price) / entry_price
                cost      = cost_pct * abs(price_now)
                cash     += cash * trade_ret - cost
                trades.append({
                    "entry_date":  dates_arr[entry_idx],
                    "exit_date":   dates_arr[i],
                    "direction":   "LONG" if position == 1 else "SHORT",
                    "entry_price": entry_price,
                    "exit_price":  price_now,
                    "pnl_pct":     trade_ret * 100,
                    "exit_reason": "SL" if hit_sl else "TP",
                    "n_bars":      i - entry_idx,
                })
                returns_list.append(trade_ret)
                position    = 0
                entry_price = None

        # ── Changement de position ──
        if target_pos != position:
            if position != 0 and entry_price is not None:
                # Clôture ancienne position
                trade_ret = position * (price_now - entry_price) / entry_price
                cost      = cost_pct * abs(price_now)
                cash     += cash * trade_ret - cost
                trades.append({
                    "entry_date":  dates_arr[entry_idx],
                    "exit_date":   dates_arr[i],
                    "direction":   "LONG" if position == 1 else "SHORT",
                    "entry_price": entry_price,
                    "exit_price":  price_now,
                    "pnl_pct":     trade_ret * 100,
                    "exit_reason": "Signal",
                    "n_bars":      i - entry_idx,
                })
                returns_list.append(trade_ret)

            if target_pos != 0:
                cost        = cost_pct * abs(price_now)
                cash       -= cost
                position    = target_pos
                entry_price = price_now
                entry_idx   = i
            else:
                position    = 0
                entry_price = None

        bar_ret = 0.0
        if position != 0 and entry_price is not None:
            bar_ret = position * (price_next - price_now) / price_now
        equity.append(cash * (1 + bar_ret) if position != 0 else cash)
    
    # Ajouter la dernière position
    positions_history.append(position)

    # Buy-and-hold benchmark
    bh_prices  = close_arr[:len(equity)]
    bh_equity  = capital_init * bh_prices / bh_prices[0]

    eq_arr     = np.array(equity)
    dates_eq   = dates_arr[:len(equity)]

    # ── Métriques ──
    total_ret   = (eq_arr[-1] / capital_init - 1) * 100
    bh_ret      = (bh_equity[-1] / capital_init - 1) * 100
    bar_returns = np.diff(eq_arr) / eq_arr[:-1]
    sharpe      = (np.mean(bar_returns) / (np.std(bar_returns) + 1e-9)) * np.sqrt(24 * 252)
    roll_max    = np.maximum.accumulate(eq_arr)
    drawdown    = (eq_arr - roll_max) / roll_max * 100
    max_dd      = drawdown.min()

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(
        columns=["entry_date","exit_date","direction","entry_price","exit_price","pnl_pct","exit_reason","n_bars"])

    n_trades  = len(trades_df)
    win_rate  = (trades_df["pnl_pct"] > 0).mean() * 100 if n_trades > 0 else 0
    avg_win   = trades_df.loc[trades_df["pnl_pct"] > 0, "pnl_pct"].mean() if n_trades > 0 and (trades_df["pnl_pct"] > 0).sum() > 0 else 0
    avg_loss  = trades_df.loc[trades_df["pnl_pct"] < 0, "pnl_pct"].mean() if n_trades > 0 and (trades_df["pnl_pct"] < 0).sum() > 0 else 0

    return {
        "dates":       dates_eq,
        "equity":      eq_arr,
        "bh_equity":   bh_equity,
        "drawdown":    drawdown,
        "trades_df":   trades_df,
        "total_ret":   total_ret,
        "bh_ret":      bh_ret,
        "sharpe":      sharpe,
        "max_dd":      max_dd,
        "n_trades":    n_trades,
        "win_rate":    win_rate,
        "avg_win":     avg_win,
        "avg_loss":    avg_loss,
        "bar_returns": bar_returns,
        "positions_history": positions_history,
        "capital_init": capital_init,
    }


st.sidebar.title("🛢️ WTI ML Dashboard")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", [
    "📈 Prix & Marché",
    "🎯 Prédictions vs Réalité",
    "📊 Performance des Modèles",
    "🔍 Feature Importance",
    "💰 Backtesting Stratégie",
])
st.sidebar.markdown("---")
st.sidebar.info(f"Horizon: **{HORIZON}h** | Folds: **{N_SPLITS}** | Test: **{TEST_SIZE}**")

# ──────────────────────────────────────────────
# LOAD
# ──────────────────────────────────────────────
with st.spinner("Chargement des données..."):
    df, df_signal, df_vol = load_data()

with st.spinner("Entraînement des modèles (cache après 1ère fois)..."):
    res_signal, res_vol, fi_signal, fi_vol = train_models(df_signal, df_vol)

# ──────────────────────────────────────────────
# PAGE 1 – PRIX & MARCHÉ
# ──────────────────────────────────────────────
if page == "📈 Prix & Marché":
    st.title("📈 Dernières Données de Prix – WTI")

    # KPI cards
    last  = df.dropna(subset=["Close"]).iloc[-1]
    prev  = df.dropna(subset=["Close"]).iloc[-2]
    chg   = last["Close"] - prev["Close"]
    chgp  = chg / prev["Close"] * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("WTI Close", f"${last['Close']:.2f}", f"{chg:+.2f} ({chgp:+.2f}%)")
    c2.metric("Brent Close", f"${last['Brent']:.2f}")
    c3.metric("USD Index",   f"{last['USD_Index']:.2f}")
    c4.metric("RSI (14)",    f"{last['RSI_14']:.1f}")
    c5.metric("Spread Brent-WTI", f"${last['Spread_Brent_WTI']:.2f}")

    st.markdown("---")

    # Date range filter
    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input("Du", value=df["Datetime"].min().date())
    with col2:
        end   = st.date_input("Au", value=df["Datetime"].max().date())

    mask = (df["Datetime"].dt.date >= start) & (df["Datetime"].dt.date <= end)
    dff  = df[mask]

    # Candlestick
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3],
                        subplot_titles=["WTI Prix (OHLC)", "Volume"])

    fig.add_trace(go.Candlestick(
        x=dff["Datetime"], open=dff["Open"], high=dff["High"],
        low=dff["Low"],    close=dff["Close"],
        name="WTI", increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350"
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=dff["Datetime"], y=dff["Volume"],
        name="Volume", marker_color="#90caf9", opacity=0.6
    ), row=2, col=1)

    fig.update_layout(height=550, xaxis_rangeslider_visible=False,
                      template="plotly_dark", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # RSI + MACD
    col1, col2 = st.columns(2)
    with col1:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=dff["Datetime"], y=dff["RSI_14"],
                                  line=dict(color="#ffa726"), name="RSI 14"))
        fig2.add_hline(y=70, line_dash="dash", line_color="red",   annotation_text="Suracheté")
        fig2.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Survendu")
        fig2.update_layout(title="RSI (14)", height=300, template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=dff["Datetime"], y=dff["MACD_slope"],
                                  line=dict(color="#ab47bc"), name="MACD Slope"))
        fig3.add_hline(y=0, line_color="white", line_dash="dot")
        fig3.update_layout(title="MACD Slope", height=300, template="plotly_dark")
        st.plotly_chart(fig3, use_container_width=True)

    # WTI vs Brent vs USD
    fig4 = make_subplots(specs=[[{"secondary_y": True}]])
    fig4.add_trace(go.Scatter(x=dff["Datetime"], y=dff["Close"],
                              name="WTI", line=dict(color="#26a69a")), secondary_y=False)
    fig4.add_trace(go.Scatter(x=dff["Datetime"], y=dff["Brent"],
                              name="Brent", line=dict(color="#ffa726")), secondary_y=False)
    fig4.add_trace(go.Scatter(x=dff["Datetime"], y=dff["USD_Index"],
                              name="USD Index", line=dict(color="#ef5350", dash="dot")), secondary_y=True)
    fig4.update_layout(title="WTI vs Brent vs USD Index", height=350, template="plotly_dark")
    fig4.update_yaxes(title_text="Prix ($)", secondary_y=False)
    fig4.update_yaxes(title_text="USD Index", secondary_y=True)
    st.plotly_chart(fig4, use_container_width=True)

# ──────────────────────────────────────────────
# PAGE 2 – PRÉDICTIONS VS RÉALITÉ
# ──────────────────────────────────────────────
elif page == "🎯 Prédictions vs Réalité":
    st.title("🎯 Prédictions vs Réalité")

    target_choice = st.radio("Target", ["BUY / SELL", "Volatilité High/Low"], horizontal=True)
    model_choice  = st.selectbox("Modèle", ["Logistic Regression", "Random Forest", "XGBoost"])

    if target_choice == "BUY / SELL":
        res    = res_signal
        df_ref = df_signal
        labels = {0: "SELL", 1: "BUY"}
        colors_map = {0: "#ef5350", 1: "#26a69a"}
    else:
        res    = res_vol
        df_ref = df_vol
        labels = {0: "Low Vol", 1: "High Vol"}
        colors_map = {0: "#90caf9", 1: "#ffa726"}

    yt = res[model_choice]["y_true"]
    yp = res[model_choice]["y_pred"]

    # On prend les dernières len(yt) dates
    dates_ref = df_ref["Datetime"].values[-len(yt):]
    close_ref = df_ref["Close"].values[-len(yt):]

    correct   = (yt == yp).astype(int)
    df_pred   = pd.DataFrame({
        "Datetime": dates_ref,
        "Close":    close_ref,
        "Réel":     [labels[v] for v in yt],
        "Prédit":   [labels[v] for v in yp],
        "Correct":  correct,
    })

    # Accuracy glissante
    window = st.slider("Fenêtre accuracy glissante (points)", 20, 200, 50)
    df_pred["Rolling_Acc"] = df_pred["Correct"].rolling(window).mean()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=["Prix WTI + Signaux (Réel vs Prédit)",
                                        f"Accuracy glissante ({window} pts)"],
                        row_heights=[0.65, 0.35])

    fig.add_trace(go.Scatter(x=df_pred["Datetime"], y=df_pred["Close"],
                             line=dict(color="white", width=1),
                             name="WTI Close"), row=1, col=1)

    # Points corrects / incorrects
    for label_val, label_name in labels.items():
        mask_ok  = (df_pred["Réel"] == label_name) & (df_pred["Correct"] == 1)
        mask_err = (df_pred["Réel"] == label_name) & (df_pred["Correct"] == 0)

        fig.add_trace(go.Scatter(
            x=df_pred[mask_ok]["Datetime"], y=df_pred[mask_ok]["Close"],
            mode="markers", marker=dict(size=5, color=colors_map[label_val], symbol="circle"),
            name=f"✓ {label_name}"
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df_pred[mask_err]["Datetime"], y=df_pred[mask_err]["Close"],
            mode="markers", marker=dict(size=5, color="gray", symbol="x"),
            name=f"✗ {label_name}"
        ), row=1, col=1)

    fig.add_trace(go.Scatter(x=df_pred["Datetime"], y=df_pred["Rolling_Acc"],
                             line=dict(color="#ffa726", width=2),
                             name="Acc glissante"), row=2, col=1)
    fig.add_hline(y=0.5, line_dash="dash", line_color="red", row=2, col=1)

    fig.update_layout(height=600, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # Stats
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy Globale", f"{res[model_choice]['global_acc']:.3f}")
    col2.metric("Acc Moyenne/Fold", f"{res[model_choice]['mean_acc']:.3f} ± {res[model_choice]['std_acc']:.3f}")
    col3.metric("Observations test", f"{len(yt):,}")

    # Distribution réel vs prédit
    col1, col2 = st.columns(2)
    with col1:
        cnt_real = pd.Series([labels[v] for v in yt]).value_counts()
        fig_r = px.bar(cnt_real, title="Distribution Réelle",
                       color=cnt_real.index,
                       color_discrete_map={v: colors_map[k] for k, v in labels.items()},
                       template="plotly_dark")
        st.plotly_chart(fig_r, use_container_width=True)

    with col2:
        cnt_pred = pd.Series([labels[v] for v in yp]).value_counts()
        fig_p = px.bar(cnt_pred, title="Distribution Prédite",
                       color=cnt_pred.index,
                       color_discrete_map={v: colors_map[k] for k, v in labels.items()},
                       template="plotly_dark")
        st.plotly_chart(fig_p, use_container_width=True)

# ──────────────────────────────────────────────
# PAGE 3 – PERFORMANCE DES MODÈLES
# ──────────────────────────────────────────────
elif page == "📊 Performance des Modèles":
    st.title("📊 Performance des Modèles")

    for target_name, res, class_names in [
        ("BUY / SELL",       res_signal, ["SELL", "BUY"]),
        ("Volatilité High/Low", res_vol, ["Low Vol", "High Vol"]),
    ]:
        st.subheader(f"🎯 Target : {target_name}")

        model_names = list(res.keys())
        colors_bar  = ["#4C72B0", "#DD8452", "#55A868"]

        # Summary table
        rows = []
        for name in model_names:
            r = res[name]
            rows.append({
                "Modèle":      name,
                "Acc Globale": f"{r['global_acc']:.4f}",
                "Acc Moyenne": f"{r['mean_acc']:.4f}",
                "Std":         f"{r['std_acc']:.4f}",
                "Acc Min":     f"{min(r['acc_folds']):.4f}",
                "Acc Max":     f"{max(r['acc_folds']):.4f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)

        # Accuracy par fold
        with col1:
            fig_fold = go.Figure()
            for i, name in enumerate(model_names):
                fig_fold.add_trace(go.Bar(
                    name=name,
                    x=[f"Fold {j+1}" for j in range(N_SPLITS)],
                    y=res[name]["acc_folds"],
                    marker_color=colors_bar[i]
                ))
            fig_fold.update_layout(
                title="Accuracy par Fold (Walk-Forward)",
                barmode="group", template="plotly_dark",
                yaxis=dict(range=[0, 1]), height=350
            )
            st.plotly_chart(fig_fold, use_container_width=True)

        # Comparaison globale
        with col2:
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(
                x=model_names,
                y=[res[n]["global_acc"] for n in model_names],
                error_y=dict(type="data", array=[res[n]["std_acc"] for n in model_names]),
                marker_color=colors_bar,
                text=[f"{res[n]['global_acc']:.3f}" for n in model_names],
                textposition="outside"
            ))
            fig_comp.update_layout(
                title="Accuracy Globale (± std)",
                template="plotly_dark",
                yaxis=dict(range=[0, 1]), height=350
            )
            st.plotly_chart(fig_comp, use_container_width=True)

        # Matrices de confusion
        st.markdown("**Matrices de Confusion**")
        cols = st.columns(3)
        for i, name in enumerate(model_names):
            with cols[i]:
                cm = res[name]["cm"]
                fig_cm = px.imshow(
                    cm, text_auto=True,
                    x=class_names, y=class_names,
                    color_continuous_scale="Blues",
                    title=name, aspect="auto"
                )
                fig_cm.update_layout(template="plotly_dark", height=280,
                                     coloraxis_showscale=False)
                st.plotly_chart(fig_cm, use_container_width=True)

        # Classification report
        with st.expander(f"Classification Report détaillé – {target_name}"):
            best = max(model_names, key=lambda n: res[n]["global_acc"])
            st.markdown(f"**Meilleur modèle : {best}**")
            report = classification_report(
                res[best]["y_true"], res[best]["y_pred"],
                target_names=class_names, zero_division=0
            )
            st.code(report)

        st.markdown("---")

# ──────────────────────────────────────────────
# PAGE 4 – FEATURE IMPORTANCE
# ──────────────────────────────────────────────
elif page == "🔍 Feature Importance":
    st.title("🔍 Feature Importance – Random Forest")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("BUY / SELL")
        fig1 = px.bar(
            fi_signal.reset_index(),
            x="importance", y="index",
            orientation="h",
            color="importance",
            color_continuous_scale="Blues",
            labels={"index": "Feature", "importance": "Importance"},
            template="plotly_dark"
        )
        fig1.update_layout(height=500, coloraxis_showscale=False,
                           yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Volatilité High/Low")
        fig2 = px.bar(
            fi_vol.reset_index(),
            x="importance", y="index",
            orientation="h",
            color="importance",
            color_continuous_scale="Oranges",
            labels={"index": "Feature", "importance": "Importance"},
            template="plotly_dark"
        )
        fig2.update_layout(height=500, coloraxis_showscale=False,
                           yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig2, use_container_width=True)


# ──────────────────────────────────────────────
# PAGE 5 – BACKTESTING STRATÉGIE (AMÉLIORÉE)
# ──────────────────────────────────────────────
elif page == "💰 Backtesting Stratégie":
    st.title("💰 Backtesting de Stratégie – Signaux ML sur WTI")
    
    # En-tête explicatif
    with st.expander("ℹ️ Comment interpréter ce backtest ?"):
        st.markdown("""
        **🎯 Objectif :** Évaluer la performance d'une stratégie de trading basée sur les prédictions ML
        
        **📊 Indicateurs clés à surveiller :**
        - **Rendement Total** : Performance globale de la stratégie
        - **Alpha** : Sur-performance par rapport au Buy & Hold (positif = bonne stratégie)
        - **Sharpe Ratio** : Rendement ajusté au risque (>1 = bon, >2 = excellent)
        - **Max Drawdown** : Plus grosse perte subie (plus faible = mieux)
        - **Win Rate** : Pourcentage de trades gagnants
        - **Profit Factor** : Gains totaux / Pertes totales (>1.5 = bonne stratégie)
        
        **⚙️ Comment optimiser :**
        1. Commencez avec les paramètres par défaut
        2. Activez le filtre de volatilité pour éviter les marchés agités
        3. Ajustez Stop-Loss et Take-Profit pour protéger les gains
        4. Comparez les 3 modèles ML dans le tableau de comparaison
        """)
    
    st.markdown("---")
    
    # ── Paramètres dans une section claire ──
    st.markdown("### ⚙️ Configuration de la Stratégie")
    
    col_conf1, col_conf2, col_conf3 = st.columns(3)
    
    with col_conf1:
        st.markdown("**🤖 Modèle Principal**")
        bt_model = st.selectbox(
            "Modèle BUY/SELL", 
            ["Logistic Regression", "Random Forest", "XGBoost"], 
            key="bt_model",
            help="Modèle ML utilisé pour générer les signaux d'achat/vente"
        )
        
        st.markdown("**💰 Capital & Coûts**")
        capital_init = st.number_input("Capital initial ($)", value=10_000, step=1_000, key="bt_cap")
        cost_pct = st.slider("Coût de transaction (%)", 0.0, 0.5, 0.05, step=0.01, key="bt_cost") / 100
    
    with col_conf2:
        st.markdown("**🛡️ Gestion des Risques**")
        use_sl = st.checkbox("Activer Stop-Loss", value=False, key="bt_sl_on")
        sl_val = st.slider("Stop-Loss (%)", 0.1, 5.0, 1.0, step=0.1, key="bt_sl", disabled=not use_sl) / 100 if use_sl else None
        
        use_tp = st.checkbox("Activer Take-Profit", value=False, key="bt_tp_on")
        tp_val = st.slider("Take-Profit (%)", 0.1, 5.0, 2.0, step=0.1, key="bt_tp", disabled=not use_tp) / 100 if use_tp else None
    
    with col_conf3:
        st.markdown("**📈 Filtrage des Signaux**")
        use_vol_f = st.checkbox("Filtre Volatilité (Low Vol only)", value=False, key="bt_vf",
                                help="Ne trade que quand la volatilité est basse (marché calme)")
        vol_model_bt = st.selectbox(
            "Modèle Volatilité", 
            ["Random Forest", "XGBoost", "Logistic Regression"], 
            key="bt_vm",
            disabled=not use_vol_f,
            help="Modèle utilisé pour prédire la volatilité"
        ) if use_vol_f else None
    
    st.markdown("---")
    
    # ── Lancement ──
    with st.spinner("🔄 Simulation en cours..."):
        bt = run_backtest(
            df_signal, res_signal, bt_model,
            capital_init  = float(capital_init),
            cost_pct      = cost_pct,
            stop_loss_pct = sl_val,
            take_profit_pct = tp_val,
            vol_filter    = use_vol_f,
            df_vol        = df_vol if use_vol_f else None,
            res_vol       = res_vol if use_vol_f else None,
            vol_model     = vol_model_bt,
        )
    
    # ── KPI Cards Améliorées ──
    st.markdown("### 📊 Synthèse de Performance")
    
    # Calcul de métriques supplémentaires
    alpha = bt["total_ret"] - bt["bh_ret"]
    profit_factor = abs(bt['avg_win'] * bt['win_rate']) / (abs(bt['avg_loss']) * (100 - bt['win_rate']) + 1e-9) if bt['avg_win'] != 0 and bt['avg_loss'] != 0 else 0
    expectancy = (bt['avg_win'] * bt['win_rate'] / 100 + bt['avg_loss'] * (1 - bt['win_rate'] / 100)) if bt['n_trades'] > 0 else 0
    
    # Cartes de performance colorées
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        color_ret = "normal" if bt["total_ret"] > bt["bh_ret"] else "inverse"
        st.metric("📈 Rendement Total", f"{bt['total_ret']:+.2f}%", delta=f"vs BH {alpha:+.2f}%", delta_color=color_ret)
        st.caption(f"Buy & Hold: {bt['bh_ret']:+.2f}%")
    
    with col2:
        color_sharpe = "normal" if bt["sharpe"] > 1 else "inverse"
        st.metric("⚡ Sharpe Ratio", f"{bt['sharpe']:.2f}", delta=">1 = bon" if bt["sharpe"] > 1 else "<1 = risqué", delta_color=color_sharpe)
        st.caption(f"Max Drawdown: {bt['max_dd']:.2f}%")
    
    with col3:
        color_win = "normal" if bt["win_rate"] > 50 else "inverse"
        st.metric("🎯 Win Rate", f"{bt['win_rate']:.1f}%", delta=f"{bt['n_trades']} trades", delta_color=color_win)
        st.caption(f"Profit Factor: {profit_factor:.2f}")
    
    with col4:
        st.metric("💰 Espérance/Trade", f"{expectancy:+.3f}%")
        if bt['n_trades'] > 0:
            st.caption(f"Gain avg: {bt['avg_win']:+.2f}% | Perte avg: {bt['avg_loss']:+.2f}%")
        else:
            st.caption("Aucun trade effectué")
    
    st.markdown("---")
    
    # ── Graphiques de Performance Améliorés ──
    st.markdown("### 📈 Visualisation de la Performance")
    
    # Graphique combiné Equity + Positions
    fig_perf = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=("Courbe d'Equity vs Buy & Hold", "Drawdown", "Position du Marché")
    )
    
    # Equity curve
    fig_perf.add_trace(
        go.Scatter(x=bt["dates"], y=bt["equity"], name="Stratégie ML",
                   line=dict(color="#26a69a", width=2), fill="tozeroy"),
        row=1, col=1
    )
    fig_perf.add_trace(
        go.Scatter(x=bt["dates"], y=bt["bh_equity"], name="Buy & Hold",
                   line=dict(color="#ffa726", width=2, dash="dash")),
        row=1, col=1
    )
    
    # Drawdown
    fig_perf.add_trace(
        go.Scatter(x=bt["dates"], y=bt["drawdown"], name="Drawdown",
                   line=dict(color="#ef5350", width=1.5), fill="tozeroy"),
        row=2, col=1
    )
    
    # Positions (couleurs différentes pour LONG/SHORT/FLAT)
    positions_array = np.array(bt["positions_history"])
    position_colors = np.where(positions_array == 1, "#26a69a", 
                               np.where(positions_array == -1, "#ef5350", "#808080"))
    
    fig_perf.add_trace(
        go.Scatter(x=bt["dates"], y=positions_array, name="Position",
                   mode="lines+markers", line=dict(color="#90caf9", width=1),
                   marker=dict(size=3, color=position_colors)),
        row=3, col=1
    )
    
    fig_perf.add_hline(y=0, line_dash="dot", line_color="gray", row=3, col=1)
    fig_perf.add_annotation(x=bt["dates"][0], y=1.05, text="LONG", showarrow=False,
                           font=dict(color="#26a69a"), row=3, col=1)
    fig_perf.add_annotation(x=bt["dates"][0], y=-1.05, text="SHORT", showarrow=False,
                           font=dict(color="#ef5350"), row=3, col=1)
    
    fig_perf.update_layout(height=700, template="plotly_dark", showlegend=True)
    fig_perf.update_xaxes(title_text="Date", row=3, col=1)
    fig_perf.update_yaxes(title_text="Equity ($)", row=1, col=1)
    fig_perf.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    fig_perf.update_yaxes(title_text="Position", row=3, col=1, tickvals=[-1,0,1], ticktext=["SHORT","FLAT","LONG"])
    
    st.plotly_chart(fig_perf, use_container_width=True)
    
    # ── Analyse des Trades ──
    if bt["n_trades"] > 0:
        st.markdown("---")
        st.markdown("### 🔬 Analyse Détaillée des Trades")
        
        # Distribution des P&L
        col_left, col_right = st.columns(2)
        
        with col_left:
            # Histogramme des P&L
            fig_pnl_dist = go.Figure()
            fig_pnl_dist.add_trace(go.Histogram(
                x=bt["trades_df"]["pnl_pct"],
                nbinsx=30,
                marker_color="#4C72B0",
                opacity=0.8,
                name="Distribution P&L"
            ))
            fig_pnl_dist.add_vline(x=0, line_color="white", line_dash="dash")
            fig_pnl_dist.add_vline(x=bt["trades_df"]["pnl_pct"].mean(), 
                                   line_color="#ffa726", line_dash="dash",
                                   annotation_text=f"μ={bt['trades_df']['pnl_pct'].mean():.2f}%")
            fig_pnl_dist.update_layout(
                title="Distribution des P&L par Trade",
                height=350,
                template="plotly_dark",
                xaxis_title="P&L (%)",
                yaxis_title="Nombre de trades"
            )
            st.plotly_chart(fig_pnl_dist, use_container_width=True)
        
        with col_right:
            # Performance par direction (LONG vs SHORT)
            perf_by_dir = bt["trades_df"].groupby("direction").agg({
                "pnl_pct": ["count", "mean", "sum"]
            }).round(2)
            perf_by_dir.columns = ["Nb Trades", "P&L Moyen (%)", "P&L Total (%)"]
            perf_by_dir = perf_by_dir.reset_index()
            
            fig_dir = go.Figure()
            fig_dir.add_trace(go.Bar(
                x=perf_by_dir["direction"],
                y=perf_by_dir["P&L Moyen (%)"],
                marker_color=["#26a69a" if d == "LONG" else "#ef5350" for d in perf_by_dir["direction"]],
                text=perf_by_dir["P&L Moyen (%)"],
                textposition="outside",
                name="P&L Moyen"
            ))
            fig_dir.update_layout(
                title="Performance par Direction",
                height=350,
                template="plotly_dark",
                xaxis_title="Direction",
                yaxis_title="P&L Moyen (%)"
            )
            st.plotly_chart(fig_dir, use_container_width=True)
        
        # Tableau des trades récents
        st.markdown("#### 📋 10 Derniers Trades")
        recent_trades = bt["trades_df"].tail(10).copy()
        recent_trades["entry_date"] = pd.to_datetime(recent_trades["entry_date"]).dt.strftime("%Y-%m-%d %H:%M")
        recent_trades["exit_date"] = pd.to_datetime(recent_trades["exit_date"]).dt.strftime("%Y-%m-%d %H:%M")
        recent_trades["pnl_pct"] = recent_trades["pnl_pct"].round(2)
        
        # Colorer les lignes selon le gain/perte
        def color_pnl(val):
            color = '#26a69a' if val > 0 else '#ef5350' if val < 0 else 'white'
            return f'color: {color}'
        
        st.dataframe(
            recent_trades[["entry_date", "exit_date", "direction", "entry_price", "exit_price", "pnl_pct", "exit_reason"]],
            use_container_width=True,
            hide_index=True
        )
    
    else:
        st.warning("⚠️ Aucun trade généré avec ces paramètres. Essayez d'ajuster les seuils ou de désactiver certains filtres.")
    
    st.markdown("---")
    
    # ── Comparaison multi-modèles (simplifiée et plus claire) ──
    st.markdown("### 🏆 Comparaison des Modèles ML")
    st.markdown("*Quel modèle performe le mieux selon vos paramètres ?*")
    
    # Sélection du métrique de comparaison
    metric_choice = st.selectbox(
        "Métrique de comparaison",
        ["Rendement Total (%)", "Sharpe Ratio", "Win Rate (%)", "Profit Factor"],
        key="compare_metric"
    )
    
    comp_data = []
    model_colors = {"Logistic Regression": "#4C72B0", "Random Forest": "#DD8452", "XGBoost": "#55A868"}
    
    for m_name in ["Logistic Regression", "Random Forest", "XGBoost"]:
        with st.spinner(f"Simulation du modèle {m_name}..."):
            bt_m = run_backtest(
                df_signal, res_signal, m_name,
                capital_init=float(capital_init),
                cost_pct=cost_pct,
                stop_loss_pct=sl_val,
                take_profit_pct=tp_val,
                vol_filter=use_vol_f,
                df_vol=df_vol if use_vol_f else None,
                res_vol=res_vol if use_vol_f else None,
                vol_model=vol_model_bt,
            )
            
            pf = abs(bt_m['avg_win'] * bt_m['win_rate']) / (abs(bt_m['avg_loss']) * (100 - bt_m['win_rate']) + 1e-9) if bt_m['avg_win'] != 0 and bt_m['avg_loss'] != 0 else 0
            
            comp_data.append({
                "Modèle": m_name,
                "Rendement Total (%)": round(bt_m["total_ret"], 2),
                "Alpha vs BH (%)": round(bt_m["total_ret"] - bt_m["bh_ret"], 2),
                "Sharpe Ratio": round(bt_m["sharpe"], 2),
                "Win Rate (%)": round(bt_m["win_rate"], 1),
                "Profit Factor": round(pf, 2),
                "Max Drawdown (%)": round(bt_m["max_dd"], 2),
                "Nb Trades": bt_m["n_trades"],
                "Couleur": model_colors[m_name]
            })
    
    comp_df = pd.DataFrame(comp_data)
    
    # Graphique de comparaison
    if metric_choice == "Rendement Total (%)":
        y_col = "Rendement Total (%)"
        y_title = "Rendement Total (%)"
    elif metric_choice == "Sharpe Ratio":
        y_col = "Sharpe Ratio"
        y_title = "Sharpe Ratio"
    elif metric_choice == "Win Rate (%)":
        y_col = "Win Rate (%)"
        y_title = "Win Rate (%)"
    else:
        y_col = "Profit Factor"
        y_title = "Profit Factor"
    
    fig_compare = go.Figure()
    for _, row in comp_df.iterrows():
        fig_compare.add_trace(go.Bar(
            name=row["Modèle"],
            x=[row["Modèle"]],
            y=[row[y_col]],
            marker_color=row["Couleur"],
            text=[f"{row[y_col]:.2f}"],
            textposition="outside"
        ))
    
    fig_compare.update_layout(
        title=f"Comparaison des Modèles - {metric_choice}",
        template="plotly_dark",
        height=400,
        yaxis_title=y_title,
        showlegend=False
    )
    st.plotly_chart(fig_compare, use_container_width=True)
    
    # Tableau complet
    st.markdown("#### 📊 Tableau Comparatif Complet")
    st.dataframe(
        comp_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Modèle": st.column_config.TextColumn("Modèle"),
            "Rendement Total (%)": st.column_config.NumberColumn("Rendement Total (%)", format="%.2f"),
            "Alpha vs BH (%)": st.column_config.NumberColumn("Alpha vs BH (%)", format="%.2f"),
            "Sharpe Ratio": st.column_config.NumberColumn("Sharpe Ratio", format="%.2f"),
            "Win Rate (%)": st.column_config.NumberColumn("Win Rate (%)", format="%.1f"),
            "Profit Factor": st.column_config.NumberColumn("Profit Factor", format="%.2f"),
            "Max Drawdown (%)": st.column_config.NumberColumn("Max Drawdown (%)", format="%.2f"),
            "Nb Trades": st.column_config.NumberColumn("Nb Trades", format="%d"),
        }
    )
    
    # Meilleur modèle recommendation
    best_model_row = comp_df.loc[comp_df[y_col].idxmax()]
    st.success(f"🏆 **Recommandation :** Le modèle **{best_model_row['Modèle']}** offre les meilleures performances selon **{metric_choice}** avec {best_model_row[y_col]:.2f}!")
    
    st.markdown("---")
    st.caption("💡 **Astuce :** Ajustez les paramètres de gestion des risques dans le panneau de configuration pour optimiser la stratégie selon votre profil.")