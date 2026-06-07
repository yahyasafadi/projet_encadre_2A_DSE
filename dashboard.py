import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG PAGE ──────────────────────────────────────────────────
st.set_page_config(
    page_title = "WTI Oil Dashboard",
    page_icon  = "🛢️",
    layout     = "wide"
)

st.title("🛢️ WTI Crude Oil — AI Prediction Dashboard")
st.markdown("Système de prédiction du prix du pétrole WTI à fréquence 1 minute")

# ── CHARGEMENT ───────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('wti_features_1min.csv', parse_dates=['Datetime'])
    df = df.sort_values('Datetime').reset_index(drop=True)
    return df

@st.cache_data
def run_model(df):
    exclude  = ['Datetime', 'target', 'WTI_Close']
    features = [c for c in df.columns if c not in exclude]
    X = df[features]
    y = df['target']

    scaler = StandardScaler()
    tscv   = TimeSeriesSplit(n_splits=5)
    results = []
    all_preds   = np.zeros(len(df))
    all_probas  = np.zeros(len(df))

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train = X.iloc[train_idx]
        X_test  = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test  = y.iloc[test_idx]

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

        y_pred       = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

        all_preds[test_idx]  = y_pred
        all_probas[test_idx] = y_pred_proba

        results.append({
            'fold'    : fold + 1,
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'auc_roc' : roc_auc_score(y_test, y_pred_proba)
        })

    return pd.DataFrame(results), model, scaler, features, all_preds, all_probas

def generate_signals(preds, probas, threshold_buy=0.6, threshold_sell=0.4):
    """Génère BUY / SELL / HOLD selon la probabilité"""
    signals = []
    for pred, proba in zip(preds, probas):
        if proba >= threshold_buy:
            signals.append('BUY')
        elif proba <= threshold_sell:
            signals.append('SELL')
        else:
            signals.append('HOLD')
    return signals

# ── CHARGEMENT ───────────────────────────────────────────────────
with st.spinner("⏳ Chargement des données..."):
    df = load_data()

st.success(f"✅ {len(df)} lignes — du {df['Datetime'].min()} au {df['Datetime'].max()}")

# ── SIDEBAR ──────────────────────────────────────────────────────
st.sidebar.header("⚙️ Paramètres")
st.sidebar.markdown("**Dashboard Information**")
st.sidebar.info(f"Last Update: {df['Datetime'].max()}")
st.sidebar.markdown("**Models Used:**")
st.sidebar.markdown("- XGBoost\n- Random Forest (baseline)")

threshold_buy  = st.sidebar.slider("Seuil BUY",  0.5, 1.0, 0.6, 0.05)
threshold_sell = st.sidebar.slider("Seuil SELL", 0.0, 0.5, 0.4, 0.05)
n_points = st.sidebar.slider("Points affichés", 100, len(df), 500, 100)

# ── PRIX ACTUEL ──────────────────────────────────────────────────
last_price = df['WTI_Close'].iloc[-1]
prev_price = df['WTI_Close'].iloc[-2]
change_pct = ((last_price - prev_price) / prev_price) * 100

# ── LANCEMENT MODÈLE ─────────────────────────────────────────────
st.header("🤖 Modèle XGBoost")

if st.button("🚀 Lancer l'entraînement et générer les signaux"):
    with st.spinner("⏳ Entraînement en cours..."):
        df_results, model, scaler, features, all_preds, all_probas = run_model(df)

    # Génération des signaux
    signals = generate_signals(all_preds, all_probas, threshold_buy, threshold_sell)
    df['signal']     = 'HOLD'
    df['signal']     = signals
    df['proba_up']   = all_probas

    last_signal = df['signal'].iloc[-1]
    last_proba  = df['proba_up'].iloc[-1]
    acc_mean    = df_results['accuracy'].mean()

    # ── MÉTRIQUES PRINCIPALES ────────────────────────────────────
    st.header("📊 Market Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💰 Current WTI Price", f"${last_price:.2f}", f"{change_pct:+.2f}%")
    col2.metric("🎯 Signal actuel",     last_signal)
    col3.metric("📊 Confidence",        f"{last_proba*100:.1f}%")
    col4.metric("✅ Model Accuracy",    f"{acc_mean*100:.2f}%")

    # ── SIGNAL PRINCIPAL ─────────────────────────────────────────
    st.subheader("🔮 AI Prediction — prochaine minute")
    if last_signal == 'BUY':
        st.success(f"📈 BUY SIGNAL | Confidence: {last_proba*100:.1f}%")
    elif last_signal == 'SELL':
        st.error(f"📉 SELL SIGNAL | Confidence: {(1-last_proba)*100:.1f}%")
    else:
        st.warning(f"⏸️ HOLD SIGNAL | Confidence: {last_proba*100:.1f}%")

    # ── GRAPHIQUE PRIX + SIGNAUX ──────────────────────────────────
    st.subheader("📈 Prix WTI avec signaux BUY/SELL")
    df_display = df.tail(n_points)

    fig = go.Figure()

    # Prix
    fig.add_trace(go.Scatter(
        x    = df_display['Datetime'],
        y    = df_display['WTI_Close'],
        name = 'WTI Close',
        line = dict(color='#2196F3', width=1)
    ))

    # BUY signals
    df_buy = df_display[df_display['signal'] == 'BUY']
    fig.add_trace(go.Scatter(
        x      = df_buy['Datetime'],
        y      = df_buy['WTI_Close'],
        mode   = 'markers',
        name   = 'BUY',
        marker = dict(color='green', size=8, symbol='triangle-up')
    ))

    # SELL signals
    df_sell = df_display[df_display['signal'] == 'SELL']
    fig.add_trace(go.Scatter(
        x      = df_sell['Datetime'],
        y      = df_sell['WTI_Close'],
        mode   = 'markers',
        name   = 'SELL',
        marker = dict(color='red', size=8, symbol='triangle-down')
    ))

    fig.update_layout(
        xaxis_title = "Datetime",
        yaxis_title = "Prix ($)",
        height      = 450,
        hovermode   = 'x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── DISTRIBUTION DES SIGNAUX ─────────────────────────────────
    st.subheader("🎯 Trading Signal Distribution")
    col1, col2 = st.columns(2)

    signal_counts = df['signal'].value_counts()
    n_buy  = signal_counts.get('BUY',  0)
    n_sell = signal_counts.get('SELL', 0)
    n_hold = signal_counts.get('HOLD', 0)

    with col1:
        st.markdown(f"- **BUY Signals** : {n_buy}")
        st.markdown(f"- **SELL Signals** : {n_sell}")
        st.markdown(f"- **HOLD Signals** : {n_hold}")

    with col2:
        fig_pie = px.pie(
            values = [n_buy, n_sell, n_hold],
            names  = ['BUY', 'SELL', 'HOLD'],
            color  = ['BUY', 'SELL', 'HOLD'],
            color_discrete_map = {
                'BUY' : 'green',
                'SELL': 'red',
                'HOLD': 'steelblue'
            },
            title = "BUY / SELL / HOLD Distribution"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── PERFORMANCE PAR FOLD ─────────────────────────────────────
    st.subheader("📊 Model Performance Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{df_results['accuracy'].mean()*100:.2f}%")
    col2.metric("F1-score", f"{df_results['f1_score'].mean()*100:.2f}%")
    col3.metric("AUC-ROC",  f"{df_results['auc_roc'].mean()*100:.2f}%")

    fig_perf = go.Figure()
    fig_perf.add_trace(go.Scatter(
        x=df_results['fold'], y=df_results['accuracy'],
        name='Accuracy', mode='lines+markers', line=dict(color='blue')
    ))
    fig_perf.add_trace(go.Scatter(
        x=df_results['fold'], y=df_results['f1_score'],
        name='F1-score', mode='lines+markers', line=dict(color='orange')
    ))
    fig_perf.add_trace(go.Scatter(
        x=df_results['fold'], y=df_results['auc_roc'],
        name='AUC-ROC', mode='lines+markers', line=dict(color='green')
    ))
    fig_perf.update_layout(
        title       = "Performance par fold (TimeSeriesSplit)",
        xaxis_title = "Fold",
        yaxis_title = "Score",
        yaxis_range = [0, 1],
        height      = 350
    )
    st.plotly_chart(fig_perf, use_container_width=True)

    # ── INDICATEURS TECHNIQUES ───────────────────────────────────
    st.subheader("🔧 Indicateurs techniques")
    df_display2 = df.tail(n_points)
    col1, col2 = st.columns(2)

    with col1:
        fig_rsi = px.line(df_display2, x='Datetime', y='RSI_14',
                          title='RSI 14', color_discrete_sequence=['purple'])
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red",   annotation_text="Surachat")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Survente")
        fig_rsi.update_layout(height=300)
        st.plotly_chart(fig_rsi, use_container_width=True)

    with col2:
        fig_macd = px.line(df_display2, x='Datetime', y='MACD_slope',
                           title='MACD Slope', color_discrete_sequence=['orange'])
        fig_macd.update_layout(height=300)
        st.plotly_chart(fig_macd, use_container_width=True)

    # ── DONNÉES BRUTES ───────────────────────────────────────────
    st.subheader("🗃️ Données brutes")
    if st.checkbox("Afficher les données"):
        st.dataframe(df.tail(100)[['Datetime', 'WTI_Close', 'RSI_14',
                                    'MACD_slope', 'sentiment_score',
                                    'signal', 'proba_up', 'target']])
else:
    # Avant le lancement — afficher quand même le prix
    st.header("📊 Market Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💰 Current WTI Price", f"${last_price:.2f}", f"{change_pct:+.2f}%")
    col2.metric("🎯 Signal actuel",     "En attente...")
    col3.metric("📊 Confidence",        "—")
    col4.metric("✅ Model Accuracy",    "—")

    st.info("👆 Cliquez sur **Lancer l'entraînement** pour générer les signaux BUY/SELL/HOLD")

    # Graphique prix simple
    st.subheader("📈 Prix WTI — 1 minute")
    df_display = df.tail(n_points)
    fig = px.line(df_display, x='Datetime', y='WTI_Close',
                  title='WTI Close Price', color_discrete_sequence=['#2196F3'])
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)