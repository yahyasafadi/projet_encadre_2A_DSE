# 🛢️ Étude des Données Horaires — WTI Crude Oil

Module **`etude_data_horaire`** du projet *projet_encadre_2A_DSE*.

Ce module construit un pipeline complet de **collecte → fusion → feature engineering → modélisation ML → backtesting → dashboard** sur des données **horaires (1h)** du pétrole **WTI**, enrichies par le **Brent**, l'**indice du dollar (USD Index)** et le **sentiment des actualités** (NLP).

L'objectif est de prédire, à un horizon de **6 heures** :
1. Un **signal de trading BUY / SELL** (les zones neutres "HOLD" sont exclues de l'entraînement) ;
2. Un régime de **volatilité High / Low**.

---

## 📂 Structure du dossier

```
etude_data_horaire/
├── oil_commodity.py              # Collecte des prix horaires (WTI, Brent, USD Index) via yfinance
├── news_api.py                   # Collecte des actualités pétrolières (Google News) agrégées par heure
├── merge_datasets.py             # Fusion prix + actualités → data_merged.csv
├── calcule_hourly_indicators.py  # Feature engineering technique + sentiment (FinBERT) → wti_features_v3.csv
├── model_training.py             # Entraînement & évaluation des modèles ML (walk-forward) → outputs/
├── dashboard.py                  # Dashboard interactif Streamlit (exploration + backtest)
│
├── oil_intraday_1h.csv           # Données brutes horaires WTI / Brent / USD Index (OHLCV)
├── news_wti_hourly_7d.csv        # Actualités "WTI crude oil" agrégées par heure
├── data_merged.csv               # Fusion prix + actualités (sortie de merge_datasets.py)
├── wti_features_v3.csv           # Features finales prêtes pour le ML (sortie de calcule_hourly_indicators.py)
├── wti_features_with_targets.csv # Version antérieure avec targets pré-calculées (3 classes SELL/HOLD/BUY) — fichier de référence/legacy
│
└── outputs/                      # Résultats générés par model_training.py
    ├── report_buysell.png            # Rapport visuel complet — modèle BUY/SELL
    ├── report_volatility.png         # Rapport visuel complet — modèle Volatilité
    ├── feat_importance_buysell.png   # Importance des features — BUY/SELL
    ├── feat_importance_volatility.png# Importance des features — Volatilité
    ├── buysell_report.png            # (variante / run antérieur)
    ├── feature_importance_buysell.png# (variante / run antérieur)
    ├── summary_all_targets.csv       # Tableau récapitulatif des accuracies (2 targets × 3 modèles)
    └── summary_buysell.csv           # Tableau récapitulatif — BUY/SELL uniquement
```

---

## 🔄 Pipeline de données

```
┌─────────────────┐     ┌──────────────────┐
│  oil_commodity.py│     │   news_api.py    │
│  (yfinance)      │     │ (Google News)    │
│  WTI/Brent/USD   │     │ titres + count   │
└────────┬─────────┘     └────────┬─────────┘
         │ oil_intraday_1h.csv    │ news_wti_hourly_7d.csv
         └───────────┬────────────┘
                      ▼
            merge_datasets.py
                      │ data_merged.csv
                      ▼
        calcule_hourly_indicators.py
        (indicateurs techniques + FinBERT sentiment)
                      │ wti_features_v3.csv
                      ▼
              model_training.py
        (LogReg / RandomForest / XGBoost
         walk-forward TimeSeriesSplit)
                      │
                      ▼
                  outputs/
        (rapports PNG + CSV récapitulatifs)

              dashboard.py
   (Streamlit — exploration interactive,
    prédictions, performance, backtesting)
```

---

## 📜 Description des scripts

### 1. `oil_commodity.py` — Collecte des prix
Télécharge via **`yfinance`** les bougies horaires (`interval="1h"`, fenêtre `period="15d"`) pour :

| Actif | Ticker |
|---|---|
| WTI Crude Oil | `CL=F` |
| US Dollar Index | `DX-Y.NYB` |
| Brent Crude Oil | `BZ=F` |

Pour chaque actif : `Open, High, Low, Close, Volume`. Les résultats sont **concaténés à l'historique existant** (`oil_intraday_1h.csv`) en supprimant les doublons d'horodatage (on garde la valeur la plus récente). Comme l'API Yahoo Finance ne fournit que ~15 jours de données horaires à la fois, ce script est **conçu pour être exécuté régulièrement** (ex. tâche planifiée) afin de construire un historique long terme.

### 2. `news_api.py` — Collecte des actualités
Utilise **`pygooglenews`** pour rechercher les actualités `"WTI crude oil"` jour par jour sur les **7 derniers jours**. Pour chaque article : titre, source, lien, date de publication. Les titres sont ensuite **agrégés par heure** (`resample("1h")`) :
- `title` → concaténation des titres de l'heure (séparateur ` | `)
- `news_count` → nombre d'articles publiés dans l'heure

Le résultat est fusionné avec l'historique existant (`news_wti_hourly_7d.csv`), même logique d'accumulation incrémentale que `oil_commodity.py`.

### 3. `merge_datasets.py` — Fusion prix + actualités
Fusionne `oil_intraday_1h.csv` (prix) et `news_wti_hourly_7d.csv` (actualités) sur l'horodatage (`left join` côté prix). Les heures sans actualité reçoivent `news_count = 0` et `title = "no_news_available"`. → **`data_merged.csv`**

### 4. `calcule_hourly_indicators.py` — Feature engineering
Calcule les indicateurs techniques et le score de sentiment NLP :

| Feature | Description |
|---|---|
| `Close_SMA20_ratio` | Close / Moyenne mobile simple 20 périodes |
| `EMA_ratio` | EMA(12) / EMA(26) |
| `RSI_14` | Relative Strength Index (14 périodes) |
| `return_12h` | Rendement sur 12 heures glissantes |
| `Spread_Brent_WTI` | Brent Close − WTI Close |
| `Corr_Brent_USD_30` | Corrélation glissante (30 périodes) Brent / USD |
| `ADX_proxy` | Ratio ATR(14)/ATR(48) lissé — proxy de force de tendance |
| `session_europe` | Flag binaire : session horaire européenne (9h–15h) |
| `MACD_slope` | Variation du MACD (EMA12-EMA26 vs sa ligne de signal) sur 3 pas |
| `volatility_ratio` | Volatilité court terme (6h) / volatilité long terme (24h) |
| `hour_sin`, `hour_cos` | Encodage cyclique de l'heure de la journée |
| `volume_ratio` | Volume / moyenne mobile du volume (24h) — détection de pics |
| `USD_return` | Rendement de l'USD Index sur 12 heures |
| `sentiment_score` | Score de sentiment moyen des titres de l'heure (**FinBERT**, `ProsusAI/finbert`), `+` positif / `-` négatif, `ffill` si pas de news |
| `sentiment_available` | Flag : sentiment réellement calculé pour cette heure (avant `ffill`) |

Le sentiment est calculé via le pipeline `transformers` (`ProsusAI/finbert`) sur chaque titre d'actualité de l'heure, puis moyenné. → **`wti_features_v3.csv`**

> ⚠️ Le script affiche `"wti_features_v4.csv exporté"` dans son message de log mais écrit bien dans `wti_features_v3.csv` (incohérence de nommage à corriger).

### 5. `model_training.py` — Entraînement & évaluation
Pipeline ML complet, sans fuite temporelle (*anti-leakage*) :

**Construction des cibles** (horizon `HORIZON = 6` heures) :
- `future_return = Close(t+6h) / Close(t) − 1`
- **Signal BUY/SELL** : `BUY (1)` si `future_return ≥ +0.3%`, `SELL (0)` si `≤ −0.3%`, sinon ligne **supprimée** (zone neutre exclue de l'entraînement classification)
- **Volatilité** : `future_vol` = écart-type glissant des rendements sur 6h (décalé) ; `High (1)` si ≥ 75ᵉ percentile, sinon `Low (0)`

**Modèles comparés** :
| Modèle | Configuration |
|---|---|
| Logistic Regression | `StandardScaler` + `C=0.1`, `class_weight="balanced"` |
| Random Forest | 200 arbres, `max_depth=8`, `min_samples_leaf=20`, `class_weight="balanced"` |
| XGBoost | 300 arbres, `max_depth=5`, `learning_rate=0.05`, `subsample/colsample=0.8` |

**Validation** : `TimeSeriesSplit` walk-forward, **5 folds**, **500 lignes** de test par fold — garantit que le modèle n'est jamais entraîné sur des données futures par rapport au test.

**Sorties** : rapports visuels (accuracy par fold, matrices de confusion, comparaison globale), feature importance (Random Forest), CSV récapitulatif → `outputs/`.

> ⚠️ Les chemins `FEATURES_PATH`, `MERGED_PATH`, `OUTPUT_DIR` sont **codés en dur** (chemins Windows locaux). À adapter avant exécution sur une autre machine (idéalement via chemins relatifs ou variables d'environnement).

### 6. `dashboard.py` — Dashboard interactif (Streamlit)
Reprend la même logique de chargement/entraînement que `model_training.py` (mise en cache via `@st.cache_data`) et expose **5 pages** de navigation :

| Page | Contenu |
|---|---|
| 📈 **Prix & Marché** | KPI temps réel (Close, Brent, USD Index, RSI, Spread), chandelier OHLC + volume, RSI, MACD slope, comparaison WTI/Brent/USD |
| 🎯 **Prédictions vs Réalité** | Sélection target (BUY/SELL ou Volatilité) + modèle, visualisation des prédictions face aux vraies classes |
| 📊 **Performance des Modèles** | Accuracy par fold, matrices de confusion, comparaison des 3 modèles |
| 🔍 **Feature Importance** | Importances Random Forest pour chaque target |
| 💰 **Backtesting Stratégie** | Simulation de trading paramétrable : capital initial, coûts de transaction, Stop-Loss / Take-Profit, filtre de volatilité ; métriques : rendement total, Alpha vs Buy & Hold, Sharpe Ratio, Max Drawdown, Win Rate, Profit Factor, espérance par trade |

**Lancer le dashboard :**
```bash
streamlit run dashboard.py
```

---

## 📊 Données

| Fichier | Lignes | Période | Description |
|---|---|---|---|
| `oil_intraday_1h.csv` | ~13 400 | 2024-01-02 → en continu | OHLCV horaire WTI / USD Index / Brent |
| `news_wti_hourly_7d.csv` | ~17 800 | accumulé en continu | Titres d'actualités agrégés par heure + compteur |
| `data_merged.csv` | ~13 400 | idem prix | Fusion prix + actualités |
| `wti_features_v3.csv` | ~13 150 | 2024-01-04 → 2026-05/06 | Features finales (16 colonnes) prêtes pour le ML |
| `wti_features_with_targets.csv` | ~13 140 | idem | Version antérieure avec `target_class` (3 classes : -1 SELL / 0 HOLD / 1 BUY) et `target_volatility` continue — conservée à titre de référence, **non utilisée** par le pipeline ML actuel |

---

## 📈 Résultats actuels (`outputs/summary_all_targets.csv`)

| Target | Modèle | Accuracy moyenne (walk-forward) |
|---|---|---|
| BUY / SELL | Logistic Regression | 0.539 ± 0.035 |
| BUY / SELL | Random Forest | 0.532 ± 0.029 |
| BUY / SELL | XGBoost | 0.526 ± 0.043 |
| Volatilité | Logistic Regression | 0.615 ± 0.093 |
| Volatilité | Random Forest | 0.692 ± 0.031 |
| Volatilité | XGBoost | **0.704 ± 0.073** |

**Lecture :**
- La prédiction du **signal directionnel BUY/SELL** reste proche du hasard (~53 %) — cohérent avec la difficulté connue de prédire la direction des prix à court terme sur un marché efficient.
- La prédiction du **régime de volatilité** est nettement plus exploitable (~70 % avec XGBoost), la volatilité étant statistiquement plus persistante/prévisible que la direction.

---

## ⚙️ Installation

```bash
pip install pandas numpy yfinance pygooglenews transformers torch \
            scikit-learn xgboost matplotlib plotly streamlit
```

> `transformers` télécharge le modèle `ProsusAI/finbert` (~440 Mo) au premier lancement de `calcule_hourly_indicators.py`.

---

## ▶️ Ordre d'exécution

```bash
# 1. Collecte des prix (à planifier régulièrement, ex. toutes les 12h)
python oil_commodity.py

# 2. Collecte des actualités (à planifier régulièrement)
python news_api.py

# 3. Fusion des deux sources
python merge_datasets.py

# 4. Calcul des features techniques + sentiment NLP
python calcule_hourly_indicators.py

# 5. Entraînement, évaluation walk-forward et génération des rapports
python model_training.py

# 6. (Optionnel) Dashboard interactif
streamlit run dashboard.py
```

---

## ⚠️ Points d'attention / limites connues

- **Chemins codés en dur** dans `model_training.py` et `dashboard.py` (chemins Windows) → à remplacer par des chemins relatifs.
- **Incohérence de log** dans `calcule_hourly_indicators.py` (`v4` affiché vs `v3` écrit).
- Collecte de prix limitée à **15 jours en arrière** par appel (limite Yahoo Finance sur l'intraday horaire) → nécessite une **exécution planifiée régulière** pour constituer un historique continu, sans quoi des trous apparaissent dans `oil_intraday_1h.csv`.
- `pygooglenews` dépend du scraping de Google News (RSS) — sensible aux changements de structure / rate limiting.
- Le signal BUY/SELL a une accuracy proche de l'aléatoire (~53 %) : à considérer comme une base de référence pour itérer (autres features, autre horizon, autre seuil de classification) plutôt qu'une stratégie prête à l'emploi.

## 💡 Pistes d'amélioration

- Tester d'autres horizons de prédiction (`HORIZON`) et seuils BUY/SELL.
- Ajouter des features macro (stocks EIA, OPEP, géopolitique) ou des features de microstructure (carnet d'ordres si disponible).
- Remplacer la classification binaire BUY/SELL par une régression du rendement futur.
- Ajouter une validation out-of-sample plus longue / un walk-forward avec ré-entraînement périodique en production.
- Versionner les modèles entraînés (ex. `joblib`/`mlflow`) plutôt que de les ré-entraîner à chaque lancement du dashboard.
