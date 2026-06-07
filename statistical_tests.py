import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── CHARGEMENT ───────────────────────────────────────────────────
print("⏳ Chargement des données 1 minute...")
df = pd.read_csv('wti_features_1min.csv', parse_dates=['Datetime'])
df = df.sort_values('Datetime').reset_index(drop=True)

exclude  = ['Datetime', 'target', 'WTI_Close']
features = [c for c in df.columns if c not in exclude]
X = df[features].dropna()

print(f"✅ {len(X)} lignes | {len(features)} features\n")

# ════════════════════════════════════════════════════════════════
# TEST 1 — ADF (Augmented Dickey-Fuller)
# Vérifie si chaque feature est stationnaire
# H0 : la série a une racine unitaire (non stationnaire)
# Si p-value < 0.05 → on rejette H0 → série STATIONNAIRE ✅
# ════════════════════════════════════════════════════════════════
print("=" * 55)
print("TEST ADF — STATIONNARITÉ DES FEATURES")
print("=" * 55)
print(f"{'Feature':<25} {'ADF Stat':>10} {'p-value':>10} {'Résultat':>12}")
print("-" * 55)

adf_results = []
for col in features:
    serie = X[col].dropna()
    if len(serie) < 20:
        continue
    if serie.std() == 0:
        print(f"{col:<25} {'N/A':>10} {'N/A':>10} {'⚠️  Constante':>20}")
        continue
    adf_stat, p_value, _, _, _, _ = adfuller(serie, autolag='AIC')
    stationnaire = "✅ Stationnaire" if p_value < 0.05 else "❌ Non stationnaire"
    adf_results.append({
        'feature'     : col,
        'adf_stat'    : adf_stat,
        'p_value'     : p_value,
        'stationnaire': p_value < 0.05
    })
    print(f"{col:<25} {adf_stat:>10.3f} {p_value:>10.4f} {stationnaire:>20}")

df_adf = pd.DataFrame(adf_results)
n_stat     = df_adf['stationnaire'].sum()
n_non_stat = (~df_adf['stationnaire']).sum()
print(f"\n📊 Résumé ADF : {n_stat} stationnaires | {n_non_stat} non stationnaires")

# ════════════════════════════════════════════════════════════════
# TEST 2 — VIF (Variance Inflation Factor)
# Vérifie la multicolinéarité entre les features
# VIF < 5  → pas de problème ✅
# VIF 5-10 → multicolinéarité modérée ⚠️
# VIF > 10 → multicolinéarité forte ❌ (feature à supprimer)
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("TEST VIF — MULTICOLINÉARITÉ DES FEATURES")
print("=" * 55)
print(f"{'Feature':<25} {'VIF':>10} {'Résultat':>15}")
print("-" * 55)

X_vif = X[features].dropna()
vif_results = []
for i, col in enumerate(X_vif.columns):
    vif = variance_inflation_factor(X_vif.values, i)
    if vif < 5:
        statut = "✅ OK"
    elif vif < 10:
        statut = "⚠️  Modéré"
    else:
        statut = "❌ Élevé"
    vif_results.append({'feature': col, 'vif': vif, 'statut': statut})
    print(f"{col:<25} {vif:>10.2f} {statut:>15}")

df_vif = pd.DataFrame(vif_results)
features_ok      = df_vif[df_vif['vif'] < 5]['feature'].tolist()
features_probleme = df_vif[df_vif['vif'] >= 10]['feature'].tolist()

print(f"\n📊 Résumé VIF :")
print(f"   ✅ Features OK (VIF<5)    : {len(features_ok)}")
print(f"   ❌ Features à risque (VIF>10) : {len(features_probleme)}")
if features_probleme:
    print(f"   → À considérer supprimer : {features_probleme}")

# ════════════════════════════════════════════════════════════════
# GRAPHIQUE — Résumé visuel
# ════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# ADF p-values
colors_adf = ['green' if p < 0.05 else 'red' for p in df_adf['p_value']]
ax1.barh(df_adf['feature'], df_adf['p_value'], color=colors_adf)
ax1.axvline(x=0.05, color='black', linestyle='--', label='Seuil 0.05')
ax1.set_xlabel('p-value')
ax1.set_title('Test ADF — p-values\n(vert = stationnaire)')
ax1.legend()

# VIF scores
colors_vif = ['green' if v < 5 else 'orange' if v < 10 else 'red'
               for v in df_vif['vif']]
ax2.barh(df_vif['feature'], df_vif['vif'], color=colors_vif)
ax2.axvline(x=5,  color='orange', linestyle='--', label='Seuil 5')
ax2.axvline(x=10, color='red',    linestyle='--', label='Seuil 10')
ax2.set_xlabel('VIF')
ax2.set_title('Test VIF — Multicolinéarité\n(vert = OK)')
ax2.legend()

plt.tight_layout()
plt.savefig('statistical_tests.png', dpi=150)
plt.show()
print("\n✅ Graphique sauvegardé : statistical_tests.png")

# ════════════════════════════════════════════════════════════════
# EXPORT des résultats
# ════════════════════════════════════════════════════════════════
df_adf.to_csv('results_adf.csv', index=False)
df_vif.to_csv('results_vif.csv', index=False)
print("✅ Résultats sauvegardés : results_adf.csv et results_vif.csv")