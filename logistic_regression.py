import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Chargement
df = pd.read_csv('wti_features_v3.csv')

# Création de la target (1 = hausse, 0 = baisse)
df['target'] = (df['WTI_Close'].shift(-1) > df['WTI_Close']).astype(int)
df = df.dropna()

# Features (tout sauf DateTime, target, WTI_Close)
exclude = ['Datetime', 'target', 'WTI_Close']
X = df[[c for c in df.columns if c not in exclude]]
y = df['target']

# Division train/test (80%/20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prédiction et accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f" Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
