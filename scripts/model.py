import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import joblib

df = pd.read_csv('data/processed/team_games_features.csv')

print(f"Data shape: {df.shape}")
print(df.head())

FEATURES = [
    "home",
    "avg_pts_last5",
    "avg_opponent_pts_last5",
    "win_rate_last5",
    "vs_opponent_win_rate"
]

x = df[FEATURES]
y = df["win"]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42, stratify=y)


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Classification Report: \n", classification_report(y_test, y_pred))


importances = pd.DataFrame({
    "Feature": FEATURES,
    "Coefficient": model.coef_[0]
}).sort_values(by="Coefficient", ascending=False)
print(importances)

plt.figure(figsize=(10, 6))
plt.barh(importances["Feature"], importances["Coefficient"])
plt.xlabel("Coefficient")
plt.ylabel("Feature")
plt.title("Feature Importances")
plt.show()


##test model
okc = df[df["team"] == "Oklahoma City Thunder"]
okc_spurs = okc[okc["opponent"] == "San Antonio Spurs"]
okc_others = okc[okc["opponent"] != "San Antonio Spurs"]

spurs_win_prob = model.predict_proba(okc_spurs[FEATURES])[:, 1].mean()
others_win_prob = model.predict_proba(okc_others[FEATURES])[:, 1].mean()

print(f"Predicted OKC win probability vs Spurs: {spurs_win_prob:.2f}")
print(f"Predicted OKC win probability vs Others: {others_win_prob:.2f}")

joblib.dump(model, 'model/nba_win_probability_model.pkl')
print("Model saved to model/nba_win_probability_model.pkl")

