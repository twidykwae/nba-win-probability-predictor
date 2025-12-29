import pandas as pd
import numpy as np

df = pd.read_csv('data/processed/team_games.csv')

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["team", "date"])

print(df.head())


ROLLING_WINDOW = 5


df["avg_pts_last5"] = (
    df.groupby("team")["points"]
    .rolling(window=ROLLING_WINDOW)
    .mean()
    .reset_index(level=0, drop=True)
)

df["avg_opponent_pts_last5"] = (
    df.groupby("team")["opponent_points"]
    .rolling(window=ROLLING_WINDOW)
    .mean()
    .reset_index(level=0, drop=True)
)

df["win_rate_last5"] = (
    df.groupby("team")["win"]
    .rolling(window=ROLLING_WINDOW)
    .mean()
    .reset_index(level=0, drop=True)
)

df["vs_opponent_win_rate"] = (
    df.groupby(["team", "opponent"])["win"]
    .expanding()
    .mean()
    .reset_index(level=[0, 1], drop=True)
)

print(df.head())

df = df.dropna()

print(df.head())
print("What I have after feature engineering: ", df.shape)

df.to_csv('data/processed/team_games_features.csv', index=False)
print("Saved to data/processed/team_games_features.csv")

print(df.columns)
print(df[df["team"] == "Oklahoma City Thunder"].head())

