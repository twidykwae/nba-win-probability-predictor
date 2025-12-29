import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

df = pd.read_csv('data/processed/team_games.csv')

print(f"Data shape: {df.shape}")
print(df.head())

okc_thunder_games = df[df['team'] == 'Oklahoma City Thunder'].copy()
print(f"Oklahoma City Thunder games shape: {okc_thunder_games.shape}")

# Add point_diff column before filtering
okc_thunder_games["point_diff"] = (okc_thunder_games["points"] - okc_thunder_games["opponent_points"])

print("OKC wins:", okc_thunder_games['win'].sum())
print("OKC losses:", (okc_thunder_games['win']==0).sum())

okc_vs_spurs = okc_thunder_games[okc_thunder_games['opponent'] == 'San Antonio Spurs']
okc_vs_others = okc_thunder_games[okc_thunder_games['opponent'] != 'San Antonio Spurs']

print("Games vs Spurs:", len(okc_vs_spurs))
print("Games vs Others:", len(okc_vs_others))


comparison = pd.DataFrame({
    "Metric": ["Points Scored", "Points Allowed", "Win Rate"],
    "Vs Spurs": [
        okc_vs_spurs['points'].mean(),
        okc_vs_spurs['opponent_points'].mean(),
        okc_vs_spurs['win'].mean()
    ],
    "Vs Others": [
        okc_vs_others['points'].mean(),
        okc_vs_others['opponent_points'].mean(),
        okc_vs_others['win'].mean()
    ]
})

print(comparison)
print("Average difference in points for OKC vs Spurs:",
      okc_vs_spurs["point_diff"].mean())
print("Average difference in points for OKC vs Others:",
      okc_vs_others["point_diff"].mean())


t_stat, p_value = ttest_ind(
    okc_vs_spurs["point_diff"],
    okc_vs_others["point_diff"],
    equal_var=False
)

print("t-stat:", t_stat)
print("p-value:", p_value)

#OKC performs significantly worse against the Spurs than against the rest of the league, and the probability that this pattern is due to chance is extremely low from the two-sample t test.