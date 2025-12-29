import pandas as pd

games = pd.read_csv('data/raw/total_stats.csv')
print(f"Initial data shape: {games.shape}")
print(games.head())
print(games.columns)

games = games.rename(columns={
    "Visitor/Neutral": "away_team",
    "PTS": "away_points",
    "Home/Neutral": "home_team",
    "PTS.1": "home_points",
    "Date": "date"
})

games = games[
    ["date", "home_team", "home_points", "away_team", "away_points"]
]

games = games.dropna(subset=["home_team", "away_team"])
games["home_points"] = games["home_points"].astype(int)
games["away_points"] = games["away_points"].astype(int)

#print(games.head())
#print(games.shape)


home = games.copy()
home["team"] = home["home_team"]
home["opponent"] = home["away_team"]
home["points"] = home["home_points"]
home["opponent_points"] = home["away_points"]
home["home"] = 1
home["win"] = (home["points"] > home["opponent_points"]).astype(int)

away = games.copy()
away["team"] = away["away_team"]
away["opponent"] = away["home_team"]
away["points"] = away["away_points"]
away["opponent_points"] = away["home_points"]
away["home"] = 0
away["win"] = (away["points"] > away["opponent_points"]).astype(int)


team_games = pd.concat([home, away], ignore_index=True)
team_games = team_games[
    ["date", "team", "opponent", "points", "opponent_points", "home", "win"]
]

print(team_games.head())
print(f"Final data shape: {team_games.shape}")
print(team_games[team_games['team'] == 'Oklahoma City Thunder'].head())

team_games.to_csv('data/processed/team_games.csv', index=False)
