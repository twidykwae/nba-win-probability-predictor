import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.stats import ttest_ind

st.title("NBA Win Probability Predictor")

df = pd.read_csv('data/processed/team_games_features.csv')
# Sidebar
st.sidebar.header("Select Parameters")

team = st.sidebar.selectbox(
    "Select Team",
    sorted(df["team"].unique())
)

team_df = df[df["team"] == team]

opponent = st.sidebar.selectbox(
    "Select Opponent",
    sorted(team_df["opponent"].unique())
)

filtered = team_df[team_df["opponent"] == opponent]

st.subheader(f"{team} vs {opponent}")

if len(filtered) == 0:
    st.warning("No games found between these teams.")
else:
    col1, col2, col3 = st.columns(3)
    
    games_played = len(filtered)
    win_rate = filtered["win"].mean()
    avg_point_diff = (filtered["points"] - filtered["opponent_points"]).mean()
    
    col1.metric("Games Played", games_played)
    col2.metric("Win Rate", f"{win_rate:.1%}")
    col3.metric("Avg Point Diff", round(avg_point_diff, 2))


if len(filtered) > 0 and len(team_df[team_df["opponent"] != opponent]) > 0:
    others = team_df[team_df["opponent"] != opponent]
    
    # Get arrays of point differences for both groups
    filtered_point_diffs = filtered["points"] - filtered["opponent_points"]
    others_point_diffs = others["points"] - others["opponent_points"]
    
    t_stat, p_value = ttest_ind(
        filtered_point_diffs,
        others_point_diffs,
        equal_var=False
    )
    
    st.subheader("Statistical Significance")
    
    st.write(f"**t-statistic:** {t_stat:.3f}")
    st.write(f"**p-value:** {p_value:.4f}")
    
    if p_value < 0.05:
        st.success("Performance difference is statistically significant.")
    else:
        st.info("No statistically significant difference detected.")


model = joblib.load("model/nba_win_probability_model.pkl")

features = filtered[[
    "home",
    "avg_pts_last5",
    "avg_opponent_pts_last5",
    "win_rate_last5",
    "vs_opponent_win_rate"
]].mean().values.reshape(1, -1)

prob = model.predict_proba(features)[0][1]

st.subheader("ML Prediction")
st.metric("Predicted Win Probability", f"{prob:.2%}")

st.info(
    "Predictions are probabilistic and based on historical performance trends. "
    "This model does not account for injuries, lineup changes, rest days, or real-time factors."
)

st.markdown("---")
st.markdown(
    "Built by **Twidy Kwae** · "
    "[View project on GitHub](https://github.com/twidykwae/nba-win-probability-predictor)",
    unsafe_allow_html=True
)
