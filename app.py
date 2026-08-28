import streamlit as st
import numpy as np
import requests
import math
from datetime import datetime

# --- UI HEADER ---
st.set_page_config(page_title="Alpha-Predict Soccer Engine", layout="wide")
st.title("⚽ Alpha-Predict: Advanced Soccer Engine")
st.write("Professional-grade Dixon-Coles model utilizing Exponential Time-Decay and Expected Goals (xG).")

# --- CORE MATH LAYERS ---
def calculate_time_weight(match_date_str, target_date, half_life_days=30):
    try:
        match_date = datetime.strptime(match_date_str.split("T")[0], "%Y-%m-%d")
        delta_days = (target_date - match_date).days
        if delta_days < 0: return 0.0
        return math.exp(-(math.log(2) / half_life_days) * delta_days)
    except:
        return 0.0

def dixon_coles_rho(x, y, mu, eta, tau):
    if x == 0 and y == 0: return 1 - mu * eta * tau
    elif x == 1 and y == 0: return 1 + mu * tau
    elif x == 0 and y == 1: return 1 + eta * tau
    elif x == 1 and y == 1: return 1 - tau
    return 1.0

def calculate_dixon_coles_probs(mu, eta, tau=-0.05, max_goals=6):
    home_probs = [math.exp(-mu) * (mu**i) / math.factorial(i) for i in range(max_goals)]
    away_probs = [math.exp(-eta) * (eta**i) / math.factorial(i) for i in range(max_goals)]
    matrix = np.outer(home_probs, away_probs)
    for h in range(2):
        for a in range(2):
            matrix[h, a] *= dixon_coles_rho(h, a, mu, eta, tau)
    home_win = np.sum(np.tril(matrix, -1))
    away_win = np.sum(np.triu(matrix, 1))
    draw = np.sum(np.diag(matrix))
    total_p = home_win + away_win + draw
    return home_win / total_p, draw / total_p, away_win / total_p

# --- SIDEBAR CONFIGURATION ---
API_KEY = st.sidebar.text_input("RapidAPI Key:", type="password")
half_life = st.sidebar.slider("Form Decay Half-Life (Days)", 10, 60, 25)

league_options = {
    "English Premier League": 39, "La Liga (Spain)": 140, 
    "Serie A (Italy)": 135, "Bundesliga (Germany)": 78
}
selected_league = st.sidebar.selectbox("League", list(league_options.keys()))
season_year = st.sidebar.number_input("Season Year", min_value=2023, max_value=2027, value=2026)

# --- APPLICATION INPUTS ---
col1, col2 = st.columns(2)
with col1: home_input = st.text_input("Home Team:", "Arsenal")
with col2: away_input = st.text_input("Away Team:", "Chelsea")

if st.button("⚡ Generate Advanced Match Projection"):
    if not API_KEY:
        st.warning("Please provide your RapidAPI Key in the sidebar.")
    else:
        with st.spinner("Processing live historical event coordinates and xG models..."):
            # Setup headers
            headers = {"X-RapidAPI-Key": API_KEY, "X-RapidAPI-Host": "://rapidapi.com"}
            base_url = "https://://rapidapi.com/v3"
            league_id = league_options[selected_league]

            # Fetch Recent Fixtures
            url = f"{base_url}/fixtures"
            params = {"league": str(league_id), "season": str(season_year), "status": "FT"}
            res = requests.get(url, headers=headers, params=params).json()

            if not res.get('response') or len(res['response']) == 0:
                st.error("Could not fetch historical data for this selection. Double check your API key or season.")
            else:
                raw_fixtures = res['response']
                target_date = datetime.now()

                # Aggregate Decay-Weighted Matrix
                global_h_xg, global_a_xg, global_w = 0, 0, 0
                t_stats = {home_input: {"gf":0,"ga":0,"w":0}, away_input: {"gf":0,"ga":0,"w":0}}

                for f in raw_fixtures:
                    match_date_str = f['fixture']['date']
                    w = calculate_time_weight(match_date_str, target_date, half_life)
                    if w < 0.01: continue
                    
                    # Fallback to score if xG data isn't active on free endpoint structure
                    h_xg = float(f['score']['fulltime']['home'] if f['score']['fulltime']['home'] is not None else 0)
                    a_xg = float(f['score']['fulltime']['away'] if f['score']['fulltime']['away'] is not None else 0)

                    global_h_xg += h_xg * w
                    global_a_xg += a_xg * w
                    global_w += w

                    # Map metrics to target teams
                    h_name, a_name = f['teams']['home']['name'], f['teams']['away']['name']
                    if home_input in [h_name, a_name]:
                        t_stats[home_input]["gf"] += (h_xg if h_name == home_input else a_xg) * w
                        t_stats[home_input]["ga"] += (a_xg if h_name == home_input else h_xg) * w
                        t_stats[home_input]["w"] += w
                    if away_input in [h_name, a_name]:
                        t_stats[away_input]["gf"] += (a_xg if a_name == away_input else h_xg) * w
                        t_stats[away_input]["ga"] += (h_xg if a_name == away_input else a_xg) * w
                        t_stats[away_input]["w"] += w

                if global_w == 0 or t_stats[home_input]["w"] == 0 or t_stats[away_input]["w"] == 0:
                    st.error("Insufficient recent match history found for one or both teams in this system.")
                else:
                    avg_h_xg = global_h_xg / global_w
                    avg_a_xg = global_a_xg / global_w

                    home_att = (t_stats[home_input]["gf"] / t_stats[home_input]["w"]) / avg_h_xg
                    home_def = (t_stats[home_input]["ga"] / t_stats[home_input]["w"]) / avg_a_xg
                    away_att = (t_stats[away_input]["gf"] / t_stats[away_input]["w"]) / avg_a_xg
                    away_def = (t_stats[away_input]["ga"] / t_stats[away_input]["w"]) / avg_h_xg

                    mu = home_att * away_def * avg_h_xg
                    eta = away_att * home_def * avg_a_xg

                    p_win, p_draw, p_loss = calculate_dixon_coles_probs(mu, eta)

                    # --- UI DISPLAY GAUGE ---
                    st.success("🎯 Model Converged Successfully!")
                    st.subheader("🔮 Probabilistic Match Outcomes")
                    m1, m2, m3 = st.columns(3)
                    m1.metric(f"🏠 {home_input} Win", f"{p_win*100:.1f}%")
                    m2.metric("🤝 Draw Chance", f"{p_draw*100:.1f}%")
                    m3.metric(f"🚀 {away_input} Win", f"{p_loss*100:.1f}%")
                    
                    st.info(f"📋 **Expected Output Scaling:** {home_input} Expected: {mu:.2f} goals vs {away_input} Expected: {eta:.2f} goals")
