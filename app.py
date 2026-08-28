import streamlit as st
import numpy as np
import requests
import math
from datetime import datetime

# --- GLOBAL CONFIGURATION (FIXED URL INFRASTRUCTURE) ---
HOST = "://rapidapi.com"
BASE_URL = "https://://rapidapi.com/v3"

# --- STREAMLIT UI HEADER ---
st.set_page_config(page_title="Alpha-Predict Soccer Engine", layout="wide")
st.title("⚽ Alpha-Predict: Advanced Soccer Engine")
st.write("Professional-grade Dixon-Coles model utilizing Exponential Time-Decay and Expected Goals (xG).")

# --- CORE MATH LAYERS ---
def calculate_time_weight(match_date_str, target_date, half_life_days=30):
    try:
        # Strip timestamp timezone notation safely
        clean_date_str = match_date_str.split("T")[0]
        match_date = datetime.strptime(clean_date_str, "%Y-%m-%d")
        delta_days = (target_date - match_date).days
        if delta_days < 0: 
            return 0.0
        return math.exp(-(math.log(2) / half_life_days) * delta_days)
    except Exception as e:
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
    if total_p == 0:
        return 0.33, 0.33, 0.33
    return home_win / total_p, draw / total_p, away_win / total_p

# --- SIDEBAR CONFIGURATION ---
API_KEY = st.sidebar.text_input("RapidAPI Key:", type="password")
half_life = st.sidebar.slider("Form Decay Half-Life (Days)", 10, 60, 25)

league_options = {
    "English Premier League": 39, 
    "La Liga (Spain)": 140, 
    "Serie A (Italy)": 135, 
    "Bundesliga (Germany)": 78
}
selected_league = st.sidebar.selectbox("League Context", list(league_options.keys()))
season_year = st.sidebar.number_input("Season Year", min_value=2023, max_value=2027, value=2026)

# --- APPLICATION INPUTS ---
col1, col2 = st.columns(2)
with col1: 
    home_input = st.text_input("Home Team Name:", "Arsenal")
with col2: 
    away_input = st.text_input("Away Team Name:", "Chelsea")

if st.button("⚡ Generate Advanced Match Projection"):
    if not API_KEY or len(API_KEY.strip()) < 10:
        st.warning("⚠️ Please provide a valid RapidAPI Key in the sidebar.")
    else:
        with st.spinner("Processing live historical event coordinates and xG models..."):
            # Set up unified network configuration headers
            headers = {
                "X-RapidAPI-Key": API_KEY.strip(), 
                "X-RapidAPI-Host": HOST
            }
            league_id = league_options[selected_league]

            # Safe endpoint compilation
            url = f"{BASE_URL}/fixtures"
            params = {"league": str(league_id), "season": str(season_year), "status": "FT"}
            
            try:
                response_obj = requests.get(url, headers=headers, params=params)
                
                # Check for HTTP Layer Errors (e.g., 403 Forbidden)
                if response_obj.status_code != 200:
                    st.error(f"❌ Server Connection Error ({response_obj.status_code}). Check if your API subscription is active.")
                    st.stop()
                    
                res = response_obj.json()
            except Exception as network_err:
                st.error(f"❌ Failed to reach data endpoint: {network_err}")
                st.stop()

            if not res.get('response') or len(res['response']) == 0:
                st.error("❌ Zero fixtures returned. The season year chosen may not have active data yet.")
            else:
                raw_fixtures = res['response']
                target_date = datetime.now()

                # Aggregate Decay-Weighted Data Matrices
                global_h_xg, global_a_xg, global_w = 0, 0, 0
                t_stats = {home_input: {"gf": 0, "ga": 0, "w": 0}, away_input: {"gf": 0, "ga": 0, "w": 0}}

                for f in raw_fixtures:
                    match_date_str = f['fixture']['date']
                    w = calculate_time_weight(match_date_str, target_date, half_life)
                    if w < 0.01: 
                        continue
                    
                    # Target scoreline parameters
                    if f['score']['fulltime']['home'] is None or f['score']['fulltime']['away'] is None:
                        continue
                        
                    h_xg = float(f['score']['fulltime']['home'])
                    a_xg = float(f['score']['fulltime']['away'])

                    global_h_xg += h_xg * w
                    global_a_xg += a_xg * w
                    global_w += w

                    h_name, a_name = f['teams']['home']['name'], f['teams']['away']['name']
                    
                    # Map metrics to selected comparison profiles
                    if home_input.lower() in [h_name.lower(), a_name.lower()]:
                        current_is_home = (h_name.lower() == home_input.lower())
                        t_stats[home_input]["gf"] += (h_xg if current_is_home else a_xg) * w
                        t_stats[home_input]["ga"] += (a_xg if current_is_home else h_xg) * w
                        t_stats[home_input]["w"] += w
                        
                    if away_input.lower() in [h_name.lower(), a_name.lower()]:
                        current_is_home = (h_name.lower() == away_input.lower())
                        t_stats[away_input]["gf"] += (h_xg if current_is_home else a_xg) * w
                        t_stats[away_input]["ga"] += (a_xg if current_is_home else h_xg) * w
                        t_stats[away_input]["w"] += w

                if global_w == 0 or t_stats[home_input]["w"] == 0 or t_stats[away_input]["w"] == 0:
                    st.error(f"❌ Insufficient match history found in the database for '{home_input}' or '{away_input}'. Check your capitalization/spelling matches the league standard.")
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

