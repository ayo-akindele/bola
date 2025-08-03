
import streamlit as st
import pandas as pd
from datetime import datetime

# URLs to fetch data from Google Sheets
HISTORICAL_URL = "https://docs.google.com/spreadsheets/d/1oZJlXF6tpLLaEDNfduHzYFvLKDw7rnyzZY17CQNl1so/gviz/tq?tqx=out:csv&gid=0"
FIXTURES_URL = "https://docs.google.com/spreadsheets/d/1oZJlXF6tpLLaEDNfduHzYFvLKDw7rnyzZY17CQNl1so/gviz/tq?tqx=out:csv&gid=1005360909"

st.set_page_config(page_title="BolaStats", layout="centered")

st.title("📊 BolaStats")
st.caption("Quick stats for quick thinking ⚡")

@st.cache_data
def load_data():
    try:
        results_df = pd.read_csv(HISTORICAL_URL)
        fixtures_df = pd.read_csv(FIXTURES_URL)
        return results_df, fixtures_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

results_df, fixtures_df = load_data()

if results_df is not None and fixtures_df is not None:
    results_df.columns = [col.strip().lower().replace(" ", "_") for col in results_df.columns]
    fixtures_df.columns = [col.strip().lower().replace(" ", "_") for col in fixtures_df.columns]

    # Convert date column
    fixtures_df["date"] = pd.to_datetime(fixtures_df["date"], errors="coerce")

    # Identify the latest available round based on today's date
    today = pd.Timestamp.today().normalize()
    round_dates = fixtures_df.groupby("round_number")["date"].max().sort_index()

    current_round = round_dates[round_dates >= today].index.min()
    if pd.isna(current_round):
        current_round = round_dates.index.max()

    gw_fixtures = fixtures_df[fixtures_df["round_number"] == current_round]

    st.subheader(f"📅 Gameweek {current_round} Predictions")

    def generate_stats(home, away):
        h2h = results_df[
            ((results_df['home_team'] == home) & (results_df['away_team'] == away)) |
            ((results_df['home_team'] == away) & (results_df['away_team'] == home))
        ].sort_values(by="match_date", ascending=False).head(5)

        if len(h2h) < 5:
            return []

        total = len(h2h)
        trends = []

        def stat_check(series, label):
            numeric_series = pd.to_numeric(series, errors='coerce')
            count = numeric_series.sum()
            if pd.notna(count) and count / total >= 0.8:
                return f"{label} in {int(count)}/{total} games"
            return None

        team_wins = 0
        for _, row in h2h.iterrows():
            if row['home_team'] == home and row['home_score'] > row['away_score']:
                team_wins += 1
            elif row['away_team'] == home and row['away_score'] > row['home_score']:
                team_wins += 1
        if team_wins / total >= 0.8:
            trends.append(f"{home} won {team_wins}/{total} recent meetings")

        if 'total_corners' in h2h.columns:
            h2h['Corners_Over_9.5'] = h2h['total_corners'] > 9.5
        if 'home_yellow_cards' in h2h.columns and 'away_yellow_cards' in h2h.columns:
            h2h['Bookings_Over_3.5'] = (h2h['home_yellow_cards'] + h2h['away_yellow_cards']) > 3.5
        if 'first_half_home' in h2h.columns and 'first_half_away' in h2h.columns:
            h2h['First_Half_Goal'] = (h2h['first_half_home'] + h2h['first_half_away']) > 0

        checks = {
            'both_teams_score': "Both teams scored",
            'over_2_5': "Over 2.5 goals",
            'Corners_Over_9.5': "Over 9.5 corners",
            'Bookings_Over_3.5': "Over 3.5 bookings",
            'First_Half_Goal': "First-half goals"
        }

        for col, label in checks.items():
            if col in h2h.columns:
                trend = stat_check(h2h[col], label)
                if trend:
                    trends.append(trend)

        return trends[:3]

    for _, row in gw_fixtures.iterrows():
        home = row['home_team']
        away = row['away_team']
        st.markdown(f"### {home} vs {away}")
        fixture_stats = generate_stats(home, away)
        if fixture_stats:
            for s in fixture_stats:
                st.markdown(f"- {s}")
        else:
            st.info("No strong trends to recommend for this game.")
else:
    st.warning("Unable to fetch data from Google Sheets. Please check links or permissions.")
