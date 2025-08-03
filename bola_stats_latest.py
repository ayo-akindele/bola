
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
    fixtures_df["date"] = pd.to_datetime(fixtures_df["date"], errors="coerce")

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

        def trend_check(condition, label):
            count = condition.sum()
            if count / total >= 0.8:
                return f"{label} in {int(count)}/{total} games"
            return None

        # Match winner logic
        wins = 0
        for _, row in h2h.iterrows():
            if row['home_team'] == home and row['home_score'] > row['away_score']:
                wins += 1
            elif row['away_team'] == home and row['away_score'] > row['home_score']:
                wins += 1
        if wins / total >= 0.8:
            trends.append((wins / total, f"{home} won {wins}/{total} recent meetings"))

        # Feature columns
        try:
            h2h['Corners_Over_9.5'] = pd.to_numeric(h2h['total_corners'], errors='coerce') > 9.5
            h2h['Bookings_Over_3.5'] = (pd.to_numeric(h2h['home_yellow_cards'], errors='coerce') +
                                        pd.to_numeric(h2h['away_yellow_cards'], errors='coerce')) > 3.5
            h2h['First_Half_Goal'] = (pd.to_numeric(h2h['first_half_home'], errors='coerce') +
                                      pd.to_numeric(h2h['first_half_away'], errors='coerce')) > 0
        except:
            pass

        market_labels = {
            'both_teams_score': "Both teams scored",
            'over_2_5': "Over 2.5 goals",
            'Corners_Over_9.5': "Over 9.5 corners",
            'Bookings_Over_3.5': "Over 3.5 bookings",
            'First_Half_Goal': "First-half goals"
        }

        for col, label in market_labels.items():
            if col in h2h.columns:
                trend = trend_check(h2h[col], label)
                if trend:
                    trends.append((h2h[col].mean(), trend))

        top_trends = [text for _, text in sorted(trends, key=lambda x: x[0], reverse=True)[:3]]
        return top_trends

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
