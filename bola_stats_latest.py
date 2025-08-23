
import streamlit as st
import pandas as pd
from datetime import datetime

# Google Sheets CSV export URLs
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

summary_top_picks = []

if results_df is not None and fixtures_df is not None:
    results_df.columns = [col.strip().lower().replace(" ", "_") for col in results_df.columns]
    fixtures_df.columns = [col.strip().lower().replace(" ", "_") for col in fixtures_df.columns]
    fixtures_df["date"] = pd.to_datetime(fixtures_df["date"], errors="coerce")
    results_df["match_date"] = pd.to_datetime(results_df["match_date"], errors="coerce")

    today = pd.Timestamp.today().normalize()
    round_dates = fixtures_df.groupby("round_number")["date"].max().sort_index()
    current_round = round_dates[round_dates >= today].index.min()
    if pd.isna(current_round):
        current_round = round_dates.index.max()
    gw_fixtures = fixtures_df[fixtures_df["round_number"] == current_round]

    st.subheader(f"📅 Gameweek {current_round} Predictions")

    def normalize_boolean(col):
        return col.astype(str).str.lower().isin(["1", "true", "yes", "y"])

    def generate_stats(home, away):
        three_seasons_ago = today.year - 3
        h2h_all = results_df[
            ((results_df['home_team'] == home) & (results_df['away_team'] == away)) |
            ((results_df['home_team'] == away) & (results_df['away_team'] == home))
        ]
        h2h = h2h_all[h2h_all["match_date"].dt.year >= three_seasons_ago].sort_values(by="match_date", ascending=False).head(5)

        if h2h.empty:
            return []

        total = len(h2h)
        trends = []

        def trend_check(condition, label):
            try:
                valid = condition.dropna()
                count = valid.sum()
                pct = count / len(valid)
                if len(valid) > 0 and pct >= 0.8:
                    return (pct, f"{label} in {int(count)}/{len(valid)} games")
            except Exception:
                return None
            return None

        # Match winner logic
        wins = 0
        for _, row in h2h.iterrows():
            if row['home_team'] == home and row['home_score'] > row['away_score']:
                wins += 1
            elif row['away_team'] == home and row['away_score'] > row['home_score']:
                wins += 1
        if total > 0 and wins / total >= 0.8:
            trends.append((wins / total, f"{home} won {wins}/{total} recent meetings"))

        # If a BTTS column is present, normalize it and derive GG/NG indicators
        if 'both_teams_score' in h2h.columns:
            # Convert common truthy strings/values into True and others into False.  Missing values
            # remain as NaN/None and will be ignored when checking trends.
            h2h['both_teams_score'] = normalize_boolean(h2h['both_teams_score'])
            # GG (Goal–Goal) is simply the normalized BTTS column; this aligns with the local
            # terminology where "GG" means both teams scored.
            h2h['gg'] = h2h['both_teams_score']
            # NG (No Goal) is the complement of BTTS: True when one or both sides failed to score.
            # We preserve missing values as None so they don't count towards the trend calculations.
            def _compute_ng(val):
                try:
                    if pd.isna(val):
                        return None
                    return not bool(val)
                except Exception:
                    return None
            h2h['ng'] = h2h['both_teams_score'].apply(_compute_ng)

        if 'over_2_5' in h2h.columns:
            h2h['over_2_5'] = normalize_boolean(h2h['over_2_5'])

        try:
            h2h['Corners_Over_9.5'] = pd.to_numeric(h2h['total_corners'], errors='coerce') > 9.5
            h2h['Bookings_Over_3.5'] = (pd.to_numeric(h2h['home_yellow_cards'], errors='coerce') +
                                        pd.to_numeric(h2h['away_yellow_cards'], errors='coerce')) > 3.5
            h2h['First_Half_Goal'] = (pd.to_numeric(h2h['first_half_home'], errors='coerce') +
                                      pd.to_numeric(h2h['first_half_away'], errors='coerce')) > 0
        except:
            pass

        # Market labels map indicator columns to human‑friendly descriptions.  We standardize on
        # "GG" for both teams scoring and "NG" for one or both teams not scoring.
        market_labels = {
            'gg': "Both teams scored (GG)",
            'ng': "Both teams failed to score (NG)",
            'over_2_5': "Over 2.5 goals",
            'Corners_Over_9.5': "Over 9.5 corners",
            'Bookings_Over_3.5': "Over 3.5 bookings",
            'First_Half_Goal': "First-half goals"
        }

        for col, label in market_labels.items():
            if col in h2h.columns:
                result = trend_check(h2h[col], label)
                if result:
                    trends.append(result)

        top_trends = sorted(trends, key=lambda x: x[0], reverse=True)[:3]
        return top_trends

    top_summary_pool = []

    for _, row in gw_fixtures.iterrows():
        home = row['home_team']
        away = row['away_team']
        st.markdown(f"### {home} vs {away}")
        fixture_stats = generate_stats(home, away)
        if fixture_stats:
            for pct, text in fixture_stats:
                st.markdown(f"- {text}")
                top_summary_pool.append((pct, f"{home} vs {away} → {text}"))
        else:
            st.info("No strong trends to recommend for this game.")

    if top_summary_pool:
        st.markdown("---")
        st.subheader("🔥 Top Picks Summary")
        top_summary_pool.sort(reverse=True)
        shown = 0
        for pct, item in top_summary_pool:
            if pct >= 0.9:
                st.markdown(f"✅ {item}")
                shown += 1
            if shown >= 3:
                break
else:
    st.warning("Unable to fetch data from Google Sheets. Please check links or permissions.")
