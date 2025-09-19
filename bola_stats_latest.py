# bola_stats_latest.py
# Modified BolaStats Streamlit App with Date-Sorted Fixtures and Today's Matches

import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# Mapping of league names to their historical and fixtures CSV filenames
LEAGUE_FILES: Dict[str, Dict[str, str]] = {
    "EPL": {"results": "EPL Historical Data.csv", "fixtures": "EPL_upcoming_fixtures.csv"},
    "La Liga": {"results": "SP1 Historical Data.csv", "fixtures": "SP1_upcoming_fixtures.csv"},
    "Serie A": {"results": "I1 Historical Data.csv", "fixtures": "I1_upcoming_fixtures.csv"},
    "Bundesliga": {"results": "D1 Historical Data.csv", "fixtures": "D1_upcoming_fixtures.csv"},
}

# Configure Streamlit
st.set_page_config(page_title="BolaStats", layout="centered")
st.title("📊 BolaStats")
st.markdown("<h4 style='margin-bottom:0; font-weight:bold;'>⚡ Quick Stats That Matter</h4>", unsafe_allow_html=True)

def load_data(league: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    config = LEAGUE_FILES.get(league)
    if not config:
        st.error(f"No configuration for league: {league}")
        return None, None

    def _load_csv(filename: str) -> pd.DataFrame:
        candidates = [filename, os.path.join(os.path.dirname(__file__), filename)]
        for path in candidates:
            if os.path.exists(path):
                return pd.read_csv(path)
        raise FileNotFoundError(f"File not found: {filename}")

    try:
        results_df = _load_csv(config["results"])
        fixtures_df = _load_csv(config["fixtures"])
    except Exception as exc:
        st.error(f"Error reading CSVs for {league}: {exc}")
        return None, None

    results_df.columns = [c.strip().lower().replace(" ", "_") for c in results_df.columns]
    fixtures_df.columns = [c.strip().lower().replace(" ", "_") for c in fixtures_df.columns]

    if "match_date" in results_df.columns:
        results_df["match_date"] = pd.to_datetime(results_df["match_date"], errors="coerce")
    if "date" in fixtures_df.columns:
        fixtures_df["date"] = pd.to_datetime(fixtures_df["date"], errors="coerce")

    rename_map = {}
    for col in fixtures_df.columns:
        if col.lower() in {"home team", "home_team", "hometeam"}:
            rename_map[col] = "home_team"
        if col.lower() in {"away team", "away_team", "awayteam"}:
            rename_map[col] = "away_team"
    if rename_map:
        fixtures_df = fixtures_df.rename(columns=rename_map)

    return results_df, fixtures_df

def generate_trends(home: str, away: str, results_df: pd.DataFrame) -> List[Tuple[float, str]]:
    three_years_ago = datetime.today().year - 3
    h2h = results_df[((results_df["home_team"] == home) & (results_df["away_team"] == away)) |
                     ((results_df["home_team"] == away) & (results_df["away_team"] == home))]
    h2h = h2h[h2h["match_date"].dt.year >= three_years_ago].sort_values(by="match_date", ascending=False).head(5)
    if h2h.empty or len(h2h) < 5:
        return []

    trends: List[Tuple[float, str]] = []
    def add_trend(cond: pd.Series, label: str) -> None:
        valid = cond.dropna()
        if valid.empty:
            return
        count = valid.sum()
        pct = count / len(valid)
        if pct >= 0.8:
            trends.append((pct, f"{label} in {int(count)}/{len(valid)} games"))

    # Win trend for home
    wins = 0
    for _, row in h2h.iterrows():
        hg = pd.to_numeric(row.get("home_score"), errors="coerce")
        ag = pd.to_numeric(row.get("away_score"), errors="coerce")
        if pd.notna(hg) and pd.notna(ag):
            if row["home_team"] == home and hg > ag:
                wins += 1
            elif row["away_team"] == home and ag > hg:
                wins += 1
    if wins / len(h2h) >= 0.8:
        trends.append((wins / len(h2h), f"{home} won {wins}/{len(h2h)} recent meetings"))

    hg = pd.to_numeric(h2h.get("home_score"), errors="coerce")
    ag = pd.to_numeric(h2h.get("away_score"), errors="coerce")
    add_trend((hg > 0) & (ag > 0), "Both teams scored (GG)")
    add_trend(~((hg > 0) & (ag > 0)), "Both teams failed to score (NG)")
    add_trend((hg + ag) > 2.5, "Over 2.5 goals")
    add_trend((hg + ag) <= 2.5, "Under 2.5 goals")

    return sorted(trends, key=lambda x: x[0], reverse=True)[:3]

EMOJI_MAP = {
    "both teams scored": "⚽",
    "both teams failed to score": "🚫",
    "over 2.5 goals": "📈",
    "under 2.5 goals": "📉",
    "clean sheet": "🧤",
    "over 9.5 corners": "🏳️",
    "under 9.5 corners": "🔻",
    "first-half goals": "⏱",
    "won": "🏆",
    "more corners": "🔺",
}

def add_time_sort_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "time" in df.columns:
        parsed_time = pd.to_datetime(df["time"], errors="coerce").dt.time
        comb = [pd.Timestamp.combine(d.date(), t) if pd.notna(d) and pd.notna(t) else d
                for d, t in zip(df["date"], parsed_time)]
        df["__dt_sort__"] = pd.to_datetime(comb, errors="coerce")
    else:
        df["__dt_sort__"] = df["date"]
    return df

today = pd.Timestamp.now(tz="Africa/Lagos").normalize().tz_localize(None)

league_options = ["All"] + list(LEAGUE_FILES.keys())
selected = st.selectbox("Select league", league_options, index=0, help="View all leagues or choose one")

if selected == "All":
    st.subheader(f"Today's Fixtures – {today.strftime('%A, %d %b %Y')}")
    any_fixtures = False
    for league, files in LEAGUE_FILES.items():
        results_df, fixtures_df = load_data(league)
        if results_df is None or fixtures_df is None or fixtures_df.empty:
            continue
        todays = fixtures_df[fixtures_df["date"].dt.normalize() == today]
        if todays.empty:
            continue
        any_fixtures = True
        todays = add_time_sort_cols(todays).sort_values("__dt_sort__")
        st.markdown(f"### {league}")
        for _, fix in todays.iterrows():
            home, away = str(fix.get("home_team")), str(fix.get("away_team"))
            match_label = f"{home} vs {away}"
            with st.expander(match_label, expanded=True):
                trends = generate_trends(home, away, results_df)
                if trends:
                    for pct, desc in trends:
                        icon = next((symbol for key, symbol in EMOJI_MAP.items() if key in desc.lower()), "•")
                        st.markdown(f"{icon} {desc}")
                else:
                    st.info("No strong trends to recommend for this game.")
    if not any_fixtures:
        st.info("No fixtures today.")

else:
    league = selected
    results_df, fixtures_df = load_data(league)
    if results_df is None or fixtures_df is None or fixtures_df.empty:
        st.stop()
    display_fixtures = fixtures_df[fixtures_df["date"].dt.normalize() >= today].copy()
    display_fixtures = add_time_sort_cols(display_fixtures).sort_values("__dt_sort__")
    st.subheader(f"{league} – Upcoming Fixtures (sorted by date)")
    if display_fixtures.empty:
        st.info("No upcoming fixtures.")
    else:
        for _, fix in display_fixtures.iterrows():
            home, away = str(fix.get("home_team")), str(fix.get("away_team"))
            match_label = f"{home} vs {away}"
            with st.expander(match_label, expanded=True):
                trends = generate_trends(home, away, results_df)
                if trends:
                    for pct, desc in trends:
                        icon = next((symbol for key, symbol in EMOJI_MAP.items() if key in desc.lower()), "•")
                        st.markdown(f"{icon} {desc}")
                else:
                    st.info("No strong trends to recommend for this game.")
