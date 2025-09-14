"""
Multi‑league BolaStats Streamlit App
===================================

This script powers a multi‑league dashboard for quick head‑to‑head
football stats.  It reads historical results and upcoming fixtures
from CSV files and surfaces succinct trends for the next round of
matches.  Trends include wins, goals (GG/NG, over/under 2.5), clean
sheets, corner totals, corner dominance and first‑half goals.  Each
trend requires at least five recent matches and an 80 % success rate.

Users can choose to view all leagues at once or filter by a single
league via a dropdown.  The data files must reside alongside this
script in your repository.  Update ``LEAGUE_FILES`` below if your
filenames differ.  The order of leagues in ``LEAGUE_FILES`` defines
their presentation order.
"""

import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# Mapping of league names to their historical and fixtures CSV
# filenames.  Ensure these files are committed to your repository so
# the Streamlit app can load them at runtime.  The order here
# determines the order on the page.
LEAGUE_FILES: Dict[str, Dict[str, str]] = {
    "EPL": {
        "results": "EPL Historical Data.csv",
        "fixtures": "EPL_upcoming_fixtures.csv",
    },
    "La Liga": {
        "results": "SP1 Historical Data.csv",
        "fixtures": "SP1_upcoming_fixtures.csv",
    },
    "Serie A": {
        "results": "I1 Historical Data.csv",
        "fixtures": "I1_upcoming_fixtures.csv",
    },
    "Bundesliga": {
        "results": "D1 Historical Data.csv",
        "fixtures": "D1_upcoming_fixtures.csv",
    },
}

# Configure Streamlit
st.set_page_config(page_title="BolaStats", layout="centered")
st.title("📊 BolaStats")
st.markdown(
    "<h4 style='margin-bottom:0; font-weight:bold;'>⚡ Quick Stats That Matter</h4>",
    unsafe_allow_html=True,
)

def load_data(league: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Load historical and fixtures data for a given league from local
    CSV files.  Returns (results_df, fixtures_df).  If files are
    missing or unreadable, returns (None, None) and displays an error.
    """
    config = LEAGUE_FILES.get(league)
    if not config:
        st.error(f"No configuration for league: {league}")
        return None, None
    def _load_csv(filename: str) -> pd.DataFrame:
        # Candidate paths: direct filename and path relative to script dir
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
    # Normalise column names
    results_df.columns = [c.strip().lower().replace(" ", "_") for c in results_df.columns]
    fixtures_df.columns = [c.strip().lower().replace(" ", "_") for c in fixtures_df.columns]
    # Parse dates
    if "match_date" in results_df.columns:
        results_df["match_date"] = pd.to_datetime(results_df["match_date"], errors="coerce")
    if "date" in fixtures_df.columns:
        fixtures_df["date"] = pd.to_datetime(fixtures_df["date"], errors="coerce")
    # Unify team column names in fixtures
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
    """
    Given a home and away team and the historical results DataFrame,
    compute up to three strong trends (80 % success over at least five
    matches).  Trends include wins, GG/NG, over/under 2.5, clean
    sheets, total corners, first‑half goals and corner dominance.
    Returns a list of (percentage, description) tuples sorted by
    descending percentage.
    """
    # Filter last three calendar years and take most recent 5 encounters
    three_years_ago = datetime.today().year - 3
    h2h = results_df[
        ((results_df["home_team"] == home) & (results_df["away_team"] == away))
        | ((results_df["home_team"] == away) & (results_df["away_team"] == home))
    ]
    h2h = (
        h2h[h2h["match_date"].dt.year >= three_years_ago]
        .sort_values(by="match_date", ascending=False)
        .head(5)
    )
    if h2h.empty or len(h2h) < 5:
        return []
    trends: List[Tuple[float, str]] = []
    # Helper to add a trend if it meets threshold
    def add_trend(cond: pd.Series, label: str) -> None:
        valid = cond.dropna()
        if valid.empty:
            return
        count = valid.sum()
        pct = count / len(valid)
        if pct >= 0.8:
            trends.append((pct, f"{label} in {int(count)}/{len(valid)} games"))
    # Win trend for home team
    wins = 0
    for _, row in h2h.iterrows():
        # Determine actual positions relative to selected home team
        r_home = row["home_team"]
        r_away = row["away_team"]
        hg = pd.to_numeric(row.get("home_score"), errors="coerce")
        ag = pd.to_numeric(row.get("away_score"), errors="coerce")
        if pd.notna(hg) and pd.notna(ag):
            if r_home == home and hg > ag:
                wins += 1
            elif r_away == home and ag > hg:
                wins += 1
    if wins / len(h2h) >= 0.8:
        trends.append((wins / len(h2h), f"{home} won {wins}/{len(h2h)} recent meetings"))
    # Goals markets
    hg = pd.to_numeric(h2h.get("home_score"), errors="coerce")
    ag = pd.to_numeric(h2h.get("away_score"), errors="coerce")
    gg = (hg > 0) & (ag > 0)
    gg = gg.where(~(hg.isna() | ag.isna()), pd.NA)
    h2h["gg"] = gg
    h2h["ng"] = gg.apply(lambda x: None if pd.isna(x) else not x)
    add_trend(h2h["gg"], "Both teams scored (GG)")
    add_trend(h2h["ng"], "Both teams failed to score (NG)")
    total_goals = hg + ag
    over25 = total_goals > 2.5
    under25 = total_goals <= 2.5
    over25 = over25.where(~total_goals.isna(), pd.NA)
    under25 = under25.where(~total_goals.isna(), pd.NA)
    add_trend(over25, "Over 2.5 goals")
    add_trend(under25, "Under 2.5 goals")
    # Clean sheet trends (team conceded zero)
    home_cs = []
    away_cs = []
    for _, row in h2h.iterrows():
        r_home = row["home_team"]
        r_away = row["away_team"]
        hg = pd.to_numeric(row.get("home_score"), errors="coerce")
        ag = pd.to_numeric(row.get("away_score"), errors="coerce")
        # Home team perspective
        if r_home == home and pd.notna(ag):
            home_cs.append(ag == 0)
        elif r_away == home and pd.notna(hg):
            home_cs.append(hg == 0)
        # Away team perspective
        if r_away == away and pd.notna(hg):
            away_cs.append(hg == 0)
        elif r_home == away and pd.notna(ag):
            away_cs.append(ag == 0)
    if home_cs:
        count = sum(bool(x) for x in home_cs)
        pct = count / len(home_cs)
        if pct >= 0.8:
            trends.append((pct, f"{home} kept a clean sheet in {count}/{len(home_cs)} games"))
    if away_cs:
        count_a = sum(bool(x) for x in away_cs)
        pct_a = count_a / len(away_cs)
        if pct_a >= 0.8:
            trends.append((pct_a, f"{away} kept a clean sheet in {count_a}/{len(away_cs)} games"))
    # Corner totals
    total_corners = pd.to_numeric(h2h.get("total_corners"), errors="coerce")
    if total_corners.notna().any():
        over_c = total_corners > 9.5
        under_c = total_corners <= 9.5
        over_c = over_c.where(~total_corners.isna(), pd.NA)
        under_c = under_c.where(~total_corners.isna(), pd.NA)
        add_trend(over_c, "Over 9.5 corners")
        add_trend(under_c, "Under 9.5 corners")
    # First half goals
    fh_home = pd.to_numeric(h2h.get("first_half_home"), errors="coerce")
    fh_away = pd.to_numeric(h2h.get("first_half_away"), errors="coerce")
    if not fh_home.empty or not fh_away.empty:
        fh_goal = (fh_home + fh_away) > 0
        fh_goal = fh_goal.where(~(fh_home.isna() | fh_away.isna()), pd.NA)
        add_trend(fh_goal, "First-half goals")
    # Corner dominance
    corner_cols = [
        ("home_corners", "away_corners"),
        ("home_corner", "away_corner"),
        ("homecorner", "awaycorner"),
        ("corners_home", "corners_away"),
    ]
    for h_col, a_col in corner_cols:
        if {h_col, a_col}.issubset(h2h.columns):
            h_num = pd.to_numeric(h2h[h_col], errors="coerce")
            a_num = pd.to_numeric(h2h[a_col], errors="coerce")
            # Determine who had more corners relative to selected home/away
            home_more = []
            away_more = []
            for _, row in h2h.iterrows():
                r_home = row["home_team"]
                r_away = row["away_team"]
                hc = pd.to_numeric(row[h_col], errors="coerce")
                ac = pd.to_numeric(row[a_col], errors="coerce")
                if pd.isna(hc) or pd.isna(ac) or hc == ac:
                    continue
                if r_home == home:
                    home_more.append(hc > ac)
                    away_more.append(ac > hc)
                else:
                    home_more.append(ac > hc)
                    away_more.append(hc > ac)
            if home_more:
                count = sum(bool(x) for x in home_more)
                pct = count / len(home_more)
                if pct >= 0.8:
                    trends.append((pct, f"{home} more corners than {away} in {count}/{len(home_more)} games"))
            if away_more:
                count_a = sum(bool(x) for x in away_more)
                pct_a = count_a / len(away_more)
                if pct_a >= 0.8:
                    trends.append((pct_a, f"{away} more corners than {home} in {count_a}/{len(away_more)} games"))
            break
    # Sort and return top three trends
    return sorted(trends, key=lambda x: x[0], reverse=True)[:3]

# Emoji icons used to highlight trend types in the UI
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

# Dropdown to select league or view all
league_options = ["All"] + list(LEAGUE_FILES.keys())
selected = st.selectbox("Select league", league_options, index=0, help="View all leagues or choose one")

if selected == "All":
    leagues_to_show = list(LEAGUE_FILES.keys())
else:
    leagues_to_show = [selected]

for league in leagues_to_show:
    results_df, fixtures_df = load_data(league)
    if results_df is None or fixtures_df is None:
        continue
    # Determine current round based on round_number and date
    if "date" in fixtures_df.columns and "round_number" in fixtures_df.columns:
        today = pd.Timestamp.today().normalize()
        round_dates = fixtures_df.groupby("round_number")["date"].max().sort_index()
        current_round = round_dates[round_dates >= today].index.min()
        if pd.isna(current_round):
            current_round = round_dates.index.max()
        display_fixtures = fixtures_df[fixtures_df["round_number"] == current_round]
    else:
        display_fixtures = fixtures_df[fixtures_df.get("date") >= pd.Timestamp.today().normalize()]
        current_round = None
    # Subheader for league
    if current_round is not None:
        st.subheader(f"{league} – Gameweek {current_round}")
    else:
        st.subheader(f"{league} – Upcoming Fixtures")
    if display_fixtures.empty:
        st.info("No upcoming fixtures available for this league.")
        continue
    top_picks: List[Tuple[float, str]] = []
    # Iterate fixtures
    for _, fix in display_fixtures.iterrows():
        home_team = fix.get("home_team")
        away_team = fix.get("away_team")
        if pd.isna(home_team) or pd.isna(away_team):
            continue
        home_team = str(home_team)
        away_team = str(away_team)
        match_label = f"{home_team} vs {away_team}"
        with st.expander(match_label, expanded=True):
            trends = generate_trends(home_team, away_team, results_df)
            if trends:
                for pct, desc in trends:
                    # choose emoji
                    icon = "•"
                    lower_desc = desc.lower()
                    for key, symbol in EMOJI_MAP.items():
                        if key in lower_desc:
                            icon = symbol
                            break
                    st.markdown(f"{icon} {desc}")
                    top_picks.append((pct, f"{match_label} → {desc}"))
            else:
                st.info("No strong trends to recommend for this game.")
    # League-level top picks summary
    if top_picks:
        st.markdown("---")
        st.markdown(f"**Top picks for {league}:**")
        top_picks.sort(key=lambda x: x[0], reverse=True)
        shown = 0
        for pct, item in top_picks:
            if pct >= 0.9:
                st.markdown(f"✅ {item}")
                shown += 1
            if shown >= 3:
                break