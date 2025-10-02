
"""
Multi‑league BolaStats Streamlit App (updated)
- Chronological fixture ordering by kick-off
- "Strongest observations (100%)" replaces "Top picks"
- Per fixture, at most one 100% stat shown (priority: GG > Over 2.5 > others), excluding First-half goals
"""

import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# Mapping of league names to their historical and fixtures CSV filenames.
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
st.markdown("<h4 style='margin-bottom:0; font-weight:bold;'>⚡ Quick Stats That Matter</h4>", unsafe_allow_html=True)

def load_data(league: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load historical and fixtures data for a league from local CSV files. Returns (results_df, fixtures_df)."""
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

    # Normalise column names
    results_df.columns = [c.strip().lower().replace(" ", "_") for c in results_df.columns]
    fixtures_df.columns = [c.strip().lower().replace(" ", "_") for c in fixtures_df.columns]

    # Parse dates (fixtures 'date' includes date+time)
    if "match_date" in results_df.columns:
        results_df["match_date"] = pd.to_datetime(results_df["match_date"], errors="coerce")
    if "date" in fixtures_df.columns:
        fixtures_df["date"] = pd.to_datetime(
            fixtures_df["date"].astype(str).str.strip(),
            errors="coerce",
            dayfirst=True,
            infer_datetime_format=True,
        )

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
    Compute up to three strong trends (>=80% over 5 H2H in last 3 seasons).
    Returns a list of (percentage, description) sorted by percentage desc.
    """
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

    def add_trend(cond: pd.Series, label: str) -> None:
        valid = cond.dropna()
        if valid.empty:
            return
        count = valid.sum()
        pct = count / len(valid)
        if pct >= 0.80:
            trends.append((pct, f"{label} in {int(count)}/{len(valid)} games"))

    # Win trend for home team
    wins = 0
    for _, row in h2h.iterrows():
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

    # Clean sheets
    home_cs = []
    away_cs = []
    for _, row in h2h.iterrows():
        r_home = row["home_team"]
        r_away = row["away_team"]
        hg = pd.to_numeric(row.get("home_score"), errors="coerce")
        ag = pd.to_numeric(row.get("away_score"), errors="coerce")
        if r_home == home and pd.notna(ag):
            home_cs.append(ag == 0)
        elif r_away == home and pd.notna(hg):
            home_cs.append(hg == 0)
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

    # Corners totals
    total_corners = pd.to_numeric(h2h.get("total_corners"), errors="coerce")
    if total_corners.notna().any():
        over_c = total_corners > 9.5
        under_c = total_corners <= 9.5
        add_trend(over_c.where(~total_corners.isna(), pd.NA), "Over 9.5 corners")
        add_trend(under_c.where(~total_corners.isna(), pd.NA), "Under 9.5 corners")

    # First-half goals (we will exclude this from "Strongest observations" later)
    fh_home = pd.to_numeric(h2h.get("first_half_home"), errors="coerce")
    fh_away = pd.to_numeric(h2h.get("first_half_away"), errors="coerce")
    if not fh_home.empty or not fh_away.empty:
        fh_goal = (fh_home + fh_away) > 0
        trends.append(((fh_goal.dropna().mean() if fh_goal.dropna().size else 0.0), "First-half goals"))

    # Corner dominance
    corner_cols = [
        ("home_corners", "away_corners"),
        ("home_corner", "away_corner"),
        ("homecorner", "awaycorner"),
        ("corners_home", "corners_away"),
    ]
    for h_col, a_col in corner_cols:
        if {h_col, a_col}.issubset(h2h.columns):
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
                pct = sum(bool(x) for x in home_more) / len(home_more)
                if pct >= 0.8:
                    trends.append((pct, f"{home} more corners than {away} in {int(pct*len(home_more))}/{len(home_more)} games"))
            if away_more:
                pct_a = sum(bool(x) for x in away_more) / len(away_more)
                if pct_a >= 0.8:
                    trends.append((pct_a, f"{away} more corners than {home} in {int(pct_a*len(away_more))}/{len(away_more)} games"))
            break

    return sorted(trends, key=lambda x: x[0], reverse=True)[:3]

# Emoji icons for trend bullets
EMOJI_MAP = {
    "both teams scored": "⚽",
    "over 2.5 goals": "📈",
    "under 2.5 goals": "📉",
    "clean sheet": "🧤",
    "over 9.5 corners": "🏳️",
    "under 9.5 corners": "🔻",
    "won": "🏆",
    "more corners": "🔺",
}

# League selector
league_options = ["All"] + list(LEAGUE_FILES.keys())
selected = st.selectbox("Select league", league_options, index=0, help="View all leagues or choose one")

leagues_to_show = list(LEAGUE_FILES.keys()) if selected == "All" else [selected]

for league in leagues_to_show:
    results_df, fixtures_df = load_data(league)
    if results_df is None or fixtures_df is None:
        continue

    # Determine current round and slice fixtures
    if "date" in fixtures_df.columns and "round_number" in fixtures_df.columns:
        today = pd.Timestamp.today().normalize()
        round_dates = fixtures_df.groupby("round_number")["date"].max().sort_index()
        current_round = round_dates[round_dates >= today].index.min()
        if pd.isna(current_round):
            current_round = round_dates.index.max()
        display_fixtures = fixtures_df[fixtures_df["round_number"] == current_round].copy()
    else:
        display_fixtures = fixtures_df[fixtures_df.get("date") >= pd.Timestamp.today().normalize()].copy()
        current_round = None

    # NEW: sort chronologically by kickoff time/date
    if "date" in display_fixtures.columns:
        display_fixtures = display_fixtures.sort_values(by=["date","home_team","away_team"], ascending=[True, True, True], kind="mergesort")

    # League header
    if current_round is not None:
        st.subheader(f"{league} – Gameweek {current_round}")
    else:
        st.subheader(f"{league} – Upcoming Fixtures")

    if display_fixtures.empty:
        st.info("No upcoming fixtures available for this league.")
        continue

    # Show fixtures with their own trends as before
    chosen_obs: List[Tuple[pd.Timestamp, str]] = []  # (kickoff_ts, text) for Strongest observations section
    for _, fix in display_fixtures.iterrows():
        home_team = fix.get("home_team")
        away_team = fix.get("away_team")
        if pd.isna(home_team) or pd.isna(away_team):
            continue
        home_team = str(home_team)
        away_team = str(away_team)
        kickoff = fix.get("date")
        ko_label = ""
        if pd.notna(kickoff):
            # Format like "Fri 04 Oct · 20:00"
            try:
                ko_label = pd.to_datetime(kickoff).strftime("%a %d %b · %H:%M")
            except Exception:
                ko_label = str(kickoff)
        match_label = f"{home_team} vs {away_team}"
        header = f"{ko_label} — {match_label}" if ko_label else match_label

        with st.expander(header, expanded=True):
            trends = generate_trends(home_team, away_team, results_df)
            if trends:
                for pct, desc in trends:
                    icon = "•"
                    low = desc.lower()
                    for key, symbol in EMOJI_MAP.items():
                        if key in low:
                            icon = symbol
                            break
                    st.markdown(f"{icon} {desc}")
            else:
                st.info("No strong trends to recommend for this game.")

        # Build the "Strongest observations (100%)" pick for this match (one per fixture)
        if 'trends' not in locals():
            trends = generate_trends(home_team, away_team, results_df)
        # Keep only 100% and exclude First-half goals
        t100 = [(p, d) for (p, d) in trends if p >= 0.9999 and "First-half goals" not in d]
        # Priority order: GG, Over 2.5, then anything else
        pick = None
        for key in ["Both teams scored (GG)", "Over 2.5 goals"]:
            for p, d in t100:
                if key in d:
                    pick = (kickoff, f"**{match_label}** → {d}")
                    break
            if pick:
                break
        if not pick and t100:
            pick = (kickoff, f"**{match_label}** → {t100[0][1]}")
        if pick:
            chosen_obs.append(pick)

    # REPLACED: Instead of "Top picks", show Strongest observations (100%)
    st.markdown("---")
    st.markdown(f"**Strongest observations (100%) — {league}**")
    if not chosen_obs:
        st.info("No fixtures have a 100% observation (excluding First‑half goals).")
    else:
        # sort chosen observations by kickoff for consistency
        chosen_obs.sort(key=lambda x: (pd.Timestamp.min if pd.isna(x[0]) else x[0]))
        for ko, text in chosen_obs:
            ko_str = ""
            if pd.notna(ko):
                try:
                    ko_str = pd.to_datetime(ko).strftime("%a %d %b · %H:%M")
                except Exception:
                    ko_str = str(ko)
            st.markdown(f"✅ {text}" + (f"  ·  {ko_str}" if ko_str else ""))
