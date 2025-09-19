# bola_stats_latest.py
# BolaStats Streamlit App: chips on homepage, single-gameweek league views, full trend set with 80%/5-min rules.

import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# ----------------------------
# Config
# ----------------------------
LEAGUE_FILES: Dict[str, Dict[str, str]] = {
    "EPL": {"results": "EPL Historical Data.csv", "fixtures": "EPL_upcoming_fixtures.csv"},
    "La Liga": {"results": "SP1 Historical Data.csv", "fixtures": "SP1_upcoming_fixtures.csv"},
    "Serie A": {"results": "I1 Historical Data.csv", "fixtures": "I1_upcoming_fixtures.csv"},
    "Bundesliga": {"results": "D1 Historical Data.csv", "fixtures": "D1_upcoming_fixtures.csv"},
}

st.set_page_config(page_title="BolaStats", layout="centered")
st.title("📊 BolaStats")
st.markdown("<h4 style='margin-bottom:0; font-weight:bold;'>⚡ Quick Stats That Matter</h4>", unsafe_allow_html=True)

# ----------------------------
# Data loading / normalization
# ----------------------------
def load_data(league: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    cfg = LEAGUE_FILES.get(league)
    if not cfg:
        st.error(f"No configuration for league: {league}")
        return None, None

    def _load_csv(filename: str) -> pd.DataFrame:
        candidates = [filename, os.path.join(os.path.dirname(__file__), filename)]
        for p in candidates:
            if os.path.exists(p):
                return pd.read_csv(p)
        raise FileNotFoundError(f"File not found: {filename}")

    try:
        results_df = _load_csv(cfg["results"])
        fixtures_df = _load_csv(cfg["fixtures"])
    except Exception as exc:
        st.error(f"Error reading CSVs for {league}: {exc}")
        return None, None

    results_df.columns = [c.strip().lower().replace(" ", "_") for c in results_df.columns]
    fixtures_df.columns = [c.strip().lower().replace(" ", "_") for c in fixtures_df.columns]

    # Parse dates
    if "match_date" in results_df.columns:
        results_df["match_date"] = pd.to_datetime(results_df["match_date"], errors="coerce")
    if "date" in fixtures_df.columns:
        fixtures_df["date"] = pd.to_datetime(fixtures_df["date"], errors="coerce")

    # Unify team columns in fixtures
    rename_map = {}
    for col in fixtures_df.columns:
        if col.lower() in {"home team", "home_team", "hometeam"}:
            rename_map[col] = "home_team"
        if col.lower() in {"away team", "away_team", "awayteam"}:
            rename_map[col] = "away_team"
    if rename_map:
        fixtures_df = fixtures_df.rename(columns=rename_map)

    return results_df, fixtures_df

# ----------------------------
# Trend engine (80% threshold, min 5 H2H)
# ----------------------------
def generate_trends(home: str, away: str, results_df: pd.DataFrame) -> List[Tuple[float, str]]:
    """
    Compute strong trends (>=80% success over at least 5 recent H2H).
    Includes: wins, GG/NG, O/U2.5, clean sheets, total corners, FH goals, corner dominance.
    Returns: list[(pct_float, description_str)] sorted by pct desc, capped to top 3.
    """
    if results_df is None or results_df.empty:
        return []
    # Last 3 calendar years, most recent 5 H2H meetings
    three_years_ago = datetime.today().year - 3
    h2h = results_df[
        ((results_df["home_team"] == home) & (results_df["away_team"] == away)) |
        ((results_df["home_team"] == away) & (results_df["away_team"] == home))
    ]
    if "match_date" not in h2h.columns:
        return []
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
        count = int(valid.sum())
        total = len(valid)
        pct = count / total
        if pct >= 0.8:  # 80% threshold
            trends.append((pct, f"{label} in {count}/{total} games"))

    # Win trend for selected home team
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
    over25 = (total_goals > 2.5).where(~total_goals.isna(), pd.NA)
    under25 = (total_goals <= 2.5).where(~total_goals.isna(), pd.NA)
    add_trend(over25, "Over 2.5 goals")
    add_trend(under25, "Under 2.5 goals")

    # Clean sheets (perspective-specific)
    home_cs, away_cs = [], []
    for _, row in h2h.iterrows():
        r_home = row["home_team"]; r_away = row["away_team"]
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
        count = sum(bool(x) for x in home_cs); pct = count / len(home_cs)
        if pct >= 0.8:
            trends.append((pct, f"{home} kept a clean sheet in {count}/{len(home_cs)} games"))
    if away_cs:
        count = sum(bool(x) for x in away_cs); pct = count / len(away_cs)
        if pct >= 0.8:
            trends.append((pct, f"{away} kept a clean sheet in {count}/{len(away_cs)} games"))

    # Total corners
    total_corners = pd.to_numeric(h2h.get("total_corners"), errors="coerce")
    if total_corners.notna().any():
        over_c = (total_corners > 9.5).where(~total_corners.isna(), pd.NA)
        under_c = (total_corners <= 9.5).where(~total_corners.isna(), pd.NA)
        add_trend(over_c, "Over 9.5 corners")
        add_trend(under_c, "Under 9.5 corners")

    # First-half goals (any)
    fh_home = pd.to_numeric(h2h.get("first_half_home"), errors="coerce")
    fh_away = pd.to_numeric(h2h.get("first_half_away"), errors="coerce")
    if fh_home is not None and fh_away is not None:
        fh_goal = (fh_home + fh_away) > 0
        fh_goal = fh_goal.where(~(fh_home.isna() | fh_away.isna()), pd.NA)
        add_trend(fh_goal, "First-half goals")

    # Corner dominance (X had more corners than Y)
    corner_pairs = [
        ("home_corners", "away_corners"),
        ("home_corner", "away_corner"),
        ("homecorner", "awaycorner"),
        ("corners_home", "corners_away"),
    ]
    for h_col, a_col in corner_pairs:
        if {h_col, a_col}.issubset(h2h.columns):
            home_more, away_more = [], []
            for _, row in h2h.iterrows():
                r_home = row["home_team"]; r_away = row["away_team"]
                hc = pd.to_numeric(row[h_col], errors="coerce")
                ac = pd.to_numeric(row[a_col], errors="coerce")
                if pd.isna(hc) or pd.isna(ac) or hc == ac:
                    continue
                # compare relative to selected sides
                if r_home == home:
                    home_more.append(hc > ac)
                    away_more.append(ac > hc)
                else:
                    home_more.append(ac > hc)
                    away_more.append(hc > ac)
            if home_more:
                c = sum(bool(x) for x in home_more); pct = c / len(home_more)
                if pct >= 0.8:
                    trends.append((pct, f"{home} more corners than {away} in {c}/{len(home_more)} games"))
            if away_more:
                c = sum(bool(x) for x in away_more); pct = c / len(away_more)
                if pct >= 0.8:
                    trends.append((pct, f"{away} more corners than {home} in {c}/{len(away_more)} games"))
            break

    # Top 3 by strength
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

# ----------------------------
# Helpers: sorting & date logic
# ----------------------------
def lagos_today() -> pd.Timestamp:
    # Normalize to date in Africa/Lagos (no DST)
    return pd.Timestamp.now(tz="Africa/Lagos").normalize().tz_localize(None)

def add_time_sort_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a sortable datetime column from date + optional time.
    Recognizes time columns: 'time', 'kickoff_time', 'kickoff', 'ko_time'.
    """
    if df.empty:
        return df
    df = df.copy()
    time_cols = [c for c in df.columns if c in {"time", "kickoff_time", "kickoff", "ko_time"}]
    if time_cols:
        tcol = time_cols[0]
        parsed_time = pd.to_datetime(df[tcol], errors="coerce").dt.time
        comb = [
            pd.Timestamp.combine(d.date(), t) if pd.notna(d) and pd.notna(t) else d
            for d, t in zip(df["date"], parsed_time)
        ]
        df["__dt_sort__"] = pd.to_datetime(comb, errors="coerce")
    else:
        df["__dt_sort__"] = df["date"]
    return df

def select_single_gameweek(fixtures_df: pd.DataFrame, today: pd.Timestamp) -> Tuple[pd.DataFrame, Optional[int]]:
    """
    Return fixtures for one gameweek:
      - If 'round_number' exists: pick the earliest round whose max-date >= today; else the last round.
      - Else: infer gameweek by ISO week of the earliest upcoming match date.
    Returns (df, round_number_or_None).
    """
    if fixtures_df is None or fixtures_df.empty:
        return fixtures_df, None

    df = fixtures_df.copy()
    df = df[df["date"].notna()]
    if df.empty:
        return df, None

    # Case 1: explicit round numbers
    if "round_number" in df.columns and df["round_number"].notna().any():
        round_dates = df.groupby("round_number")["date"].max().sort_index()
        upcoming = round_dates[round_dates.dt.normalize() >= today]
        current_round = int(upcoming.index.min()) if not upcoming.empty else int(round_dates.index.max())
        out = df[df["round_number"] == current_round].copy()
        return out, current_round

    # Case 2: infer by ISO week of the next upcoming match date
    upcoming_mask = df["date"].dt.normalize() >= today
    if not upcoming_mask.any():
        # if nothing upcoming, show most recent ISO week in data
        last_day = df["date"].max().normalize()
        week = int(last_day.isocalendar().week)
    else:
        first_day = df.loc[upcoming_mask, "date"].min().normalize()
        week = int(first_day.isocalendar().week)

    df["__iso_week__"] = df["date"].dt.isocalendar().week.astype(int)
    out = df[df["__iso_week__"] == week].copy()
    out.drop(columns=["__iso_week__"], inplace=True, errors="ignore")
    return out, None

# ----------------------------
# UI
# ----------------------------
today = lagos_today()

league_options = ["All"] + list(LEAGUE_FILES.keys())
selected = st.selectbox("Select league", league_options, index=0, help="View all leagues or choose one")

# ---- Homepage: Today / Tomorrow / Weekend fixtures across all leagues ----
if selected == "All":
    # Toggle chips (Segmented Control if available, else radio fallback)
    try:
        selected_range = st.segmented_control(
            "Show fixtures for",
            options=["Today", "Tomorrow", "Weekend"],
            default="Today",
        )
    except Exception:
        selected_range = st.radio(
            "Show fixtures for",
            options=["Today", "Tomorrow", "Weekend"],
            index=0,
            horizontal=True,
        )

    # Build date set for filter (Africa/Lagos)
    base_today = lagos_today()
    if selected_range == "Today":
        date_set = {base_today}
        subtitle = base_today.strftime("%A, %d %b %Y")
    elif selected_range == "Tomorrow":
        d = base_today + pd.Timedelta(days=1)
        date_set = {d}
        subtitle = d.strftime("%A, %d %b %Y")
    else:
        # Weekend: Fri–Sun of the upcoming weekend relative to today
        weekday = base_today.weekday()  # Mon=0 ... Sun=6
        days_until_fri = (4 - weekday) % 7
        fri = base_today + pd.Timedelta(days=days_until_fri)
        sat = fri + pd.Timedelta(days=1)
        sun = fri + pd.Timedelta(days=2)
        date_set = {fri, sat, sun}
        subtitle = f"{fri.strftime('%d %b')} – {sun.strftime('%d %b %Y')}"

    st.subheader(f"Fixtures – {selected_range} ({subtitle})")

    any_fixtures = False
    for league in LEAGUE_FILES.keys():
        results_df, fixtures_df = load_data(league)
        if results_df is None or fixtures_df is None or fixtures_df.empty:
            continue

        # Filter to the selected dates (normalize fixture dates first)
        fx = fixtures_df.copy()
        fx["__date_only__"] = fx["date"].dt.normalize()
        fx = fx[fx["__date_only__"].isin(date_set)].copy()
        if fx.empty:
            continue
        any_fixtures = True

        fx = add_time_sort_cols(fx).sort_values(["__dt_sort__", "home_team", "away_team"])
        st.markdown(f"### {league}")
        for _, fix in fx.iterrows():
            home = str(fix.get("home_team") or "")
            away = str(fix.get("away_team") or "")
            if not home or not away:
                continue
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
        st.info("No fixtures for the selected range across the configured leagues.")

# ---- League views: SINGLE GAMEWEEK (current/upcoming), header "League – Gameweek X trends" ----
else:
    league = selected
    results_df, fixtures_df = load_data(league)
    if results_df is None or fixtures_df is None or fixtures_df.empty:
        st.stop()

    display_fixtures, current_round = select_single_gameweek(fixtures_df, today)
    display_fixtures = add_time_sort_cols(display_fixtures).sort_values(["__dt_sort__", "home_team", "away_team"])

    header = f"{league} – Gameweek {int(current_round)} trends" if current_round is not None else f"{league} – Gameweek trends"
    st.subheader(header)

    if display_fixtures.empty:
        st.info("No fixtures available for the current gameweek.")
    else:
        for _, fix in display_fixtures.iterrows():
            home = str(fix.get("home_team") or "")
            away = str(fix.get("away_team") or "")
            if not home or not away:
                continue
            match_label = f"{home} vs {away}"
            with st.expander(match_label, expanded=True):
                trends = generate_trends(home, away, results_df)
                if trends:
                    for pct, desc in trends:
                        icon = next((symbol for key, symbol in EMOJI_MAP.items() if key in desc.lower()), "•")
                        st.markdown(f"{icon} {desc}")
                else:
                    st.info("No strong trends to recommend for this game.")
