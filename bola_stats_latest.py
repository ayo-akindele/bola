
"""
BolaPredict — Fixtures (enhanced per‑fixture trends)
----------------------------------------------------
Adds to the per‑game bullets (>=80% over last up to 5 H2H in last 3 seasons):
- Home/Away **win dominance**
- Home/Away **more corners** (correctly mapped; ignores ties/NaNs)
- Home/Away **clean sheets**
Also keeps: GG/NG, Over/Under 2.5, Over/Under 9.5 corners, First‑half goals.
No "Strongest observations" list.
"""

import os
from datetime import datetime, timedelta, time as dtime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import streamlit as st

LOCAL_TZ = "Africa/Lagos"

LEAGUE_FILES: Dict[str, Dict[str, str]] = {
    "EPL": {"results": "EPL Historical Data.csv", "fixtures": "EPL_upcoming_fixtures.csv"},
    "La Liga": {"results": "SP1 Historical Data.csv", "fixtures": "SP1_upcoming_fixtures.csv"},
    "Serie A": {"results": "I1 Historical Data.csv", "fixtures": "I1_upcoming_fixtures.csv"},
    "Bundesliga": {"results": "D1 Historical Data.csv", "fixtures": "D1_upcoming_fixtures.csv"},
}

st.set_page_config(page_title="BolaPredict — Fixtures", layout="centered")
st.title("📅 BolaPredict Fixtures")
st.caption("⚡ Quick stats that matter. Fixtures are listed strictly by kick‑off time (Africa/Lagos). Each game shows recent H2H trends (no 'strongest list').")

# High-contrast radio "chips" (dark & light)
st.markdown(
    """
    <style>
    div[role="radiogroup"] > label {
        border: 1px solid rgba(239,68,68,0.45);
        padding: 10px 14px;
        border-radius: 10px;
        margin-right: 8px;
        margin-bottom: 8px;
        background: rgba(239,68,68,0.08);
        color: #ef4444;
        cursor: pointer;
        font-weight: 600;
    }
    div[role="radiogroup"] > label[data-checked="true"] {
        background: rgba(239,68,68,0.18);
        border-color: #ef4444;
        color: #ef4444;
        box-shadow: inset 0 0 0 1px #ef4444;
    }
    div[role="radiogroup"] svg { stroke: #ef4444 !important; fill: #ef4444 !important; }
    .league-header { margin-top: 18px; margin-bottom: 6px; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------- helpers -------------
def _load_csv_nearby(filename: str) -> pd.DataFrame:
    for path in [filename, os.path.join(os.path.dirname(__file__), filename)]:
        if os.path.exists(path):
            return pd.read_csv(path)
    raise FileNotFoundError(f"File not found: {filename}")

def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def _parse_fixtures_datetime(fixtures_df: pd.DataFrame) -> pd.DataFrame:
    df = fixtures_df.copy()
    if "date" in df.columns:
        date_parsed = pd.to_datetime(df["date"].astype(str).str.strip(), errors="coerce", dayfirst=True, infer_datetime_format=True)
    else:
        date_parsed = pd.to_datetime(pd.NaT)
    time_col = next((c for c in ["time","kickoff_time","kick_off_time","ko","kickoff"] if c in df.columns), None)
    if time_col:
        t = df[time_col].astype(str).str.strip()
        combo = pd.to_datetime(date_parsed.dt.strftime("%Y-%m-%d") + " " + t, errors="coerce", infer_datetime_format=True)
        kickoff = combo.where(combo.notna(), date_parsed)
    else:
        kickoff = date_parsed
    try:
        kickoff = kickoff.dt.tz_localize(LOCAL_TZ)
    except TypeError:
        kickoff = kickoff.dt.tz_convert(LOCAL_TZ)
    df["kickoff_dt"] = kickoff
    return df

def load_league(league: str):
    conf = LEAGUE_FILES.get(league, {})
    res = _normalize(_load_csv_nearby(conf["results"]))
    fix = _normalize(_load_csv_nearby(conf["fixtures"]))
    for col in list(fix.columns):
        if col.lower() == "home team":
            fix = fix.rename(columns={col: "home_team"})
        if col.lower() == "away team":
            fix = fix.rename(columns={col: "away_team"})
        if col.lower() in {"round_no","rnd"}:
            fix = fix.rename(columns={col: "round_number"})
    if "match_date" in res.columns:
        res["match_date"] = pd.to_datetime(res["match_date"], errors="coerce")
    fix = _parse_fixtures_datetime(fix)
    return res, fix

def pick_current_round(fixtures: pd.DataFrame) -> Optional[int]:
    if "round_number" not in fixtures.columns or "kickoff_dt" not in fixtures.columns:
        return None
    today = pd.Timestamp.now(tz=LOCAL_TZ).normalize()
    grp = fixtures.groupby("round_number")["kickoff_dt"].max().sort_index()
    for r, mx in grp.items():
        if pd.notna(mx) and mx >= today:
            return r
    return grp.index.max() if len(grp) else None

def next_friday_window(now: pd.Timestamp):
    now = now.tz_convert(LOCAL_TZ)
    days_ahead = (4 - now.weekday()) % 7  # 4=Fri
    fri = (now + timedelta(days=days_ahead)).date()
    start = pd.Timestamp.combine(fri, dtime(0,0)).tz_localize(LOCAL_TZ)
    end   = start + timedelta(days=3, hours=23, minutes=59, seconds=59)
    return start, end

def date_mask(df: pd.DataFrame, window: str, now: pd.Timestamp) -> pd.Series:
    if "kickoff_dt" not in df.columns:
        return pd.Series(False, index=df.index)
    dates = df["kickoff_dt"].dt.tz_convert(LOCAL_TZ).dt.date
    if window == "Today":
        return dates == now.date()
    if window == "Tomorrow":
        return dates == (now + pd.Timedelta(days=1)).date()
    if window == "Weekend (Fri–Mon)":
        start, end = next_friday_window(now)
        return (df["kickoff_dt"] >= start) & (df["kickoff_dt"] <= end)
    return pd.Series(True, index=df.index)

def format_ko(ts: Optional[pd.Timestamp]) -> str:
    if ts is None or pd.isna(ts):
        return ""
    try:
        return pd.to_datetime(ts).tz_convert(LOCAL_TZ).strftime("%a %d %b · %H:%M")
    except Exception:
        return str(ts)

# ------------- trends -------------
def compute_trends(results_df: pd.DataFrame, home: str, away: str) -> List[Tuple[float, str]]:
    """Return list of (pct, text) trends (>=0.80), sorted desc. Uses last up to 5 H2H in past 3 seasons."""
    trends: List[Tuple[float, str]] = []
    if results_df is None or results_df.empty:
        return trends

    three_years_ago = datetime.today().year - 3
    h2h = results_df[((results_df["home_team"] == home) & (results_df["away_team"] == away)) |
                     ((results_df["home_team"] == away) & (results_df["away_team"] == home))].copy()
    if "match_date" in h2h.columns:
        h2h = h2h[h2h["match_date"].dt.year >= three_years_ago].sort_values("match_date", ascending=False).head(5)
    if h2h.empty:
        return trends

    def add_bool(series: pd.Series, label: str):
        valid = series.dropna()
        if valid.empty:
            return
        pct = float(valid.mean())
        if pct >= 0.80:
            trends.append((pct, f"{label} in {int(pct*len(valid))}/{len(valid)} games"))

    # Goals
    hs = pd.to_numeric(h2h.get("home_score"), errors="coerce")
    as_ = pd.to_numeric(h2h.get("away_score"), errors="coerce")
    gg = (hs > 0) & (as_ > 0)
    gg = gg.where(~(hs.isna() | as_.isna()), pd.NA)
    add_bool(gg, "Both teams scored (GG)")
    add_bool(gg.apply(lambda x: None if pd.isna(x) else not x), "Both teams failed to score (NG)")

    total = hs + as_
    add_bool((total > 2.5).where(~total.isna(), pd.NA), "Over 2.5 goals")
    add_bool((total <= 2.5).where(~total.isna(), pd.NA), "Under 2.5 goals")

    # Win dominance
    wins_home_team = []
    wins_away_team = []
    for _, row in h2h.iterrows():
        rh, ra = row["home_team"], row["away_team"]
        hsc = pd.to_numeric(row.get("home_score"), errors="coerce")
        asc = pd.to_numeric(row.get("away_score"), errors="coerce")
        if pd.isna(hsc) or pd.isna(asc):
            continue
        if hsc == asc:
            continue
        winner = rh if hsc > asc else ra
        wins_home_team.append(winner == home)
        wins_away_team.append(winner == away)
    if wins_home_team:
        pct = sum(bool(x) for x in wins_home_team) / len(wins_home_team)
        if pct >= 0.80:
            trends.append((pct, f"{home} won {int(pct*len(wins_home_team))}/{len(wins_home_team)} recent meetings"))
    if wins_away_team:
        pct = sum(bool(x) for x in wins_away_team) / len(wins_away_team)
        if pct >= 0.80:
            trends.append((pct, f"{away} won {int(pct*len(wins_away_team))}/{len(wins_away_team)} recent meetings"))

    # Corners total O/U 9.5
    tc = pd.to_numeric(h2h.get("total_corners"), errors="coerce")
    if tc.notna().any():
        add_bool((tc > 9.5).where(~tc.isna(), pd.NA), "Over 9.5 corners")
        add_bool((tc <= 9.5).where(~tc.isna(), pd.NA), "Under 9.5 corners")

    # Corner dominance (venue-aware; ignore ties & NaNs)
    corner_pairs = [
        ("home_corners","away_corners"),
        ("home_corner","away_corner"),
        ("homecorner","awaycorner"),
        ("corners_home","corners_away"),
    ]
    for hc_col, ac_col in corner_pairs:
        if {hc_col, ac_col}.issubset(h2h.columns):
            hc = pd.to_numeric(h2h[hc_col], errors="coerce")
            ac = pd.to_numeric(h2h[ac_col], errors="coerce")

            # Map corners to the current fixture teams regardless of venue in the historical row
            home_corners_series = np.where(h2h["home_team"] == home, hc,
                                           np.where(h2h["away_team"] == home, ac, np.nan))
            away_corners_series = np.where(h2h["home_team"] == away, hc,
                                           np.where(h2h["away_team"] == away, ac, np.nan))

            home_corners_series = pd.to_numeric(home_corners_series, errors="coerce")
            away_corners_series = pd.to_numeric(away_corners_series, errors="coerce")

            mask_valid = (~home_corners_series.isna()) & (~away_corners_series.isna()) & (home_corners_series != away_corners_series)
            if mask_valid.any():
                home_more = (home_corners_series > away_corners_series)[mask_valid]
                away_more = (away_corners_series > home_corners_series)[mask_valid]
                add_bool(home_more, f"{home} more corners than {away}")
                add_bool(away_more, f"{away} more corners than {home}")
            break  # use the first valid pair

    # Clean sheets (team‑specific, venue‑aware)
    cs_home = []  # home team kept opp = 0
    cs_away = []  # away team kept opp = 0
    for _, row in h2h.iterrows():
        rh, ra = row["home_team"], row["away_team"]
        hsc = pd.to_numeric(row.get("home_score"), errors="coerce")
        asc = pd.to_numeric(row.get("away_score"), errors="coerce")
        if pd.isna(hsc) or pd.isna(asc):
            continue
        if rh == home:
            cs_home.append(asc == 0)
        elif ra == home:
            cs_home.append(hsc == 0)
        if rh == away:
            cs_away.append(asc == 0)
        elif ra == away:
            cs_away.append(hsc == 0)
    if cs_home:
        pct = sum(bool(x) for x in cs_home) / len(cs_home)
        if pct >= 0.80:
            trends.append((pct, f"{home} kept a clean sheet in {int(pct*len(cs_home))}/{len(cs_home)} games"))
    if cs_away:
        pct = sum(bool(x) for x in cs_away) / len(cs_away)
        if pct >= 0.80:
            trends.append((pct, f"{away} kept a clean sheet in {int(pct*len(cs_away))}/{len(cs_away)} games"))

    # First‑half goals (informational)
    fh_home = pd.to_numeric(h2h.get("first_half_home"), errors="coerce")
    fh_away = pd.to_numeric(h2h.get("first_half_away"), errors="coerce")
    if not fh_home.empty or not fh_away.empty:
        fh_any = ((fh_home + fh_away) > 0).where(~(fh_home.isna() | fh_away.isna()), pd.NA)
        add_bool(fh_any, "First-half goals")

    return sorted(trends, key=lambda x: x[0], reverse=True)

# ------------- controls -------------
league_choice = st.selectbox("League", ["All"] + list(LEAGUE_FILES.keys()), index=0)
time_window = st.radio("Time window", ["Today", "Tomorrow", "Weekend (Fri–Mon)", "All in round"], horizontal=True, index=3)

now = pd.Timestamp.now(tz=LOCAL_TZ)
leagues = list(LEAGUE_FILES.keys()) if league_choice == "All" else [league_choice]

def header_text(lg: str, rnd: Optional[int]) -> str:
    return f"{lg} — {'Gameweek ' + str(rnd) if time_window == 'All in round' and rnd is not None else time_window}"

def pick_round(fixtures: pd.DataFrame) -> Optional[int]:
    return pick_current_round(fixtures)

# ------------- content -------------
for lg in leagues:
    res_df, fx_df = load_league(lg)
    if res_df is None or fx_df is None or fx_df.empty:
        continue

    round_id = pick_round(fx_df)
    view = fx_df.copy()
    if time_window == "All in round" and round_id is not None:
        view = view[view["round_number"] == round_id].copy()

    m = date_mask(view, time_window, now)
    view = view[m] if m.any() else view.iloc[0:0]
    if view.empty:
        continue

    view = view.sort_values(by=["kickoff_dt","home_team","away_team"], ascending=[True,True,True], kind="mergesort")

    st.markdown(f"<div class='league-header'>{header_text(lg, round_id)}</div>", unsafe_allow_html=True)

    for _, r in view.iterrows():
        home = str(r.get("home_team","")).strip()
        away = str(r.get("away_team","")).strip()
        ko   = r.get("kickoff_dt")
        header = f"{format_ko(ko)} — {home} vs {away}" if ko is not None else f"{home} vs {away}"
        with st.expander(header, expanded=True):
            trends = compute_trends(res_df, home, away)
            # show up to 5 bullets to surface new categories without clutter
            shown = 0
            for _, text in trends:
                st.markdown(f"• {text}")
                shown += 1
                if shown >= 5:
                    break
            if shown == 0:
                st.info("No strong trends to show for this fixture.")
