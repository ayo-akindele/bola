"""
BolaPredict — Fixtures (min H2H threshold for trends)
-----------------------------------------------------
- MIN_H2H = 4 (set to 5 if you want stricter). Any metric needs at least MIN_H2H valid games.
- Added Ligue 1 (France) support using:
    • Results:    F1 Historical Data.xlsx
    • Fixtures:   F1_upcoming_fixtures.csv
- More robust loader that accepts both CSV and Excel (first sheet).
- Smarter column normalization for varied provider schemas.
"""

import os
from datetime import datetime, timedelta, time as dtime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import streamlit as st

LOCAL_TZ = "Africa/Lagos"
MIN_H2H = 4  # <-- change to 5 for stricter requirement

LEAGUE_FILES: Dict[str, Dict[str, str]] = {
    "EPL": {"results": "EPL Historical Data.csv", "fixtures": "EPL_upcoming_fixtures.csv"},
    "La Liga": {"results": "SP1 Historical Data.csv", "fixtures": "SP1_upcoming_fixtures.csv"},
    "Serie A": {"results": "I1 Historical Data.csv", "fixtures": "I1_upcoming_fixtures.csv"},
    "Bundesliga": {"results": "D1 Historical Data.csv", "fixtures": "D1_upcoming_fixtures.csv"},
    # NEW — France Ligue 1 (code F1)
    "Ligue 1": {"results": "F1 Historical Data.xlsx", "fixtures": "F1_upcoming_fixtures.csv"},
}

st.set_page_config(page_title="BolaPredict — Fixtures", layout="centered")
st.title("📅 BolaPredict Fixtures")
st.caption(
    "⚡ Quick stats that matter. Trends in the most recent H2H meetings across EPL, La Liga, Bundesliga, Serie A & Ligue 1. "
    "Fixtures are listed strictly by kick‑off time (Africa/Lagos)."
)

# Chip styling
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

# -------------------- IO helpers --------------------

def _exists_nearby(filename: str) -> Optional[str]:
    for path in [filename, os.path.join(os.path.dirname(__file__), filename)]:
        if os.path.exists(path):
            return path
    return None


def _load_table_nearby(filename: str) -> pd.DataFrame:
    """Load CSV or Excel from current dir or module dir.
    - For Excel, use the first sheet.
    - Keep raw columns; normalization done separately.
    """
    path = _exists_nearby(filename)
    if not path:
        raise FileNotFoundError(f"File not found: {filename}")

    ext = os.path.splitext(path)[1].lower()
    if ext in [".csv", ".txt", ".tsv"]:
        # Allow for weird encodings silently
        try:
            return pd.read_csv(path)
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="latin-1")
    elif ext in [".xlsx", ".xlsm", ".xls"]:
        try:
            return pd.read_excel(path)
        except Exception:
            # some providers require engine explicitly
            return pd.read_excel(path, engine="openpyxl")
    else:
        raise ValueError(f"Unsupported file extension for {filename}")


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


# -------------------- Schema mappers --------------------

RESULTS_MAP = {
    # canonical : list of known variants
    "match_date": ["match_date", "date", "matchdate"],
    "home_team": ["home_team", "home", "hometeam", "home_team_name", "home_club"],
    "away_team": ["away_team", "away", "awayteam", "away_team_name", "away_club"],
    "home_score": ["home_score", "hs", "fthg", "home_goals", "home_goal", "hg"],
    "away_score": ["away_score", "as", "ftag", "away_goals", "away_goal", "ag"],
    "home_corners": ["home_corners", "homecorner", "corners_home", "hc", "home_c"],
    "away_corners": ["away_corners", "awaycorner", "corners_away", "ac", "away_c"],
    "total_corners": ["total_corners", "corners_total", "totalcorner", "corner_total", "corners"],
    "first_half_home": [
        "first_half_home", "h1_home", "fh_home", "ht_home_goals", "home_ht_goals", "home_first_half_goals", "h1hg"
    ],
    "first_half_away": [
        "first_half_away", "h1_away", "fh_away", "ht_away_goals", "away_ht_goals", "away_first_half_goals", "h1ag"
    ],
}


def _standardize_results(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_cols(df_raw)

    # Map to canonical names where present
    colmap = {}
    for canon, variants in RESULTS_MAP.items():
        for v in variants:
            if v in df.columns:
                colmap[v] = canon
                break
    df = df.rename(columns=colmap)

    # Compute missing derived fields
    if "match_date" in df.columns:
        df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")

    # If home/away score split not present but total goals or result code exist, try to infer (best‑effort)
    # We keep it simple: rely on common fields; otherwise leave as NaN.

    # Total corners from components
    if "total_corners" not in df.columns:
        if {"home_corners", "away_corners"}.issubset(df.columns):
            df["total_corners"] = pd.to_numeric(df["home_corners"], errors="coerce") + pd.to_numeric(
                df["away_corners"], errors="coerce"
            )

    # First‑half goals from halves if provider used H1HG/H1AG numeric fields
    if "first_half_home" not in df.columns and "h1hg" in df_raw.columns:
        df["first_half_home"] = pd.to_numeric(df_raw["h1hg"], errors="coerce")
    if "first_half_away" not in df.columns and "h1ag" in df_raw.columns:
        df["first_half_away"] = pd.to_numeric(df_raw["h1ag"], errors="coerce")

    # Ensure numeric types for core metrics
    for c in ["home_score", "away_score", "home_corners", "away_corners", "total_corners", "first_half_home", "first_half_away"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def _standardize_fixtures(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_cols(df_raw)

    # Normalize key names
    rename_pairs = {}
    for c in list(df.columns):
        lc = c.lower()
        if lc in {"home team", "hometeam"}:
            rename_pairs[c] = "home_team"
        elif lc in {"away team", "awayteam"}:
            rename_pairs[c] = "away_team"
        elif lc in {"round_no", "rnd", "round"}:
            rename_pairs[c] = "round_number"
    if rename_pairs:
        df = df.rename(columns=rename_pairs)

    # Parse kickoff datetime in local TZ
    df = _parse_fixtures_datetime(df)
    return df


# -------------------- Datetime helpers --------------------

def _parse_fixtures_datetime(fixtures_df: pd.DataFrame) -> pd.DataFrame:
    df = fixtures_df.copy()
    # Date
    if "date" in df.columns:
        date_parsed = pd.to_datetime(
            df["date"].astype(str).str.strip(), errors="coerce", dayfirst=True, infer_datetime_format=True
        )
    elif "match_date" in df.columns:
        date_parsed = pd.to_datetime(df["match_date"], errors="coerce")
    else:
        date_parsed = pd.to_datetime(pd.NaT)

    # Time
    time_col = next((c for c in ["time", "kickoff_time", "kick_off_time", "ko", "kickoff"] if c in df.columns), None)
    if time_col:
        t = df[time_col].astype(str).str.strip()
        combo = pd.to_datetime(
            date_parsed.dt.strftime("%Y-%m-%d") + " " + t, errors="coerce", infer_datetime_format=True
        )
        kickoff = combo.where(combo.notna(), date_parsed)
    else:
        kickoff = date_parsed

    try:
        kickoff = kickoff.dt.tz_localize(LOCAL_TZ)
    except TypeError:
        kickoff = kickoff.dt.tz_convert(LOCAL_TZ)

    df["kickoff_dt"] = kickoff
    return df


# -------------------- Loaders --------------------

def load_league(league: str):
    conf = LEAGUE_FILES.get(league, {})
    if not conf:
        return None, None
    res_raw = _load_table_nearby(conf["results"])  # CSV or Excel
    fix_raw = _load_table_nearby(conf["fixtures"])  # CSV

    res = _standardize_results(res_raw)
    fix = _standardize_fixtures(fix_raw)

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
    start = pd.Timestamp.combine(fri, dtime(0, 0)).tz_localize(LOCAL_TZ)
    end = start + timedelta(days=3, hours=23, minutes=59, seconds=59)
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


# -------------------- Trends engine --------------------

def compute_trends(results_df: pd.DataFrame, home: str, away: str) -> List[Tuple[float, str]]:
    """Return list of (pct, text) trends (>=0.80), sorted desc. Uses last up to 5 H2H in past 3 seasons.
       Any metric needs at least MIN_H2H valid games to be considered.
    """
    trends: List[Tuple[float, str]] = []
    if results_df is None or results_df.empty:
        return trends

    three_years_ago = datetime.today().year - 3
    h2h = results_df[
        ((results_df.get("home_team") == home) & (results_df.get("away_team") == away))
        | ((results_df.get("home_team") == away) & (results_df.get("away_team") == home))
    ].copy()

    if "match_date" in h2h.columns:
        h2h = h2h[pd.to_datetime(h2h["match_date"], errors="coerce").dt.year >= three_years_ago]
        h2h = h2h.sort_values("match_date", ascending=False).head(5)

    if len(h2h) < MIN_H2H:
        return trends

    def add_bool(series: pd.Series, label: str):
        valid = series.dropna()
        if len(valid) < MIN_H2H:
            return
        pct = float(valid.mean())
        if pct >= 0.80:
            trends.append((pct, f"{label} in {int(round(pct * len(valid)))}/{len(valid)} games"))

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
        rh, ra = row.get("home_team"), row.get("away_team")
        hsc = pd.to_numeric(row.get("home_score"), errors="coerce")
        asc = pd.to_numeric(row.get("away_score"), errors="coerce")
        if pd.isna(hsc) or pd.isna(asc) or hsc == asc:
            continue
        winner = rh if hsc > asc else ra
        wins_home_team.append(winner == home)
        wins_away_team.append(winner == away)
    if len(wins_home_team) >= MIN_H2H:
        pct = sum(bool(x) for x in wins_home_team) / len(wins_home_team)
        if pct >= 0.80:
            trends.append((pct, f"{home} won {int(round(pct * len(wins_home_team)))}/{len(wins_home_team)} recent meetings"))
    if len(wins_away_team) >= MIN_H2H:
        pct = sum(bool(x) for x in wins_away_team) / len(wins_away_team)
        if pct >= 0.80:
            trends.append((pct, f"{away} won {int(round(pct * len(wins_away_team)))}/{len(wins_away_team)} recent meetings"))

    # Corners total O/U 9.5
    tc = pd.to_numeric(h2h.get("total_corners"), errors="coerce")
    if tc.notna().sum() >= MIN_H2H:
        add_bool((tc > 9.5).where(~tc.isna(), pd.NA), "Over 9.5 corners")
        add_bool((tc <= 9.5).where(~tc.isna(), pd.NA), "Under 9.5 corners")

    # Corner dominance (venue-aware; ignore ties & NaNs)
    corner_pairs = [
        ("home_corners", "away_corners"),
        ("home_corner", "away_corner"),
        ("homecorner", "awaycorner"),
        ("corners_home", "corners_away"),
    ]
    for hc_col, ac_col in corner_pairs:
        if {hc_col, ac_col}.issubset(h2h.columns):
            hc = pd.to_numeric(h2h[hc_col], errors="coerce")
            ac = pd.to_numeric(h2h[ac_col], errors="coerce")

            home_arr = np.where(h2h["home_team"] == home, hc, np.where(h2h["away_team"] == home, ac, np.nan))
            away_arr = np.where(h2h["home_team"] == away, hc, np.where(h2h["away_team"] == away, ac, np.nan))

            home_ser = pd.to_numeric(pd.Series(home_arr), errors="coerce")
            away_ser = pd.to_numeric(pd.Series(away_arr), errors="coerce")

            mask_valid = home_ser.notna() & away_ser.notna() & (home_ser != away_ser)
            if mask_valid.sum() >= MIN_H2H:
                add_bool((home_ser > away_ser)[mask_valid], f"{home} more corners than {away}")
                add_bool((away_ser > home_ser)[mask_valid], f"{away} more corners than {home}")
            break  # first valid pair only

    # Clean sheets (team‑specific, venue‑aware)
    cs_home = []
    cs_away = []
    for _, row in h2h.iterrows():
        rh, ra = row.get("home_team"), row.get("away_team")
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
    if len(cs_home) >= MIN_H2H:
        pct = sum(bool(x) for x in cs_home) / len(cs_home)
        if pct >= 0.80:
            trends.append((pct, f"{home} kept a clean sheet in {int(round(pct * len(cs_home)))}/{len(cs_home)} games"))
    if len(cs_away) >= MIN_H2H:
        pct = sum(bool(x) for x in cs_away) / len(cs_away)
        if pct >= 0.80:
            trends.append((pct, f"{away} kept a clean sheet in {int(round(pct * len(cs_away)))}/{len(cs_away)} games"))

    # First‑half goals (informational)
    fh_home = pd.to_numeric(h2h.get("first_half_home"), errors="coerce")
    fh_away = pd.to_numeric(h2h.get("first_half_away"), errors="coerce")
    fh_valid = (~fh_home.isna()) & (~fh_away.isna())
    if fh_valid.sum() >= MIN_H2H:
        fh_any = ((fh_home + fh_away) > 0).where(fh_valid, pd.NA)
        add_bool(fh_any, "First-half goals")

    return sorted(trends, key=lambda x: x[0], reverse=True)


# -------------------- UI --------------------

league_choice = st.selectbox("League", ["All"] + list(LEAGUE_FILES.keys()), index=0)
time_window = st.radio("Time window", ["Today", "Tomorrow", "Weekend (Fri–Mon)", "All in round"], horizontal=True, index=3)

now = pd.Timestamp.now(tz=LOCAL_TZ)
leagues = list(LEAGUE_FILES.keys()) if league_choice == "All" else [league_choice]


def header_text(lg: str, rnd: Optional[int]) -> str:
    return f"{lg} — {'Gameweek ' + str(rnd) if time_window == 'All in round' and rnd is not None else time_window}"


def pick_round(fixtures: pd.DataFrame) -> Optional[int]:
    return pick_current_round(fixtures)


# -------------------- Content --------------------

for lg in leagues:
    try:
        res_df, fx_df = load_league(lg)
    except Exception as e:
        st.warning(f"{lg}: {e}")
        continue

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

    view = view.sort_values(by=["kickoff_dt", "home_team", "away_team"], ascending=[True, True, True], kind="mergesort")

    st.markdown(f"<div class='league-header'>{header_text(lg, round_id)}</div>", unsafe_allow_html=True)

    for _, r in view.iterrows():
        home = str(r.get("home_team", "")).strip()
        away = str(r.get("away_team", "")).strip()
        ko = r.get("kickoff_dt")
        header = f"{format_ko(ko)} — {home} vs {away}" if ko is not None else f"{home} vs {away}"
        with st.expander(header, expanded=True):
            trends = compute_trends(res_df, home, away)
            shown = 0
            for _, text in trends:
                st.markdown(f"• {text}")
                shown += 1
                if shown >= 5:
                    break
            if shown == 0:
                st.info("No strong trends to show for this fixture.")
