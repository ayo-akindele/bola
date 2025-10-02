
"""
BolaStats — Fixtures by Date (Simplified)
----------------------------------------
- Filters: Today / Tomorrow / Weekend (Fri–Mon) / All in round
- Chronological sorting by kickoff time (Africa/Lagos)
- Input: per‑league CSVs where Column B = Date, Column E = Time
- No "strongest trends" or recommendations — just clean fixture listings
"""

import os
from datetime import datetime, timedelta, time as dtime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

LOCAL_TZ = "Africa/Lagos"

LEAGUE_FILES: Dict[str, Dict[str, str]] = {
    "EPL": {"results": "EPL Historical Data.csv", "fixtures": "EPL_upcoming_fixtures.csv"},
    "La Liga": {"results": "SP1 Historical Data.csv", "fixtures": "SP1_upcoming_fixtures.csv"},
    "Serie A": {"results": "I1 Historical Data.csv", "fixtures": "I1_upcoming_fixtures.csv"},
    "Bundesliga": {"results": "D1 Historical Data.csv", "fixtures": "D1_upcoming_fixtures.csv"},
}

# ------------- Streamlit chrome -------------
st.set_page_config(page_title="BolaStats — Fixtures", layout="centered")
st.title("📅 BolaPredict Fixtures")
st.caption("Fixtures are listed strictly by kick‑off time (Africa/Lagos).")

# Subtle CSS to make radio look like rectangular chips
st.markdown(
    """
    <style>
    /* radio chips */
    div[role="radiogroup"] > label {
        border: 1px solid #e5e7eb;
        padding: 8px 12px;
        border-radius: 8px;
        margin-right: 8px;
        margin-bottom: 6px;
        background: #ffffff;
        cursor: pointer;
    }
    div[role="radiogroup"] > label[data-checked="true"] {
        background: #f1f5f9;
        border-color: #94a3b8;
        box-shadow: inset 0 0 0 1px #94a3b8;
    }
    /* list aesthetics */
    .fixture-row {
        padding: 10px 12px;
        border-radius: 10px;
        border: 1px solid #eef2f7;
        margin-bottom: 8px;
        background: #ffffff;
    }
    .fixture-time {
        font-weight: 600;
        margin-right: 10px;
    }
    .fixture-teams {
        font-weight: 500;
    }
    .league-header {
        margin-top: 20px;
        margin-bottom: 6px;
        font-weight: 700;
    }
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
    """
    Build tz-aware kickoff_dt from 'date' + 'time'.
    - 'date' may be date-only or date+time; 'time' takes precedence when present.
    """
    df = fixtures_df.copy()

    # base date
    if "date" in df.columns:
        date_parsed = pd.to_datetime(df["date"].astype(str).str.strip(),
                                     errors="coerce", dayfirst=True, infer_datetime_format=True)
    else:
        date_parsed = pd.to_datetime(pd.NaT)

    # time column (from Column E)
    time_col = next((c for c in ["time", "kickoff_time", "kick_off_time", "ko", "kickoff"] if c in df.columns), None)

    if time_col:
        time_str = df[time_col].astype(str).str.strip()
        combo = pd.to_datetime(date_parsed.dt.strftime("%Y-%m-%d") + " " + time_str,
                               errors="coerce", infer_datetime_format=True)
        kickoff = combo.where(combo.notna(), date_parsed)
    else:
        kickoff = date_parsed

    try:
        kickoff = kickoff.dt.tz_localize(LOCAL_TZ)
    except TypeError:
        kickoff = kickoff.dt.tz_convert(LOCAL_TZ)

    df["kickoff_dt"] = kickoff
    return df

def load_league(league: str) -> pd.DataFrame:
    conf = LEAGUE_FILES.get(league)
    if not conf:
        st.error(f"No configuration for league: {league}")
        return pd.DataFrame()
    try:
        fx = _normalize(_load_csv_nearby(conf["fixtures"]))
    except Exception as exc:
        st.error(f"{league}: {exc}")
        return pd.DataFrame()

    # Common renames
    for col in list(fx.columns):
        if col.lower() == "home team":
            fx = fx.rename(columns={col: "home_team"})
        if col.lower() == "away team":
            fx = fx.rename(columns={col: "away_team"})
        if col.lower() in {"round_no", "rnd"}:
            fx = fx.rename(columns={col: "round_number"})
        if col.lower() == "time":
            fx = fx.rename(columns={col: "time"})  # normalized by _normalize anyway

    fx = _parse_fixtures_datetime(fx)
    return fx

def pick_current_round(fixtures: pd.DataFrame) -> Optional[int]:
    if "round_number" not in fixtures.columns or "kickoff_dt" not in fixtures.columns:
        return None
    today = pd.Timestamp.now(tz=LOCAL_TZ).normalize()
    grp = fixtures.groupby("round_number")["kickoff_dt"].max().sort_index()
    for r, mx in grp.items():
        if pd.notna(mx) and mx >= today:
            return r
    return grp.index.max() if len(grp) else None

def next_friday_window(now: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
    now = now.tz_convert(LOCAL_TZ)
    days_ahead = (4 - now.weekday()) % 7  # 4=Fri
    fri = (now + timedelta(days=days_ahead)).date()
    start = pd.Timestamp.combine(fri, dtime(0,0)).tz_localize(LOCAL_TZ)
    end   = start + timedelta(days=3, hours=23, minutes=59, seconds=59)  # Fri->Mon 23:59:59
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
    # All in round: no time filtering
    return pd.Series(True, index=df.index)

def format_ko(ts: Optional[pd.Timestamp]) -> str:
    if ts is None or pd.isna(ts):
        return ""
    try:
        return pd.to_datetime(ts).tz_convert(LOCAL_TZ).strftime("%a %d %b · %H:%M")
    except Exception:
        return str(ts)

# ------------- controls -------------
league_choice = st.selectbox("League", ["All"] + list(LEAGUE_FILES.keys()), index=0)

time_window = st.radio(
    "Time window",
    ["Today", "Tomorrow", "Weekend (Fri–Mon)", "All in round"],
    horizontal=True,
    index=3
)

now = pd.Timestamp.now(tz=LOCAL_TZ)
leagues = list(LEAGUE_FILES.keys()) if league_choice == "All" else [league_choice]

# ------------- content -------------
for lg in leagues:
    fx = load_league(lg)
    if fx.empty:
        continue

    # scope to round for "All in round"
    round_id = pick_current_round(fx)
    view = fx.copy()
    if time_window == "All in round" and round_id is not None:
        view = view[view["round_number"] == round_id].copy()

    # time mask
    m = date_mask(view, time_window, now)
    view = view[m] if m.any() else view.iloc[0:0]
    if view.empty:
        continue

    # sort
    view = view.sort_values(by=["kickoff_dt","home_team","away_team"], ascending=[True,True,True], kind="mergesort")

    # header
    hdr = f"{lg} — {'Gameweek ' + str(round_id) if time_window == 'All in round' and round_id is not None else time_window}"
    st.markdown(f"<div class='league-header'>{hdr}</div>", unsafe_allow_html=True)

    # list fixtures
    for _, r in view.iterrows():
        home = str(r.get("home_team","")).strip()
        away = str(r.get("away_team","")).strip()
        ko   = r.get("kickoff_dt")
        row  = f"<div class='fixture-row'><span class='fixture-time'>{format_ko(ko)}</span><span class='fixture-teams'>{home} vs {away}</span></div>"
        st.markdown(row, unsafe_allow_html=True)
