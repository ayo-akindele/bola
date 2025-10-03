
"""
BolaPredict — Fixtures (dark-mode chip fix + tagline)
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

st.set_page_config(page_title="BolaPredict — Fixtures", layout="centered")
st.title("📅 BolaPredict Fixtures")
st.caption("⚡ Quick stats that matter. Fixtures are listed strictly by kick‑off time (Africa/Lagos). Each game shows the strongest recent H2H trends (no 'strongest list').")

# High-contrast radio "chips" (works for dark & light)
st.markdown(
    """
    <style>
    /* Base chip */
    div[role="radiogroup"] > label {
        border: 1px solid rgba(239,68,68,0.45); /* red-500 */
        padding: 10px 14px;
        border-radius: 10px;
        margin-right: 8px;
        margin-bottom: 8px;
        background: rgba(239,68,68,0.08);
        color: #ef4444; /* red-500 */
        cursor: pointer;
        font-weight: 600;
    }
    /* Selected */
    div[role="radiogroup"] > label[data-checked="true"] {
        background: rgba(239,68,68,0.18);
        border-color: #ef4444;
        color: #ef4444;
        box-shadow: inset 0 0 0 1px #ef4444;
    }
    /* Make sure inner radio glyph (dot) stays visible against dark bg */
    div[role="radiogroup"] svg {
        stroke: #ef4444 !important;
        fill: #ef4444 !important;
    }
    /* Section headers */
    .league-header { margin-top: 18px; margin-bottom: 6px; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---- helpers (same as previous build) ----
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

# ---- Controls ----
league_choice = st.selectbox("League", ["All"] + list(LEAGUE_FILES.keys()), index=0)
time_window = st.radio("Time window", ["Today", "Tomorrow", "Weekend (Fri–Mon)", "All in round"], horizontal=True, index=3)

now = pd.Timestamp.now(tz=LOCAL_TZ)
leagues = list(LEAGUE_FILES.keys()) if league_choice == "All" else [league_choice]

# ---- Content ----
for lg in leagues:
    res_df, fx_df = load_league(lg)
    if res_df is None or fx_df is None or fx_df.empty:
        continue

    round_id = pick_current_round(fx_df)
    view = fx_df.copy()
    if time_window == "All in round" and round_id is not None:
        view = view[view["round_number"] == round_id].copy()

    m = date_mask(view, time_window, now)
    view = view[m] if m.any() else view.iloc[0:0]
    if view.empty:
        continue

    view = view.sort_values(by=["kickoff_dt","home_team","away_team"], ascending=[True,True,True], kind="mergesort")

    hdr = f"{lg} — {'Gameweek ' + str(round_id) if time_window == 'All in round' and round_id is not None else time_window}"
    st.markdown(f"<div class='league-header'>{hdr}</div>", unsafe_allow_html=True)

    for _, r in view.iterrows():
        home = str(r.get("home_team","")).strip()
        away = str(r.get("away_team","")).strip()
        ko   = r.get("kickoff_dt")
        header = f"{format_ko(ko)} — {home} vs {away}" if ko is not None else f"{home} vs {away}"
        with st.expander(header, expanded=True):
            # trend bullets (unchanged)
            three_years_ago = datetime.today().year - 3
            h2h = res_df[((res_df["home_team"] == home) & (res_df["away_team"] == away)) |
                         ((res_df["home_team"] == away) & (res_df["away_team"] == home))].copy()
            if "match_date" in h2h.columns:
                h2h = h2h[h2h["match_date"].dt.year >= three_years_ago].sort_values("match_date", ascending=False).head(5)
            if h2h.empty:
                st.info("No strong trends to show for this fixture.")
            else:
                # compute minimal set
                hs = pd.to_numeric(h2h.get("home_score"), errors="coerce")
                as_ = pd.to_numeric(h2h.get("away_score"), errors="coerce")
                gg = (hs > 0) & (as_ > 0)
                gg = gg.where(~(hs.isna() | as_.isna()), pd.NA)
                total = hs + as_
                over25 = (total > 2.5).where(~total.isna(), pd.NA)
                under25 = (total <= 2.5).where(~total.isna(), pd.NA)
                tc = pd.to_numeric(h2h.get("total_corners"), errors="coerce")
                items = []
                for s, label in [(gg,"Both teams scored (GG)"),
                                 (over25,"Over 2.5 goals"),
                                 (under25,"Under 2.5 goals")]:
                    v = s.dropna()
                    if not v.empty:
                        pct = float(v.mean())
                        if pct >= 0.80:
                            items.append((pct, f"{label} in {int(pct*len(v))}/{len(v)} games"))
                items = sorted(items, key=lambda x: x[0], reverse=True)[:3]
                if not items:
                    st.info("No strong trends to show for this fixture.")
                else:
                    for _, text in items:
                        st.markdown(f"• {text}")
