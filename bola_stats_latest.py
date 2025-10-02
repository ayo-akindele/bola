
"""
BolaStats — Multi‑league Fixtures & Strongest Observations (patched)
- Robust date filters (Today / Tomorrow) using date equality in Africa/Lagos.
- Weekend window remains next Fri–Mon.
- Strongest picks no longer require 5 H2H; 100% is allowed at 3/3, 4/4, or 5/5 (First‑half goals excluded).
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

st.set_page_config(page_title="BolaStats", layout="centered")
st.title("📊 BolaStats")
st.caption("Fixtures sorted by kick-off. Strongest observations show 100% trends (First-half goals excluded).")

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
    # parse date first
    if "date" in df.columns:
        date_parsed = pd.to_datetime(df["date"].astype(str).str.strip(), errors="coerce", dayfirst=True, infer_datetime_format=True)
    else:
        date_parsed = pd.to_datetime(pd.NaT)
    # prefer explicit time column if present
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

def load_league(league: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    conf = LEAGUE_FILES.get(league)
    if not conf:
        st.error(f"No configuration for league: {league}")
        return None, None
    try:
        res = _normalize(_load_csv_nearby(conf["results"]))
        fx  = _normalize(_load_csv_nearby(conf["fixtures"]))
    except Exception as exc:
        st.error(f"{league}: {exc}")
        return None, None
    if "match_date" in res.columns:
        res["match_date"] = pd.to_datetime(res["match_date"], errors="coerce")
    # common renames
    for col in list(fx.columns):
        if col.lower() == "home team":
            fx = fx.rename(columns={col: "home_team"})
        if col.lower() == "away team":
            fx = fx.rename(columns={col: "away_team"})
        if col.lower() in {"round_no","rnd"}:
            fx = fx.rename(columns={col: "round_number"})
    fx = _parse_fixtures_datetime(fx)
    return res, fx

def next_friday_window(now: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
    now = now.tz_convert(LOCAL_TZ)
    days_ahead = (4 - now.weekday()) % 7  # 4=Fri
    fri = (now + timedelta(days=days_ahead)).date()
    start = pd.Timestamp.combine(fri, dtime(0,0)).tz_localize(LOCAL_TZ)
    end   = start + timedelta(days=3, hours=23, minutes=59, seconds=59)  # through Mon 23:59:59
    return start, end

def pick_current_round(fixtures: pd.DataFrame) -> Optional[int]:
    if "round_number" not in fixtures.columns or "kickoff_dt" not in fixtures.columns:
        return None
    today = pd.Timestamp.now(tz=LOCAL_TZ).normalize()
    grp = fixtures.groupby("round_number")["kickoff_dt"].max().sort_index()
    for r, mx in grp.items():
        if pd.notna(mx) and mx >= today:
            return r
    return grp.index.max() if len(grp) else None

def compute_trends(results_df: pd.DataFrame, home: str, away: str) -> List[Tuple[float,str]]:
    three_years_ago = datetime.today().year - 3
    h2h = results_df[((results_df["home_team"] == home) & (results_df["away_team"] == away)) |
                     ((results_df["home_team"] == away) & (results_df["away_team"] == home))].copy()
    if "match_date" in h2h.columns:
        h2h = h2h[h2h["match_date"].dt.year >= three_years_ago].sort_values("match_date", ascending=False).head(5)
    if h2h.empty:  # allow 1..5, not only 5
        return []
    trends: List[Tuple[float,str]] = []

    def add_trend(s: pd.Series, label: str):
        valid = s.dropna()
        if valid.empty:
            return
        pct = float(valid.mean())
        if pct >= 0.80:
            trends.append((pct, f"{label} in {int(pct*len(valid))}/{len(valid)} games"))

    hs = pd.to_numeric(h2h.get("home_score"), errors="coerce")
    as_ = pd.to_numeric(h2h.get("away_score"), errors="coerce")
    gg = (hs > 0) & (as_ > 0)
    gg = gg.where(~(hs.isna() | as_.isna()), pd.NA)
    add_trend(gg, "Both teams scored (GG)")
    add_trend(gg.apply(lambda x: None if pd.isna(x) else not x), "Both teams failed to score (NG)")

    total = hs + as_
    add_trend((total > 2.5).where(~total.isna(), pd.NA), "Over 2.5 goals")
    add_trend((total <= 2.5).where(~total.isna(), pd.NA), "Under 2.5 goals")

    # Corners total
    tc = pd.to_numeric(h2h.get("total_corners"), errors="coerce")
    if tc.notna().any():
        add_trend((tc > 9.5).where(~tc.isna(), pd.NA), "Over 9.5 corners")
        add_trend((tc <= 9.5).where(~tc.isna(), pd.NA), "Under 9.5 corners")

    # First-half goals (stored for info; excluded from strongest picks)
    fh_home = pd.to_numeric(h2h.get("first_half_home"), errors="coerce")
    fh_away = pd.to_numeric(h2h.get("first_half_away"), errors="coerce")
    if not fh_home.empty or not fh_away.empty:
        fh_any = ((fh_home + fh_away) > 0).where(~(fh_home.isna() | fh_away.isna()), pd.NA)
        v = fh_any.dropna()
        if not v.empty:
            trends.append((float(v.mean()), "First-half goals"))

    return sorted(trends, key=lambda x: x[0], reverse=True)[:3]

def pick_strongest(trends: List[Tuple[float,str]]) -> Optional[str]:
    t100 = [(p, d) for (p, d) in trends if p >= 0.9999 and "First-half goals" not in d]
    if not t100:
        return None
    for key in ["Both teams scored (GG)", "Over 2.5 goals"]:
        for _, d in t100:
            if key in d:
                return d
    return t100[0][1]

def format_ko(ts: Optional[pd.Timestamp]) -> str:
    if ts is None or pd.isna(ts):
        return ""
    try:
        return pd.to_datetime(ts).tz_convert(LOCAL_TZ).strftime("%a %d %b · %H:%M")
    except Exception:
        return str(ts)

# ------------- sidebar -------------
league_choice = st.selectbox("League", ["All"] + list(LEAGUE_FILES.keys()), index=0)
time_filter = st.radio("Time window", ["Today", "Tomorrow", "Weekend (Fri–Mon)", "All in round"], horizontal=True, index=3)

now = pd.Timestamp.now(tz=LOCAL_TZ)
leagues = list(LEAGUE_FILES.keys()) if league_choice == "All" else [league_choice]

tab_fx, tab_strong, tab_weekend = st.tabs(["📅 Fixtures", "💪 Strongest (this window)", "🌟 Weekend Strongest"])

def date_mask(df: pd.DataFrame, window: str) -> pd.Series:
    if "kickoff_dt" not in df.columns:
        return pd.Series(False, index=df.index)
    dates = df["kickoff_dt"].dt.tz_convert(LOCAL_TZ).dt.date
    if window == "Today":
        target = now.date()
        return dates == target
    if window == "Tomorrow":
        target = (now + pd.Timedelta(days=1)).date()
        return dates == target
    if window == "Weekend (Fri–Mon)":
        start, end = next_friday_window(now)
        return (df["kickoff_dt"] >= start) & (df["kickoff_dt"] <= end)
    # All in round: don't filter by time
    return pd.Series(True, index=df.index)

# ---- Fixtures tab ----
with tab_fx:
    for lg in leagues:
        res, fx = load_league(lg)
        if res is None or fx is None:
            continue
        rnd = pick_current_round(fx)
        view = fx.copy()
        if time_filter == "All in round" and rnd is not None:
            view = view[view["round_number"] == rnd].copy()
        m = date_mask(view, time_filter)
        view = view[m] if m.any() else view.iloc[0:0]
        if view.empty:
            continue
        view = view.sort_values(by=["kickoff_dt","home_team","away_team"], ascending=[True,True,True], kind="mergesort")
        hdr = f"{lg} — {'Gameweek ' + str(rnd) if time_filter == 'All in round' and rnd is not None else time_filter}"
        st.subheader(hdr)
        for _, r in view.iterrows():
            home = str(r.get("home_team","")).strip()
            away = str(r.get("away_team","")).strip()
            header = f"{format_ko(r.get('kickoff_dt'))} — {home} vs {away}"
            with st.expander(header, expanded=True):
                t = compute_trends(res, home, away)
                if t:
                    for p, d in t:
                        st.markdown(f"• {d}")
                else:
                    st.info("No strong trends to show for this fixture.")

# ---- Strongest (this window) ----
with tab_strong:
    picks = []
    for lg in leagues:
        res, fx = load_league(lg)
        if res is None or fx is None:
            continue
        rnd = pick_current_round(fx)
        view = fx.copy()
        if time_filter == "All in round" and rnd is not None:
            view = view[view["round_number"] == rnd].copy()
        m = date_mask(view, time_filter)
        view = view[m] if m.any() else view.iloc[0:0]
        if view.empty:
            continue
        view = view.sort_values(by=["kickoff_dt","home_team","away_team"], ascending=[True,True,True], kind="mergesort")
        for _, r in view.iterrows():
            home = str(r.get("home_team","")).strip()
            away = str(r.get("away_team","")).strip()
            ko   = r.get("kickoff_dt")
            s    = pick_strongest(compute_trends(res, home, away))
            if s:
                picks.append((ko, lg, f"**{home} vs {away}** — {s}"))
    st.subheader(f"Strongest observations (100%) — {time_filter}")
    if not picks:
        st.info("No 100% observations for this window (First‑half goals excluded).")
    else:
        picks.sort(key=lambda x: (pd.Timestamp.min if pd.isna(x[0]) else x[0]))
        for ko, lg, text in picks:
            st.markdown(f"✅ {text} · {lg} · {format_ko(ko)}")

# ---- Weekend Strongest ----
with tab_weekend:
    start, end = next_friday_window(now)
    picks = []
    for lg in LEAGUE_FILES.keys():
        res, fx = load_league(lg)
        if res is None or fx is None:
            continue
        m = (fx["kickoff_dt"].notna()) & (fx["kickoff_dt"] >= start) & (fx["kickoff_dt"] <= end)
        view = fx[m].copy()
        if view.empty:
            continue
        view = view.sort_values(by=["kickoff_dt","home_team","away_team"], ascending=[True,True,True], kind="mergesort")
        for _, r in view.iterrows():
            home = str(r.get("home_team","")).strip()
            away = str(r.get("away_team","")).strip()
            ko   = r.get("kickoff_dt")
            s    = pick_strongest(compute_trends(res, home, away))
            if s:
                picks.append((ko, lg, f"**{home} vs {away}** — {s}"))
    st.subheader("Weekend Strongest (next Fri–Mon) — All leagues")
    if not picks:
        st.info("No 100% observations for the upcoming weekend (First‑half goals excluded).")
    else:
        picks.sort(key=lambda x: (pd.Timestamp.min if pd.isna(x[0]) else x[0]))
        for ko, lg, text in picks:
            st.markdown(f"✅ {text} · {lg} · {format_ko(ko)}")
