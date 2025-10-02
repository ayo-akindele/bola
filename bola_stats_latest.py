
"""
BolaStats — Multi‑league Fixtures & Strongest Observations
---------------------------------------------------------
- Kick-off time built from 'Date' (Column B) + 'Time' (Column E), Africa/Lagos.
- Filters: Today / Tomorrow / Weekend (Fri–Mon) / All in round.
- Chronological ordering.
- "Strongest observations (100%)" replaces "Top picks":
    Priority per match: GG > Over 2.5 > others (exclude First‑half goals).
- Extra tab: "Weekend Strongest" aggregates 100% observations across all leagues for the next weekend.
"""

import os
from datetime import datetime, timedelta, time as dtime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

LOCAL_TZ = "Africa/Lagos"

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

st.set_page_config(page_title="BolaStats", layout="centered")
st.title("📊 BolaStats")
st.caption("Fixtures sorted by kick-off. Strongest observations show 100% trends (5/5), excluding First-half goals.")

# ---------------- helpers ----------------
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

    # time column (from new Column E)
    time_col = next((c for c in ["time", "kickoff_time", "kick_off_time", "ko", "kickoff"] if c in df.columns), None)

    if time_col:
        time_str = df[time_col].astype(str).str.strip()
        combo = pd.to_datetime(date_parsed.dt.strftime("%Y-%m-%d") + " " + time_str,
                               errors="coerce", infer_datetime_format=True)
        kickoff = combo
        need_fallback = kickoff.isna() & date_parsed.notna()
        kickoff = kickoff.where(~need_fallback, date_parsed)
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
        fx = _normalize(_load_csv_nearby(conf["fixtures"]))
    except Exception as exc:
        st.error(f"{league}: {exc}")
        return None, None

    # unify columns
    if "match_date" in res.columns:
        res["match_date"] = pd.to_datetime(res["match_date"], errors="coerce")

    # fixture teams & round
    rename_map = {}
    for col in fx.columns:
        if col.lower() in {"home team"}:
            rename_map[col] = "home_team"
        if col.lower() in {"away team"}:
            rename_map[col] = "away_team"
        if col.lower() in {"round_no", "rnd"}:
            rename_map[col] = "round_number"
    if rename_map:
        fx = fx.rename(columns=rename_map)

    fx = _parse_fixtures_datetime(fx)

    return res, fx

def next_friday_window(now: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Next Fri 00:00 to Mon 23:59:59 (Africa/Lagos)."""
    now = now.tz_convert(LOCAL_TZ)
    days_ahead = (4 - now.weekday()) % 7  # 4 = Friday
    fri = (now + timedelta(days=days_ahead)).date()
    start = pd.Timestamp.combine(fri, dtime(0,0)).tz_localize(LOCAL_TZ)
    end = start + timedelta(days=3, hours=23, minutes=59, seconds=59)  # Fri->Mon 23:59:59
    return start, end

def pick_current_round(fixtures: pd.DataFrame) -> Optional[int]:
    if "round_number" not in fixtures.columns or "kickoff_dt" not in fixtures.columns:
        return None
    today = pd.Timestamp.now(tz=LOCAL_TZ).normalize()
    grp = fixtures.groupby("round_number")["kickoff_dt"].max().sort_index()
    # first round with max >= today
    for r, mx in grp.items():
        if pd.notna(mx) and mx >= today:
            return r
    # fallback to latest
    return grp.index.max() if len(grp) else None

def compute_trends(results_df: pd.DataFrame, home: str, away: str) -> List[Tuple[float, str]]:
    three_years_ago = datetime.today().year - 3
    h2h = results_df[((results_df["home_team"] == home) & (results_df["away_team"] == away)) |
                     ((results_df["home_team"] == away) & (results_df["away_team"] == home))].copy()
    if "match_date" in h2h.columns:
        h2h = h2h[h2h["match_date"].dt.year >= three_years_ago].sort_values("match_date", ascending=False).head(5)
    if h2h.empty or len(h2h) < 5:
        return []

    trends: List[Tuple[float, str]] = []

    def add_trend(series_bool: pd.Series, label: str):
        valid = series_bool.dropna()
        if valid.empty:
            return
        pct = valid.mean()
        if pct >= 0.80:
            trends.append((float(pct), f"{label} in {int(pct*len(valid))}/{len(valid)} games"))

    hg = pd.to_numeric(h2h.get("home_score"), errors="coerce")
    ag = pd.to_numeric(h2h.get("away_score"), errors="coerce")

    # GG / NG
    gg = (hg > 0) & (ag > 0)
    gg = gg.where(~(hg.isna() | ag.isna()), pd.NA)
    ng = gg.apply(lambda x: None if pd.isna(x) else not x)
    add_trend(gg, "Both teams scored (GG)")
    add_trend(ng, "Both teams failed to score (NG)")

    # O/U 2.5
    total = hg + ag
    over25 = (total > 2.5).where(~total.isna(), pd.NA)
    under25 = (total <= 2.5).where(~total.isna(), pd.NA)
    add_trend(over25, "Over 2.5 goals")
    add_trend(under25, "Under 2.5 goals")

    # Clean sheets
    # (quick venue-aware check)
    home_cs = []
    away_cs = []
    for _, r in h2h.iterrows():
        rh, ra = r["home_team"], r["away_team"]
        hs = pd.to_numeric(r.get("home_score"), errors="coerce")
        as_ = pd.to_numeric(r.get("away_score"), errors="coerce")
        if pd.notna(hs) and pd.notna(as_):
            if rh == home:
                home_cs.append(as_ == 0)
            else:
                home_cs.append(hs == 0)
            if ra == away:
                away_cs.append(hs == 0)
            else:
                away_cs.append(as_ == 0)
    if home_cs:
        pct = sum(bool(x) for x in home_cs) / len(home_cs)
        if pct >= 0.80:
            trends.append((pct, f"{home} kept a clean sheet in {int(pct*len(home_cs))}/{len(home_cs)} games"))
    if away_cs:
        pct = sum(bool(x) for x in away_cs) / len(away_cs)
        if pct >= 0.80:
            trends.append((pct, f"{away} kept a clean sheet in {int(pct*len(away_cs))}/{len(away_cs)} games"))

    # Corners total
    total_corners = pd.to_numeric(h2h.get("total_corners"), errors="coerce")
    if total_corners.notna().any():
        add_trend((total_corners > 9.5).where(~total_corners.isna(), pd.NA), "Over 9.5 corners")
        add_trend((total_corners <= 9.5).where(~total_corners.isna(), pd.NA), "Under 9.5 corners")

    # First-half goals (for display only; will be excluded from 100% list)
    fh_home = pd.to_numeric(h2h.get("first_half_home"), errors="coerce")
    fh_away = pd.to_numeric(h2h.get("first_half_away"), errors="coerce")
    if not fh_home.empty or not fh_away.empty:
        fh_any = ((fh_home + fh_away) > 0).where(~(fh_home.isna() | fh_away.isna()), pd.NA)
        # Store but later we exclude from strongest picks
        valid = fh_any.dropna()
        if not valid.empty:
            pct = valid.mean()
            trends.append((pct, "First-half goals"))

    # Corner dominance (venue-aware mapping)
    for h_col, a_col in [("home_corners","away_corners"),("home_corner","away_corner"),
                         ("homecorner","awaycorner"),("corners_home","corners_away")]:
        if {h_col, a_col}.issubset(h2h.columns):
            home_more = []
            away_more = []
            for _, r in h2h.iterrows():
                rh, ra = r["home_team"], r["away_team"]
                hc = pd.to_numeric(r[h_col], errors="coerce")
                ac = pd.to_numeric(r[a_col], errors="coerce")
                if pd.isna(hc) or pd.isna(ac) or hc == ac:
                    continue
                if rh == home:
                    home_more.append(hc > ac)
                    away_more.append(ac > hc)
                else:
                    home_more.append(ac > hc)
                    away_more.append(hc > ac)
            if home_more:
                pct = sum(bool(x) for x in home_more) / len(home_more)
                if pct >= 0.80:
                    trends.append((pct, f"{home} more corners than {away} in {int(pct*len(home_more))}/{len(home_more)} games"))
            if away_more:
                pct = sum(bool(x) for x in away_more) / len(away_more)
                if pct >= 0.80:
                    trends.append((pct, f"{away} more corners than {home} in {int(pct*len(away_more))}/{len(away_more)} games"))
            break

    return sorted(trends, key=lambda x: x[0], reverse=True)[:3]

def pick_strongest_for_match(trends: List[Tuple[float,str]]) -> Optional[str]:
    """Return ONE 100% observation text by priority: GG > Over 2.5 > others; exclude First-half goals."""
    t100 = [(p, d) for (p, d) in trends if p >= 0.9999 and "First-half goals" not in d]
    if not t100:
        return None
    # Priority order
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

# ---------------- sidebar controls ----------------
league_choice = st.selectbox("League", ["All"] + list(LEAGUE_FILES.keys()), index=0)
time_filter = st.radio("Time window", ["Today", "Tomorrow", "Weekend (Fri–Mon)", "All in round"], horizontal=True, index=3)

# ---------------- build views ----------------
now = pd.Timestamp.now(tz=LOCAL_TZ)
leagues = list(LEAGUE_FILES.keys()) if league_choice == "All" else [league_choice]

# Tabs: Fixtures | Strongest (for window) | Weekend Strongest
tab_fx, tab_strong, tab_weekend = st.tabs(["📅 Fixtures", "💪 Strongest (this window)", "🌟 Weekend Strongest"])

def time_window_mask(df: pd.DataFrame, window: str) -> pd.Series:
    if "kickoff_dt" not in df.columns:
        return pd.Series(False, index=df.index)
    if window == "Today":
        start = now.normalize()
        end = start + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    elif window == "Tomorrow":
        start = now.normalize() + pd.Timedelta(days=1)
        end = start + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    elif window == "Weekend (Fri–Mon)":
        start, end = next_friday_window(now)
    else:
        # All in round: no additional time slicing
        return pd.Series(True, index=df.index)
    return (df["kickoff_dt"].notna()) & (df["kickoff_dt"] >= start) & (df["kickoff_dt"] <= end)

# ---------- Fixtures tab ----------
with tab_fx:
    for league in leagues:
        res_df, fx_df = load_league(league)
        if res_df is None or fx_df is None:
            continue

        # Determine current round to scope "All in round"
        round_id = pick_current_round(fx_df)
        view = fx_df.copy()
        if time_filter == "All in round" and round_id is not None:
            view = view[view["round_number"] == round_id].copy()

        mask = time_window_mask(view, time_filter)
        view = view[mask] if mask.any() else view.iloc[0:0]
        if view.empty:
            continue

        # Sort by kickoff
        view = view.sort_values(by=["kickoff_dt","home_team","away_team"], ascending=[True,True,True], kind="mergesort")

        # Header per league
        hdr = f"{league} — "
        if time_filter == "All in round" and round_id is not None:
            hdr += f"Gameweek {round_id}"
        else:
            hdr += time_filter
        st.subheader(hdr)

        for _, r in view.iterrows():
            home = str(r.get("home_team","")).strip()
            away = str(r.get("away_team","")).strip()
            ko = r.get("kickoff_dt")
            header = f"{format_ko(ko)} — {home} vs {away}" if ko else f"{home} vs {away}"

            with st.expander(header, expanded=True):
                t = compute_trends(res_df, home, away)
                if t:
                    for p, d in t:
                        st.markdown(f"• {d}")
                else:
                    st.info("No strong trends to show for this fixture.")

# ---------- Strongest (this window) tab ----------
with tab_strong:
    picks = []
    for league in leagues:
        res_df, fx_df = load_league(league)
        if res_df is None or fx_df is None:
            continue
        round_id = pick_current_round(fx_df)
        view = fx_df.copy()
        if time_filter == "All in round" and round_id is not None:
            view = view[view["round_number"] == round_id].copy()
        mask = time_window_mask(view, time_filter)
        view = view[mask] if mask.any() else view.iloc[0:0]
        if view.empty:
            continue
        # order
        view = view.sort_values(by=["kickoff_dt","home_team","away_team"], ascending=[True,True,True], kind="mergesort")
        for _, r in view.iterrows():
            home = str(r.get("home_team","")).strip()
            away = str(r.get("away_team","")).strip()
            ko = r.get("kickoff_dt")
            t = compute_trends(res_df, home, away)
            pick_text = pick_strongest_for_match(t)
            if pick_text:
                picks.append((ko, league, f"**{home} vs {away}** → {pick_text}"))

    st.subheader(f"Strongest observations (100%) — {time_filter}")
    if not picks:
        st.info("No 100% observations for this window (First‑half goals excluded).")
    else:
        picks.sort(key=lambda x: (pd.Timestamp.min if pd.isna(x[0]) else x[0]))
        for ko, league, text in picks:
            st.markdown(f"✅ {text}  ·  {league}  ·  {format_ko(ko)}")

# ---------- Weekend Strongest tab ----------
with tab_weekend:
    start, end = next_friday_window(now)
    picks = []
    for league in LEAGUE_FILES.keys():  # always aggregate all leagues here
        res_df, fx_df = load_league(league)
        if res_df is None or fx_df is None:
            continue
        mask = (fx_df["kickoff_dt"].notna()) & (fx_df["kickoff_dt"] >= start) & (fx_df["kickoff_dt"] <= end)
        view = fx_df[mask].copy()
        if view.empty:
            continue
        view = view.sort_values(by=["kickoff_dt","home_team","away_team"], ascending=[True,True,True], kind="mergesort")
        for _, r in view.iterrows():
            home = str(r.get("home_team","")).strip()
            away = str(r.get("away_team","")).strip()
            ko = r.get("kickoff_dt")
            t = compute_trends(res_df, home, away)
            pick_text = pick_strongest_for_match(t)
            if pick_text:
                picks.append((ko, league, f"**{home} vs {away}** → {pick_text}"))

    st.subheader("Weekend Strongest (next Fri–Mon) — All leagues")
    if not picks:
        st.info("No 100% observations for the upcoming weekend (First‑half goals excluded).")
    else:
        picks.sort(key=lambda x: (pd.Timestamp.min if pd.isna(x[0]) else x[0]))
        for ko, league, text in picks:
            st.markdown(f"✅ {text}  ·  {league}  ·  {format_ko(ko)}")
