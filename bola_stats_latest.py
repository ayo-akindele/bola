
"""
BolaPredict — Fixtures (Mobile-First, Inline Trends, Highlights)
----------------------------------------------------------------
- Netherlands Eredivisie (N1) supported
  • Results:  N1 Historical Data.csv
  • Fixtures: N1_upcoming_fixtures.csv
- Mobile-friendly, single-column cards with inline trends
- NEW: Highlights section (GG, Over 2.5, First-half goals)
- Randomizer removed (not fit for purpose)
"""

import os
from datetime import datetime, timedelta, time as dtime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import streamlit as st

# -------------------- Config --------------------
LOCAL_TZ = "Africa/Lagos"
MIN_H2H = 4  # set 5 for stricter H2H requirement

LEAGUE_FILES: Dict[str, Dict[str, str]] = {
    "EPL 🇬🇧": {"results": "EPL Historical Data.csv", "fixtures": "EPL_upcoming_fixtures.csv"},
    "La Liga 🇪🇸": {"results": "SP1 Historical Data.csv", "fixtures": "SP1_upcoming_fixtures.csv"},
    "Serie A 🇮🇹": {"results": "I1 Historical Data.csv", "fixtures": "I1_upcoming_fixtures.csv"},
    "Bundesliga 🇩🇪": {"results": "D1 Historical Data.csv", "fixtures": "D1_upcoming_fixtures.csv"},
    "Ligue 1 🇫🇷": {"results": "F1 Historical Data.csv", "fixtures": "F1_upcoming_fixtures.csv"},
    "Eredivisie 🇳🇱": {"results": "N1 Historical Data.csv", "fixtures": "N1_upcoming_fixtures.csv"},
}

# -------------------- Page Setup --------------------
st.set_page_config(page_title="BolaPredict — Fixtures", layout="centered")
st.title("⚡ BolaPredict — Quick Stats That Matter")
st.caption("Head-to-head trends and fixtures. Times in Africa/Lagos.")

# Mobile-friendly CSS tweaks
st.markdown(
    """
    <style>
/* FIX SELECTBOX TEXT VISIBILITY FOR DARK MODE */
/* Closed state */
.stSelectbox div[data-baseweb="select"] {
    background-color: #ffffff !important;
    color: #000000 !important;
}

/* Open dropdown items */
.stSelectbox div[role="listbox"] {
    background-color: #1f2937 !important;  /* dark background */
    color: #ffffff !important; /* white text */
}

.stSelectbox div[role="option"] {
    background-color: #1f2937 !important;
    color: #ffffff !important;
}

/* Chevron icon */
.stSelectbox svg {
    stroke: #000000 !important;
}

/* Label */
.stSelectbox label, .stSelectbox p {
    color: #1f2937 !important;
    font-weight: 600 !important;
}



    /* Force light theme even when device is in dark mode */
    html, body, .block-container {
      background-color: #ffffff !important;
      color: #000000 !important;
    }
    /* Ensure text is always readable */
    * {
      color: #1f2937 !important;
    }
    .block-container { padding-top: 0.6rem; padding-bottom: 2rem; max-width: 700px; }
    h1, h2, h3, h4 { line-height: 1.2; }
    .league-header {
      margin-top: 12px;
      margin-bottom: 6px;
      padding: 6px 10px;
      background-color: rgba(239,68,68,0.08);
      border: 1px solid rgba(239,68,68,0.15);
      border-radius: 10px;
      font-weight: 700;
      color: #b91c1c;
      display: inline-block;
    }
    .fixture-card {
      border: 1px solid rgba(0,0,0,0.08);
      border-radius: 12px;
      padding: 10px 12px;
      margin-bottom: 10px;
      background: white;
    }
    .fixture-title { font-weight: 700; margin-bottom: 6px; }
    .fixture-sub { color: #6b7280; font-size: 0.95rem; margin-bottom: 6px; }
    .trend-bullets { margin: 0 0 2px 0; padding-left: 18px; }
    .trend-bullets li { margin: 2px 0; }
    .pill { display:inline-block; padding: 2px 8px; border-radius: 999px; font-size: 0.8rem; margin-right: 6px; border:1px solid rgba(0,0,0,0.12); }
    .pill-goals { background: rgba(34,197,94,0.12); }
    .pill-fh { background: rgba(59,130,246,0.12); }
    .pill-over { background: rgba(250,204,21,0.18); }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- Helper Functions --------------------
def _exists_nearby(filename: str) -> Optional[str]:
    for path in [filename, os.path.join(os.path.dirname(__file__), filename)]:
        if os.path.exists(path):
            return path
    return None

def _load_csv_nearby(filename: str) -> pd.DataFrame:
    path = _exists_nearby(filename)
    if not path:
        raise FileNotFoundError(f"File not found: {filename}")
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

RESULTS_MAP = {
    "match_date": ["match_date", "date", "matchdate"],
    "home_team": ["home_team", "home", "hometeam", "home_team_name", "home_club"],
    "away_team": ["away_team", "away", "awayteam", "away_team_name", "away_club"],
    "home_score": ["home_score", "hs", "fthg", "home_goals", "home_goal", "hg"],
    "away_score": ["away_score", "as", "ftag", "away_goals", "away_goal", "ag"],
    "home_corners": ["home_corners", "homecorner", "corners_home", "hc", "home_c"],
    "away_corners": ["away_corners", "awaycorner", "corners_away", "ac", "away_c"],
    "total_corners": ["total_corners", "corners_total", "totalcorner", "corner_total", "corners"],
    "first_half_home": ["first_half_home", "h1_home", "fh_home", "ht_home_goals", "home_ht_goals", "home_first_half_goals", "h1hg"],
    "first_half_away": ["first_half_away", "h1_away", "fh_away", "ht_away_goals", "away_ht_goals", "away_first_half_goals", "h1ag"],
}

FIXTURES_RENAMES = {
    "home team": "home_team", "hometeam": "home_team",
    "away team": "away_team", "awayteam": "away_team",
    "round_no": "round_number", "rnd": "round_number", "round": "round_number",
}

def _standardize_results(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_cols(df_raw)

    # map to canonical names if present
    colmap = {}
    for canon, variants in RESULTS_MAP.items():
        for v in variants:
            if v in df.columns:
                colmap[v] = canon
                break
    if colmap:
        df = df.rename(columns=colmap)

    # parse dates
    if "match_date" in df.columns:
        df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce", dayfirst=True)

    # total corners if missing
    if "total_corners" not in df.columns and {"home_corners", "away_corners"}.issubset(df.columns):
        df["total_corners"] = pd.to_numeric(df["home_corners"], errors="coerce") + pd.to_numeric(df["away_corners"], errors="coerce")

    # ensure numeric types
    for c in ["home_score","away_score","home_corners","away_corners","total_corners","first_half_home","first_half_away"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

def _parse_fixtures_datetime(fixtures_df: pd.DataFrame) -> pd.DataFrame:
    df = fixtures_df.copy()

    # try several ways to build kickoff_dt
    date_parsed = None
    for date_col in ["kickoff_dt", "datetime", "match_datetime", "match_date", "date"]:
        if date_col in df.columns:
            date_parsed = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
            break
    if date_parsed is None:
        date_parsed = pd.to_datetime(pd.NaT)

    time_col = next((c for c in ["time", "kickoff_time", "kick_off_time", "ko", "kickoff"] if c in df.columns), None)
    if time_col and "date" in df.columns:
        t = df[time_col].astype(str).str.strip()
        combo = pd.to_datetime(
            pd.to_datetime(df["date"], errors="coerce", dayfirst=True).dt.strftime("%Y-%m-%d") + " " + t,
            errors="coerce",
        )
        kickoff = combo.where(combo.notna(), date_parsed)
    else:
        kickoff = date_parsed

    try:
        kickoff = kickoff.dt.tz_localize(LOCAL_TZ)
    except Exception:
        try:
            kickoff = kickoff.dt.tz_convert(LOCAL_TZ)
        except Exception:
            pass

    df["kickoff_dt"] = kickoff
    return df

def _standardize_fixtures(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_cols(df_raw)
    # rename common alternates
    ren = {c: FIXTURES_RENAMES[c] for c in df.columns if c in FIXTURES_RENAMES}
    if ren:
        df = df.rename(columns=ren)

    df = _parse_fixtures_datetime(df)
    return df

# -------------------- Loaders --------------------
def load_league(league: str):
    conf = LEAGUE_FILES.get(league, {})
    if not conf:
        return None, None
    try:
        res_raw = _load_csv_nearby(conf["results"])
        fix_raw = _load_csv_nearby(conf["fixtures"])
    except Exception as e:
        st.warning(f"{league}: {e}")
        return None, None

    res = _standardize_results(res_raw)
    fix = _standardize_fixtures(fix_raw)

    return res, fix

# -------------------- Utilities --------------------
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
    return pd.Series(True, index=df.index)  # All / All in round

def format_ko(ts: Optional[pd.Timestamp]) -> str:
    if ts is None or pd.isna(ts):
        return ""
    try:
        return pd.to_datetime(ts).tz_convert(LOCAL_TZ).strftime("%a %d %b · %H:%M")
    except Exception:
        return str(ts)

# -------------------- Trends engine --------------------
def compute_trends(results_df: pd.DataFrame, home: str, away: str) -> List[Tuple[float, str, str]]:
    """Return list of (pct, text, tag) 'strong' trends (>=0.80), sorted desc.
       Uses last up to 5 H2H in past 3 seasons. Needs at least MIN_H2H valid games.
       tag ∈ {"gg","over25","firsthalf","wins","corners"}
    """
    trends: List[Tuple[float, str, str]] = []
    if results_df is None or results_df.empty:
        return trends

    three_years_ago = datetime.today().year - 3
    h2h = results_df[
        ((results_df.get("home_team") == home) & (results_df.get("away_team") == away)) |
        ((results_df.get("home_team") == away) & (results_df.get("away_team") == home))
    ].copy()

    if "match_date" in h2h.columns:
        h2h = h2h[pd.to_datetime(h2h["match_date"], errors="coerce").dt.year >= three_years_ago]
        h2h = h2h.sort_values("match_date", ascending=False).head(5)

    if len(h2h) < MIN_H2H:
        return trends

    def add_bool(series: pd.Series, label: str, tag: str):
        valid = series.dropna()
        if len(valid) < MIN_H2H:
            return
        pct = float(valid.mean())
        if pct >= 0.80:
            trends.append((pct, f"{label} in {int(round(pct * len(valid)))}/{len(valid)} games", tag))

    # Goals / GG / Over 2.5
    hs = pd.to_numeric(h2h.get("home_score"), errors="coerce")
    as_ = pd.to_numeric(h2h.get("away_score"), errors="coerce")
    gg = (hs > 0) & (as_ > 0)
    gg = gg.where(~(hs.isna() | as_.isna()), pd.NA)
    add_bool(gg, "Both teams scored (GG)", "gg")

    total = hs + as_
    add_bool((total > 2.5).where(~total.isna(), pd.NA), "Over 2.5 goals", "over25")

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
            trends.append((pct, f"{home} won {int(round(pct * len(wins_home_team)))}/{len(wins_home_team)} recent meetings", "wins"))
    if len(wins_away_team) >= MIN_H2H:
        pct = sum(bool(x) for x in wins_away_team) / len(wins_away_team)
        if pct >= 0.80:
            trends.append((pct, f"{away} won {int(round(pct * len(wins_away_team)))}/{len(wins_away_team)} recent meetings", "wins"))

    # Corners total O/U 9.5
    tc = pd.to_numeric(h2h.get("total_corners"), errors="coerce")
    if tc.notna().sum() >= MIN_H2H:
        add_bool((tc > 9.5).where(~tc.isna(), pd.NA), "Over 9.5 corners", "corners")

    # First-half goals
    fh_home = pd.to_numeric(h2h.get("first_half_home"), errors="coerce")
    fh_away = pd.to_numeric(h2h.get("first_half_away"), errors="coerce")
    fh_valid = (~fh_home.isna()) & (~fh_away.isna())
    if fh_valid.sum() >= MIN_H2H:
        fh_any = ((fh_home + fh_away) > 0).where(fh_valid, pd.NA)
        add_bool(fh_any, "First-half goals", "firsthalf")

    return sorted(trends, key=lambda x: x[0], reverse=True)

# -------------------- Top Controls --------------------
col1, col2 = st.columns([1.3, 1])
with col1:
    league_choice = st.selectbox("League", ["All"] + list(LEAGUE_FILES.keys()), index=0)
with col2:
    time_window = st.selectbox("When", ["All in round", "Today", "Tomorrow", "Weekend (Fri–Mon)", "All"], index=0)

# Fixed defaults for clean UI
max_trends = 3       # bullets per match
max_fixtures = 15    # fixtures per league
show_debug = False   # hide debug info

now = pd.Timestamp.now(tz=LOCAL_TZ)
leagues = list(LEAGUE_FILES.keys()) if league_choice == "All" else [league_choice]

# -------------------- Load once --------------------
league_data = {}
for lg in leagues:
    res_df, fx_df = load_league(lg)
    league_data[lg] = (res_df, fx_df)

# -------------------- Highlights (GG / Over 2.5 / 1st-half) --------------------

# -------------------- Highlights (GG / Over 2.5 / 1st-half) --------------------
highlight_rows = []  # (ko, title, lg, pills_html, score)
for lg in leagues:
    res_df, fx_df = league_data.get(lg, (None, None))
    if res_df is None or fx_df is None or fx_df.empty:
        continue

    view = fx_df.copy()
    round_id = pick_current_round(view)
    if time_window == "All in round" and round_id is not None and "round_number" in view.columns:
        view = view[view["round_number"] == round_id].copy()
    if time_window in {"Today", "Tomorrow", "Weekend (Fri–Mon)", "All"}:
        m = date_mask(view, time_window if time_window != "All" else "__ALL__", now)
        view = view[m] if m.any() else view.iloc[0:0]

    if view.empty:
        continue

    for _, r in view.iterrows():
        home = str(r.get("home_team", "")).strip()
        away = str(r.get("away_team", "")).strip()
        ko = r.get("kickoff_dt")
        tlist = compute_trends(res_df, home, away)
        tags = set(t[2] for t in tlist)  # available tags
        # Focus tags of interest
        wanted = {"gg","over25","firsthalf"}
        present = tags.intersection(wanted)
        score = len(present)  # how many of the 3 are present?
        if score >= 2:
            pills = []
            if "gg" in present: pills.append('<span class="pill pill-goals">GG</span>')
            if "over25" in present: pills.append('<span class="pill pill-over">O2.5</span>')
            if "firsthalf" in present: pills.append('<span class="pill pill-fh">1st Half</span>')
            # badge showing count
            pills.append(f'<span class="pill"> {score}/3 </span>')
            highlight_rows.append((ko, f"{home} vs {away}", lg, " ".join(pills), score))

if highlight_rows:
    # Sort: (1) by score desc (3/3 first), (2) by kickoff time
    highlight_rows.sort(key=lambda x: (-x[4], pd.NaT if x[0] is None else x[0]))
    st.subheader("✨ Highlights — GG / O2.5 / 1st‑Half (≥2/3)")
    for ko, title, lg, pills, score in highlight_rows[:24]:
        ko_txt = format_ko(ko)
        st.markdown(f"- **{title}** — {ko_txt} · _{lg}_ {pills}", unsafe_allow_html=True)
    st.markdown("---")

# -------------------- Render Fixtures (inline cards) --------------------
for lg in leagues:
    res_df, fx_df = league_data.get(lg, (None, None))
    if res_df is None or fx_df is None or fx_df.empty:
        st.info(f"{lg}: No fixtures available.")
        continue

    view = fx_df.copy()
    round_id = pick_current_round(view)
    if time_window == "All in round" and round_id is not None and "round_number" in view.columns:
        view = view[view["round_number"] == round_id].copy()
    if time_window in {"Today", "Tomorrow", "Weekend (Fri–Mon)", "All"}:
        m = date_mask(view, time_window if time_window != "All" else "__ALL__", now)
        view = view[m] if m.any() else view.iloc[0:0]

    if view.empty:
        st.warning(f"{lg}: No fixtures match the current filters.")
        continue

    view = view.sort_values(by=["kickoff_dt", "home_team", "away_team"], ascending=[True, True, True], kind="mergesort")
    if len(view) > max_fixtures:
        view = view.head(max_fixtures)

    hdr = f"{lg} — {'Gameweek ' + str(round_id) if round_id is not None else 'Fixtures'}"
    st.markdown(f"<div class='league-header'>{hdr}</div>", unsafe_allow_html=True)

    for _, r in view.iterrows():
        home = str(r.get("home_team", "")).strip()
        away = str(r.get("away_team", "")).strip()
        ko = r.get("kickoff_dt")
        title = f"{home} vs {away}"
        sub = f"{format_ko(ko)}"

        trends = compute_trends(res_df, home, away)
        bullets = [t[1] for t in trends[:max_trends]]
        if not bullets:
            bullets = ["No strong trends (need ≥4 H2H in last 3 seasons with data)."]

        st.markdown(
            f"<div class='fixture-card'><div class='fixture-title'>{title}</div>"
            f"<div class='fixture-sub'>{sub}</div>"
            f"<ul class='trend-bullets'>{''.join([f'<li>{b}</li>' for b in bullets])}</ul></div>",
            unsafe_allow_html=True,
        )

    if show_debug:
        with st.expander(f"🔎 {lg} debug"):
            st.write("Results columns:", list(res_df.columns))
            st.dataframe(res_df.head(3))
            st.write("Fixtures columns:", list(fx_df.columns))
            st.dataframe(fx_df.head(3))
