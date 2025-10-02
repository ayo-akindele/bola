
import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from pathlib import Path

APP_TITLE = "⚽ BolaPredict — Fixtures & Trends (v2)"
LOCAL_TZ = "Africa/Lagos"

st.set_page_config(page_title="BolaPredict", layout="centered")
st.title(APP_TITLE)
st.caption("Fixtures are sorted by actual kick-off time. If a league file has only dates (no times), we still sort correctly by date; leagues with times get true time-order. Strength Trends shows games with any 100% trend (excluding First-half goals).")

# ----------------------
# Data loading helpers
# ----------------------
def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def parse_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create tz-aware kickoff_dt from either a single 'date' string containing time or separate date/time columns."""
    df = df.copy()
    # Normalize potential columns
    candidates_full = [c for c in ["date", "datetime", "fixture_date", "kickoff"] if c in df.columns]
    time_candidates = [c for c in ["kickoff_time", "kick_off_time", "kickoff", "kick_off", "time", "ko"] if c in df.columns]
    date_candidates = [c for c in ["date", "match_date", "fixture_date"] if c in df.columns]

    kickoff = None

    if candidates_full:
        s = df[candidates_full[0]].astype(str).str.strip()
        kickoff = pd.to_datetime(s, errors="coerce", dayfirst=True, infer_datetime_format=True)
    elif date_candidates:
        d = pd.to_datetime(df[date_candidates[0]].astype(str).str.strip(), errors="coerce", dayfirst=True, infer_datetime_format=True)
        if time_candidates:
            t = df[time_candidates[0]].astype(str).str.strip()
            # Combine naive
            combo = pd.to_datetime(d.dt.strftime("%Y-%m-%d") + " " + t, errors="coerce")
            kickoff = combo
        else:
            kickoff = d
    else:
        df["kickoff_dt"] = pd.NaT
        return df

    # Localize/convert to Africa/Lagos
    try:
        kickoff = kickoff.dt.tz_localize(LOCAL_TZ)
    except TypeError:
        kickoff = kickoff.dt.tz_convert(LOCAL_TZ)

    df["kickoff_dt"] = kickoff
    return df

def pick_upcoming_slice(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Return fixtures for the next 'round':
       - If round_number exists, pick the next round by date.
       - Otherwise, pick fixtures within the next 8 days starting today.
    """
    df = fixtures.copy()
    now = pd.Timestamp.now(tz=LOCAL_TZ).normalize()

    # Ensure a date column for grouping logic
    if "kickoff_dt" in df.columns:
        df["date_only"] = df["kickoff_dt"].dt.date
    else:
        # try parsing date
        if "date" in df.columns:
            tmp = pd.to_datetime(df["date"], errors="coerce", dayfirst=True, infer_datetime_format=True)
        else:
            tmp = pd.to_datetime(pd.NaT)
        df["date_only"] = tmp.dt.date

    if "round_number" in df.columns and df["round_number"].notna().any():
        # choose round with max date >= today
        grp = df.groupby("round_number")["date_only"].max().dropna()
        upcoming = [r for r in grp.index if pd.to_datetime(grp.loc[r]) >= pd.to_datetime(now.date())]
        target_round = upcoming[0] if upcoming else (grp.index.max() if len(grp) else None)
        return df[df["round_number"] == target_round].copy() if target_round is not None else df.iloc[0:0].copy()
    else:
        # 8-day window around this weekend
        window_end = now + timedelta(days=8)
        mask = (df["kickoff_dt"].notna()) & (df["kickoff_dt"] >= now) & (df["kickoff_dt"] <= window_end)
        subset = df[mask].copy()
        if subset.empty:
            # fallback: future fixtures
            subset = df[df["kickoff_dt"] >= now].copy()
        return subset

# ----------------------
# Trend logic (H2H, last 5, threshold)
# ----------------------
def compute_h2h_trends(results_df: pd.DataFrame, home: str, away: str, today_ts: pd.Timestamp) -> list:
    """Return list of (pct, label) trends for last 5 H2H in the last 3 seasons, with >=80% threshold.
       Markets covered: GG/NG, O2.5/U2.5, Over/Under 9.5 corners, Over/Under 3.5 bookings, First-half goal>0, Corner dominance.
    """
    three_seasons_ago = today_ts.year - 3
    h2h_all = results_df[
        ((results_df['home_team'] == home) & (results_df['away_team'] == away)) |
        ((results_df['home_team'] == away) & (results_df['away_team'] == home))
    ].copy()

    h2h = h2h_all[h2h_all["match_date"].dt.year >= three_seasons_ago].sort_values(by="match_date", ascending=False).head(5)
    if h2h.empty:
        return []

    trends = []

    def trend_check(series_bool, label):
        try:
            valid = series_bool.dropna()
            if len(valid) == 0:
                return None
            count = valid.sum()
            pct = float(count) / float(len(valid))
            if pct >= 0.80:
                return (pct, f"{label} in {int(count)}/{len(valid)} games")
        except Exception:
            return None
        return None

    # Scores to build GG/NG
    if {'home_score', 'away_score'}.issubset(h2h.columns):
        h = pd.to_numeric(h2h['home_score'], errors='coerce')
        a = pd.to_numeric(h2h['away_score'], errors='coerce')
        gg = (h > 0) & (a > 0)
        gg = gg.where(~(h.isna() | a.isna()), None)
        ng = gg.apply(lambda x: None if pd.isna(x) else not x)
        res = trend_check(gg, "Both teams scored (GG)")
        if res: trends.append(res)
        res = trend_check(ng, "Both teams failed to score (NG)")
        if res: trends.append(res)

        total_goals = h + a
        over_2_5 = total_goals > 2.5
        under_2_5 = total_goals <= 2.5
        over_2_5 = over_2_5.where(~total_goals.isna(), None)
        under_2_5 = under_2_5.where(~total_goals.isna(), None)
        res = trend_check(over_2_5, "Over 2.5 goals")
        if res: trends.append(res)
        res = trend_check(under_2_5, "Under 2.5 goals")
        if res: trends.append(res)

    # Corners thresholds (total)
    for corners_col in ["total_corners", "corners_total", "totalcorner"]:
        if corners_col in h2h.columns:
            c = pd.to_numeric(h2h[corners_col], errors='coerce')
            over = (c > 9.5).where(~c.isna(), None)
            under = (c <= 9.5).where(~c.isna(), None)
            res = trend_check(over, "Over 9.5 corners")
            if res: trends.append(res)
            res = trend_check(under, "Under 9.5 corners")
            if res: trends.append(res)
            break

    # Bookings thresholds (yellow cards sum)
    home_yc_col = next((c for c in ["home_yellow_cards", "home_yellows", "yc_home"] if c in h2h.columns), None)
    away_yc_col = next((c for c in ["away_yellow_cards", "away_yellows", "yc_away"] if c in h2h.columns), None)
    if home_yc_col and away_yc_col:
        yc = pd.to_numeric(h2h[home_yc_col], errors='coerce') + pd.to_numeric(h2h[away_yc_col], errors='coerce')
        over = (yc > 3.5).where(~yc.isna(), None)
        under = (yc <= 3.5).where(~yc.isna(), None)
        res = trend_check(over, "Over 3.5 bookings")
        if res: trends.append(res)
        res = trend_check(under, "Under 3.5 bookings")
        if res: trends.append(res)

    # First-half goal > 0
    fh_home = next((c for c in ["first_half_home", "fh_home", "home_ht_goals"] if c in h2h.columns), None)
    fh_away = next((c for c in ["first_half_away", "fh_away", "away_ht_goals"] if c in h2h.columns), None)
    if fh_home and fh_away:
        fh = pd.to_numeric(h2h[fh_home], errors='coerce') + pd.to_numeric(h2h[fh_away], errors='coerce')
        fh_goal = (fh > 0).where(~fh.isna(), None)
        res = trend_check(fh_goal, "First-half goals")
        if res: trends.append(res)

    # Corner dominance (map to current fixture home/away logically)
    corner_pairs = [
        ('home_corners', 'away_corners'),
        ('home_corner', 'away_corner'),
        ('homecorner', 'awaycorner'),
        ('corners_home', 'corners_away')
    ]
    for h_col, a_col in corner_pairs:
        if {h_col, a_col}.issubset(h2h.columns):
            hc = pd.to_numeric(h2h[h_col], errors='coerce')
            ac = pd.to_numeric(h2h[a_col], errors='coerce')

            # Map corners to the side labels for this specific fixture
            home_team_corners = np.where(
                h2h['home_team'] == home, hc,
                np.where(h2h['away_team'] == home, ac, np.nan)
            )
            away_team_corners = np.where(
                h2h['home_team'] == away, hc,
                np.where(h2h['away_team'] == away, ac, np.nan)
            )

            home_team_corners = pd.to_numeric(home_team_corners, errors='coerce')
            away_team_corners = pd.to_numeric(away_team_corners, errors='coerce')

            mask_bad = (
                pd.isna(home_team_corners) |
                pd.isna(away_team_corners) |
                (home_team_corners == away_team_corners)
            )

            home_more = pd.Series(home_team_corners > away_team_corners).where(~mask_bad, None)
            away_more = pd.Series(away_team_corners > home_team_corners).where(~mask_bad, None)

            res = trend_check(home_more, f"{home} more corners than {away}")
            if res: trends.append(res)
            res = trend_check(away_more, f"{away} more corners than {home}")
            if res: trends.append(res)
            break

    # Sort strongest first and cap display to top 3
    trends = sorted(trends, key=lambda x: x[0], reverse=True)[:3]
    return trends

# ----------------------
# Inputs
# ----------------------
st.sidebar.header("Data Sources")

upload_mode = st.sidebar.radio("Fixtures source", ["Upload multiple CSVs", "Single URL"], index=0)

fixtures_all = []
if upload_mode == "Upload multiple CSVs":
    fx_files = st.sidebar.file_uploader("Upload one or more FIXTURES CSVs", type=["csv"], accept_multiple_files=True, key="fx_multi")
    if fx_files:
        for f in fx_files:
            try:
                df = pd.read_csv(f)
                df = normalize_cols(df)
                df = parse_datetime_columns(df)
                # League tagging
                if "league" not in df.columns or df["league"].isna().all():
                    league_name = Path(f.name).stem.split("_")[0]  # e.g., "EPL_upcoming_fixtures" -> "EPL"
                    df["league"] = league_name
                fixtures_all.append(df)
            except Exception as e:
                st.error(f"Error reading {f.name}: {e}")
else:
    fixtures_url = st.sidebar.text_input("Fixtures CSV URL (single)")
    if fixtures_url:
        try:
            df = pd.read_csv(fixtures_url)
            df = normalize_cols(df)
            df = parse_datetime_columns(df)
            if "league" not in df.columns or df["league"].isna().all():
                df["league"] = "League"
            fixtures_all.append(df)
        except Exception as e:
            st.error(f"Error reading fixtures URL: {e}")

if not fixtures_all:
    st.warning("Load at least one fixtures CSV to continue.")
    st.stop()

# Concatenate all fixtures
fixtures_df = pd.concat(fixtures_all, ignore_index=True)

# Minimal required columns
for must in ["home_team", "away_team"]:
    if must not in fixtures_df.columns:
        st.error(f"Missing required column in fixtures: '{must}'")
        st.stop()

# Optional results (for trends)
results_df = None
rs_mode = st.sidebar.radio("Historical results source (for trends)", ["Skip", "Upload CSV", "URL"], index=0)
if rs_mode == "Upload CSV":
    rs = st.sidebar.file_uploader("Upload HISTORICAL RESULTS CSV", type=["csv"], key="rs1")
    if rs is not None:
        results_df = normalize_cols(pd.read_csv(rs))
elif rs_mode == "URL":
    results_url  = st.sidebar.text_input("Historical Results CSV URL")
    if results_url:
        results_df = normalize_cols(pd.read_csv(results_url))

# Prepare results date if present
if results_df is not None and "match_date" in results_df.columns:
    results_df["match_date"] = pd.to_datetime(results_df["match_date"], errors="coerce", dayfirst=True, infer_datetime_format=True)

# Prepare fixtures
today_lagos = pd.Timestamp.now(tz=LOCAL_TZ)
upcoming = pick_upcoming_slice(fixtures_df)

# League filter
league_col = "league" if "league" in upcoming.columns else None
if league_col:
    leagues = ["All"] + sorted([x for x in upcoming[league_col].dropna().unique().tolist() if str(x).strip() != ""])
    selected = st.selectbox("Filter by league (optional)", leagues, index=0)
    if selected != "All":
        upcoming = upcoming[upcoming[league_col] == selected]

# Sort strictly by kickoff time, with stable tie-breakers
sort_cols = ["kickoff_dt"]
for c in ["date", "home_team", "away_team"]:
    if c in upcoming.columns and c not in sort_cols:
        sort_cols.append(c)
upcoming_sorted = upcoming.sort_values(by=sort_cols, ascending=True, kind="mergesort")

# Tabs for views
tab_fixtures, tab_strength = st.tabs(["📅 Fixtures (by kick-off time)", "💪 Strength Trends (100%)"])

with tab_fixtures:
    st.subheader("Upcoming Fixtures")
    for _, row in upcoming_sorted.iterrows():
        home = row.get("home_team", "")
        away = row.get("away_team", "")
        league = row.get("league", "")
        ko = row.get("kickoff_dt")
        ko_label = ko.strftime("%a %d %b · %H:%M") if pd.notna(ko) else ""

        header = f"{ko_label} — {home} vs {away}" if ko_label else f"{home} vs {away}"
        suffix = f"  ·  {league}" if str(league).strip() else ""
        with st.expander(header + suffix, expanded=True):
            st.markdown(f"- **Match**: {home} vs {away}")
            if ko_label:
                st.markdown(f"- **Kick-off**: {ko_label} (Africa/Lagos)")
            if league:
                st.markdown(f"- **League**: {league}")

            # Trend callouts
            if results_df is not None:
                trends = compute_h2h_trends(results_df, home, away, today_lagos)
                if trends:
                    st.markdown("**Top trends (last 5 H2H, ≥80%)**")
                    emoji_map = {
                        "Both teams scored": "⚽",
                        "Both teams failed to score": "🚫",
                        "Over 2.5 goals": "📈",
                        "Under 2.5 goals": "📉",
                        "Over 9.5 corners": "🏳️",
                        "Under 9.5 corners": "🔻",
                        "Over 3.5 bookings": "🟨",
                        "Under 3.5 bookings": "🟩",
                        "more corners": "🔺",
                        "First-half goals": "⏱"
                    }
                    for pct, text in trends:
                        icon = "•"
                        for key, symbol in emoji_map.items():
                            if key in text:
                                icon = symbol
                                break
                        st.markdown(f"{icon} {text}")
                else:
                    st.info("No strong H2H trends (≥80%) found.")
            else:
                st.info("Upload/enter historical results to see trend callouts.")

with tab_strength:
    st.subheader("100% Strength Trends (excluding First-half goals)")
    if results_df is None:
        st.info("Upload/enter historical results to evaluate Strength Trends.")
    else:
        strong_rows = []
        for _, row in upcoming_sorted.iterrows():
            home = row.get("home_team", "")
            away = row.get("away_team", "")
            league = row.get("league", "")
            trends = compute_h2h_trends(results_df, home, away, today_lagos)
            # Pick only trends with pct == 1.0 and NOT first-half goals
            t100 = [(pct, label) for (pct, label) in trends if (pct >= 0.9999) and ("First-half goals" not in label)]
            if t100:
                strong_rows.append((row.get("kickoff_dt"), home, away, league, t100))

        if not strong_rows:
            st.info("No matches hit a 100% trend (excluding First-half goals) in the selected window.")
        else:
            # Sort by KO just in case
            strong_rows.sort(key=lambda x: (pd.Timestamp.min if pd.isna(x[0]) else x[0]))
            for ko, home, away, league, tlist in strong_rows:
                ko_label = ko.strftime("%a %d %b · %H:%M") if pd.notna(ko) else ""
                title = f"{home} vs {away}"
                if league:
                    title += f"  ·  {league}"
                if ko_label:
                    title += f"  ·  {ko_label}"
                st.markdown(f"### {title}")
                for pct, label in tlist:
                    st.markdown(f"✅ {label}")
