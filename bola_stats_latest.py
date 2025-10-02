
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, time as dtime
import numpy as np  # Needed for vectorized operations when mapping corner counts to the correct team

# ------------------------------------------------------------------
# DATA SOURCES
# ------------------------------------------------------------------
# Replace these with your GitHub raw CSVs if needed.
HISTORICAL_URL = "https://docs.google.com/spreadsheets/d/1oZJlXF6tpLLaEDNfduHzYFvLKDw7rnyzZY17CQNl1so/gviz/tq?tqx=out:csv&gid=0"
FIXTURES_URL = "https://docs.google.com/spreadsheets/d/1oZJlXF6tpLLaEDNfduHzYFvLKDw7rnyzZY17CQNl1so/gviz/tq?tqx=out:csv&gid=1005360909"

LOCAL_TZ = "Africa/Lagos"

st.set_page_config(page_title="BolaStats", layout="centered")
st.title("📊 BolaStats")
st.markdown("<h4 style='margin-bottom:0; font-weight:bold;'>⚡ Quick Stats That Matter</h4>", unsafe_allow_html=True)

# ------------------------------------------------------------------
# LOADERS & HELPERS
# ------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data():
    try:
        results_df = pd.read_csv(HISTORICAL_URL)
        fixtures_df = pd.read_csv(FIXTURES_URL)
        return results_df, fixtures_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def _build_kickoff_dt(fixtures_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a tz-aware kickoff_dt from the fixtures' date/time field.
    Expectation: Column B 'Date' includes both date and time for all leagues (e.g. 'Fri 04 Oct 20:00' or '15/08/2025 16:30').
    If your header isn't exactly 'date', we try common variants.
    """
    df = fixtures_df.copy()
    # Try common column names for the combined date+time
    dt_col = next((c for c in ["date", "datetime", "fixture_date", "kickoff"] if c in df.columns), None)
    if dt_col is None:
        df["kickoff_dt"] = pd.NaT
        return df

    parsed = pd.to_datetime(df[dt_col].astype(str).str.strip(), errors="coerce", dayfirst=True, infer_datetime_format=True)

    # Localize to Africa/Lagos (if naive), else convert
    try:
        parsed = parsed.dt.tz_localize(LOCAL_TZ)
    except TypeError:
        parsed = parsed.dt.tz_convert(LOCAL_TZ)

    df["kickoff_dt"] = parsed
    return df

def _next_weekday(dt, target_weekday: int):
    """Return dt advanced to the next target weekday (0=Mon,...,6=Sun). If dt is already that weekday, keep dt."""
    days_ahead = (target_weekday - dt.weekday()) % 7
    return dt + timedelta(days=days_ahead)

def _weekend_window(now_ts):
    """Weekend window: Fri 00:00 through Mon 23:59:59 (next Fri from 'now')."""
    # Anchor to date in LOCAL_TZ
    now = now_ts.tz_convert(LOCAL_TZ)
    # Start = next Friday 00:00
    fri = _next_weekday(now, 4)  # 4=Fri
    start = pd.Timestamp.combine(fri.date(), dtime(0, 0)).tz_localize(LOCAL_TZ)
    # End = Monday 23:59:59 following that Friday
    mon = fri + timedelta(days=3)  # Fri->Sat(1)->Sun(2)->Mon(3)
    end = pd.Timestamp.combine(mon.date(), dtime(23, 59, 59)).tz_localize(LOCAL_TZ)
    return start, end

# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
results_df, fixtures_df = load_data()

if results_df is None or fixtures_df is None:
    st.warning("Unable to fetch data. Please check your CSV links or permissions.")
    st.stop()

results_df = _normalize_columns(results_df)
fixtures_df = _normalize_columns(fixtures_df)

# Parse kickoff and base date
fixtures_df = _build_kickoff_dt(fixtures_df)
if "date" in fixtures_df.columns:
    fixtures_df["date_parsed"] = pd.to_datetime(fixtures_df["date"], errors="coerce", dayfirst=True, infer_datetime_format=True)
results_df["match_date"] = pd.to_datetime(results_df.get("match_date"), errors="coerce")

# --- Determine current round by upcoming date ---
now = pd.Timestamp.now(tz=LOCAL_TZ)
tmp = fixtures_df.copy()
tmp["date_only"] = fixtures_df["kickoff_dt"].dt.date if "kickoff_dt" in fixtures_df.columns else fixtures_df["date_parsed"].dt.date
round_dates = tmp.groupby("round_number")["date_only"].max().sort_index()
current_round = None
if len(round_dates):
    upcoming = [r for r in round_dates.index if pd.to_datetime(round_dates.loc[r]) >= pd.to_datetime(now.date())]
    current_round = upcoming[0] if upcoming else round_dates.index.max()

if current_round is None:
    st.warning("No rounds found in fixtures.")
    st.stop()

gw_fixtures = fixtures_df[fixtures_df["round_number"] == current_round].copy()

# ----------------- USER FILTERS (TIME & LEAGUE) -----------------
time_filter = st.radio(
    "Show fixtures for:",
    ["Today", "Tomorrow", "Weekend (Fri–Mon)", "All upcoming in round"],
    horizontal=True,
    index=3
)

# League filter (if present)
league_col = next((c for c in ["league", "competition", "tournament"] if c in gw_fixtures.columns), None)
if league_col and gw_fixtures[league_col].notna().any():
    leagues = ["All"] + sorted([x for x in gw_fixtures[league_col].dropna().astype(str).unique().tolist() if x.strip() != ""])
    picked = st.selectbox("Filter by league", leagues, index=0)
else:
    picked = "All"

df_view = gw_fixtures.copy()
if time_filter == "Today":
    start = pd.Timestamp(now.date(), tz=LOCAL_TZ)
    end = start + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    mask = (df_view["kickoff_dt"].notna()) & (df_view["kickoff_dt"] >= start) & (df_view["kickoff_dt"] <= end)
    df_view = df_view[mask]
elif time_filter == "Tomorrow":
    start = pd.Timestamp(now.date(), tz=LOCAL_TZ) + pd.Timedelta(days=1)
    end = start + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    mask = (df_view["kickoff_dt"].notna()) & (df_view["kickoff_dt"] >= start) & (df_view["kickoff_dt"] <= end)
    df_view = df_view[mask]
elif time_filter == "Weekend (Fri–Mon)":
    w_start, w_end = _weekend_window(now)
    mask = (df_view["kickoff_dt"].notna()) & (df_view["kickoff_dt"] >= w_start) & (df_view["kickoff_dt"] <= w_end)
    df_view = df_view[mask]
# else: All upcoming in round -> no change

if picked != "All" and league_col:
    df_view = df_view[df_view[league_col].astype(str) == picked]

# ----------------- ORDER BY KICK-OFF TIME (chronological) -----------------
sort_cols = []
if "kickoff_dt" in df_view.columns:
    sort_cols.append("kickoff_dt")  # strict KO order
elif "date_parsed" in df_view.columns:
    sort_cols.append("date_parsed")
for extra in ["home_team", "away_team"]:
    if extra in df_view.columns and extra not in sort_cols:
        sort_cols.append(extra)
if sort_cols:
    df_view = df_view.sort_values(by=sort_cols, ascending=True, kind="mergesort")

st.subheader(f"📅 Gameweek {current_round} | Fixtures ({time_filter}) — sorted by kick-off time")

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
    "First-half goals": "⏱",
    "won": "🏆"
}

def normalize_boolean(col):
    return col.astype(str).str.lower().isin(["1", "true", "yes", "y"])

def generate_stats(home, away):
    three_seasons_ago = now.year - 3
    h2h_all = results_df[
        ((results_df['home_team'] == home) & (results_df['away_team'] == away)) |
        ((results_df['home_team'] == away) & (results_df['away_team'] == home))
    ]
    h2h = h2h_all[h2h_all["match_date"].dt.year >= three_seasons_ago].sort_values(by="match_date", ascending=False).head(5)

    if h2h.empty:
        return []

    total = len(h2h)
    trends = []

    def trend_check(condition, label):
        try:
            valid = condition.dropna()
            count = valid.sum()
            pct = count / len(valid)
            if len(valid) > 0 and pct >= 0.8:
                return (pct, f"{label} in {int(count)}/{len(valid)} games")
        except Exception:
            return None
        return None

    # Match winner logic
    wins = 0
    for _, row in h2h.iterrows():
        if row['home_team'] == home and row['home_score'] > row['away_score']:
            wins += 1
        elif row['away_team'] == home and row['away_score'] > row['home_score']:
            wins += 1
    if total > 0 and wins / total >= 0.8:
        trends.append((wins / total, f"{home} won {wins}/{total} recent meetings"))

    # GG/NG and O/U 2.5
    if {'home_score', 'away_score'}.issubset(h2h.columns):
        home_goals = pd.to_numeric(h2h['home_score'], errors='coerce')
        away_goals = pd.to_numeric(h2h['away_score'], errors='coerce')
        gg = (home_goals > 0) & (away_goals > 0)
        gg = gg.where(~(home_goals.isna() | away_goals.isna()), None)
        h2h['gg'] = gg
        h2h['ng'] = h2h['gg'].apply(lambda x: None if pd.isna(x) else not x)

        total_goals = home_goals + away_goals
        h2h['over_2_5'] = total_goals > 2.5
        h2h['under_2_5'] = total_goals <= 2.5
        h2h.loc[total_goals.isna(), ['over_2_5', 'under_2_5']] = None

    # Corners O/U 9.5
    if 'total_corners' in h2h.columns:
        corners_num = pd.to_numeric(h2h['total_corners'], errors='coerce')
        h2h['Corners_Over_9.5'] = corners_num > 9.5
        h2h['Corners_Under_9.5'] = corners_num <= 9.5
        h2h.loc[corners_num.isna(), ['Corners_Over_9.5', 'Corners_Under_9.5']] = None

    # Bookings O/U 3.5
    if {'home_yellow_cards', 'away_yellow_cards'}.issubset(h2h.columns):
        home_yc = pd.to_numeric(h2h['home_yellow_cards'], errors='coerce')
        away_yc = pd.to_numeric(h2h['away_yellow_cards'], errors='coerce')
        total_bookings = home_yc + away_yc
        h2h['Bookings_Over_3.5'] = total_bookings > 3.5
        h2h['Bookings_Under_3.5'] = total_bookings <= 3.5
        h2h.loc[total_bookings.isna(), ['Bookings_Over_3.5', 'Bookings_Under_3.5']] = None

    # First-half goal > 0
    if {'first_half_home', 'first_half_away'}.issubset(h2h.columns):
        h2h['First_Half_Goal'] = (pd.to_numeric(h2h['first_half_home'], errors='coerce') +
                                  pd.to_numeric(h2h['first_half_away'], errors='coerce')) > 0

    market_labels = {
        'gg': "Both teams scored (GG)",
        'ng': "Both teams failed to score (NG)",
        'over_2_5': "Over 2.5 goals",
        'under_2_5': "Under 2.5 goals",
        'Corners_Over_9.5': "Over 9.5 corners",
        'Corners_Under_9.5': "Under 9.5 corners",
        'Bookings_Over_3.5': "Over 3.5 bookings",
        'Bookings_Under_3.5': "Under 3.5 bookings",
        'First_Half_Goal': "First-half goals"
    }

    for col, label in market_labels.items():
        if col in h2h.columns:
            result = trend_check(h2h[col], label)
            if result:
                trends.append(result)

    # Corner dominance trends with correct team mapping
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

            res_home = (home_more.dropna().mean() if home_more.dropna().size else 0.0)
            res_away = (away_more.dropna().mean() if away_more.dropna().size else 0.0)
            if res_home >= 0.80:
                trends.append((res_home, f"{home} more corners than {away} in {int(res_home*home_more.dropna().size)}/{home_more.dropna().size} games"))
            if res_away >= 0.80:
                trends.append((res_away, f"{away} more corners than {home} in {int(res_away*away_more.dropna().size)}/{away_more.dropna().size} games"))
            break

    top_trends = sorted(trends, key=lambda x: x[0], reverse=True)[:3]
    return top_trends

top_summary_pool = []

# ----------------- FIXTURE LIST (by kick-off time) -----------------
if df_view.empty:
    st.info("No fixtures match the selected time window/league.")
else:
    for _, row in df_view.iterrows():
        home = row['home_team']
        away = row['away_team']

        # Header with formatted KO time
        if 'kickoff_dt' in df_view.columns and pd.notna(row.get("kickoff_dt")):
            ko_str = row["kickoff_dt"].strftime("%a %d %b · %H:%M")
            header = f"{ko_str} — {home} vs {away}"
        elif pd.notna(row.get("date_parsed")):
            date_str = pd.to_datetime(row["date_parsed"]).strftime("%a %d %b")
            header = f"{date_str} — {home} vs {away}"
        else:
            header = f"{home} vs {away}"

        with st.expander(header, expanded=True):
            fixture_stats = generate_stats(home, away)
            if fixture_stats:
                for pct, text in fixture_stats:
                    icon = "•"
                    for key, symbol in emoji_map.items():
                        if key in text:
                            icon = symbol
                            break
                    st.markdown(f"{icon} {text}")
                    top_summary_pool.append((pct, f"{home} vs {away} → {text}"))
            else:
                st.info("No strong trends to recommend for this game.")

# ----------------- TOP PICKS (90%+) -----------------
if top_summary_pool:
    st.markdown("---")
    st.subheader("🔥 Top Picks Summary (≥90%)")
    top_summary_pool.sort(reverse=True)
    shown = 0
    for pct, item in top_summary_pool:
        if pct >= 0.9:
            st.markdown(f"✅ {item}")
            shown += 1
        if shown >= 3:
            break

# ----------------- MAX TRENDS (100%) -----------------
# Exclude "First-half goals" from the qualifying trends.
st.markdown("---")
st.subheader("💪 Max Trends (100%) — excluding First‑half goals")
any_max = False
for _, row in df_view.iterrows():
    home = row['home_team']
    away = row['away_team']
    trends = generate_stats(home, away)
    max_trends = [(p, t) for (p, t) in trends if p >= 0.9999 and "First-half goals" not in t]
    if max_trends:
        any_max = True
        if 'kickoff_dt' in df_view.columns and pd.notna(row.get("kickoff_dt")):
            ko_str = row["kickoff_dt"].strftime("%a %d %b · %H:%M")
            st.markdown(f"**{home} vs {away}** · {ko_str}")
        else:
            st.markdown(f"**{home} vs {away}**")
        for _, text in max_trends:
            st.markdown(f"✅ {text}")
if not any_max:
    st.info("No fixtures hit a 100% trend (excluding First‑half goals) for the selected view.")
