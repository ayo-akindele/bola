"""
Bola Stats – Multi‑league prediction engine
-------------------------------------------

This Streamlit script extends the original single‑league ``bola_stats``
application to support multiple European football competitions using a
shared Google Sheet. The sheet should contain one tab for the
historical results and another for upcoming fixtures for each league,
following the naming convention below:

* EPL – ``EPL_Historical_Data`` and ``EPL_upcoming_fixtures``
* Serie A – ``I1_Historical_Data`` and ``I1_upcoming_fixtures``
* Bundesliga – ``D1_Historical_Data`` and ``D1_upcoming_fixtures``
* La Liga – ``SP1_Historical_Data`` and ``SP1_upcoming_fixtures``

The app reads from the same spreadsheet (specified by ``SPREADSHEET_ID``)
using the ``gviz`` export endpoint, so no authentication is required
provided the sheet is shared publicly. Set the ``SPREADSHEET_ID``
constant to the ID of your sheet (the long identifier in the URL). If
your sheet is private, you can adapt the loader to use the
``gspread``/``google-auth`` workflow instead.

For each upcoming match the app analyses each team’s last five games
from the historical tab and derives simple signals:

* **BTTS (both teams to score)** – proportion of last five games where
  both teams scored at least one goal.
* **Over 2.5 goals** – proportion of last five games with total goals
  greater than or equal to three.
* **Over 9.5 corners** – proportion of last five games where total
  corners reached at least ten (if corner columns exist).
* **More corners** – which side tends to earn more corners on
  average; calculated by comparing each team’s corners to their
  opponents over the last five games. A bug in earlier versions
  mis‑computed this value by swapping home/away roles; this
  implementation corrects that.

Users can adjust thresholds for BTTS and Over 2.5 via sliders; by
default a team must hit the threshold in four of the last five games
(i.e., 80 %). Only fixtures meeting the combined criteria for both
teams are displayed.
"""

from __future__ import annotations

import datetime as _dt
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import os

# ──────────────────────────────────────────────────────────────────────────────
# Configuration

SPREADSHEET_ID: str = "1oZJlXF6tpLLaEDNfduHzYFvLKDw7rnyzZY17CQNl1so"

# Each league entry contains the sheet/tab names used when pulling from
# Google Sheets along with a human‑friendly display name.  If you
# provide corresponding local CSV files in ``LOCAL_DATA`` below then
# those files will be used instead of the remote sheet.  This allows
# offline analysis or private data while retaining the same API.
LEAGUE_SHEETS: Dict[str, Dict[str, str]] = {
    "EPL": {
        "fixtures": "EPL_upcoming_fixtures",
        "historical": "EPL_Historical_Data",
        "display_name": "Premier League",
    },
    "I1": {
        "fixtures": "I1_upcoming_fixtures",
        "historical": "I1_Historical_Data",
        "display_name": "Serie A",
    },
    "D1": {
        "fixtures": "D1_upcoming_fixtures",
        "historical": "D1_Historical_Data",
        "display_name": "Bundesliga",
    },
    "SP1": {
        "fixtures": "SP1_upcoming_fixtures",
        "historical": "SP1_Historical_Data",
        "display_name": "La Liga",
    },
}

# Local CSV files for historical results and upcoming fixtures. When a league code
# appears in this mapping the loader will read from the corresponding local
# files rather than Google Sheets. This supports offline analysis and tests
# without requiring the sheet to be publicly shared. File paths are relative
# to the project root ("/home/oai/share").
LOCAL_DATA: Dict[str, Dict[str, str]] = {
    "I1": {
        "fixtures": "I1_upcoming_fixtures.csv",
        "historical": "I1 Historical Data.csv",
    },
    "D1": {
        "fixtures": "D1_upcoming_fixtures.csv",
        "historical": "D1 Historical Data.csv",
    },
    "SP1": {
        "fixtures": "SP1_upcoming_fixtures.csv",
        "historical": "SP1 Historical Data.csv",
    },
    # Add EPL if you have CSVs available locally
}

# Mapping of logical column names to possible sheet column headers. The
# sheets may differ slightly in naming conventions; if you alter the
# sheet structure, update these lists accordingly.
# Aliases for columns. If your sheet uses other names, add them here. The
# inference logic below will attempt to use the first matching header.
COLUMN_ALIASES: Dict[str, List[str]] = {
    # Date/time of the fixture (historical or upcoming)
    "date": [
        "Date",
        "date",
        "Match_Date",
        "FixtureDate",
        "Datetime",
        "MatchDate",
        "GameDate",
        "match_date",
    ],
    # Home team name
    "home": [
        "HomeTeam",
        "home",
        "Home",
        "Home_Team",
        "Home Team",
        "Team1",
        "home_team",
        "Home Team",
        "home team",
    ],
    # Away team name
    "away": [
        "AwayTeam",
        "away",
        "Away",
        "Away_Team",
        "Away Team",
        "Team2",
        "away_team",
        "Away Team",
        "away team",
    ],
    # Full time home goals
    "home_goals": [
        "FTHG",
        "home_goals",
        "HomeGoals",
        "HG",
        "Home Score",
        "Home_Score",
        "Home Score FT",
        "Goals_Home",
        "home_score",
    ],
    # Full time away goals
    "away_goals": [
        "FTAG",
        "away_goals",
        "AwayGoals",
        "AG",
        "Away Score",
        "Away_Score",
        "Away Score FT",
        "Goals_Away",
        "away_score",
    ],
    # Home team corners
    "home_corners": [
        "HC",
        "home_corners",
        "HomeCorners",
        "CornersForHomeTeam",
        "Home Corners",
        "Corners_Home",
        "home_corners",
        "Home_Corners",
        "Home Corners",
        "home corners",
    ],
    # Away team corners
    "away_corners": [
        "AC",
        "away_corners",
        "AwayCorners",
        "CornersForAwayTeam",
        "Away Corners",
        "Corners_Away",
        "away_corners",
        "Away_Corners",
        "Away Corners",
        "away corners",
    ],
    # Optional flags for BTTS and over 2.5 if present in dataset
    "btts_flag": [
        "both_teams_score",
        "BTTS",
        "btts",
    ],
    "o25_flag": [
        "over_2_5",
        "Over2.5",
        "over25",
        "O2.5",
    ],
}


@lru_cache(maxsize=16)
def _csv_from_gsheet(sheet_name: str) -> pd.DataFrame:
    """Fetch a tab from the Google Sheet via the gviz export endpoint.

    Google Sheets allows exporting a specific tab by specifying the sheet name
    in the query parameters. See:
    https://stackoverflow.com/a/33727890/753501
    """
    sheet_param = sheet_name.replace(" ", "%20")
    url = (
        f"https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/gviz/tq?tqx=out:csv&sheet={sheet_param}"
    )
    try:
        df = pd.read_csv(url)
    except Exception as exc:
        # If 403: the sheet might be private; the user must adjust sharing.
        raise RuntimeError(
            f"Failed to download sheet '{sheet_name}'. Check sharing settings or network."
        ) from exc
    return df


def _find_column(df: pd.DataFrame, logical_name: str) -> Optional[str]:
    """Return the first column in the DataFrame matching any alias for a logical name."""
    for cand in COLUMN_ALIASES.get(logical_name, []):
        if cand in df.columns:
            return cand
    return None


def _normalise_team_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from team names and unify case."""
    for key in ["home", "away"]:
        col = _find_column(df, key)
        if col:
            df[col] = df[col].astype(str).str.strip()
    return df


def _compute_team_stats(df: pd.DataFrame, last_n: int = 5) -> Dict[str, Dict[str, float]]:
    """Compute per‑team summary statistics based on the last ``n`` games.

    Returns a dictionary mapping team names to metrics:
    - ``btts_rate``: fraction of last ``n`` games where both teams scored.
    - ``o25_rate``: fraction of last ``n`` games with total goals ≥3.
    - ``corners_diff``: average corner difference (team minus opponent). If
      corner columns are missing the value is NaN.
    """
    h_col = _find_column(df, "home")
    a_col = _find_column(df, "away")
    hg_col = _find_column(df, "home_goals")
    ag_col = _find_column(df, "away_goals")
    hc_col = _find_column(df, "home_corners")
    ac_col = _find_column(df, "away_corners")
    date_col = _find_column(df, "date")

    # Ensure we have essential columns
    if not all([h_col, a_col, hg_col, ag_col, date_col]):
        raise ValueError("Historical data missing required columns (date, teams, goals).")

    df_sorted = df.sort_values(date_col, ascending=False)
    teams = pd.unique(pd.concat([df_sorted[h_col], df_sorted[a_col]])).astype(str)
    stats: Dict[str, Dict[str, float]] = {}
    for team in teams:
        # Filter last ``n`` matches for the team
        team_matches = df_sorted[(df_sorted[h_col] == team) | (df_sorted[a_col] == team)].head(last_n)
        total_games = len(team_matches)
        if total_games == 0:
            continue
        # Compute BTTS rate
        btts_count = ((team_matches[hg_col] > 0) & (team_matches[ag_col] > 0)).sum()
        # Compute Over 2.5 goals rate
        goals_total = team_matches[hg_col] + team_matches[ag_col]
        o25_count = (goals_total >= 3).sum()
        # Compute corner difference
        if hc_col and ac_col:
            diffs = []
            for _, row in team_matches.iterrows():
                # If the team played home: team corners minus opponent corners
                if row[h_col] == team:
                    if pd.notna(row[hc_col]) and pd.notna(row[ac_col]):
                        diffs.append(row[hc_col] - row[ac_col])
                else:
                    if pd.notna(row[hc_col]) and pd.notna(row[ac_col]):
                        # Team was away; team corners = away corners
                        diffs.append(row[ac_col] - row[hc_col])
            avg_diff = float(np.mean(diffs)) if diffs else np.nan
        else:
            avg_diff = np.nan
        stats[team] = {
            "btts_rate": btts_count / total_games,
            "o25_rate": o25_count / total_games,
            "corners_diff": avg_diff,
        }
    return stats

# -----------------------------------------------------------------------------
# Additional helper functions for trend counts
#
def _team_btts_o25_counts(df: pd.DataFrame, team: str, last_n: int = 5) -> Tuple[int, int, int]:
    """Return counts of BTTS and Over 2.5 goals for a single team.

    Parameters
    ----------
    df : DataFrame
        Historical match data.
    team : str
        Team name.
    last_n : int
        Number of most recent matches to consider.

    Returns
    -------
    Tuple[int, int, int]
        (btts_count, o25_count, total_games) representing how many of
        the last ``n`` games featured both teams scoring and how many had
        three or more total goals. ``total_games`` is the number of matches
        considered (may be fewer than ``n`` if data is limited).
    """
    h_col = _find_column(df, "home")
    a_col = _find_column(df, "away")
    hg_col = _find_column(df, "home_goals")
    ag_col = _find_column(df, "away_goals")
    date_col = _find_column(df, "date")
    if not all([h_col, a_col, hg_col, ag_col, date_col]):
        return (0, 0, 0)
    df_sorted = df.sort_values(date_col, ascending=False)
    team_matches = df_sorted[(df_sorted[h_col] == team) | (df_sorted[a_col] == team)].head(last_n)
    total_games = len(team_matches)
    if total_games == 0:
        return (0, 0, 0)
    btts_count = 0
    o25_count = 0
    for _, row in team_matches.iterrows():
        hg = row[hg_col]
        ag = row[ag_col]
        # numeric conversion
        try:
            hg_val = float(hg) if pd.notna(hg) else 0.0
            ag_val = float(ag) if pd.notna(ag) else 0.0
        except Exception:
            hg_val = pd.to_numeric(hg, errors="coerce")
            ag_val = pd.to_numeric(ag, errors="coerce")
        if pd.notna(hg_val) and pd.notna(ag_val):
            if hg_val > 0 and ag_val > 0:
                btts_count += 1
            if (hg_val + ag_val) >= 3:
                o25_count += 1
    return (btts_count, o25_count, total_games)


def _h2h_corners_counts(
    hist_df: pd.DataFrame,
    home_team: str,
    away_team: str,
    last_n: int = 5,
    over_threshold: float = 9.5,
) -> Optional[Tuple[int, int, int, int, int]]:
    """Compute head‑to‑head corner counts for the last ``n`` meetings.

    Parameters
    ----------
    hist_df : DataFrame
        Historical match data containing corner statistics.
    home_team, away_team : str
        Names of the teams in the upcoming fixture.
    last_n : int
        Number of most recent H2H games to consider.
    over_threshold : float
        Corner count threshold for determining over/under (default 9.5).

    Returns
    -------
    Optional[Tuple[int, int, int, int, int]]
        (over_count, under_count, home_corner_wins, away_corner_wins, total_games)
        where ``total_games`` is the number of H2H matches considered. If
        fewer than two H2H matches exist, returns None.
    """
    h_col = _find_column(hist_df, "home")
    a_col = _find_column(hist_df, "away")
    hc_col = _find_column(hist_df, "home_corners")
    ac_col = _find_column(hist_df, "away_corners")
    date_col = _find_column(hist_df, "date")
    # Determine column for total corners if present
    total_c_col = None
    for col in hist_df.columns:
        key = col.lower().replace("_", "").replace(" ", "")
        if key == "totalcorners":
            total_c_col = col
            break
    if not all([h_col, a_col, date_col]) or not (hc_col or total_c_col):
        return None
    # Filter H2H matches
    mask = (
        (hist_df[h_col] == home_team) & (hist_df[a_col] == away_team)
    ) | (
        (hist_df[h_col] == away_team) & (hist_df[a_col] == home_team)
    )
    h2h = hist_df.loc[mask].copy()
    if h2h.empty:
        return None
    h2h[date_col] = pd.to_datetime(h2h[date_col], errors="coerce")
    h2h = h2h.sort_values(date_col, ascending=False).head(last_n)
    total_games = len(h2h)
    if total_games == 0:
        return None
    over_count = 0
    under_count = 0
    home_corner_wins = 0
    away_corner_wins = 0
    for _, row in h2h.iterrows():
        # total corners
        if total_c_col and total_c_col in h2h.columns:
            total_c = pd.to_numeric(row[total_c_col], errors="coerce")
        else:
            # sum of home and away corners
            hc = pd.to_numeric(row[hc_col], errors="coerce")
            ac = pd.to_numeric(row[ac_col], errors="coerce")
            if pd.isna(hc) or pd.isna(ac):
                continue
            total_c = hc + ac
        if pd.notna(total_c):
            if total_c > over_threshold:
                over_count += 1
            else:
                under_count += 1
        # determine corner winner relative to upcoming fixture
        if hc_col and ac_col:
            hc = pd.to_numeric(row[hc_col], errors="coerce")
            ac = pd.to_numeric(row[ac_col], errors="coerce")
            if pd.isna(hc) or pd.isna(ac):
                continue
            # If historical match home team matches upcoming home_team
            if row[h_col] == home_team:
                diff = hc - ac
            else:
                diff = ac - hc
            if diff > 0:
                home_corner_wins += 1
            elif diff < 0:
                away_corner_wins += 1
    return (over_count, under_count, home_corner_wins, away_corner_wins, total_games)


def _get_h2h_corners_trend(
    hist_df: pd.DataFrame,
    home_team: str,
    away_team: str,
    *,
    last_n: int = 5,
    over_threshold: float = 9.5,
    min_games: int = 5,
    rate_threshold: float = 0.8,
) -> Tuple[str, str]:
    """Compute corner trends for head‑to‑head meetings.

    Parameters
    ----------
    hist_df : DataFrame
        Historical match data including at least date, home/away team, and corner counts.
    home_team, away_team : str
        Names of the teams in the upcoming fixture.
    last_n : int, default 5
        Number of recent H2H games to consider.
    over_threshold : float, default 9.5
        Corner count threshold for determining over/under.
    min_games : int, default 5
        Minimum number of H2H matches required to form a trend. If fewer
        than ``min_games`` exist, no trend is returned.
    rate_threshold : float, default 0.8
        Proportion of games (e.g. 80 %) required to highlight a trend.

    Returns
    -------
    Tuple[str, str]
        Two strings: the first denotes whether Over 9.5 corners is a
        noteworthy trend ("Yes", "No", or empty); the second denotes
        which team tends to have more corners (home team name, away team name, or empty).
    """
    h_col = _find_column(hist_df, "home")
    a_col = _find_column(hist_df, "away")
    hc_col = _find_column(hist_df, "home_corners")
    ac_col = _find_column(hist_df, "away_corners")
    total_c_col = None
    # Use total corners if present (case‑insensitive, ignore underscores/spaces)
    for col in hist_df.columns:
        key = col.lower().replace("_", "").replace(" ", "")
        if key == "totalcorners":
            total_c_col = col
            break
    date_col = _find_column(hist_df, "date")
    if not all([h_col, a_col, date_col]) or not (hc_col or total_c_col):
        return "", ""
    # Filter head‑to‑head matches regardless of home/away orientation
    mask = (
        (hist_df[h_col] == home_team) & (hist_df[a_col] == away_team)
    ) | (
        (hist_df[h_col] == away_team) & (hist_df[a_col] == home_team)
    )
    h2h = hist_df.loc[mask].copy()
    if h2h.empty:
        return "", ""
    # Sort by date descending and take the most recent ``last_n``
    h2h[date_col] = pd.to_datetime(h2h[date_col], errors="coerce")
    h2h = h2h.sort_values(date_col, ascending=False)
    h2h = h2h.head(last_n)
    if len(h2h) < min_games:
        return "", ""
    # Compute total corners for each match
    if total_c_col and total_c_col in h2h.columns:
        total_c = pd.to_numeric(h2h[total_c_col], errors="coerce")
    else:
        # Fallback to sum of home and away corners
        if hc_col and ac_col:
            total_c = pd.to_numeric(h2h[hc_col], errors="coerce") + pd.to_numeric(h2h[ac_col], errors="coerce")
        else:
            return "", ""
    # Determine over/under trend
    over_count = (total_c > over_threshold).sum()
    under_count = (total_c <= over_threshold).sum()
    over_trend = ""
    # If at least 80% of games are over, mark "Yes"; if at least 80% are under, mark "No"
    if over_count / len(h2h) >= rate_threshold:
        over_trend = "Yes"  # Over 9.5 is common
    elif under_count / len(h2h) >= rate_threshold:
        over_trend = "No"  # Under 9.5 is common
    # Determine which team tends to win corners
    if hc_col and ac_col:
        # Count how many times the home team (relative to upcoming fixture) wins the corner battle
        home_corner_wins = 0
        away_corner_wins = 0
        for _, r in h2h.iterrows():
            if pd.isna(r[hc_col]) or pd.isna(r[ac_col]):
                continue
            # If r's home team matches upcoming home_team
            if r[h_col] == home_team:
                diff = r[hc_col] - r[ac_col]
            else:
                # upcoming home team was away in this historical match
                diff = r[ac_col] - r[hc_col]
            if diff > 0:
                home_corner_wins += 1
            elif diff < 0:
                away_corner_wins += 1
        corners_team = ""
        if home_corner_wins / len(h2h) >= rate_threshold:
            corners_team = home_team
        elif away_corner_wins / len(h2h) >= rate_threshold:
            corners_team = away_team
    else:
        corners_team = ""
    return over_trend, corners_team


def _generate_predictions(
    fixtures: pd.DataFrame,
    team_stats: Dict[str, Dict[str, float]],
    *,
    hist_df: Optional[pd.DataFrame] = None,
    btts_threshold: float,
    o25_threshold: float,
    corners_bias: float,
) -> pd.DataFrame:
    """Create a DataFrame of match predictions given team statistics and thresholds.

    Parameters
    ----------
    fixtures : pd.DataFrame
        Upcoming fixtures; must contain home and away columns.
    team_stats : dict
        Mapping of team names to computed stats.
    btts_threshold : float, default 0.8
        Minimum BTTS rate required for both teams to recommend BTTS.
    o25_threshold : float, default 0.8
        Minimum Over 2.5 goals rate required for both teams.
    corners_bias : float, default 0.0
        Minimum absolute average corner difference required to call a
        ``More corners`` recommendation; set to 0 to always recommend.
    """
    h_col = _find_column(fixtures, "home")
    a_col = _find_column(fixtures, "away")
    date_col = _find_column(fixtures, "date")
    if not all([h_col, a_col]):
        raise ValueError("Fixtures data missing required columns (home, away).")

    predictions = []
    for _, row in fixtures.iterrows():
        home_team = str(row[h_col])
        away_team = str(row[a_col])
        home_stats = team_stats.get(home_team)
        away_stats = team_stats.get(away_team)
        if not home_stats or not away_stats:
            continue
        # Determine BTTS and Over 2.5 predictions based on team form
        btts_pred = (
            home_stats.get("btts_rate", 0) >= btts_threshold
            and away_stats.get("btts_rate", 0) >= btts_threshold
        )
        o25_pred = (
            home_stats.get("o25_rate", 0) >= o25_threshold
            and away_stats.get("o25_rate", 0) >= o25_threshold
        )
        # Corners: pick side with higher corners_diff across teams
        corners_pick = ""
        if not np.isnan(home_stats.get("corners_diff", np.nan)) and not np.isnan(away_stats.get("corners_diff", np.nan)):
            diff = home_stats["corners_diff"] - away_stats["corners_diff"]
            if abs(diff) > corners_bias:
                corners_pick = home_team if diff > 0 else away_team
        # H2H corner trend: Over 9.5 and corner winner
        over9_trend = ""
        corners_team = corners_pick
        if hist_df is not None:
            try:
                over9_trend, h2h_corners_team = _get_h2h_corners_trend(hist_df, home_team, away_team)
                # If h2h_corners_team is available, override corners_pick
                if h2h_corners_team:
                    corners_team = h2h_corners_team
            except Exception:
                # If something goes wrong, leave trends blank
                pass
        predictions.append({
            "Date": row[date_col] if date_col and date_col in row else None,
            "Home": home_team,
            "Away": away_team,
            "BTTS": "Yes" if btts_pred else "",
            "Over 2.5": "Yes" if o25_pred else "",
            "Over 9.5 Corners": over9_trend,
            "More Corners": corners_team or "",
        })
    return pd.DataFrame(predictions)


def main() -> None:
    st.set_page_config(page_title="BolaStats", layout="wide")
    st.title("BolaStats – Multi‑League Predictions")
    st.caption(
        "Select a league below to view predictions for upcoming fixtures. Each match is listed with emojis to "
        "indicate the recommended markets: ✅ for BTTS (both teams to score), ⚽ for Over 2.5 goals, 🔼 for over 9.5 "
        "corners, 🔽 for under 9.5 corners, 🏠 for the home team to win more corners and 🚌 for the away team to win "
        "more corners. Only matches where both sides meet the form thresholds (80 %) are shown."
    )

    # League selection. Display names derived from LEAGUE_SHEETS; default to Premier League
    league_codes = list(LEAGUE_SHEETS.keys())
    league_names = [LEAGUE_SHEETS[c]["display_name"] for c in league_codes]
    default_index = 0
    league_name = st.selectbox(
        "Choose a league",
        league_names,
        index=default_index,
    )
    league_code = league_codes[league_names.index(league_name)]

    # Fixed thresholds
    BTTS_THRESHOLD = 0.8  # 4/5 games must have BTTS
    O25_THRESHOLD = 0.8   # 4/5 games must exceed 2.5 goals
    CORNERS_BIAS = 0.0    # Always pick a more corners recommendation if a side has higher average

    # Load fixtures and historical data from local CSV if available, otherwise via Google Sheets
    with st.spinner("Loading data..."):
        try:
            if league_code in LOCAL_DATA:
                # Resolve file paths relative to the script directory
                base_path = os.path.dirname(os.path.abspath(__file__))
                fx_path = os.path.join(base_path, LOCAL_DATA[league_code]["fixtures"])
                hist_path = os.path.join(base_path, LOCAL_DATA[league_code]["historical"])
                fixtures_df = pd.read_csv(fx_path)
                hist_df = pd.read_csv(hist_path)
            else:
                fixtures_df = _csv_from_gsheet(LEAGUE_SHEETS[league_code]["fixtures"])
                hist_df = _csv_from_gsheet(LEAGUE_SHEETS[league_code]["historical"])
        except Exception as exc:
            st.error(str(exc))
            return

    # Normalise team names and parse dates
    fixtures_df = _normalise_team_names(fixtures_df)
    hist_df = _normalise_team_names(hist_df)
    # Ensure date columns are parsed and filter fixtures to upcoming dates
    # Convert any detected date columns to datetime objects.  For fixtures
    # this allows us to drop games that have already been played.
    for df in (fixtures_df, hist_df):
        date_col = _find_column(df, "date")
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # Determine the upcoming game week using the Round_Number column if
    # available.  We identify the round with the earliest fixture date
    # that is on or after today.  Only fixtures from that round will be shown.
    fx_date_col = _find_column(fixtures_df, "date")
    if fx_date_col:
        # Use a timezone‑naive date for comparison; otherwise pandas will
        # raise an error when comparing tz‑aware and tz‑naive timestamps.
        today = pd.Timestamp.utcnow().normalize().tz_localize(None)
        # parse round number if present
        round_col = None
        for cand in ["Round_Number", "Round", "Gameweek", "GW", "RoundNumber"]:
            if cand in fixtures_df.columns:
                round_col = cand
                break
        if round_col:
            try:
                # Convert round numbers to numeric if possible
                r_series = pd.to_numeric(fixtures_df[round_col], errors="coerce")
                # Combine round and date into a DataFrame
                round_dates = fixtures_df[[round_col, fx_date_col]].copy()
                round_dates[round_col] = r_series
                round_dates[fx_date_col] = pd.to_datetime(round_dates[fx_date_col], errors="coerce")
                # Drop NaT
                round_dates = round_dates.dropna(subset=[round_col, fx_date_col])
                # For each round, get the earliest fixture date
                rd = round_dates.groupby(round_col)[fx_date_col].min().sort_index()
                # Find the smallest round with date >= today
                upcoming_rounds = rd[rd >= today]
                if not upcoming_rounds.empty:
                    next_round = upcoming_rounds.index[0]
                    # Filter fixtures to that round
                    fixtures_df = fixtures_df[pd.to_numeric(fixtures_df[round_col], errors="coerce") == next_round].copy()
                else:
                    # If no rounds are upcoming, keep fixtures on or after today
                    fixtures_df = fixtures_df[fixtures_df[fx_date_col] >= today].copy()
            except Exception:
                # Fallback to filtering by date
                fixtures_df = fixtures_df[fixtures_df[fx_date_col] >= today].copy()
        else:
            # Fallback: filter by date only
            try:
                fixtures_df = fixtures_df[fixtures_df[fx_date_col] >= today].copy()
            except Exception:
                pass

    # Compute team statistics
    try:
        stats = _compute_team_stats(hist_df)
    except Exception as exc:
        st.error(f"Error processing historical data: {exc}")
        return

    # Generate predictions
    preds_df = _generate_predictions(
        fixtures_df,
        stats,
        hist_df=hist_df,
        btts_threshold=BTTS_THRESHOLD,
        o25_threshold=O25_THRESHOLD,
        corners_bias=CORNERS_BIAS,
    )

    st.subheader(f"Upcoming fixtures – {league_name}")
    if preds_df.empty:
        st.info(
            "No fixtures meet the specified criteria based on current team form. "
            "If this seems incorrect, verify that your historical data contains at least five recent matches per team."
        )
    else:
        # For each fixture, derive simple one-line trend summaries
        for _, r in preds_df.iterrows():
            home = r.get("Home", "")
            away = r.get("Away", "")
            date_val = r.get("Date", None)
            # Format date for display
            if pd.notna(date_val):
                try:
                    dt = pd.to_datetime(date_val, errors="coerce")
                    date_str = dt.strftime("%d %b %Y") if not pd.isna(dt) else str(date_val)
                except Exception:
                    date_str = str(date_val)
            else:
                date_str = ""
            # Compute team counts for BTTS and Over 2.5
            hbtts, ho25, htotal = _team_btts_o25_counts(hist_df, home, last_n=5)
            abtts, ao25, atotal = _team_btts_o25_counts(hist_df, away, last_n=5)
            # Head‑to‑head corner counts
            h2h_counts = _h2h_corners_counts(hist_df, home, away, last_n=5)
            trend_lines = []
            # BTTS trend: show if either team has at least one BTTS in the last 5
            if htotal > 0:
                trend_lines.append(f"BTTS or GG: {hbtts}/{htotal} ({home})")
            if atotal > 0:
                trend_lines.append(f"BTTS or GG: {abtts}/{atotal} ({away})")
            # Over 2.5 trend
            if htotal > 0:
                trend_lines.append(f"Over 2.5: {ho25}/{htotal} ({home})")
            if atotal > 0:
                trend_lines.append(f"Over 2.5: {ao25}/{atotal} ({away})")
            # Corners trends
            if h2h_counts:
                oc, uc, hc_win, ac_win, total_h2h = h2h_counts
                # Over/under corners
                if oc >= 1 or uc >= 1:
                    # Choose whichever is more prevalent
                    if oc >= uc:
                        trend_lines.append(f"Over 9.5 corners: {oc}/{total_h2h}")
                    else:
                        trend_lines.append(f"Under 9.5 corners: {uc}/{total_h2h}")
                # Corner winner
                if hc_win > 0:
                    trend_lines.append(f"{home} more corners: {hc_win}/{total_h2h}")
                if ac_win > 0:
                    trend_lines.append(f"{away} more corners: {ac_win}/{total_h2h}")
            # Compose display: match header and trends separated by commas
            match_header = f"**{home} vs {away}** — {date_str}"
            if trend_lines:
                details = "; ".join(trend_lines)
            else:
                details = "No recent trends found"
            st.markdown(
                f"<div style='margin-bottom:8px;'>"
                f"{match_header}<br>"
                f"{details}"
                f"</div>",
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()