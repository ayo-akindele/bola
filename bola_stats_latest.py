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

# ──────────────────────────────────────────────────────────────────────────────
# Configuration

SPREADSHEET_ID: str = "1oZJlXF6tpLLaEDNfduHzYFvLKDw7rnyzZY17CQNl1so"

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
    ],
    # Away team corners
    "away_corners": [
        "AC",
        "away_corners",
        "AwayCorners",
        "CornersForAwayTeam",
        "Away Corners",
        "Corners_Away",
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
    st.set_page_config(page_title="BolaStats Multi‑League", layout="wide")
    st.title("BolaStats – Multi‑League Predictions")
    st.markdown(
        """
        The landing page defaults to the **Premier League**, showing quick stats based on each team's last five
        matches. You can switch to other leagues using the drop‑down below. Predictions are based on fixed
        thresholds (80 % BTTS and over 2.5 goals rates) and recommend matches that meet both sides’ criteria.
        """
    )

    # League selection. Default to Premier League (first entry).
    league_codes = list(LEAGUE_SHEETS.keys())
    league_display_names = [LEAGUE_SHEETS[c]["display_name"] for c in league_codes]
    default_index = 0  # ensure EPL appears first for landing page
    league_choice = st.selectbox(
        "Choose a league",
        league_display_names,
        index=default_index,
        help="Select which league's stats and predictions to display.",
    )
    league_code = league_codes[league_display_names.index(league_choice)]

    # Fixed thresholds; adjust these values in code if needed.
    BTTS_THRESHOLD = 0.8  # 4/5 games must have BTTS
    O25_THRESHOLD = 0.8   # 4/5 games must exceed 2.5 goals
    CORNERS_BIAS = 0.0    # Always pick a more corners recommendation if a side has higher average

    # Load data
    with st.spinner("Loading data..."):
        try:
            fixtures_df = _csv_from_gsheet(LEAGUE_SHEETS[league_code]["fixtures"])
            hist_df = _csv_from_gsheet(LEAGUE_SHEETS[league_code]["historical"])
        except Exception as exc:
            st.error(str(exc))
            return

    # Normalise team names
    fixtures_df = _normalise_team_names(fixtures_df)
    hist_df = _normalise_team_names(hist_df)

    # Compute team stats
    try:
        stats = _compute_team_stats(hist_df)
    except Exception as exc:
        st.error(f"Error processing historical data: {exc}")
        return

    # Generate predictions using fixed thresholds
    preds_df = _generate_predictions(
        fixtures_df,
        stats,
        hist_df=hist_df,
        btts_threshold=BTTS_THRESHOLD,
        o25_threshold=O25_THRESHOLD,
        corners_bias=CORNERS_BIAS,
    )
    if preds_df.empty:
        st.info(
            "No fixtures meet the specified criteria based on current team form. "
            "If this seems incorrect, verify that your historical data contains at least five recent matches per team."
        )
    else:
        st.dataframe(preds_df.reset_index(drop=True), use_container_width=True)


if __name__ == "__main__":
    main()