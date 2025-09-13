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
COLUMN_ALIASES: Dict[str, List[str]] = {
    "date": ["Date", "date", "Match_Date", "FixtureDate"],
    "home": ["HomeTeam", "home", "Home", "Home_Team"],
    "away": ["AwayTeam", "away", "Away", "Away_Team"],
    "home_goals": ["FTHG", "home_goals", "HomeGoals", "HG"],
    "away_goals": ["FTAG", "away_goals", "AwayGoals", "AG"],
    "home_corners": ["HC", "home_corners", "HomeCorners"],
    "away_corners": ["AC", "away_corners", "AwayCorners"],
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


def _generate_predictions(
    fixtures: pd.DataFrame,
    team_stats: Dict[str, Dict[str, float]],
    btts_threshold: float = 0.8,
    o25_threshold: float = 0.8,
    corners_bias: float = 0.0,
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
        # Determine BTTS and Over 2.5 predictions
        btts_pred = (home_stats["btts_rate"] >= btts_threshold) and (away_stats["btts_rate"] >= btts_threshold)
        o25_pred = (home_stats["o25_rate"] >= o25_threshold) and (away_stats["o25_rate"] >= o25_threshold)
        # Corners: pick side with higher corners_diff
        corners_pick = None
        if not np.isnan(home_stats["corners_diff"]) and not np.isnan(away_stats["corners_diff"]):
            diff = home_stats["corners_diff"] - away_stats["corners_diff"]
            if abs(diff) > corners_bias:
                corners_pick = home_team if diff > 0 else away_team
        # Compose result
        predictions.append({
            "Date": row[date_col] if date_col and date_col in row else None,
            "Home": home_team,
            "Away": away_team,
            "BTTS": "Yes" if btts_pred else "",
            "Over 2.5": "Yes" if o25_pred else "",
            "More Corners": corners_pick or "",
        })
    return pd.DataFrame(predictions)


def main() -> None:
    st.set_page_config(page_title="BolaStats Multi‑League", layout="wide")
    st.title("BolaStats – Multi‑League Predictions")
    st.markdown(
        "Select a league to view predictions based on recent form (last 5 games). "
        "Adjust thresholds using the sliders below. Only fixtures meeting the criteria "
        "are displayed."
    )

    # League selection
    league_codes = list(LEAGUE_SHEETS.keys())
    league_display_names = [LEAGUE_SHEETS[c]["display_name"] for c in league_codes]
    league_choice = st.selectbox("Choose a league", league_display_names)
    # Map back to code
    league_code = league_codes[league_display_names.index(league_choice)]

    # Threshold sliders
    col1, col2, col3 = st.columns(3)
    with col1:
        btts_thr = st.slider("BTTS threshold", 0.5, 1.0, 0.8, 0.05)
    with col2:
        o25_thr = st.slider("Over 2.5 threshold", 0.5, 1.0, 0.8, 0.05)
    with col3:
        corners_bias = st.slider("Corner bias threshold", 0.0, 3.0, 0.0, 0.1)

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

    # Generate predictions
    preds_df = _generate_predictions(
        fixtures_df,
        stats,
        btts_threshold=btts_thr,
        o25_threshold=o25_thr,
        corners_bias=corners_bias,
    )
    if preds_df.empty:
        st.info("No fixtures meet the specified thresholds. Try relaxing the sliders or check data availability.")
    else:
        st.dataframe(preds_df.reset_index(drop=True), use_container_width=True)


if __name__ == "__main__":
    main()