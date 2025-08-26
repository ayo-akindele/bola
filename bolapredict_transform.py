"""
bolapredict_transform.py
=======================

This module provides a simple command‑line utility for updating the
trend flags in a Bolapredict data set.  The existing Bolapredict
pipeline computes trend counts for positive outcomes — such as
`GG` (both teams scored), `Over 2.5 goals` and `Over 9.5 corners`
— over the last five matches.  However, the inverse outcomes (no goal
for at least one team, under 2.5 goals and under 9.5 corners) can
also be important indicators.  Since the negative outcome occurs
whenever the positive outcome does not, we do not need to recalculate
historical match statistics; we can simply infer the counts and
trends by taking the inverse.

The script defined here expects a CSV file containing at least the
following columns for each fixture:

    * ``GG_count_last5``
    * ``O25_count_last5`` (count of matches with total goals ≥3 in the last 5)
    * ``O95corn_count_last5`` (count of matches with total corners ≥10 in the last 5)

It will compute the complementary counts and trend flags:

    * ``NG_count_last5`` – number of matches where at least one team failed to score
    * ``U25_count_last5`` – number of matches with total goals < 3
    * ``U95corn_count_last5`` – number of matches with total corners < 10
    * Boolean trend flags for each of the six outcomes listed above

Thresholds for what constitutes a “trend” can be customised.  By
default, a positive trend is flagged if the positive outcome occurred
in at least three of the last five matches.  Under‑trends are simply
the inverse — i.e. an under trend exists when an over trend does not.

Usage::

    python bolapredict_transform.py input.csv output.csv

If the second argument is omitted the script will print the updated
data to stdout in CSV format.  Thresholds can be customised by
editing the ``DEFAULT_THRESHOLDS`` mapping near the bottom of this file.
"""

from __future__ import annotations

import sys
import pandas as pd
from typing import Dict, Any


# -----------------------------------------------------------------------------
# Configuration
#
# Define the minimum number of occurrences in the last five matches for a
# positive outcome to be considered a trend.  If ``GG_count_last5 >= 3`` then
# ``GG_trend`` will be ``True`` and ``NG_trend`` will be ``False``.  If
# ``GG_count_last5 < 3`` then ``NG_trend`` will be ``True``.  Adjust these
# thresholds if your analysis uses a different cutoff (e.g. four out of five).
# -----------------------------------------------------------------------------
DEFAULT_THRESHOLDS: Dict[str, int] = {
    "GG": 3,
    "O25": 3,
    "O95corn": 3,
}


def update_trends(df: pd.DataFrame, thresholds: Dict[str, int]) -> pd.DataFrame:
    """Add inverse counts and trend flags to the Bolapredict DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        A Bolapredict results DataFrame.  Must contain the columns
        ``GG_count_last5``, ``O25_count_last5`` and ``O95corn_count_last5``.
    thresholds : dict
        A mapping specifying the minimum number of positive outcomes required
        for a trend flag to be ``True``.  Keys correspond to the prefixes of
        the count columns (``GG``, ``O25``, ``O95corn``).

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with the following additional columns:

        * ``NG_count_last5``, ``U25_count_last5``, ``U95corn_count_last5``
        * ``GG_trend``, ``NG_trend``, ``O25_trend``, ``U25_trend``,
          ``O95corn_trend``, ``U95corn_trend``
    """
    df = df.copy()

    # Ensure required columns are present
    required_cols = ["GG_count_last5", "O25_count_last5", "O95corn_count_last5"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns in input DataFrame: {', '.join(missing)}"
        )

    # Compute inverse counts — since there are always 5 matches in the window
    df["NG_count_last5"] = 5 - df["GG_count_last5"]
    df["U25_count_last5"] = 5 - df["O25_count_last5"]
    df["U95corn_count_last5"] = 5 - df["O95corn_count_last5"]

    # Positive trend flags
    df["GG_trend"] = df["GG_count_last5"] >= thresholds.get("GG", 3)
    df["O25_trend"] = df["O25_count_last5"] >= thresholds.get("O25", 3)
    df["O95corn_trend"] = df["O95corn_count_last5"] >= thresholds.get("O95corn", 3)

    # Inverse trend flags — simply the logical complement of the over/positive trend
    df["NG_trend"] = ~df["GG_trend"]
    df["U25_trend"] = ~df["O25_trend"]
    df["U95corn_trend"] = ~df["O95corn_trend"]

    return df


def main(argv: list[str]) -> None:
    """Entry point for the command‑line interface.

    Reads a CSV file, updates trend columns, and writes the result to
    another CSV file or stdout.
    """
    if not argv or len(argv) < 1:
        print(
            "Usage: python bolapredict_transform.py <input.csv> [<output.csv>]",
            file=sys.stderr,
        )
        sys.exit(1)

    input_path = argv[0]
    output_path = argv[1] if len(argv) > 1 else None

    # Load data
    df = pd.read_csv(input_path)

    # Update trends using default thresholds; customize if needed
    updated_df = update_trends(df, DEFAULT_THRESHOLDS)

    # Write output
    if output_path:
        updated_df.to_csv(output_path, index=False)
    else:
        # Print to stdout
        updated_df.to_csv(sys.stdout, index=False)


if __name__ == "__main__":
    main(sys.argv[1:])