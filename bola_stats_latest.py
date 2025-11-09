"""
BolaPredict — Fixtures (min H2H threshold for trends)
-----------------------------------------------------
- MIN_H2H = 4 (set to 5 if you want stricter). Any metric needs at least MIN_H2H valid games.
- Added Ligue 1 (France) support using CSV files:
    • Results:    F1 Historical Data.csv
    • Fixtures:   F1_upcoming_fixtures.csv
- CSV‑only loader (no openpyxl dependency).
"""

import os
from datetime import datetime, timedelta, time as dtime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import streamlit as st

LOCAL_TZ = "Africa/Lagos"
MIN_H2H = 4  # <-- change to 5 for stricter requirement

LEAGUE_FILES: Dict[str, Dict[str, str]] = {
    "EPL": {"results": "EPL Historical Data.csv", "fixtures": "EPL_upcoming_fixtures.csv"},
    "La Liga": {"results": "SP1 Historical Data.csv", "fixtures": "SP1_upcoming_fixtures.csv"},
    "Serie A": {"results": "I1 Historical Data.csv", "fixtures": "I1_upcoming_fixtures.csv"},
    "Bundesliga": {"results": "D1 Historical Data.csv", "fixtures": "D1_upcoming_fixtures.csv"},
    # NEW — France Ligue 1 (code F1)
    "Ligue 1": {"results": "F1 Historical Data.csv", "fixtures": "F1_upcoming_fixtures.csv"},
}

st.set_page_config(page_title="BolaPredict — Fixtures", layout="centered")
st.title("📅 BolaPredict Fixtures")
st.caption(
    "⚡ Quick stats that matter. Trends in the most recent H2H meetings across EPL, La Liga, Bundesliga, Serie A & Ligue 1. "
    "Fixtures are listed strictly by kick‑off time (Africa/Lagos)."
)

# Chip styling
st.markdown(
    """
    <style>
    div[role="radiogroup"] > label {
        border: 1px solid rgba(239,68,68,0.45);
        padding: 10px 14px;
        border-radius: 10px;
        margin-right: 8px;
        margin-bottom: 8px;
        background: rgba(239,68,68,0.08);
        color: #ef4444;
        cursor: pointer;
        font-weight: 600;
    }
    div[role="radiogroup"] > label[data-checked="true"] {
        background: rgba(239,68,68,0.18);
        border-color: #ef4444;
        color: #ef4444;
        box-shadow: inset 0 0 0 1px #ef4444;
    }
    div[role="radiogroup"] svg { stroke: #ef4444 !important; fill: #ef4444 !important; }
    .league-header { margin-top: 18px; margin-bottom: 6px; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------- IO helpers --------------------

def _exists_nearby(filename: str) -> Optional[str]:
    for path in [filename, os.path.join(os.path.dirname(__file__), filename)]:
        if os.path.exists(path):
            return path
    return None


def _load_table_nearby(filename: str) -> pd.DataFrame:
    """Load CSV only (simpler, no Excel dependencies)."""
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

# (rest of the code remains the same as before)
