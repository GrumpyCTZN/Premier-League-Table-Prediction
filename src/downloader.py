"""
src/downloader.py
=================
Fetches Premier League (E0) and Championship (E1) CSV data
from football-data.co.uk for seasons 2015/16 to 2024/25.

Refactored from pl_prediction_v2.py — logic preserved exactly.
"""

from __future__ import annotations

import io
from typing import Optional

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PL_SEASONS: list[str] = [
    "1516", "1617", "1718", "1819", "1920",
    "2021", "2122", "2223", "2324", "2425",
]

CHAMP_FALLBACK_MAP: dict[str, str] = {
    "1516": "1415",
    "1617": "1516",
    "1718": "1617",
    "1819": "1718",
    "1920": "1819",
    "2021": "1920",
    "2122": "2021",
    "2223": "2122",
    "2324": "2223",
    "2425": "2324",
}

BASE_URL  = "https://www.football-data.co.uk/mmz4281/{season}/E0.csv"
CHAMP_URL = "https://www.football-data.co.uk/mmz4281/{season}/E1.csv"

KEEP_COLS: list[str] = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_season(season_code: str) -> str:
    """
    Convert a season code like '2324' to a readable label '2023/24'.

    Parameters
    ----------
    season_code : str
        Four-character season code, e.g. ``'2324'``.

    Returns
    -------
    str
        Human-readable label, e.g. ``'2023/24'``.
    """
    if len(season_code) != 4:
        return season_code
    start   = season_code[:2]
    end     = season_code[2:]
    century = "20" if int(start) < 50 else "19"
    return f"{century}{start}/{end}"

# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _fetch_csv(url: str, season_label: str, league: str) -> Optional[pd.DataFrame]:
    """Download one CSV; return None on failure."""
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(
            io.StringIO(resp.text),
            usecols=lambda c: c in KEEP_COLS,
            on_bad_lines="skip",
        )
        df["Season"] = season_label
        df["League"] = league
        return df
    except Exception as exc:
        print(f"    ✗  {url}  — {exc}")
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_all_data(pl_seasons: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download PL (E0) data for every season in pl_seasons.
    Also download the Championship (E1) season that immediately preceded
    each PL season, for use as a promoted-team fallback.

    Parameters
    ----------
    pl_seasons : list[str]
        List of PL season codes to download, e.g. ``["2324", "2425"]``.

    Returns
    -------
    pl_df : pd.DataFrame
        All PL match rows, with ``Season`` and ``League`` columns added.
    champ_df : pd.DataFrame
        All Championship match rows, with ``Season``, ``League``, and
        ``PL_Season`` columns added.
    """
    pl_frames, champ_frames = [], []

    for season in pl_seasons:
        print(f"  PL   {season}...", end=" ")
        df = _fetch_csv(BASE_URL.format(season=season), season, "PL")
        if df is not None:
            pl_frames.append(df)
            print(f"✓  ({len(df)} matches)")

        champ_season = CHAMP_FALLBACK_MAP[season]
        print(f"  Champ {champ_season} (fallback for {season})...", end=" ")
        df = _fetch_csv(CHAMP_URL.format(season=champ_season), champ_season, "Championship")
        if df is not None:
            champ_frames.append(df)
            df["PL_Season"] = season          # tag which PL season this fallback serves
            print(f"✓  ({len(df)} matches)")

    pl_df    = pd.concat(pl_frames,    ignore_index=True) if pl_frames    else pd.DataFrame()
    champ_df = pd.concat(champ_frames, ignore_index=True) if champ_frames else pd.DataFrame()
    return pl_df, champ_df