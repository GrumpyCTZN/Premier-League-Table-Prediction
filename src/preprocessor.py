"""
src/preprocessor.py
===================
Cleans raw match DataFrames and standardises team names.

Refactored from pl_prediction_v2.py — logic preserved exactly.
"""

from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RENAME_MAP: dict[str, str] = {
    "Manchester United":        "Man United",
    "Manchester City":          "Man City",
    "Tottenham Hotspur":        "Tottenham",
    "Spurs":                    "Tottenham",
    "Newcastle United":         "Newcastle",
    "Brighton and Hove Albion": "Brighton",
    "Brighton & Hove Albion":   "Brighton",
    "Wolverhampton Wanderers":  "Wolves",
    "West Bromwich Albion":     "West Brom",
    "Sheffield United":         "Sheffield Utd",
    "Queens Park Rangers":      "QPR",
    "Huddersfield Town":        "Huddersfield",
    "Stoke City":               "Stoke",
    "Swansea City":             "Swansea",
    "Cardiff City":             "Cardiff",
    "Norwich City":             "Norwich",
    "Nottingham Forest":        "Nott'm Forest",
    "Nottm Forest":             "Nott'm Forest",
    "Blackburn Rovers":         "Blackburn",
    "Bolton Wanderers":         "Bolton",
    "Derby County":             "Derby",
    "Preston North End":        "Preston",
    "Rotherham United":         "Rotherham",
    "Coventry City":            "Coventry",
    "Luton Town":               "Luton",
    "Birmingham City":          "Birmingham",
    "Sunderland AFC":           "Sunderland",
    "Middlesbrough FC":         "Middlesbrough",
    "Watford FC":               "Watford",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop incomplete rows, fix dtypes, and standardise team names.

    Steps applied (in order):
    1. Drop rows where any of ``FTHG``, ``FTAG``, ``HomeTeam``, or
       ``AwayTeam`` is null.
    2. Cast ``FTHG`` and ``FTAG`` to ``int``.
    3. Apply ``RENAME_MAP`` to both ``HomeTeam`` and ``AwayTeam`` columns
       to normalise variant spellings to a single canonical name.
    4. Reset the index.

    Parameters
    ----------
    df : pd.DataFrame
        Raw match DataFrame as returned by :func:`downloader.download_all_data`.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with standardised team names and reset index.
    """
    df = df.dropna(subset=["FTHG", "FTAG", "HomeTeam", "AwayTeam"]).copy()
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)
    df["HomeTeam"] = df["HomeTeam"].replace(RENAME_MAP)
    df["AwayTeam"]  = df["AwayTeam"].replace(RENAME_MAP)
    return df.reset_index(drop=True)