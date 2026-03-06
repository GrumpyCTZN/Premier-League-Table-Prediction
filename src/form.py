"""
src/form.py
===========
Tracks each team's last-5-match form and computes a dampened
multiplier for use in Head-to-Head simulations (Page 3).

New module — not present in pl_prediction_v2.py.
"""

from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_POINTS: dict[str, int] = {"W": 3, "D": 1, "L": 0}
_MAX_POINTS: int = 15          # 5 wins × 3 pts
_WINDOW: int = 5


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_form_multiplier(results: list[str]) -> float:
    """
    Convert a team's last-5-match results into a dampened goal multiplier.

    Scoring
    -------
    * W = 3 pts,  D = 1 pt,  L = 0 pts  →  max possible = 15

    Formula
    -------
    ::

        raw        = sum(points) / 15        # 0.0 – 1.0
        nudge      = (raw - 0.5) * 0.2       # max ±0.10
        multiplier = 1.0 + nudge             # range: 0.90 – 1.10

    A perfect 5W run gives +10%; a 5L run gives −10%.

    Parameters
    ----------
    results : list[str]
        Up to 5 most recent results, each ``"W"``, ``"D"``, or ``"L"``.
        Fewer than 5 entries are handled gracefully (treated as partial
        evidence — proportional to games played).

    Returns
    -------
    float
        Multiplier in the range ``[0.90, 1.10]``.
    """
    if not results:
        return 1.0

    total_pts = sum(_POINTS.get(r, 0) for r in results)
    # Scale against the max possible for however many games we have
    max_pts   = len(results) * 3
    raw       = total_pts / max_pts if max_pts > 0 else 0.5
    nudge     = (raw - 0.5) * 0.2
    return round(1.0 + nudge, 6)


def seed_form_window(
    pl_df: pd.DataFrame,
    team: str,
    season_before_target: str,
) -> list[str]:
    """
    Return a team's last-5 results from the season immediately before the
    target season, so the live simulation starts with real momentum.

    Only matches from ``season_before_target`` are considered.  Results are
    sorted by ``Date`` ascending and the final 5 are taken.

    Parameters
    ----------
    pl_df : pd.DataFrame
        Cleaned PL match data (all seasons), with a ``Season`` column.
    team : str
        Canonical team name.
    season_before_target : str
        Season code of the season immediately before the target, e.g.
        ``"2324"`` when predicting ``"2425"``.

    Returns
    -------
    list[str]
        List of up to 5 result strings, e.g. ``["W", "W", "D", "L", "W"]``,
        ordered oldest-first.  Returns an empty list if the team has no
        matches in that season (e.g. newly promoted).
    """
    season_df = pl_df[pl_df["Season"] == season_before_target].copy()

    if season_df.empty:
        return []

    # Ensure Date is parsed for sorting
    season_df["Date"] = pd.to_datetime(season_df["Date"], dayfirst=True, errors="coerce")
    season_df = season_df.dropna(subset=["Date"]).sort_values("Date")

    results: list[str] = []

    for _, row in season_df.iterrows():
        if row["HomeTeam"] == team:
            if row["FTHG"] > row["FTAG"]:
                results.append("W")
            elif row["FTHG"] < row["FTAG"]:
                results.append("L")
            else:
                results.append("D")
        elif row["AwayTeam"] == team:
            if row["FTAG"] > row["FTHG"]:
                results.append("W")
            elif row["FTAG"] < row["FTHG"]:
                results.append("L")
            else:
                results.append("D")

    return results[-_WINDOW:]