"""
src/feature_engineering.py
===========================
Head-to-head lookup, Championship fallback strengths, and team
strength ratio computation.

Refactored from pl_prediction_v2.py — existing logic preserved exactly.
compute_team_strengths() is new (not in v2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHAMP_DISCOUNT: float = 0.35
H2H_WEIGHT: float = 0.25


# ---------------------------------------------------------------------------
# Head-to-head
# ---------------------------------------------------------------------------

def build_h2h_table(pl_df: pd.DataFrame) -> pd.DataFrame:
    """
    For every ordered pair (HomeTeam, AwayTeam) that has ever met in the
    PL dataset, compute:

    * ``h2h_home_goals`` — avg goals the HOME team scored in that fixture
    * ``h2h_away_goals`` — avg goals the AWAY team scored in that fixture
    * ``h2h_matches``    — number of meetings

    Pairs with fewer than 2 meetings have their goal averages set to NaN
    so they are treated as "no H2H signal" by :func:`get_h2h_adjustment`.

    Parameters
    ----------
    pl_df : pd.DataFrame
        Cleaned PL match data (output of :func:`preprocessor.clean_data`).

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by ``(HomeTeam, AwayTeam)``.
    """
    h2h = (
        pl_df
        .groupby(["HomeTeam", "AwayTeam"])
        .agg(
            h2h_home_goals=("FTHG", "mean"),
            h2h_away_goals=("FTAG", "mean"),
            h2h_matches   =("FTHG", "count"),
        )
        .reset_index()
    )

    h2h.loc[h2h["h2h_matches"] < 2, ["h2h_home_goals", "h2h_away_goals"]] = np.nan
    return h2h.set_index(["HomeTeam", "AwayTeam"])


def get_h2h_adjustment(
    h2h_table: pd.DataFrame,
    home_team: str,
    away_team: str,
    mu_home: float,
    mu_away: float,
) -> tuple[float, float]:
    """
    Blend the model's base expected goals with the H2H historical averages.

    Formula::

        adjusted_mu = (1 - H2H_WEIGHT) * model_mu  +  H2H_WEIGHT * h2h_avg

    If no H2H record exists (pair not in table, or fewer than 2 meetings),
    ``model_mu`` is returned unchanged.

    Parameters
    ----------
    h2h_table : pd.DataFrame
        Output of :func:`build_h2h_table`.
    home_team : str
        Home team canonical name.
    away_team : str
        Away team canonical name.
    mu_home : float
        Model's base expected goals for the home team.
    mu_away : float
        Model's base expected goals for the away team.

    Returns
    -------
    tuple[float, float]
        ``(adj_mu_home, adj_mu_away)``
    """
    try:
        row = h2h_table.loc[(home_team, away_team)]
        if pd.isna(row["h2h_home_goals"]):
            return mu_home, mu_away
        adj_home = (1 - H2H_WEIGHT) * mu_home + H2H_WEIGHT * row["h2h_home_goals"]
        adj_away = (1 - H2H_WEIGHT) * mu_away + H2H_WEIGHT * row["h2h_away_goals"]
        return adj_home, adj_away
    except KeyError:
        return mu_home, mu_away


# ---------------------------------------------------------------------------
# Championship fallback
# ---------------------------------------------------------------------------

def build_championship_fallback(
    champ_df: pd.DataFrame,
    league_avg_home: float,
    league_avg_away: float,
) -> dict[tuple[str, str], dict[str, float]]:
    """
    For each team in the Championship data, compute season-level attack and
    defense averages, apply ``CHAMP_DISCOUNT`` (0.85), and store them keyed
    by ``(team_name, pl_season)``.

    Parameters
    ----------
    champ_df : pd.DataFrame
        Cleaned Championship match data with a ``PL_Season`` column
        indicating which PL season each Championship season is a fallback for.
    league_avg_home : float
        Average home goals per match in the PL training set.
    league_avg_away : float
        Average away goals per match in the PL training set.

    Returns
    -------
    dict
        ``{ (team, pl_season): {"home_attack": x, "home_defense": x,
                                "away_attack": x, "away_defense": x} }``
    """
    fallback: dict[tuple[str, str], dict[str, float]] = {}

    for pl_season, group in champ_df.groupby("PL_Season"):
        home_stats = (
            group
            .groupby("HomeTeam")
            .agg(
                h_goals_for    =("FTHG", "mean"),
                h_goals_against=("FTAG", "mean"),
            )
            .reset_index()
            .rename(columns={"HomeTeam": "Team"})
        )

        away_stats = (
            group
            .groupby("AwayTeam")
            .agg(
                a_goals_for    =("FTAG", "mean"),
                a_goals_against=("FTHG", "mean"),
            )
            .reset_index()
            .rename(columns={"AwayTeam": "Team"})
        )

        merged = home_stats.merge(away_stats, on="Team", how="outer")

        for _, row in merged.iterrows():
            team = row["Team"]
            fallback[(team, pl_season)] = {
                "home_attack":  (row["h_goals_for"]     * CHAMP_DISCOUNT) / league_avg_home,
                "home_defense": (row["h_goals_against"] * CHAMP_DISCOUNT) / league_avg_away,
                "away_attack":  (row["a_goals_for"]     * CHAMP_DISCOUNT) / league_avg_away,
                "away_defense": (row["a_goals_against"] * CHAMP_DISCOUNT) / league_avg_home,
            }

    return fallback


# ---------------------------------------------------------------------------
# Team strength ratios (NEW — not in v2)
# ---------------------------------------------------------------------------

def compute_team_strengths(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute normalised attack and defense strength ratios for every team
    across the full historical dataset.

    Four ratios are calculated per team:

    * ``home_attack_str``  = avg home goals scored   / league avg home goals
    * ``home_defense_str`` = avg home goals conceded / league avg away goals
    * ``away_attack_str``  = avg away goals scored   / league avg away goals
    * ``away_defense_str`` = avg away goals conceded / league avg home goals

    A ratio > 1.0 means above average; < 1.0 means below average.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned PL match data (full history, all seasons).

    Returns
    -------
    pd.DataFrame
        One row per team with columns:
        ``Team``, ``home_attack_str``, ``home_defense_str``,
        ``away_attack_str``, ``away_defense_str``.
    """
    league_avg_home = df["FTHG"].mean()
    league_avg_away = df["FTAG"].mean()

    home_stats = (
        df
        .groupby("HomeTeam")
        .agg(
            home_goals_scored  =("FTHG", "mean"),
            home_goals_conceded=("FTAG", "mean"),
        )
        .reset_index()
        .rename(columns={"HomeTeam": "Team"})
    )

    away_stats = (
        df
        .groupby("AwayTeam")
        .agg(
            away_goals_scored  =("FTAG", "mean"),
            away_goals_conceded=("FTHG", "mean"),
        )
        .reset_index()
        .rename(columns={"AwayTeam": "Team"})
    )

    merged = home_stats.merge(away_stats, on="Team", how="outer")

    merged["home_attack_str"]  = merged["home_goals_scored"]   / league_avg_home
    merged["home_defense_str"] = merged["home_goals_conceded"] / league_avg_away
    merged["away_attack_str"]  = merged["away_goals_scored"]   / league_avg_away
    merged["away_defense_str"] = merged["away_goals_conceded"] / league_avg_home

    return merged[
        ["Team", "home_attack_str", "home_defense_str",
         "away_attack_str", "away_defense_str"]
    ].reset_index(drop=True)