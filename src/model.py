"""
src/model.py
============
Model A — Poisson GLM using historical match data only.

GLM formula:  goals ~ home + C(team) + C(opponent)
Train on seasons 1–9, validate on season 10 (2024/25).
No betting data used under any circumstance.

Refactored from pl_prediction_v2.py — logic preserved exactly.
"""

from __future__ import annotations

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from src.feature_engineering import get_h2h_adjustment, CHAMP_DISCOUNT


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def build_poisson_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape match data to one row per team per match (Dixon-Coles format).

    Each match produces two rows:
    * Home team row — ``home=1``, goals = ``FTHG``
    * Away team row — ``home=0``, goals = ``FTAG``

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned PL match data.

    Returns
    -------
    pd.DataFrame
        Columns: ``team``, ``opponent``, ``goals``, ``home``.
    """
    home_rows = df[["HomeTeam", "AwayTeam", "FTHG"]].copy()
    home_rows.columns = ["team", "opponent", "goals"]
    home_rows["home"] = 1

    away_rows = df[["AwayTeam", "HomeTeam", "FTAG"]].copy()
    away_rows.columns = ["team", "opponent", "goals"]
    away_rows["home"] = 0

    return pd.concat([home_rows, away_rows], ignore_index=True)


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------

def fit_poisson_model(poisson_df: pd.DataFrame):
    """
    Fit a Poisson GLM on the reshaped match dataset.

    Formula::

        goals ~ home + C(team) + C(opponent)

    * ``C(team)``     → attack coefficient per team
    * ``C(opponent)`` → defense coefficient per team
    * ``home``        → universal home advantage scalar

    Parameters
    ----------
    poisson_df : pd.DataFrame
        Output of :func:`build_poisson_dataset`.

    Returns
    -------
    statsmodels GLM result object
        Fitted model ready for :func:`predict_base_goals`.
    """
    model = smf.glm(
        formula="goals ~ home + C(team) + C(opponent)",
        data=poisson_df,
        family=sm.families.Poisson(),
    ).fit(disp=False)
    print(f"    Converged : {model.converged}")
    print(f"    Pseudo R² : {1 - model.llf / model.llnull:.4f}")
    return model


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_base_goals(
    model,
    home_team: str,
    away_team: str,
) -> tuple[float, float]:
    """
    Get the Poisson GLM's raw expected goals for a single fixture.

    Parameters
    ----------
    model : statsmodels GLM result
        Fitted model from :func:`fit_poisson_model`.
    home_team : str
        Canonical home team name.
    away_team : str
        Canonical away team name.

    Returns
    -------
    tuple[float, float]
        ``(mu_home, mu_away)``
    """
    home_pred = pd.DataFrame({"team": [home_team], "opponent": [away_team], "home": [1]})
    away_pred = pd.DataFrame({"team": [away_team], "opponent": [home_team], "home": [0]})
    mu_home = model.predict(home_pred).values[0]
    mu_away = model.predict(away_pred).values[0]
    return mu_home, mu_away


# ---------------------------------------------------------------------------
# Promoted team injection
# ---------------------------------------------------------------------------

def inject_promoted_teams(
    model,
    fixtures: pd.DataFrame,
    champ_fallback: dict,
    target_pl_season: str,
    train_df: pd.DataFrame,
) -> set[str]:
    """
    Identify teams in the fixture list that the Poisson model has never seen.

    Prints a summary of each promoted team and the Championship-derived
    strength values that will be used in their place, or flags if no
    Championship data is available (league-average fallback will apply).

    Parameters
    ----------
    model : statsmodels GLM result
        Fitted Poisson model (used only to determine known teams via
        training data).
    fixtures : pd.DataFrame
        Target-season fixture list with ``HomeTeam`` and ``AwayTeam`` columns.
    champ_fallback : dict
        Output of :func:`feature_engineering.build_championship_fallback`.
    target_pl_season : str
        Season code for the season being predicted, e.g. ``"2425"``.
    train_df : pd.DataFrame
        The training split of PL data used to fit the model.

    Returns
    -------
    set[str]
        Set of team names that are promoted (unseen by the GLM).
    """
    pl_teams_in_training = set(train_df["HomeTeam"]) | set(train_df["AwayTeam"])
    fixture_teams = set(fixtures["HomeTeam"]) | set(fixtures["AwayTeam"])
    promoted = fixture_teams - pl_teams_in_training

    if promoted:
        print(f"\n  Promoted teams detected (no PL history): {promoted}")
        for team in promoted:
            key = (team, target_pl_season)
            if key in champ_fallback:
                s = champ_fallback[key]
                print(
                    f"    {team:25s}  home_atk={s['home_attack']:.2f}  "
                    f"away_atk={s['away_attack']:.2f}  "
                    f"home_def={s['home_defense']:.2f}  "
                    f"away_def={s['away_defense']:.2f}  "
                    f"[Championship × {CHAMP_DISCOUNT}]"
                )
            else:
                print(f"    {team:25s}  — no Championship data found, using league average")
    else:
        print("  No newly promoted teams detected.")

    return promoted


# ---------------------------------------------------------------------------
# Pre-compute expected goals for all fixtures
# ---------------------------------------------------------------------------

def precompute_expected_goals(
    fixtures: pd.DataFrame,
    model,
    h2h_table: pd.DataFrame,
    champ_fallback: dict,
    promoted_teams: set,
    target_pl_season: str,
    league_avg_home: float,
    league_avg_away: float,
) -> pd.DataFrame:
    """
    For every fixture, compute ``mu_home`` and ``mu_away`` including H2H
    blending.  For promoted teams, uses Championship-derived strength
    ratios instead of the GLM.

    Adds four columns to the returned DataFrame:

    * ``mu_home_base`` — raw model expected goals (home), before H2H blend
    * ``mu_away_base`` — raw model expected goals (away), before H2H blend
    * ``mu_home``      — H2H-adjusted expected goals (home)
    * ``mu_away``      — H2H-adjusted expected goals (away)

    Parameters
    ----------
    fixtures : pd.DataFrame
        Fixture list with at least ``HomeTeam`` and ``AwayTeam`` columns.
    model : statsmodels GLM result
        Fitted Poisson model from :func:`fit_poisson_model`.
    h2h_table : pd.DataFrame
        Output of :func:`feature_engineering.build_h2h_table`.
    champ_fallback : dict
        Output of :func:`feature_engineering.build_championship_fallback`.
    promoted_teams : set[str]
        Output of :func:`inject_promoted_teams`.
    target_pl_season : str
        Season code for the season being predicted.
    league_avg_home : float
        Average home goals per match in the PL training set.
    league_avg_away : float
        Average away goals per match in the PL training set.

    Returns
    -------
    pd.DataFrame
        Copy of ``fixtures`` with the four ``mu_*`` columns appended.
    """
    mu_home_base_list, mu_away_base_list = [], []
    mu_home_adj_list,  mu_away_adj_list  = [], []

    for _, row in fixtures.iterrows():
        ht, at = row["HomeTeam"], row["AwayTeam"]

        ht_promoted = ht in promoted_teams
        at_promoted = at in promoted_teams

        if not ht_promoted and not at_promoted:
            try:
                mu_h, mu_a = predict_base_goals(model, ht, at)
            except Exception:
                mu_h, mu_a = league_avg_home, league_avg_away
        else:
            def get_str(team: str, pl_season: str) -> dict:
                """
                Return Championship strength dict with a promoted team cap.
                Attack is capped at 0.85 of league average to reflect the
                step-up in quality from Championship to Premier League.
                Defense is floored at 1.15 to reflect vulnerability.
                """
                PROMOTED_ATTACK_CAP    = 0.85
                PROMOTED_DEFENSE_FLOOR = 1.15

                key = (team, pl_season)
                if key in champ_fallback:
                    s = champ_fallback[key]
                    return {
                        "home_attack":  min(s["home_attack"],  PROMOTED_ATTACK_CAP),
                        "home_defense": max(s["home_defense"], PROMOTED_DEFENSE_FLOOR),
                        "away_attack":  min(s["away_attack"],  PROMOTED_ATTACK_CAP),
                        "away_defense": max(s["away_defense"], PROMOTED_DEFENSE_FLOOR),
                    }
                # No Championship data at all — use weak defaults
                return {
                    "home_attack":  0.75,
                    "home_defense": 1.20,
                    "away_attack":  0.70,
                    "away_defense": 1.25,
                }

            ht_str = get_str(ht, target_pl_season)
            at_str = get_str(at, target_pl_season)

            mu_h = league_avg_home * ht_str["home_attack"]  * at_str["away_defense"]
            mu_a = league_avg_away * at_str["away_attack"]  * ht_str["home_defense"]

        mu_home_base_list.append(mu_h)
        mu_away_base_list.append(mu_a)

        mu_h_adj, mu_a_adj = get_h2h_adjustment(h2h_table, ht, at, mu_h, mu_a)
        mu_home_adj_list.append(mu_h_adj)
        mu_away_adj_list.append(mu_a_adj)

    fixtures = fixtures.copy()
    fixtures["mu_home_base"] = mu_home_base_list
    fixtures["mu_away_base"] = mu_away_base_list
    fixtures["mu_home"]      = mu_home_adj_list
    fixtures["mu_away"]      = mu_away_adj_list
    return fixtures