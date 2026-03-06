"""
src/model_b.py
==============
Model B — Poisson GLM augmented with a betting-market implied probability
signal.

Academic framing:
    Tests whether betting market efficiency adds predictive signal beyond
    historical stats alone.  Grounded in Dixon & Coles (1997).

Formula:  goals ~ home + implied_prob_home + C(team) + C(opponent)

New module — not present in pl_prediction_v2.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


# ---------------------------------------------------------------------------
# Implied probability extraction
# ---------------------------------------------------------------------------

def extract_implied_probs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute fair implied probabilities from Bet365 odds, removing the
    bookmaker overround (vig).

    Raw implied probabilities
    -------------------------
    ::

        raw_h = 1 / B365H
        raw_d = 1 / B365D
        raw_a = 1 / B365A

    Overround removal (normalisation)
    ----------------------------------
    ::

        overround          = raw_h + raw_d + raw_a
        implied_prob_home  = raw_h / overround
        implied_prob_draw  = raw_d / overround
        implied_prob_away  = raw_a / overround

    Parameters
    ----------
    df : pd.DataFrame
        Match DataFrame containing ``B365H``, ``B365D``, and ``B365A``
        columns.  Rows missing any of these columns are left with ``NaN``
        in the output probability columns.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with three additional columns:
        ``implied_prob_home``, ``implied_prob_draw``, ``implied_prob_away``.
    """
    df = df.copy()
    raw_h = 1.0 / df["B365H"]
    raw_d = 1.0 / df["B365D"]
    raw_a = 1.0 / df["B365A"]
    overround = raw_h + raw_d + raw_a
    df["implied_prob_home"] = raw_h / overround
    df["implied_prob_draw"] = raw_d / overround
    df["implied_prob_away"] = raw_a / overround
    return df


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def build_model_b_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape match data to one row per team per match, adding
    ``implied_prob_home`` as a feature column.

    The implied probability is meaningful only for the home-team row;
    away-team rows receive ``0`` so the GLM coefficient captures the
    home-perspective market signal exclusively.

    Parameters
    ----------
    df : pd.DataFrame
        Match DataFrame that has already had implied probabilities computed
        by :func:`extract_implied_probs` (must contain ``implied_prob_home``).

    Returns
    -------
    pd.DataFrame
        Columns: ``team``, ``opponent``, ``goals``, ``home``,
        ``implied_prob_home``.
    """
    home_rows = df[["HomeTeam", "AwayTeam", "FTHG", "implied_prob_home"]].copy()
    home_rows.columns = ["team", "opponent", "goals", "implied_prob_home"]
    home_rows["home"] = 1

    away_rows = df[["AwayTeam", "HomeTeam", "FTAG"]].copy()
    away_rows.columns = ["team", "opponent", "goals"]
    away_rows["home"] = 0
    away_rows["implied_prob_home"] = 0.0   # not applicable for away-team rows

    return pd.concat([home_rows, away_rows], ignore_index=True)


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------

def fit_model_b(model_b_df: pd.DataFrame):
    """
    Fit a Poisson GLM with an additional betting-market signal feature.

    Formula::

        goals ~ home + implied_prob_home + C(team) + C(opponent)

    Same Poisson family and fitting approach as Model A.

    Parameters
    ----------
    model_b_df : pd.DataFrame
        Output of :func:`build_model_b_dataset`.

    Returns
    -------
    statsmodels GLM result object
        Fitted Model B ready for :func:`predict_base_goals_b`.
    """
    model = smf.glm(
        formula="goals ~ home + implied_prob_home + C(team) + C(opponent)",
        data=model_b_df,
        family=sm.families.Poisson(),
    ).fit(disp=False)
    print(f"    [Model B] Converged : {model.converged}")
    print(f"    [Model B] Pseudo R² : {1 - model.llf / model.llnull:.4f}")
    return model


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_base_goals_b(
    model_b,
    home_team: str,
    away_team: str,
    implied_prob_home: float,
) -> tuple[float, float]:
    """
    Get Model B's raw expected goals for a single fixture.

    Identical to :func:`model.predict_base_goals` but additionally passes
    ``implied_prob_home`` as an input feature.  The away-team prediction
    uses ``implied_prob_home=0`` (consistent with dataset construction).

    Parameters
    ----------
    model_b : statsmodels GLM result
        Fitted Model B from :func:`fit_model_b`.
    home_team : str
        Canonical home team name.
    away_team : str
        Canonical away team name.
    implied_prob_home : float
        Fair implied probability of a home win (overround-removed).

    Returns
    -------
    tuple[float, float]
        ``(mu_home, mu_away)``
    """
    home_pred = pd.DataFrame({
        "team":               [home_team],
        "opponent":           [away_team],
        "home":               [1],
        "implied_prob_home":  [implied_prob_home],
    })
    away_pred = pd.DataFrame({
        "team":               [away_team],
        "opponent":           [home_team],
        "home":               [0],
        "implied_prob_home":  [0.0],
    })
    mu_home = model_b.predict(home_pred).values[0]
    mu_away = model_b.predict(away_pred).values[0]
    return mu_home, mu_away