"""
src/evaluator.py
================
Evaluation metrics for comparing Model A and Model B against the
actual 2024/25 Premier League results.

New module — not present in pl_prediction_v2.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.metrics import log_loss

from src.model import predict_base_goals
from src.model_b import predict_base_goals_b


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------

def compute_mae(
    predicted_table: pd.DataFrame,
    actual_table: pd.DataFrame,
) -> float:
    """
    Mean Absolute Error of predicted final points vs actual final points.

    Parameters
    ----------
    predicted_table : pd.DataFrame
        Output of :func:`simulator.build_predicted_table`.
        Must contain columns ``Team`` and ``Avg Points``.
    actual_table : pd.DataFrame
        Actual season results.  Must contain columns ``Team`` and ``Points``.

    Returns
    -------
    float
        MAE across all teams present in both tables.
    """
    merged = predicted_table[["Team", "Avg Points"]].merge(
        actual_table[["Team", "Points"]],
        on="Team",
        how="inner",
    )
    mae = (merged["Avg Points"] - merged["Points"]).abs().mean()
    return round(float(mae), 4)


def compute_logloss(
    model,
    fixtures_with_odds: pd.DataFrame,
    model_type: str = "A",
) -> float:
    """
    Compute log-loss by comparing each fixture's predicted outcome
    probabilities against the actual full-time result (FTR).

    Outcome probabilities are computed by integrating the Poisson PMF
    over scorelines 0–9 goals each::

        P(H) = sum_{hg > ag}  Pois(hg|mu_h) × Pois(ag|mu_a)
        P(D) = sum_{hg == ag} Pois(hg|mu_h) × Pois(ag|mu_a)
        P(A) = sum_{hg < ag}  Pois(hg|mu_h) × Pois(ag|mu_a)

    Fixtures involving promoted teams not seen during GLM training are
    skipped gracefully — league averages are used as a fallback instead
    of crashing.

    Parameters
    ----------
    model : statsmodels GLM result
        Fitted Model A or Model B.
    fixtures_with_odds : pd.DataFrame
        Fixture DataFrame containing ``HomeTeam``, ``AwayTeam``, ``FTR``,
        and (for Model B) ``implied_prob_home``.
    model_type : str, optional
        ``"A"`` (default) or ``"B"`` — determines which prediction function
        is called.

    Returns
    -------
    float
        Sklearn log-loss (lower is better).
    """
    MAX_GOALS = 10

    # Derive league averages from model training data as fallback
    # for promoted teams not seen by the GLM
    try:
        league_avg_home = float(
            np.exp(model.params.get("Intercept", 0) + model.params.get("home", 0))
        )
        league_avg_away = float(np.exp(model.params.get("Intercept", 0)))
    except Exception:
        league_avg_home = 1.5
        league_avg_away = 1.2

    y_true, y_pred = [], []

    for _, row in fixtures_with_odds.iterrows():
        ht  = row["HomeTeam"]
        at  = row["AwayTeam"]
        ftr = row.get("FTR")

        if ftr not in ("H", "D", "A"):
            continue

        # Try model prediction — fall back to league averages for
        # promoted teams the GLM has never seen
        try:
            if model_type == "B":
                imp = row.get("implied_prob_home", 0.5)
                mu_h, mu_a = predict_base_goals_b(model, ht, at, float(imp))
            else:
                mu_h, mu_a = predict_base_goals(model, ht, at)
        except Exception:
            mu_h, mu_a = league_avg_home, league_avg_away

        # Integrate Poisson PMF over scoreline grid
        hg_range = np.arange(MAX_GOALS)
        ag_range = np.arange(MAX_GOALS)
        p_hg     = poisson.pmf(hg_range, mu_h)
        p_ag     = poisson.pmf(ag_range, mu_a)
        matrix   = np.outer(p_hg, p_ag)

        p_home_win = float(np.tril(matrix, -1).sum())
        p_draw     = float(np.trace(matrix))
        p_away_win = float(np.triu(matrix, 1).sum())

        total = p_home_win + p_draw + p_away_win
        if total == 0:
            continue
        probs = [p_home_win / total, p_draw / total, p_away_win / total]

        if ftr == "H":
            outcome = [1, 0, 0]
        elif ftr == "D":
            outcome = [0, 1, 0]
        else:
            outcome = [0, 0, 1]

        y_true.append(outcome)
        y_pred.append(probs)

    if not y_true:
        return float("nan")

    return round(float(log_loss(y_true, y_pred)), 6)

def compute_rank_accuracy(
    predicted_table: pd.DataFrame,
    actual_table: pd.DataFrame,
    tolerance: int = 2,
) -> float:
    """
    Percentage of teams whose predicted finishing position is within
    ±*tolerance* places of their actual finishing position.

    Parameters
    ----------
    predicted_table : pd.DataFrame
        Output of :func:`simulator.build_predicted_table`.
        Must contain ``Team``; position is inferred from row order
        (index is 1-based ``Pos``).
    actual_table : pd.DataFrame
        Actual final standings.  Must contain ``Team``; position is
        inferred from row order (1-based).
    tolerance : int, optional
        Allowed position error.  Default is 2.

    Returns
    -------
    float
        Accuracy as a percentage (0–100), rounded to 1 decimal place.
    """
    # Build position maps
    pred_pos = {
        row["Team"]: pos
        for pos, row in predicted_table.reset_index().iterrows()
    }
    actual_pos = {
        row["Team"]: pos + 1
        for pos, row in actual_table.reset_index().iterrows()
    }

    # Re-derive predicted positions as 1-based from sorted order
    predicted_sorted = (
        predicted_table
        .reset_index(drop=True)
        .reset_index()
        .rename(columns={"index": "pred_pos"})
    )
    predicted_sorted["pred_pos"] += 1

    correct = 0
    total   = 0

    for _, row in predicted_sorted.iterrows():
        team     = row["Team"]
        pred_p   = int(row["pred_pos"])
        actual_p = actual_pos.get(team)
        if actual_p is not None:
            total += 1
            if abs(pred_p - actual_p) <= tolerance:
                correct += 1

    if total == 0:
        return 0.0
    return round(100.0 * correct / total, 1)


# ---------------------------------------------------------------------------
# Combined comparison
# ---------------------------------------------------------------------------

def compare_models(
    model_a_results: dict,
    model_b_results: dict,
    actual_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run all three metrics for both models and return a summary DataFrame
    for display on Page 4.

    Parameters
    ----------
    model_a_results : dict
        Must contain keys:

        * ``"predicted_table"`` — output of :func:`simulator.build_predicted_table`
        * ``"model"``           — fitted Model A GLM object
        * ``"fixtures"``        — fixture DataFrame with ``FTR`` column

    model_b_results : dict
        Same keys as *model_a_results*; model is the fitted Model B GLM.

    actual_table : pd.DataFrame
        Actual final standings with columns ``Team`` and ``Points``.

    Returns
    -------
    pd.DataFrame
        Columns: ``Metric``, ``Model A``, ``Model B``, ``Winner``.
        Rows: MAE, Log-Loss, Rank Accuracy (±2).
    """
    mae_a = compute_mae(model_a_results["predicted_table"], actual_table)
    mae_b = compute_mae(model_b_results["predicted_table"], actual_table)

    ll_a  = compute_logloss(model_a_results["model"],
                            model_a_results["fixtures"], model_type="A")
    ll_b  = compute_logloss(model_b_results["model"],
                            model_b_results["fixtures"], model_type="B")

    ra_a  = compute_rank_accuracy(model_a_results["predicted_table"], actual_table)
    ra_b  = compute_rank_accuracy(model_b_results["predicted_table"], actual_table)

    def winner_lower(a: float, b: float) -> str:
        """Return label for the model with the lower (better) value."""
        if a < b:
            return "Model A"
        elif b < a:
            return "Model B"
        return "Tie"

    def winner_higher(a: float, b: float) -> str:
        """Return label for the model with the higher (better) value."""
        if a > b:
            return "Model A"
        elif b > a:
            return "Model B"
        return "Tie"

    summary = pd.DataFrame([
        {
            "Metric":   "MAE (points)",
            "Model A":  mae_a,
            "Model B":  mae_b,
            "Winner":   winner_lower(mae_a, mae_b),
        },
        {
            "Metric":   "Log-Loss",
            "Model A":  ll_a,
            "Model B":  ll_b,
            "Winner":   winner_lower(ll_a, ll_b),
        },
        {
            "Metric":   "Rank Accuracy ±2 (%)",
            "Model A":  ra_a,
            "Model B":  ra_b,
            "Winner":   winner_higher(ra_a, ra_b),
        },
    ])

    return summary