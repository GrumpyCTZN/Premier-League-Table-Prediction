"""
src/simulator.py
================
Monte Carlo season simulation.

simulate_season()   — refactored from pl_prediction_v2.py, logic unchanged.
build_predicted_table() — refactored from pl_prediction_v2.py, logic unchanged.
simulate_once()     — new generator for the Streamlit live-simulation page.
"""

from __future__ import annotations

from typing import Generator, Union

import numpy as np
import pandas as pd
from scipy.stats import poisson


# ---------------------------------------------------------------------------
# Batch simulation (from v2 — unchanged)
# ---------------------------------------------------------------------------

def simulate_season(
    fixtures: pd.DataFrame,
    n_simulations: int = 1000,
) -> pd.DataFrame:
    """
    Run *n_simulations* full season simulations using pre-computed expected
    goals.

    Each simulation independently samples Poisson-distributed scorelines
    for every fixture and accumulates points using standard football rules
    (W=3, D=1, L=0).

    Parameters
    ----------
    fixtures : pd.DataFrame
        Fixture list with ``HomeTeam``, ``AwayTeam``, ``mu_home``, and
        ``mu_away`` columns (output of
        :func:`model.precompute_expected_goals`).
    n_simulations : int, optional
        Number of Monte Carlo iterations.  Default is 1 000.

    Returns
    -------
    pd.DataFrame
        Shape ``(n_simulations, n_teams)``.  Each cell contains the total
        points that team earned in that simulation.
    """
    teams = sorted(set(fixtures["HomeTeam"]) | set(fixtures["AwayTeam"]))
    results = {team: np.zeros(n_simulations) for team in teams}

    mu_home_arr = fixtures["mu_home"].values
    mu_away_arr = fixtures["mu_away"].values
    ht_arr      = fixtures["HomeTeam"].values
    at_arr      = fixtures["AwayTeam"].values

    print(f"  Running {n_simulations} simulations over {len(fixtures)} fixtures...")

    for sim in range(n_simulations):
        season_pts = {team: 0 for team in teams}

        hg_all = poisson.rvs(mu_home_arr)
        ag_all = poisson.rvs(mu_away_arr)

        for i in range(len(fixtures)):
            ht, at = ht_arr[i], at_arr[i]
            hg, ag = hg_all[i], ag_all[i]
            if hg > ag:
                season_pts[ht] += 3
            elif hg < ag:
                season_pts[at] += 3
            else:
                season_pts[ht] += 1
                season_pts[at] += 1

        for team in teams:
            results[team][sim] = season_pts[team]

        if (sim + 1) % 200 == 0:
            print(f"    {sim + 1}/{n_simulations} done...")

    return pd.DataFrame(results)


def build_predicted_table(sim_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise Monte Carlo results into a readable predicted league table.

    Columns produced:

    * ``Team``         — team name
    * ``Avg Points``   — mean final points across all simulations
    * ``Std Dev``      — standard deviation of final points
    * ``Min Pts``      — minimum points in any single simulation
    * ``Max Pts``      — maximum points in any single simulation
    * ``Top-4 %``      — % of simulations the team finished in the top 4
    * ``Title %``      — % of simulations the team finished 1st
    * ``Relegation %`` — % of simulations the team finished in the bottom 3

    Parameters
    ----------
    sim_df : pd.DataFrame
        Output of :func:`simulate_season`.

    Returns
    -------
    pd.DataFrame
        Sorted by ``Avg Points`` descending.  Index is 1-based position.
    """
    ranks = sim_df.rank(axis=1, ascending=False, method="min")

    table = pd.DataFrame({
        "Team":          sim_df.columns,
        "Avg Points":    sim_df.mean().round(1),
        "Std Dev":       sim_df.std().round(1),
        "Min Pts":       sim_df.min().astype(int),
        "Max Pts":       sim_df.max().astype(int),
        "Top-4 %":       (ranks <= 4).mean().mul(100).round(1),
        "Title %":       (ranks == 1).mean().mul(100).round(1),
        "Relegation %":  (ranks >= (len(sim_df.columns) - 2)).mean().mul(100).round(1),
    })

    table = (
        table
        .sort_values("Avg Points", ascending=False)
        .reset_index(drop=True)
    )
    table.index += 1
    table.index.name = "Pos"
    return table


# ---------------------------------------------------------------------------
# Live single-season generator (NEW — not in v2)
# ---------------------------------------------------------------------------

def _build_standings(records: dict[str, dict]) -> pd.DataFrame:
    """
    Convert the running records dict into a sorted standings DataFrame.

    Parameters
    ----------
    records : dict
        Keyed by team name; each value is a dict with keys:
        ``Played``, ``Won``, ``Drawn``, ``Lost``, ``GF``, ``GA``, ``Points``.

    Returns
    -------
    pd.DataFrame
        Columns: ``Pos``, ``Team``, ``Played``, ``Won``, ``Drawn``, ``Lost``,
        ``GF``, ``GA``, ``GD``, ``Points``.  Sorted by Points desc, then GD
        desc, then GF desc (standard PL tiebreakers).
    """
    rows = []
    for team, r in records.items():
        rows.append({
            "Team":   team,
            "Played": r["Played"],
            "Won":    r["Won"],
            "Drawn":  r["Drawn"],
            "Lost":   r["Lost"],
            "GF":     r["GF"],
            "GA":     r["GA"],
            "GD":     r["GF"] - r["GA"],
            "Points": r["Points"],
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["Points", "GD", "GF"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    df.insert(0, "Pos", range(1, len(df) + 1))
    return df


def simulate_once(
    fixtures: pd.DataFrame,
) -> Generator[Union[pd.DataFrame, dict], None, None]:
    """
    Run exactly ONE full season simulation in chronological fixture order,
    yielding live standings after every matchday (~10 fixtures).

    Designed for the Streamlit live-simulation page (Page 6).  The caller
    should iterate over this generator and update a ``st.empty()``
    placeholder on each yield.

    Yield behaviour
    ---------------
    * After each matchday batch of fixtures: yields a ``pd.DataFrame``
      with current standings (columns: ``Pos``, ``Team``, ``Played``,
      ``Won``, ``Drawn``, ``Lost``, ``GF``, ``GA``, ``GD``, ``Points``).
    * When the full season ends: yields a ``dict``::

          {"final": True, "table": <final standings DataFrame>}

      The ``"table"`` value is the same standings DataFrame as the last
      intermediate yield, but signals to the UI that the season is over.

    Matchday grouping
    -----------------
    Fixtures are grouped by ``matchday`` column if present; otherwise they
    are split into sequential batches of 10 (approximating one round of
    fixtures in a 20-team league).

    Goal/upset tracking
    -------------------
    The generator accumulates ``GF`` per team (for the "top scorer" banner)
    and tracks the biggest upset — the match where the underdog (team with
    the lower ``mu``) won by the largest goals margin relative to the
    expected goals difference.  Both are embedded in the final yield dict::

        {
            "final":         True,
            "table":         <DataFrame>,
            "top_scorer":    <team_name: str>,
            "biggest_upset": {
                "home":       <team>,
                "away":       <team>,
                "score":      "<hg>–<ag>",
                "mu_diff":    <float>,   # expected goals advantage of the favourite
                "margin":     <int>,     # actual goals margin of the upset
            }
        }

    Parameters
    ----------
    fixtures : pd.DataFrame
        Fixture list with ``HomeTeam``, ``AwayTeam``, ``mu_home``, and
        ``mu_away`` columns (output of
        :func:`model.precompute_expected_goals`).
        May optionally contain a ``matchday`` column for precise grouping.

    Yields
    ------
    pd.DataFrame or dict
        Intermediate standings DataFrames, then a final dict.
    """
    teams = sorted(set(fixtures["HomeTeam"]) | set(fixtures["AwayTeam"]))

    # Initialise running records
    records: dict[str, dict] = {
        team: {"Played": 0, "Won": 0, "Drawn": 0, "Lost": 0,
               "GF": 0, "GA": 0, "Points": 0}
        for team in teams
    }

    # Upset tracking
    biggest_upset: dict = {}
    biggest_upset_score: float = -1.0

    # ── Determine matchday grouping ──────────────────────────────────────
    BATCH_SIZE = 10

    if "matchday" in fixtures.columns:
        matchday_groups = [
            grp for _, grp in fixtures.groupby("matchday", sort=True)
        ]
    else:
        # Split into sequential batches of ~10
        n = len(fixtures)
        matchday_groups = [
            fixtures.iloc[i: i + BATCH_SIZE]
            for i in range(0, n, BATCH_SIZE)
        ]

    total_matchdays = len(matchday_groups)

    # ── Simulate matchday by matchday ────────────────────────────────────
    for md_idx, md_fixtures in enumerate(matchday_groups, start=1):
        mu_home_arr = md_fixtures["mu_home"].values
        mu_away_arr = md_fixtures["mu_away"].values
        ht_arr      = md_fixtures["HomeTeam"].values
        at_arr      = md_fixtures["AwayTeam"].values

        hg_all = poisson.rvs(mu_home_arr)
        ag_all = poisson.rvs(mu_away_arr)

        for i in range(len(md_fixtures)):
            ht, at = ht_arr[i], at_arr[i]
            hg, ag = int(hg_all[i]), int(ag_all[i])
            mu_h   = float(mu_home_arr[i])
            mu_a   = float(mu_away_arr[i])

            # Update records
            records[ht]["Played"] += 1
            records[at]["Played"] += 1
            records[ht]["GF"]     += hg
            records[ht]["GA"]     += ag
            records[at]["GF"]     += ag
            records[at]["GA"]     += hg

            if hg > ag:
                records[ht]["Won"]    += 1
                records[at]["Lost"]   += 1
                records[ht]["Points"] += 3
                winner, loser = ht, at
                winner_was_home = True
            elif hg < ag:
                records[at]["Won"]    += 1
                records[ht]["Lost"]   += 1
                records[at]["Points"] += 3
                winner, loser = at, ht
                winner_was_home = False
            else:
                records[ht]["Drawn"]  += 1
                records[at]["Drawn"]  += 1
                records[ht]["Points"] += 1
                records[at]["Points"] += 1
                winner = None

            # ── Upset detection ──────────────────────────────────────────
            # An upset = underdog won.  Favourite = team with higher mu.
            if winner is not None:
                favourite_mu   = max(mu_h, mu_a)
                underdog_mu    = min(mu_h, mu_a)
                mu_diff        = favourite_mu - underdog_mu
                actual_margin  = abs(hg - ag)
                # Upset score: only count when the underdog actually won
                favourite_won_home = (mu_h >= mu_a) and (hg > ag)
                favourite_won_away = (mu_a >  mu_h) and (ag > hg)
                is_upset = not (favourite_won_home or favourite_won_away)

                if is_upset:
                    upset_score = mu_diff + actual_margin  # bigger = more shocking
                    if upset_score > biggest_upset_score:
                        biggest_upset_score = upset_score
                        biggest_upset = {
                            "home":     ht,
                            "away":     at,
                            "score":    f"{hg}–{ag}",
                            "mu_diff":  round(mu_diff, 3),
                            "margin":   actual_margin,
                        }

        # Yield current standings after this matchday
        standings = _build_standings(records)
        standings.attrs["matchday"]       = md_idx
        standings.attrs["total_matchdays"] = total_matchdays
        yield standings

    # ── Final yield ──────────────────────────────────────────────────────
    final_table  = _build_standings(records)
    top_scorer   = max(records, key=lambda t: records[t]["GF"])

    yield {
        "final":         True,
        "table":         final_table,
        "top_scorer":    top_scorer,
        "biggest_upset": biggest_upset if biggest_upset else None,
    }