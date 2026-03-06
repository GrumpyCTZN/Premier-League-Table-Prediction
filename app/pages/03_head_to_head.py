"""
Page 3 — Head to Head Simulation
==================================
Simulate any single fixture 1 000 times using the Poisson GLM base mu
blended with a form multiplier.  Displays win/draw/loss probabilities,
a scoreline heatmap, and the historical H2H record.

No betting odds are used on this page.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import poisson

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model       import predict_base_goals
from src.form        import compute_form_multiplier, seed_form_window
from src.downloader  import PL_SEASONS

# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

N_SIMS   = 1000
MAX_SHOW = 6          # scoreline heatmap axis range: 0 to MAX_SHOW-1


@st.cache_data(show_spinner="Simulating fixture…")
def simulate_fixture(
    home_team: str,
    away_team: str,
    mu_home_base: float,
    mu_away_base: float,
    home_form_mult: float,
    away_form_mult: float,
    n_sims: int = N_SIMS,
) -> dict:
    """
    Simulate a single fixture *n_sims* times.

    Probability blend per simulation::

        adjusted_mu = 0.70 × GLM_base_mu + 0.30 × form_multiplier × GLM_base_mu
                    = GLM_base_mu × (0.70 + 0.30 × form_multiplier)

    Parameters
    ----------
    home_team, away_team : str
    mu_home_base, mu_away_base : float
        Raw GLM expected goals.
    home_form_mult, away_form_mult : float
        Form multipliers from :func:`src.form.compute_form_multiplier`.
    n_sims : int

    Returns
    -------
    dict with keys:
        p_home, p_draw, p_away : float (probabilities)
        scoreline_matrix       : np.ndarray shape (MAX_SHOW, MAX_SHOW)
        home_goals_arr         : np.ndarray
        away_goals_arr         : np.ndarray
    """
    mu_h = mu_home_base * (0.70 + 0.30 * home_form_mult)
    mu_a = mu_away_base * (0.70 + 0.30 * away_form_mult)

    hg = poisson.rvs(mu_h, size=n_sims)
    ag = poisson.rvs(mu_a, size=n_sims)

    p_home = float((hg > ag).mean())
    p_draw = float((hg == ag).mean())
    p_away = float((hg < ag).mean())

    # Scoreline probability matrix (capped at MAX_SHOW-1 goals each side)
    matrix = np.zeros((MAX_SHOW, MAX_SHOW))
    for h, a in zip(np.minimum(hg, MAX_SHOW - 1), np.minimum(ag, MAX_SHOW - 1)):
        matrix[h, a] += 1
    matrix /= n_sims

    return dict(
        p_home           = p_home,
        p_draw           = p_draw,
        p_away           = p_away,
        scoreline_matrix = matrix,
        home_goals_arr   = hg,
        away_goals_arr   = ag,
    )


def _donut_chart(home_team: str, away_team: str,
                 p_home: float, p_draw: float, p_away: float) -> go.Figure:
    """Donut chart for Win/Draw/Loss probabilities."""
    fig = go.Figure(go.Pie(
        labels   = [f"{home_team} Win", "Draw", f"{away_team} Win"],
        values   = [p_home, p_draw, p_away],
        hole     = 0.55,
        marker   = dict(colors=["#3a86ff", "#adb5bd", "#ff4444"]),
        textinfo = "label+percent",
        hovertemplate="%{label}: %{percent}<extra></extra>",
    ))
    fig.update_layout(
        title         = "Match Outcome Probabilities",
        height        = 380,
        paper_bgcolor = "#0e1117",
        font          = dict(color="white"),
        showlegend    = False,
    )
    return fig


def _heatmap_fig(matrix: np.ndarray,
                 home_team: str, away_team: str) -> go.Figure:
    """Scoreline probability heatmap."""
    labels = [str(i) for i in range(MAX_SHOW)]
    fig = go.Figure(go.Heatmap(
        z           = matrix,
        x           = labels,
        y           = labels,
        colorscale  = "Blues",
        text        = [[f"{matrix[r, c]:.1%}" for c in range(MAX_SHOW)]
                        for r in range(MAX_SHOW)],
        texttemplate= "%{text}",
        showscale   = True,
        hovertemplate=(
            f"Score: {home_team} %{{y}} – %{{x}} {away_team}<br>"
            "Probability: %{z:.2%}<extra></extra>"
        ),
    ))
    fig.update_layout(
        title         = "Scoreline Probability Heatmap",
        xaxis_title   = f"{away_team} Goals",
        yaxis_title   = f"{home_team} Goals",
        height        = 420,
        paper_bgcolor = "#0e1117",
        plot_bgcolor  = "#0e1117",
        font          = dict(color="white"),
    )
    return fig


def _h2h_record(pl_df: pd.DataFrame,
                home_team: str, away_team: str) -> dict:
    """Extract historical H2H record (all matches, both directions)."""
    mask = (
        ((pl_df["HomeTeam"] == home_team) & (pl_df["AwayTeam"] == away_team)) |
        ((pl_df["HomeTeam"] == away_team) & (pl_df["AwayTeam"] == home_team))
    )
    subset = pl_df[mask]
    if subset.empty:
        return {"home_w": 0, "home_d": 0, "home_l": 0,
                "away_w": 0, "away_d": 0, "away_l": 0,
                "total": 0}

    # home_team's perspective when playing at home
    ht_home = subset[(subset["HomeTeam"] == home_team)]
    ht_away = subset[(subset["AwayTeam"] == home_team)]

    return {
        "home_w": int((ht_home["FTHG"] > ht_home["FTAG"]).sum()),
        "home_d": int((ht_home["FTHG"] == ht_home["FTAG"]).sum()),
        "home_l": int((ht_home["FTHG"] < ht_home["FTAG"]).sum()),
        "away_w": int((ht_away["FTAG"] > ht_away["FTHG"]).sum()),
        "away_d": int((ht_away["FTAG"] == ht_away["FTHG"]).sum()),
        "away_l": int((ht_away["FTAG"] < ht_away["FTHG"]).sum()),
        "total":  len(subset),
    }


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Head to Head", page_icon="⚔️", layout="wide")
st.title("⚔️ Head to Head Simulation")
st.caption(f"Simulate any fixture {N_SIMS:,} times · Form-adjusted Poisson GLM · No betting data")

if "model_a" not in st.session_state or "data" not in st.session_state:
    st.warning("Please launch the app from `app/app.py` so data loads correctly.")
    st.stop()

model_a  = st.session_state["model_a"]
data     = st.session_state["data"]
pl_df    = data["pl_df"]
fixtures = model_a["fixtures"]
all_teams = sorted(set(fixtures["HomeTeam"]) | set(fixtures["AwayTeam"]))

# Team selection
col1, col2 = st.columns(2)
home_team = col1.selectbox("🏠 Home Team", all_teams, index=0)
away_team = col2.selectbox("✈️ Away Team", all_teams, index=1)

if home_team == away_team:
    st.warning("Please select two different teams.")
    st.stop()

simulate_btn = st.button("⚽ Simulate 1 000 Times", type="primary")

if simulate_btn:
    # Get base expected goals from Model A GLM
    try:
        mu_h_base, mu_a_base = predict_base_goals(
            model_a["model_a"], home_team, away_team
        )
    except Exception:
        avg_h = data["league_avg_home"]
        avg_a = data["league_avg_away"]
        mu_h_base, mu_a_base = avg_h, avg_a
        st.info("Team not in GLM training data — using league averages.")

    # Form multipliers
    season_before = PL_SEASONS[-2]   # season before the target
    home_form = seed_form_window(pl_df, home_team, season_before)
    away_form = seed_form_window(pl_df, away_team, season_before)
    home_mult = compute_form_multiplier(home_form)
    away_mult = compute_form_multiplier(away_form)

    # Run simulation
    result = simulate_fixture(
        home_team, away_team,
        mu_h_base, mu_a_base,
        home_mult, away_mult,
        N_SIMS,
    )

    # ── Layout ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader(f"{home_team}  vs  {away_team}")

    # Outcome metrics
    m1, m2, m3 = st.columns(3)
    m1.metric(f"🔵 {home_team} Win",  f"{result['p_home']:.1%}")
    m2.metric("⚪ Draw",              f"{result['p_draw']:.1%}")
    m3.metric(f"🔴 {away_team} Win",  f"{result['p_away']:.1%}")

    # Form info
    st.caption(
        f"Form ({home_team}): {' '.join(home_form) if home_form else 'N/A'}  "
        f"| Multiplier: ×{home_mult:.3f}     "
        f"Form ({away_team}): {' '.join(away_form) if away_form else 'N/A'}  "
        f"| Multiplier: ×{away_mult:.3f}"
    )

    # Donut + Heatmap
    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            _donut_chart(home_team, away_team,
                         result["p_home"], result["p_draw"], result["p_away"]),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(
            _heatmap_fig(result["scoreline_matrix"], home_team, away_team),
            use_container_width=True,
        )

    # Most likely scoreline
    mat = result["scoreline_matrix"]
    best_h, best_a = np.unravel_index(mat.argmax(), mat.shape)
    st.info(
        f"🎯 **Most likely scoreline:** "
        f"{home_team} {best_h} – {best_a} {away_team}  "
        f"({mat[best_h, best_a]:.1%} probability)"
    )

    # Historical H2H record
    st.markdown("---")
    st.subheader("Historical Head-to-Head Record (PL data)")
    rec = _h2h_record(pl_df, home_team, away_team)

    if rec["total"] == 0:
        st.info("No historical meetings found in the dataset.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Meetings", rec["total"])
        c2.metric(
            f"{home_team} Record",
            f"W{rec['home_w']+rec['away_w']} "
            f"D{rec['home_d']+rec['away_d']} "
            f"L{rec['home_l']+rec['away_l']}"
        )
        c3.metric(f"{home_team} at Home",
                  f"W{rec['home_w']} D{rec['home_d']} L{rec['home_l']}")
        c4.metric(f"{home_team} Away",
                  f"W{rec['away_w']} D{rec['away_d']} L{rec['away_l']}")
else:
    st.info("Select two teams and press **Simulate 1 000 Times** to run the fixture.")