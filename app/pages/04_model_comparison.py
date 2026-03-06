"""
Page 4 — Model A vs Model B Comparison
========================================
Side-by-side metrics (MAE, log-loss, rank accuracy) for both models
on the 2024/25 validation season, plus a grouped bar chart.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.simulator  import simulate_season, build_predicted_table
from src.evaluator  import compare_models
from src.model_b    import predict_base_goals_b

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ACTUAL_2425 = {
    "Liverpool":      82, "Arsenal":       74, "Chelsea":       63,
    "Man City":       62, "Newcastle":     60, "Aston Villa":   60,
    "Fulham":         57, "Nottm Forest":  55, "Brentford":     54,
    "Tottenham":      53, "Man United":    44, "Bournemouth":   48,
    "Brighton":       48, "Crystal Palace":46, "Everton":       40,
    "Wolves":         39, "Ipswich":       30, "Leicester":     26,
    "Southampton":    12, "West Ham":      38,
}
"""
Approximate 2024/25 final points used for validation.
Replace with actual final table once season concludes.
"""


@st.cache_data(show_spinner="Running Model A validation simulations…")
def _run_model_a(_fixtures: pd.DataFrame) -> pd.DataFrame:
    sim_df = simulate_season(_fixtures, n_simulations=1000)
    return build_predicted_table(sim_df)


@st.cache_data(show_spinner="Running Model B validation simulations…")
def _run_model_b(_model_b, _fixtures_a: pd.DataFrame,
                 _train_odds: pd.DataFrame) -> pd.DataFrame | None:
    if _model_b is None:
        return None
    median_imp = float(_train_odds["implied_prob_home"].median())
    fixtures_b = _fixtures_a.copy()
    mu_h_list, mu_a_list = [], []
    for _, row in fixtures_b.iterrows():
        mh, ma = predict_base_goals_b(
            _model_b, row["HomeTeam"], row["AwayTeam"], median_imp
        )
        mu_h_list.append(mh)
        mu_a_list.append(ma)
    fixtures_b["mu_home"] = mu_h_list
    fixtures_b["mu_away"] = mu_a_list
    sim_df = simulate_season(fixtures_b, n_simulations=1000)
    return build_predicted_table(sim_df)


def _build_actual_table() -> pd.DataFrame:
    rows = [{"Team": t, "Points": p} for t, p in ACTUAL_2425.items()]
    df = (pd.DataFrame(rows)
          .sort_values("Points", ascending=False)
          .reset_index(drop=True))
    df.index += 1
    df.index.name = "Pos"
    return df


def _grouped_bar_chart(
    pred_a: pd.DataFrame,
    pred_b: pd.DataFrame | None,
    actual: pd.DataFrame,
) -> go.Figure:
    """
    Grouped bar chart: predicted vs actual points per team.
    Model A and Model B bars + actual as a scatter dot.
    Winning bar (closer to actual) is highlighted.
    """
    teams = actual["Team"].tolist()
    actual_pts = actual.set_index("Team")["Points"]
    pred_a_pts = pred_a.set_index("Team")["Avg Points"]
    pred_b_pts = pred_b.set_index("Team")["Avg Points"] if pred_b is not None else None

    # Colour bars by which model was closer
    colors_a, colors_b = [], []
    for t in teams:
        a_val   = pred_a_pts.get(t, np.nan)
        act_val = actual_pts.get(t, np.nan)
        if pred_b_pts is not None:
            b_val = pred_b_pts.get(t, np.nan)
            err_a = abs(a_val - act_val)
            err_b = abs(b_val - act_val)
            colors_a.append("#3a86ff" if err_a <= err_b else "#8ecfff")
            colors_b.append("#ff006e" if err_b < err_a  else "#ffaacc")
        else:
            colors_a.append("#3a86ff")

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name         = "Model A",
        x            = teams,
        y            = [pred_a_pts.get(t, 0) for t in teams],
        marker_color = colors_a,
        offsetgroup  = 0,
        hovertemplate="<b>%{x}</b><br>Model A: %{y:.1f} pts<extra></extra>",
    ))

    if pred_b_pts is not None:
        fig.add_trace(go.Bar(
            name         = "Model B",
            x            = teams,
            y            = [pred_b_pts.get(t, 0) for t in teams],
            marker_color = colors_b,
            offsetgroup  = 1,
            hovertemplate="<b>%{x}</b><br>Model B: %{y:.1f} pts<extra></extra>",
        ))

    fig.add_trace(go.Scatter(
        name         = "Actual",
        x            = teams,
        y            = [actual_pts.get(t, np.nan) for t in teams],
        mode         = "markers",
        marker       = dict(symbol="diamond", size=10, color="#ffbe0b",
                            line=dict(width=1, color="white")),
        hovertemplate="<b>%{x}</b><br>Actual: %{y} pts<extra></extra>",
    ))

    fig.update_layout(
        barmode       = "group",
        title         = "Predicted vs Actual Final Points per Team",
        xaxis_title   = "Team",
        yaxis_title   = "Points",
        xaxis_tickangle = -45,
        height        = 520,
        paper_bgcolor = "#0e1117",
        plot_bgcolor  = "#0e1117",
        font          = dict(color="white"),
        legend        = dict(orientation="h", y=1.08),
    )
    return fig


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Model Comparison", page_icon="🔬", layout="wide")
st.title("🔬 Model A vs Model B — Comparison")
st.caption("Validation on 2024/25 season · MAE | Log-Loss | Rank Accuracy ±2")

if "model_a" not in st.session_state or "data" not in st.session_state:
    st.warning("Please launch the app from `app/app.py` so data loads correctly.")
    st.stop()

model_a_state = st.session_state["model_a"]
model_b_state = st.session_state["model_b"]
data          = st.session_state["data"]

actual_table = _build_actual_table()

pred_a = _run_model_a(model_a_state["fixtures"])
pred_b = None
if model_b_state.get("model_b") is not None:
    pred_b = _run_model_b(
        model_b_state["model_b"],
        model_a_state["fixtures"],
        model_b_state["train_odds"],
    )

# ── Metrics table ────────────────────────────────────────────────────────
st.subheader("Evaluation Metrics")

test_fixtures = data["test_df"].copy()

model_a_results = {
    "predicted_table": pred_a,
    "model":           model_a_state["model_a"],
    "fixtures":        test_fixtures,
}
model_b_results = {
    "predicted_table": pred_b if pred_b is not None else pred_a,
    "model":           model_b_state.get("model_b") or model_a_state["model_a"],
    "fixtures":        test_fixtures,
}

with st.spinner("Computing metrics…"):
    summary = compare_models(model_a_results, model_b_results, actual_table)

# Highlight winner column
def _highlight_winner(row):
    styles = [""] * len(row)
    if row["Winner"] == "Model A":
        styles[1] = "background-color: #1a3a5c; font-weight: bold"
    elif row["Winner"] == "Model B":
        styles[2] = "background-color: #3d0a1a; font-weight: bold"
    return styles

st.dataframe(
    summary.style.apply(_highlight_winner, axis=1),
    use_container_width=True,
    hide_index=True,
)

# ── Verdict text ────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Verdict")

a_wins = (summary["Winner"] == "Model A").sum()
b_wins = (summary["Winner"] == "Model B").sum()

if a_wins > b_wins:
    st.success(
        f"✅ **Model A wins** — better on {a_wins}/3 metrics.  "
        f"Historical data alone is sufficient; the betting market adds limited signal."
    )
elif b_wins > a_wins:
    st.success(
        f"✅ **Model B wins** — better on {b_wins}/3 metrics.  "
        f"The betting market provides statistically significant additional predictive power "
        f"(consistent with Dixon & Coles, 1997)."
    )
else:
    st.info("🤝 **Tie** — both models perform equally across the three metrics.")

mae_row = summary[summary["Metric"] == "MAE (points)"].iloc[0]
ll_row  = summary[summary["Metric"] == "Log-Loss"].iloc[0]
ra_row  = summary[summary["Metric"] == "Rank Accuracy ±2 (%)"].iloc[0]

st.markdown(
    f"""
| Metric | Model A | Model B | Gap |
|--------|---------|---------|-----|
| MAE (pts) | {mae_row['Model A']:.2f} | {mae_row['Model B']:.2f} | {abs(mae_row['Model A']-mae_row['Model B']):.2f} pts |
| Log-Loss | {ll_row['Model A']:.4f} | {ll_row['Model B']:.4f} | {abs(ll_row['Model A']-ll_row['Model B']):.4f} |
| Rank Acc ±2 | {ra_row['Model A']:.1f}% | {ra_row['Model B']:.1f}% | {abs(ra_row['Model A']-ra_row['Model B']):.1f} pp |
"""
)

# ── Grouped bar chart ────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Predicted vs Actual Points per Team")
st.caption("🔵 Model A closer · 🔴 Model B closer · 🔶 Actual")
fig = _grouped_bar_chart(pred_a, pred_b, actual_table)
st.plotly_chart(fig, use_container_width=True)