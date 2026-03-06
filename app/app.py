"""
app/app.py
==========
Streamlit entry point for the Premier League Season Predictor.

Responsibilities:
  - Sidebar navigation to all 6 pages
  - Load and cache data + both models once at startup
  - Store all shared objects in st.session_state for pages to consume

Run with:
    streamlit run app/app.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

# ---------------------------------------------------------------------------
# Make src/ importable when running from project root or app/
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.downloader          import download_all_data, PL_SEASONS, format_season
from src.preprocessor        import clean_data
from src.feature_engineering import (
    build_h2h_table,
    build_championship_fallback,
    compute_team_strengths,
)
from src.model import (
    build_poisson_dataset,
    fit_poisson_model,
    inject_promoted_teams,
    precompute_expected_goals,
)
from src.model_b import (
    extract_implied_probs,
    build_model_b_dataset,
    fit_model_b,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PL Season Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Cached data + model loader
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Downloading match data…")
def load_data() -> dict:
    """
    Download, clean, and return all shared data objects.

    Cached so the network round-trip only happens once per session.

    Returns
    -------
    dict with keys:
        pl_df         — cleaned full PL DataFrame (all seasons)
        champ_df      — cleaned Championship DataFrame (all fallback seasons)
        train_df      — PL data excluding the target season
        test_df       — PL data for the target season only
        fixtures      — unique fixture list for the target season
        h2h_table     — head-to-head lookup table
        champ_fallback— Championship strength fallback dict
        promoted_teams— set of promoted team names
        league_avg_home / league_avg_away — float scalars
        target_season — season code string, e.g. "2425"
        team_strengths— DataFrame of attack/defense strength ratios
    """
    pl_raw, champ_raw = download_all_data(PL_SEASONS)

    pl_df    = clean_data(pl_raw)

    # Restore PL_Season BEFORE cleaning so row counts stay aligned
    if "PL_Season" in champ_raw.columns:
        champ_raw_with_season = champ_raw.copy()
    else:
        champ_raw_with_season = champ_raw.copy()
        champ_raw_with_season["PL_Season"] = None

    champ_df = clean_data(champ_raw_with_season)

    target_season   = PL_SEASONS[-1]          # "2425"
    train_df        = pl_df[pl_df["Season"] != target_season].copy()
    test_df         = pl_df[pl_df["Season"] == target_season].copy()

    league_avg_home = float(train_df["FTHG"].mean())
    league_avg_away = float(train_df["FTAG"].mean())

    h2h_table       = build_h2h_table(pl_df)
    champ_fallback  = build_championship_fallback(
        champ_df, league_avg_home, league_avg_away
    )
    team_strengths  = compute_team_strengths(pl_df)

    fixtures = (
        test_df[["HomeTeam", "AwayTeam"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    return dict(
        pl_df          = pl_df,
        champ_df       = champ_df,
        train_df       = train_df,
        test_df        = test_df,
        fixtures       = fixtures,
        h2h_table      = h2h_table,
        champ_fallback = champ_fallback,
        team_strengths = team_strengths,
        league_avg_home= league_avg_home,
        league_avg_away= league_avg_away,
        target_season  = target_season,
    )


@st.cache_data(show_spinner="Fitting Model A (Poisson GLM)…")
def load_model_a(_data: dict) -> dict:
    """
    Fit Model A and pre-compute expected goals for all target-season fixtures.

    The leading underscore on ``_data`` prevents Streamlit from trying to
    hash the dict (which contains non-hashable DataFrames).

    Returns
    -------
    dict with keys:
        model_a   — fitted statsmodels GLM result
        fixtures  — fixture DataFrame with mu_home / mu_away columns
        promoted_teams — set[str]
    """
    train_df        = _data["train_df"]
    fixtures        = _data["fixtures"].copy()
    h2h_table       = _data["h2h_table"]
    champ_fallback  = _data["champ_fallback"]
    target_season   = _data["target_season"]
    league_avg_home = _data["league_avg_home"]
    league_avg_away = _data["league_avg_away"]

    poisson_df     = build_poisson_dataset(train_df)
    model_a        = fit_poisson_model(poisson_df)
    promoted_teams = inject_promoted_teams(
        model_a, fixtures, champ_fallback, target_season, train_df
    )
    fixtures_a = precompute_expected_goals(
        fixtures, model_a, h2h_table, champ_fallback,
        promoted_teams, target_season,
        league_avg_home, league_avg_away,
    )

    return dict(
        model_a        = model_a,
        fixtures       = fixtures_a,
        promoted_teams = promoted_teams,
    )


@st.cache_data(show_spinner="Fitting Model B (GLM + market signal)…")
def load_model_b(_data: dict) -> dict:
    """
    Fit Model B on training data that includes Bet365 odds columns.

    Returns
    -------
    dict with keys:
        model_b   — fitted statsmodels GLM result
        fixtures  — fixture DataFrame (same as Model A fixtures — odds used
                    only during GLM training, not fixture pre-computation)
    """
    train_df = _data["train_df"].copy()

    # Only use rows that have complete odds data
    odds_cols = ["B365H", "B365D", "B365A"]
    if not all(c in train_df.columns for c in odds_cols):
        # Odds not available — return None so pages can gracefully degrade
        return dict(model_b=None, fixtures=None)

    train_odds = train_df.dropna(subset=odds_cols).copy()
    train_odds = extract_implied_probs(train_odds)

    model_b_df = build_model_b_dataset(train_odds)
    model_b    = fit_model_b(model_b_df)

    # Re-use Model A fixtures (mu values differ via training only)
    # Model B fixture mu_home/mu_away are computed on-demand in pages
    # using predict_base_goals_b with per-fixture implied_prob_home.
    return dict(model_b=model_b, train_odds=train_odds)


# ---------------------------------------------------------------------------
# Bootstrap session state
# ---------------------------------------------------------------------------

def _bootstrap() -> None:
    """Load all shared state into st.session_state on first run."""
    if "data" not in st.session_state:
        data = load_data()
        st.session_state["data"] = data

    if "model_a" not in st.session_state:
        model_a = load_model_a(st.session_state["data"])
        st.session_state["model_a"] = model_a

    if "model_b" not in st.session_state:
        model_b = load_model_b(st.session_state["data"])
        st.session_state["model_b"] = model_b


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

PAGES = {
    "⚽  Predicted League Table":     "pages/01_league_table.py",
    "📊  Points Distribution":        "pages/02_points_distribution.py",
    "⚔️   Head to Head":              "pages/03_head_to_head.py",
    "🔬  Model A vs Model B":         "pages/04_model_comparison.py",
    "🔍  Team Deep Dive":             "pages/05_team_deep_dive.py",
    "▶️   Live Season Simulation":    "pages/06_live_simulation.py",
}


def _render_sidebar() -> None:
    """Render the sidebar with project branding and navigation links."""
    with st.sidebar:
        st.image(
            "https://upload.wikimedia.org/wikipedia/en/f/f2/Premier_League_Logo.svg",
            width=120,
        )
        st.title("PL Season Predictor")
        st.caption(
            "Foundation of Data Science · IOE Pulchowk Campus\n\n"
            "Poisson GLM + Monte Carlo (1 000 iterations)"
        )
        st.markdown("---")
        st.markdown("### Navigation")
        for label, path in PAGES.items():
            st.page_link(path, label=label)
        st.markdown("---")
        st.caption("Data: football-data.co.uk · Seasons 2015/16 – 2024/25")


# ---------------------------------------------------------------------------
# Home / landing page
# ---------------------------------------------------------------------------

def _render_home() -> None:
    """Render the app landing page shown at the root URL."""
    st.title("⚽ Premier League Season Predictor")
    st.subheader("Foundation of Data Science · Tribhuvan University, IOE Pulchowk Campus")

    st.markdown(
        """
        This app uses a **Poisson Generalised Linear Model** combined with
        **Monte Carlo simulation** (1 000 iterations) to predict the final
        Premier League standings for the 2024/25 season.

        ---

        ### How it works
        | Step | Description |
        |------|-------------|
        | 1 | Download 10 seasons of PL + Championship data from football-data.co.uk |
        | 2 | Clean and standardise team names |
        | 3 | Fit a Poisson GLM: `goals ~ home + C(team) + C(opponent)` |
        | 4 | Inject Championship-derived strengths for promoted teams |
        | 5 | Apply head-to-head adjustments per fixture |
        | 6 | Run 1 000 independent Monte Carlo season simulations |
        | 7 | Compare Model A (historical only) vs Model B (+ betting market signal) |

        ---

        ### Pages
        | Page | Description |
        |------|-------------|
        | ⚽ Predicted League Table | Final standings with title / top-4 / relegation probabilities |
        | 📊 Points Distribution | Overlapping histograms for selected teams |
        | ⚔️ Head to Head | Simulate any fixture 1 000 times with form multiplier |
        | 🔬 Model A vs Model B | MAE, log-loss, and rank accuracy comparison |
        | 🔍 Team Deep Dive | Strength gauges, form badges, full H2H record |
        | ▶️ Live Season Simulation | Watch an entire season unfold matchday by matchday |

        ---
        """
    )

    # Show data loading status
    if "data" in st.session_state:
        data = st.session_state["data"]
        col1, col2, col3 = st.columns(3)
        col1.metric("PL Matches Loaded",    f"{len(data['pl_df']):,}")
        col2.metric("Training Matches",     f"{len(data['train_df']):,}")
        col3.metric("Target Season", format_season(data["target_season"]))
    else:
        st.info("Loading data… please wait.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_bootstrap()
_render_sidebar()
_render_home()