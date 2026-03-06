"""
Page 6 — Live Season Simulation
=================================
Watch an entire Premier League season unfold matchday by matchday using the
simulate_once() generator.  Every Monte Carlo run is a different universe —
the randomness is made visceral and visible.
"""

from __future__ import annotations

from pathlib import Path
import sys
import time

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.simulator import simulate_once

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

_CL_BLUE   = "#1a3a5c"
_EL_AMBER  = "#5c3d00"
_REL_RED   = "#5c1a1a"
_DEFAULT   = "#1e1e1e"
_CHANGED   = "#2a4a2a"   # flash green for rows that just gained points


def _row_color(pos: int, n_teams: int) -> str:
    if pos <= 4:
        return _CL_BLUE
    if pos <= 6:
        return _EL_AMBER
    if pos > n_teams - 3:
        return _REL_RED
    return _DEFAULT


def _style_live_table(
    df: pd.DataFrame,
    changed_teams: set[str],
) -> pd.io.formats.style.Styler:
    """
    Apply row background colours; highlight teams that just gained points
    with a bright green flash.
    """
    n = len(df)

    def row_bg(row):
        team  = row["Team"]
        pos   = int(row["Pos"])
        color = _CHANGED if team in changed_teams else _row_color(pos, n)
        return [f"background-color: {color}; color: white"] * len(row)

    return df.style.apply(row_bg, axis=1)


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Live Simulation", page_icon="▶️", layout="wide")
st.title("▶️ Live Season Simulation")
st.caption(
    "Every press of the button is a brand-new universe. "
    "Watch 380 fixtures resolve one matchday at a time."
)

if "model_a" not in st.session_state:
    st.warning("Please launch the app from `app/app.py` so data loads correctly.")
    st.stop()

fixtures = st.session_state["model_a"]["fixtures"]

# ── Legend ───────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.markdown("🔵 **Top 4** — Champions League")
c2.markdown("🟡 **5th–6th** — Europa League")
c3.markdown("⚪ **7th–17th** — Mid-table")
c4.markdown("🔴 **Bottom 3** — Relegation")

st.markdown("---")

# ── Simulate button ───────────────────────────────────────────────────────
if st.button("▶  Simulate Season", type="primary"):

    progress_bar   = st.progress(0, text="Starting simulation…")
    table_slot     = st.empty()       # standings table placeholder
    matchday_label = st.empty()       # "Matchday X of 38"

    prev_points: dict[str, int] = {}  # track which teams gained points

    for payload in simulate_once(fixtures):

        # ── Final yield ────────────────────────────────────────────────
        if isinstance(payload, dict) and payload.get("final"):
            final_table   = payload["table"]
            top_scorer    = payload.get("top_scorer")
            biggest_upset = payload.get("biggest_upset")

            progress_bar.progress(1.0, text="Full Time!")
            matchday_label.empty()

            # Lock in final table
            table_slot.dataframe(
                _style_live_table(final_table, set()),
                use_container_width=True,
                height=620,
            )

            # "Full Time" banner
            st.success("⏱️ **Full Time — Season Complete!**")

            col1, col2 = st.columns(2)

            # Top scorer (team with most GF)
            with col1:
                st.markdown("### 🥇 Top Scoring Team")
                if top_scorer:
                    ts_row = final_table[final_table["Team"] == top_scorer]
                    if not ts_row.empty:
                        gf = int(ts_row.iloc[0]["GF"])
                        st.metric(top_scorer, f"{gf} goals scored")
                    else:
                        st.write(top_scorer)

            # Biggest upset
            with col2:
                st.markdown("### 😱 Biggest Upset")
                if biggest_upset:
                    u = biggest_upset
                    st.markdown(
                        f"**{u['home']}  {u['score']}  {u['away']}**  \n"
                        f"Expected goals advantage of favourite: **{u['mu_diff']:.2f}**  \n"
                        f"Actual winning margin: **{u['margin']} goal(s)**"
                    )
                else:
                    st.write("No significant upset recorded.")

            break   # generator exhausted

        # ── Intermediate matchday yield ────────────────────────────────
        standings: pd.DataFrame = payload

        md        = standings.attrs.get("matchday", "?")
        total_md  = standings.attrs.get("total_matchdays", 38)
        progress  = min(int(md) / max(int(total_md), 1), 1.0)

        progress_bar.progress(progress, text=f"Matchday {md} of {total_md}…")
        matchday_label.markdown(f"### Matchday {md} of {total_md}")

        # Determine which teams just gained points (highlight flash)
        current_points = dict(zip(standings["Team"], standings["Points"]))
        changed = {
            team for team, pts in current_points.items()
            if pts != prev_points.get(team, 0)
        }
        prev_points = current_points

        table_slot.dataframe(
            _style_live_table(standings, changed),
            use_container_width=True,
            height=620,
        )

        time.sleep(0.08)   # brief pause so the UI animates smoothly

else:
    st.info(
        "Press **▶ Simulate Season** to run a complete season simulation "
        "live — matchday by matchday.  Press again for a brand-new universe."
    )