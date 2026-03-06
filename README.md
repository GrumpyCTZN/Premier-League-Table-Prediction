# ⚽ Premier League Season Predictor

**Foundation of Data Science** · Tribhuvan University, IOE Pulchowk Campus

Predicts the final Premier League standings using a **Poisson GLM** and **Monte Carlo simulation** (1 000 iterations), with a six-page Streamlit web interface.

---

## Project Structure

```
premier-league-predictor/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── downloader.py          # Fetch E0 + E1 CSVs from football-data.co.uk
│   ├── preprocessor.py        # Clean + standardise team names
│   ├── feature_engineering.py # H2H table, Championship fallback, strength ratios
│   ├── model.py               # Model A — Poisson GLM (historical only)
│   ├── model_b.py             # Model B — Poisson GLM + betting market signal
│   ├── form.py                # Last-5 form multiplier
│   ├── simulator.py           # Monte Carlo simulation + live generator
│   └── evaluator.py           # MAE, log-loss, rank accuracy
├── app/
│   ├── app.py                 # Streamlit entry point
│   └── pages/
│       ├── 01_league_table.py
│       ├── 02_points_distribution.py
│       ├── 03_head_to_head.py
│       ├── 04_model_comparison.py
│       ├── 05_team_deep_dive.py
│       └── 06_live_simulation.py
├── notebooks/
│   └── main.ipynb
├── tests/
├── requirements.txt
└── README.md
```

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/Bipin018/premier-league-predictor.git
cd premier-league-predictor

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app/app.py
```

## Pages

| # | Page | Description |
|---|------|-------------|
| 1 | ⚽ Predicted League Table | Final standings · title / top-4 / relegation % |
| 2 | 📊 Points Distribution | Overlapping histograms for 2–6 selected teams |
| 3 | ⚔️ Head to Head | Simulate any fixture 1 000× with form multiplier |
| 4 | 🔬 Model A vs Model B | MAE · log-loss · rank accuracy comparison |
| 5 | 🔍 Team Deep Dive | Gauges · form badges · full H2H record |
| 6 | ▶️ Live Simulation | Watch a full season unfold matchday by matchday |

## Models

| Model | Formula | Extra feature |
|-------|---------|--------------|
| A | `goals ~ home + C(team) + C(opponent)` | None |
| B | `goals ~ home + implied_prob_home + C(team) + C(opponent)` | Bet365 implied probability |

Both models are trained on seasons 2015/16 – 2023/24 and validated on 2024/25.

## Data Source

[football-data.co.uk](https://www.football-data.co.uk) · Seasons 2015/16 to 2024/25  
Leagues: E0 (Premier League) · E1 (Championship — promoted team fallback)