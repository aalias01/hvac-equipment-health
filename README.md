# HVAC Building Anomaly Scoring

Ranks unusual building-level chilled-water operation from ASHRAE meter, weather, and building-metadata rows, then explains each relative score with SHAP. The COP and delta-T fields are HVAC-informed proxies built for this dataset, not measured equipment telemetry or validated failure indicators.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7-orange)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-teal)](https://fastapi.tiangolo.com/)
[![CI](https://github.com/aalias01/hvac-equipment-health/actions/workflows/ci.yml/badge.svg)](https://github.com/aalias01/hvac-equipment-health/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**[Live demo](https://hvac.alvinalias.com)** | **[API docs](https://alvinalias-portfolio-ml-api.hf.space/hvac/docs)**

## Why

Building energy data can reveal unusual operating patterns even when failure labels are unavailable. This project tests that narrower question on hourly chilled-water demand and outdoor weather. It does not establish early equipment degradation or diagnose a specific compressor, refrigerant circuit, or coil fault.

## The features

| Feature | Formula | What it signals |
|---------|---------|-----------------|
| Cooling-efficiency proxy | Chilled-water meter reading converted to kWh, divided by a weather-based denominator | A normalized meter-demand feature; direct input power is unavailable, so this is not measured COP |
| Outdoor-to-setpoint proxy | `max(outdoor air temperature - 13°C, 0)` | Load context only; supply and return air temperatures are unavailable |
| Air-to-dew-point proxy | `max(outdoor air temperature - dew point, 0)` | Ambient moisture/temperature spread; condenser and evaporator temperatures are unavailable |
| Load-ratio proxy | Chilled-water demand / square-footage-based capacity estimate | Relative demand against a rule-of-thumb building capacity, not measured equipment loading |
| Runtime proxy | Fraction of recent hours with nonzero chilled-water demand | Building-meter activity, not compressor runtime telemetry |
| Rolling proxy statistics | 24-hour and 7-day means, spreads, and deviations | Temporal context for the anomaly detector, not proven fault lead time |

## Results

| Metric | Value | Notes |
|--------|-------|-------|
| Readings scored | 2,876,400 | Hourly chilled-water readings (4.18M raw, filtered for rolling-history coverage) |
| Buildings analyzed | 497 | Buildings with at least 90 days of meter coverage |
| Anomaly rate (contamination=0.05) | 5.0% (143,820 readings) | Sensitivity-checked at 0.02 / 0.05 / 0.10; every point flagged at 0.02 stays flagged at 0.05 |
| Relative display scores | 35.0 to 74.0, median 58.6 | Per-building mean of the reading-level anomaly display; bands are not condition classes |
| Top SHAP feature | `rolling_cop_std_24h` | 24-hour variability of the cooling-efficiency proxy contributed most to detector scores |
| LOF vs Isolation Forest agreement | 91.3% | On a 100k-reading comparison sample |

![Relative anomaly-score distribution](figures/03_health_score_distribution.png)

![SHAP summary](figures/03_shap_summary.png)

See the [model card](models/MODEL_CARD.md) for score meaning, detector limits, and data provenance.

## How it works

```
ASHRAE data (hourly chilled-water meter + weather + building metadata)
        |
feature engineering (src/features.py)
  explicitly labeled efficiency, delta-T, load, runtime proxies + rolling stats
        |
anomaly detection (src/scorer.py)
  Isolation Forest (primary), LOF (comparison), contamination 0.05
        |
relative anomaly score 0-100, normalized per building system
  SHAP explains which proxy feature drove each score
        |
FastAPI (shared HF Space)  <->  vanilla JS operations wall (Vercel)
  scored fleet snapshot, curated demo readings, detector cross-check
```

## Tech stack

Python 3.11, Pandas, NumPy, scikit-learn 1.7 (Isolation Forest, LOF), SHAP TreeExplainer, FastAPI on a shared Hugging Face Docker Space, vanilla HTML/CSS/JS operations wall on Vercel. Environments: `environment.yml` for local conda work and `requirements.txt` for pip serving.

## Dataset

**ASHRAE Great Energy Predictor III** ([Kaggle competition](https://www.kaggle.com/c/ashrae-energy-prediction)): 1,000+ buildings, hourly meter readings plus weather, 2016 to 2017, covering chilled water, electricity, hot water, and steam meters. Used under the competition's data rules.

## Run it locally

```bash
git clone https://github.com/aalias01/hvac-equipment-health
cd hvac-equipment-health
conda env create -f environment.yml
conda activate hvac-health

# data (requires kaggle.json in ~/.kaggle/)
kaggle competitions download -c ashrae-energy-prediction -p data/raw/
```

Run the notebooks in order: `01_eda.ipynb`, `02_feature_engineering.ipynb`, `03_anomaly_detection.ipynb`. Notebook 03 writes the model artifacts the API needs:

```
models/isolation_forest.joblib
models/isolation_forest_scaler.joblib
models/lof_model.joblib
models/scorer_meta.json
models/unit_baselines.joblib
```

Build the curated demo readings from real feature rows:

```bash
python scripts/curate_demo_readings.py
```

That writes `models/demo_readings.json`, a small API runtime artifact used by `/demo-readings`.

Then start the API and open the frontend:

```bash
uvicorn api.main:app --reload
# docs at http://localhost:8000/docs
```

Open `frontend/index.html`. It points at the deployed API by default; for a local API, run `localStorage.setItem("HVAC_API_BASE", "http://localhost:8000")` in the browser console. The fleet board is a scored snapshot from `models/unit_baselines.joblib`, not live telemetry. The score button draws from complete historical readings in `models/demo_readings.json` instead of placeholder sensor defaults. The API starts in degraded mode until the model artifacts exist.

Run tests and lint:

```bash
python -m pip install -r requirements-dev.txt
ruff check .
pytest -q
```

## Limitations

- ASHRAE meters are building-level, so "unit" here means a building's chilled-water system, not an individual compressor.
- Direct electrical input, supply/return air temperatures, condenser/evaporator temperatures, equipment runtime, alarms, work orders, and failure outcomes are absent. The named COP and delta-T fields are proxies and must not be presented as those measurements.
- Anomaly detection is unsupervised; there are no ground-truth failure labels in this dataset to compute precision or recall against.
- The shared Hugging Face CPU Space sleeps after extended inactivity; the first request to this route can take a moment while the service wakes and loads its models.

## Deployment

From the portfolio workspace, run `bash portfolio_ml_api/scripts/sync_from_portfolio.sh`, commit the changes in `portfolio_ml_api`, and push its `main` branch. GitHub Actions deploys the shared Hugging Face Docker Space. This service is mounted at `/hvac`. Vercel serves `frontend/` at the live demo URL.

## Project structure

```
├── notebooks/       # 01 EDA, 02 features, 03 anomaly detection + scoring
├── src/             # features.py, scorer.py
├── api/             # FastAPI: main.py, schemas.py, predictor.py
├── frontend/        # operations wall (Vercel)
├── scripts/         # reproducible curation utilities
├── models/          # API runtime artifacts (committed)
├── figures/         # generated plots
└── data/            # gitignored; download from Kaggle
```

Built by [Alvin Alias](https://github.com/aalias01), MS Data Science, University of Washington. 3 years HVAC product development at Rheem Manufacturing.
