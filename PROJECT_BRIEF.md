# Project Brief — HVAC Equipment Health Scoring

| Priority Score | Tier | Recommended Ship Slot | Effort |
|----------------|------|----------------------|--------|
| **4.10** | **P1** | **Order #4** *(after CMAPSS · Retail Returns · RAG)* | 10–14 hrs base · +6–8 hrs for PySpark variant (skip if Retail Returns shipped) |

**Score breakdown** — ED 4 · DIFF 5 · SC 3 · DSS 5 · BV 4 · EE 3
**Lane:** A (Industrial — Manufacturing · HVAC · Building Systems)
**Target companies:** Honeywell, Siemens, Johnson Controls, Boeing Facilities, Rheem, Daikin

**Conditions to re-rank:**
- If applying primarily to retail Lane B targets: deprioritize (lower ED for retail roles)
- If a Rheem / Honeywell / Johnson Controls opportunity surfaces: promote to P0 immediately — this is the strongest domain-credibility play in the portfolio
- DIFF score is the highest in the portfolio (5/5) — keep this one defended as a moat

---

## Problem Statement

HVAC systems in commercial buildings and manufacturing plants fail in predictable ways — but most operators react after failure rather than before it. Given operational sensor data (temperatures, pressures, flow rates, power draw), can we score the health of each unit, detect anomalies early, and surface actionable insights through a dashboard?

This mirrors exactly what Alvin observed at Rheem — but packaged as a reproducible, end-to-end data science system.

---

## Why This Project for Alvin

- **Unmatched domain credibility:** 3 years at Rheem designing HVAC systems. Knows what failing looks like physically — feature engineering will be defensible and non-obvious to generic DS candidates.
- **COP and delta-T features:** Coefficient of Performance and temperature differential are the two most important efficiency signals in refrigeration. Only someone with this background would know to engineer them.
- **Fills the unsupervised learning gap:** CMAPSS is supervised (labeled RUL). HVAC is unsupervised (no failure labels). Different paradigm, different skills, same domain credibility.

---

## Dataset

**Primary:** ASHRAE Great Energy Predictor III
- Kaggle: https://www.kaggle.com/c/ashrae-energy-prediction
- Real building energy/HVAC data from 1,000+ buildings
- Hourly meter readings + weather data

**Alternative (simpler start):** HVAC Fault Detection Dataset
- UCI or Kaggle: "HVAC fault detection"
- Smaller, cleaner, faster to get moving

**Last-resort fallback:** Synthetic HVAC sensor data
- Generate realistic sensor streams using domain knowledge (COP curves, refrigerant physics)
- Only use if both real datasets above prove unworkable — flag clearly in README
- Hiring-manager research strongly favors real data; treat synthetic as a labelled fallback, not the default

---

## Tech Stack

| Layer | Tool |
|-------|------|
| Data wrangling (base) | Pandas, NumPy |
| Data wrangling (scale variant) | **PySpark on Databricks Free Edition** (optional Phase 5) |
| EDA visualization | Matplotlib, Seaborn |
| Anomaly detection | Scikit-learn: Isolation Forest, Local Outlier Factor |
| Health scoring | Rule-based from anomaly scores + domain thresholds |
| Interpretability | SHAP (on the scoring model) |
| API | FastAPI (Render) |
| Frontend | Custom HTML/CSS/JS (Vercel) — health score gauge, sensor trends, alert table |
| Optional dashboard | PowerBI (alternate to custom frontend if Retail Returns hasn't shipped yet) |
| Environment | conda (environment.yml) |

---

## Domain-Driven Features to Engineer

| Feature | Formula / Description | Physical meaning |
|---------|----------------------|------------------|
| COP | Cooling output / power input | Single best efficiency indicator in refrigeration |
| delta-T supply | T_supply_air - T_return_air | Measures heat exchange effectiveness |
| delta-T refrigerant | T_condenser - T_evaporator | Refrigerant circuit efficiency |
| Load ratio | Actual load / rated capacity | Operating near limits = higher stress |
| Runtime fraction | Hours running / hours in period | High runtime + poor efficiency = degradation signal |
| Rolling COP deviation | COP vs. 30-day rolling mean | Trend-based degradation indicator |

---

## Deliverables

1. `notebooks/01_eda.ipynb` — sensor distributions, correlation, time-series trends
2. `notebooks/02_feature_engineering.ipynb` — COP, delta-T, rolling stats, load ratio
3. `notebooks/03_anomaly_detection.ipynb` — Isolation Forest, LOF, threshold tuning, health score
4. `src/features.py` — domain feature engineering functions
5. `src/scorer.py` — health score computation + anomaly flagging
6. `api/main.py` — FastAPI: POST /score → health score + anomaly flags + SHAP
7. `frontend/` — dashboard: unit selector, health gauge, trend charts, alert table
8. `README.md` — GitHub-ready with screenshots and live demo link

---

## Project Phases

### Phase 1 — Data + EDA (2–3 hrs)
- [ ] Load and understand dataset schema (sensors, frequency, units)
- [ ] Plot distributions, correlations, time-series by unit
- [ ] Identify data quality issues (missing, outliers, sensor dropouts)
- [ ] Annotated EDA notebook with findings

### Phase 2 — Feature Engineering (2–3 hrs)
- [ ] Implement COP, delta-T, load ratio features
- [ ] Rolling statistics: 24-hr and 7-day windows
- [ ] Time-based features: hour of day, day of week, season
- [ ] Document every feature with physical interpretation in notebook

### Phase 3 — Anomaly Detection + Health Scoring (3–4 hrs)
- [ ] Isolation Forest: train, tune contamination parameter
- [ ] LOF as comparison
- [ ] Convert anomaly scores to 0–100 health score (inverse of anomaly severity)
- [ ] Validate: do flagged anomalies correspond to physically unusual operating points?
- [ ] SHAP on the scoring model — which sensors drive each unit's score?
- [ ] Save model artifacts with joblib

### Phase 4 — API + Frontend (3–4 hrs)
- [ ] FastAPI: POST /score with sensor readings → health score + anomaly flags + top SHAP features
- [ ] Frontend: health gauge (0–100), time-series chart, anomaly alert table
- [ ] Deploy: FastAPI to Render, frontend to Vercel
- [ ] Update README with live demo link and screenshots

### Phase 5 — *Optional:* PySpark / Databricks Variant (6–8 hrs)
*Skip this phase if Retail Returns Intelligence has already shipped — that project is the primary PySpark showcase and this would be redundant. Run this phase if HVAC ships first OR if applying to a Honeywell/Siemens role that explicitly asks for big-data scale.*

- [ ] Set up Databricks Free Edition workspace
- [ ] Reimplement feature engineering pipeline in PySpark
- [ ] Apply medallion architecture: Bronze (raw sensor) → Silver (cleaned + resampled) → Gold (engineered features per unit)
- [ ] Validate parity vs. Pandas on a sample
- [ ] Add Databricks notebook link in README
- [ ] Resume bullet edit: replace "Pandas" with "Pandas + PySpark on Databricks"

---

## Interview Talking Points

1. *"I chose Isolation Forest because HVAC faults are rare and unlabeled in practice — I've seen this at Rheem. IForest doesn't require labeled failure data, which is the realistic industrial scenario."*
2. *"The COP feature was my idea — it's the ratio of cooling output to power input, the single most important efficiency signal in refrigeration systems. Most DS candidates wouldn't know to engineer that."*
3. *"I set the contamination parameter to 0.05 based on industry rule of thumb — roughly 5% of operating points are genuinely anomalous. I validated this against physically unusual readings in the EDA."*
4. *"The health score is interpretable by operations staff — it's a 0–100 gauge, not a probability. I deliberately translated the model output into engineering language."*

---

## Success Criteria

- [ ] GitHub repo public with live Vercel demo link in README
- [ ] Health score dashboard renders without errors
- [ ] Can explain every modeling decision in an interview
- [ ] SHAP analysis ties sensor importance to physical HVAC behavior
- [ ] Resume bullet updated: *"Built end-to-end HVAC equipment health scoring system using Isolation Forest anomaly detection on real operational data; engineered domain-specific features (COP, delta-T) — features only an HVAC engineer would know to build — and deployed as an interactive dashboard."*

---

*Brief created: April 2026 · Updated April 2026 · May 2026 (ship slot corrected to #4 — RAG promoted to #3 in Apr 2026 strategic pass; PySpark phase marked skip since Retail Returns shipped) | Priority Score 4.10 · Tier P1 · Ship slot #4*
