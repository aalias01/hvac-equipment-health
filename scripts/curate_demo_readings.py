#!/usr/bin/env python3
"""Curate realistic demo readings from the local HVAC study data.

The deployed demo should not score a random unit with placeholder rolling
features. This script rebuilds real feature rows from the raw ASHRAE-derived
CSVs, scores a deterministic sample against each unit's own baseline, and
writes a small balanced scenario file for the frontend.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import (  # noqa: E402
    AIR_TEMP_COL,
    BUILDING_ID_COL,
    CLOUD_COVER_COL,
    DEW_TEMP_COL,
    FEATURE_COLS,
    METER_COL,
    METER_READING_COL,
    TIMESTAMP_COL,
    WIND_SPEED_COL,
    add_cop_features,
    add_delta_t_features,
    add_load_features,
    add_rolling_features,
    add_time_features,
)
from src.scorer import Scorer  # noqa: E402


RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
OUTPUT_PATH = MODEL_DIR / "demo_readings.json"
RANDOM_STATE = 20260706
SAMPLE_SIZE = 80_000

SCENARIO_TARGETS = {
    "normal": {
        "title": "Normal operation",
        "range": (90, 100.1),
        "target": 95,
        "count": 3,
        "tier": "healthy",
        "anomaly_flag": 0,
    },
    "watch": {
        "title": "Watch trend",
        "range": (70, 90),
        "target": 78,
        "count": 3,
        "tier": "monitor",
        "anomaly_flag": 0,
    },
    "elevated": {
        "title": "Elevated concern",
        "range": (50, 70),
        "target": 60,
        "count": 3,
        "tier": "warning",
        "anomaly_flag": 0,
    },
    "high_attention": {
        "title": "High attention",
        "range": (0, 50),
        "target": 25,
        "count": 3,
        "tier": "critical",
        "anomaly_flag": 1,
    },
}

INTEGER_READING_FIELDS = {"hour_of_day", "day_of_week", "is_weekend", "month"}


def load_study_subset() -> pd.DataFrame:
    """Load chilled-water meter rows for the qualifying units only."""
    qualifying = pd.read_csv(PROCESSED_DIR / "qualifying_buildings.csv")
    building_ids = set(qualifying[BUILDING_ID_COL].astype("int16"))

    train_dtypes = {
        BUILDING_ID_COL: "int16",
        METER_COL: "int8",
        METER_READING_COL: "float32",
    }
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        RAW_DIR / "train.csv",
        dtype=train_dtypes,
        parse_dates=[TIMESTAMP_COL],
        chunksize=2_000_000,
    ):
        mask = (chunk[METER_COL] == 1) & chunk[BUILDING_ID_COL].isin(building_ids)
        if mask.any():
            chunks.append(chunk.loc[mask, [BUILDING_ID_COL, TIMESTAMP_COL, METER_READING_COL]])

    if not chunks:
        raise RuntimeError("No chilled-water readings found for qualifying buildings.")

    df = pd.concat(chunks, ignore_index=True)
    meta = pd.read_csv(
        RAW_DIR / "building_metadata.csv",
        usecols=["site_id", BUILDING_ID_COL, "primary_use", "square_feet", "year_built"],
        dtype={
            "site_id": "int8",
            BUILDING_ID_COL: "int16",
            "square_feet": "float32",
            "year_built": "float32",
        },
    )
    weather = pd.read_csv(
        RAW_DIR / "weather_train.csv",
        usecols=["site_id", TIMESTAMP_COL, AIR_TEMP_COL, DEW_TEMP_COL, WIND_SPEED_COL, CLOUD_COVER_COL],
        dtype={
            "site_id": "int8",
            AIR_TEMP_COL: "float32",
            DEW_TEMP_COL: "float32",
            WIND_SPEED_COL: "float32",
            CLOUD_COVER_COL: "float32",
        },
        parse_dates=[TIMESTAMP_COL],
    )

    df = df.merge(meta, on=BUILDING_ID_COL, how="left")
    df = df.merge(weather, on=["site_id", TIMESTAMP_COL], how="left")
    return df.sort_values([BUILDING_ID_COL, TIMESTAMP_COL]).reset_index(drop=True)


def build_feature_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Run the same feature engineering pipeline used for model training."""
    df = add_cop_features(df)
    df = add_delta_t_features(df)
    df = add_load_features(df)
    df = add_rolling_features(df)
    df = add_time_features(df)
    df = df.dropna(subset=["rolling_cop_mean_24h", "rolling_cop_mean_168h"])

    medians = df[FEATURE_COLS].median(numeric_only=True)
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(medians)
    return df.dropna(subset=FEATURE_COLS).reset_index(drop=True)


def score_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Score a deterministic sample with per-unit normalization."""
    scorer = Scorer.load(str(MODEL_DIR))
    sample_n = min(SAMPLE_SIZE, len(df))
    sample = df.sample(n=sample_n, random_state=RANDOM_STATE).copy()
    X = sample[FEATURE_COLS].astype(np.float32)
    scores = scorer.score(X, building_ids=sample[BUILDING_ID_COL])
    scores = scores.drop(columns=[BUILDING_ID_COL], errors="ignore")
    return pd.concat(
        [
            sample[[BUILDING_ID_COL, TIMESTAMP_COL]].reset_index(drop=True),
            X.reset_index(drop=True),
            scores.reset_index(drop=True),
        ],
        axis=1,
    )


def choose_scenarios(scored: pd.DataFrame) -> list[dict]:
    """Pick balanced, detector-agreed scenarios across score bands."""
    scorer = Scorer.load(str(MODEL_DIR))
    scenarios: list[dict] = []
    used_buildings: set[str] = set()

    for key, spec in SCENARIO_TARGETS.items():
        low, high = spec["range"]
        candidates = scored[
            (scored["health_score"] >= low)
            & (scored["health_score"] < high)
            & (scored["if_lof_agree"].fillna(0).astype(int) == 1)
        ].copy()
        if "anomaly_flag" in spec:
            candidates = candidates[candidates["anomaly_flag"].astype(int) == spec["anomaly_flag"]]

        if candidates.empty:
            candidates = scored[
                (scored["health_score"] >= low)
                & (scored["health_score"] < high)
            ].copy()
        if candidates.empty:
            raise RuntimeError(f"No scored candidates found for {key}.")

        candidates["distance"] = (candidates["health_score"] - spec["target"]).abs()
        candidates = candidates.sort_values(["distance", "building_id", TIMESTAMP_COL])

        selected = []
        for _, row in candidates.iterrows():
            bid = str(row[BUILDING_ID_COL])
            if bid in used_buildings and len(candidates[BUILDING_ID_COL].unique()) >= spec["count"]:
                continue

            reading = reading_from_row(row)
            features = {feature: reading[feature] for feature in FEATURE_COLS}
            result = scorer.score_single(features, building_id=reading["building_id"])
            if result["health_tier"] != spec["tier"]:
                continue
            if "anomaly_flag" in spec and int(result["anomaly_flag"]) != spec["anomaly_flag"]:
                continue
            if result.get("if_lof_agree") is not None and int(result["if_lof_agree"]) != 1:
                continue

            selected.append((row, reading, result))
            used_buildings.add(bid)
            if len(selected) == spec["count"]:
                break

        if len(selected) < spec["count"]:
            raise RuntimeError(f"Only found {len(selected)} candidates for {key}.")

        for row, reading, result in selected:
            scenarios.append(
                {
                    "id": f"{key}-{len([s for s in scenarios if s['band'] == key]) + 1}",
                    "band": key,
                    "title": spec["title"],
                    "source_unit_id": str(row[BUILDING_ID_COL]),
                    "source_timestamp": row[TIMESTAMP_COL].strftime("%Y-%m-%d %H:%M:%S"),
                    "reading": reading,
                    "expected": {
                        "health_score": round(float(result["health_score"]), 1),
                        "health_tier": str(result["health_tier"]),
                        "anomaly_flag": int(result["anomaly_flag"]),
                        "lof_flag": None if pd.isna(result.get("lof_flag")) else int(result["lof_flag"]),
                        "if_lof_agree": None if pd.isna(result.get("if_lof_agree")) else int(result["if_lof_agree"]),
                    },
                }
            )

    return scenarios


def reading_from_row(row: pd.Series) -> dict:
    """Create the exact JSON reading that the API will receive."""
    reading = {"building_id": str(row[BUILDING_ID_COL])}
    for feature in FEATURE_COLS:
        value = round(float(row[feature]), 4)
        reading[feature] = int(value) if feature in INTEGER_READING_FIELDS else value
    return reading


def main() -> None:
    df = load_study_subset()
    features = build_feature_rows(df)
    scored = score_sample(features)
    scenarios = choose_scenarios(scored)

    payload = {
        "generated": "2026-07-06",
        "source": "Raw ASHRAE chilled-water rows rebuilt with src.features",
        "scoring": "Each scenario is a complete historical feature row scored against that unit's own baseline.",
        "random_state": RANDOM_STATE,
        "scenario_count": len(scenarios),
        "bands": [
            {"id": key, "title": value["title"], "range": list(value["range"])}
            for key, value in SCENARIO_TARGETS.items()
        ],
        "scenarios": scenarios,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {OUTPUT_PATH.relative_to(ROOT)} with {len(scenarios)} scenarios.")


if __name__ == "__main__":
    main()
