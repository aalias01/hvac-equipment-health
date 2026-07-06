import json
from pathlib import Path

import pytest

from src.scorer import Scorer, score_to_tier


MODEL_DIR = Path("models")


@pytest.mark.skipif(
    not (MODEL_DIR / "isolation_forest.joblib").exists(),
    reason="model artifacts are not present",
)
def test_committed_scorer_artifacts_roundtrip():
    meta = json.loads((MODEL_DIR / "scorer_meta.json").read_text())
    scorer = Scorer.load(str(MODEL_DIR))
    features = {
        "cop_proxy": 3.0,
        "delta_t_supply_proxy": 9.0,
        "delta_t_refrigerant_proxy": 17.0,
        "load_ratio": 0.7,
        "rolling_cop_mean_24h": 3.0,
        "rolling_cop_std_24h": 0.1,
        "rolling_load_mean_24h": 0.7,
        "rolling_cop_mean_168h": 3.0,
        "cop_deviation_from_baseline": 0.0,
        "air_temperature": 20.0,
        "dew_temperature": 15.0,
        "wind_speed": 2.0,
        "hour_of_day": 12,
        "day_of_week": 2,
        "is_weekend": 0,
        "month": 6,
    }

    result = scorer.score_single(features, building_id="7")
    factors = scorer.top_shap_factors(features, top_n=5)

    assert 0 <= result["health_score"] <= 100
    assert result["health_tier"] == score_to_tier(result["health_score"])
    assert len(factors) == 5
    assert len(scorer.feature_names) == len(meta["feature_names"])
    assert len(scorer.feature_names) == 16
    for factor in factors:
        expected = "worsens_health" if factor["shap_value"] > 0 else "improves_health"
        assert factor["direction"] == expected
