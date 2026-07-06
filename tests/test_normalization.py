from src.scorer import Scorer


def test_raw_to_health_score_uses_unit_percentiles_and_clips():
    scorer = Scorer()
    scorer._unit_score_stats = {
        "unit-1": {
            "p5": 1.0,
            "p95": 3.0,
        }
    }

    assert scorer._raw_to_health_score(1.0, "unit-1") < 0.001
    assert scorer._raw_to_health_score(3.0, "unit-1") > 99.999
    assert scorer._raw_to_health_score(0.0, "unit-1") == 0.0
    assert scorer._raw_to_health_score(4.0, "unit-1") == 100.0


def test_raw_to_health_score_uses_global_fallback_for_unknown_unit():
    scorer = Scorer()

    assert scorer._raw_to_health_score(-0.5, "missing") == 0.0
    assert scorer._raw_to_health_score(0.0, "missing") == 100.0
    assert scorer._raw_to_health_score(0.5, "missing") == 100.0
