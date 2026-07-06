from src.scorer import score_to_tier


def test_score_to_tier_boundaries():
    assert score_to_tier(90) == "healthy"
    assert score_to_tier(89.9) == "monitor"
    assert score_to_tier(70) == "monitor"
    assert score_to_tier(69.9) == "warning"
    assert score_to_tier(50) == "warning"
    assert score_to_tier(49.9) == "critical"
