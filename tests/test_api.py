from fastapi.testclient import TestClient

from api.main import app


VALID_READING = {
    "building_id": "7",
    "cop_proxy": 3.0,
    "delta_t_supply_proxy": 9.0,
    "delta_t_refrigerant_proxy": 17.0,
    "load_ratio": 0.7,
    "air_temperature": 20.0,
    "hour_of_day": 12,
}


def test_health_includes_tiers_and_snapshot_flag():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["tiers"] == {"healthy": 90, "monitor": 70, "warning": 50}
    assert data["fleet_snapshot"] is True
    assert data["scorer_loaded"] is True
    assert data["feature_count"] == 16


def test_units_counts_match_unit_list():
    with TestClient(app) as client:
        response = client.get("/units")

    assert response.status_code == 200
    data = response.json()
    units = data["units"]
    assert data["total"] == len(units)
    assert data["n_critical"] == sum(u["health_tier"] == "critical" for u in units)
    assert data["n_warning"] == sum(u["health_tier"] == "warning" for u in units)
    assert data["n_monitor"] == sum(u["health_tier"] == "monitor" for u in units)
    assert data["n_healthy"] == sum(u["health_tier"] == "healthy" for u in units)
    assert data["snapshot_generated"] == "2026-06-11"


def test_score_accepts_valid_reading_and_rejects_malformed_reading():
    with TestClient(app) as client:
        valid = client.post("/score?shap=false", json=VALID_READING)
        malformed = client.post("/score?shap=false", json={"building_id": "7"})

    assert valid.status_code == 200
    assert 0 <= valid.json()["health_score"] <= 100
    assert malformed.status_code == 422


def test_score_accepts_numeric_building_id_from_units():
    reading = {**VALID_READING, "building_id": 1084}

    with TestClient(app) as client:
        response = client.post("/score?shap=false", json=reading)

    assert response.status_code == 200
    assert response.json()["building_id"] == "1084"


def test_score_batch_returns_one_result_per_input():
    with TestClient(app) as client:
        response = client.post("/score/batch?shap=false", json=[VALID_READING, VALID_READING])

    assert response.status_code == 200
    assert len(response.json()) == 2
