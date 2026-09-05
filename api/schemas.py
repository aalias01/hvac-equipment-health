"""
api/schemas.py: Pydantic models for building-level HVAC anomaly scoring.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class SensorReading(BaseModel):
    """
    A single processed meter/weather feature row for one building system.

    In a production BMS integration, readings arrive at 15-minute or 1-hour intervals.
    For the Kaggle/demo setup, readings are constructed from notebook-processed features.
    """
    building_id: Optional[str] = Field(
        default=None,
        description="Unit/building identifier. If provided, enables per-unit normalization.",
        examples=["building_001"],
    )

    @field_validator("building_id", mode="before")
    @classmethod
    def coerce_building_id(cls, value):
        """Accept numeric IDs returned by /units while storing a stable string ID."""
        if value is None:
            return None
        return str(value)

    # Primary engineered features (required for meaningful score)
    cop_proxy: float = Field(..., description="Weather-normalized chilled-water demand proxy; not measured COP", ge=0, le=10)
    delta_t_supply_proxy: float = Field(
        ..., description="Outdoor-to-fixed-setpoint proxy; no supply/return sensors", ge=0
    )
    delta_t_refrigerant_proxy: float = Field(
        ..., description="Outdoor-air to dew-point proxy; no refrigerant sensors", ge=0
    )
    load_ratio: float = Field(
        ..., description="Meter demand / square-footage-based capacity estimate", ge=0, le=2.0
    )

    # Rolling features (optional — set to 0 if not available)
    rolling_cop_mean_24h: Optional[float] = Field(default=None, ge=0)
    rolling_cop_std_24h: Optional[float] = Field(default=None, ge=0)
    rolling_load_mean_24h: Optional[float] = Field(default=None, ge=0)
    rolling_cop_mean_168h: Optional[float] = Field(default=None, ge=0)
    cop_deviation_from_baseline: Optional[float] = Field(default=None)

    # Weather context (optional)
    air_temperature: Optional[float] = Field(default=None, description="Outdoor air temp (°C)")
    dew_temperature: Optional[float] = Field(default=None, description="Dew point temp (°C)")
    wind_speed: Optional[float] = Field(default=None, ge=0)

    # Time features (optional — hour of day 0–23, day of week 0–6)
    hour_of_day: Optional[int] = Field(default=None, ge=0, le=23)
    day_of_week: Optional[int] = Field(default=None, ge=0, le=6)
    is_weekend: Optional[int] = Field(default=None, ge=0, le=1)
    month: Optional[int] = Field(default=None, ge=1, le=12)

    model_config = {
        "json_schema_extra": {
            "example": {
                "building_id": "building_001",
                "cop_proxy": 3.2,
                "delta_t_supply_proxy": 9.5,
                "delta_t_refrigerant_proxy": 18.0,
                "load_ratio": 0.72,
                "rolling_cop_mean_24h": 3.1,
                "rolling_cop_std_24h": 0.15,
                "rolling_load_mean_24h": 0.68,
                "rolling_cop_mean_168h": 3.05,
                "cop_deviation_from_baseline": 0.15,
                "air_temperature": 28.5,
                "dew_temperature": 18.0,
                "wind_speed": 3.2,
                "hour_of_day": 14,
                "day_of_week": 2,
                "is_weekend": 0,
                "month": 7,
            }
        }
    }


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class SHAPFactor(BaseModel):
    """A single SHAP explanation factor for the relative anomaly score."""
    feature: str
    shap_value: float = Field(..., description="Signed contribution to the relative score")
    direction: str = Field(
        ...,
        description="Legacy direction label for API compatibility; describes score movement, not equipment condition",
        examples=["worsens_health", "improves_health"],
    )
    feature_value: float


class ScoreResponse(BaseModel):
    """Relative anomaly-score response for one building system."""
    building_id: Optional[str] = None
    health_score: float = Field(
        ...,
        ge=0, le=100,
        description="0–100 rank-relative display score; presentation bands are not fault classes",
    )
    health_tier: str = Field(
        ...,
        description="Legacy presentation-band name; not a fault or condition class",
        examples=["healthy", "monitor", "warning", "critical"],
    )
    anomaly_flag: int = Field(..., description="1 = anomalous operating point, 0 = normal")
    iforest_score: float = Field(..., description="Raw Isolation Forest decision function score")
    lof_flag: Optional[int] = Field(default=None, description="1 = LOF anomaly (if model loaded)")
    if_lof_agree: Optional[int] = Field(default=None, description="1 = IF and LOF agree")
    top_shap_factors: Optional[list[SHAPFactor]] = Field(
        default=None,
        description="Top SHAP factors explaining this building's relative score",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "building_id": "building_001",
                "health_score": 84.2,
                "health_tier": "monitor",
                "anomaly_flag": 0,
                "iforest_score": 0.042,
                "lof_flag": 0,
                "if_lof_agree": 1,
                "top_shap_factors": [
                    {
                        "feature": "cop_deviation_from_baseline",
                        "shap_value": 0.021,
                        "direction": "worsens_health",
                        "feature_value": -0.15,
                    }
                ],
            }
        }
    }


class UnitListResponse(BaseModel):
    """Study-snapshot summaries sorted by legacy display score, low first."""
    units: list[dict]
    n_critical: int
    n_warning: int
    n_monitor: int
    n_healthy: int
    total: int
    snapshot_generated: Optional[str] = None


class DemoBand(BaseModel):
    """Score band included in the curated demo scenario set."""
    id: str
    title: str
    range: list[float]


class DemoExpectedScore(BaseModel):
    """Expected score metadata captured when the demo scenario was curated."""
    health_score: float
    health_tier: str
    anomaly_flag: int
    lof_flag: Optional[int] = None
    if_lof_agree: Optional[int] = None


class DemoScenario(BaseModel):
    """A complete historical reading used by the frontend demo."""
    id: str
    band: str
    title: str
    source_unit_id: str
    source_timestamp: str
    reading: SensorReading
    expected: DemoExpectedScore


class DemoScenariosResponse(BaseModel):
    """Curated demo readings sampled from the study distribution."""
    generated: str
    source: str
    scoring: str
    random_state: int
    scenario_count: int
    bands: list[DemoBand]
    scenarios: list[DemoScenario]


class HealthResponse(BaseModel):
    """API health check."""
    status: str
    scorer_loaded: bool
    feature_count: int
    contamination: float
    tiers: dict[str, int]
    fleet_snapshot: bool
    version: str = "0.1.0"
