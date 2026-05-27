"""
src/scorer.py — Health score computation and anomaly detection for HVAC units.

Pipeline:
    Feature matrix (from src/features.py)
        → Isolation Forest (primary anomaly detector)
        → LOF comparison
        → Anomaly score → 0–100 health score (inverted, per-unit normalized)
        → SHAP explanations (which sensors drive this unit's score)

Health score interpretation:
    90–100: Healthy — normal operating range
    70–89:  Monitor — slightly degraded efficiency, watch trends
    50–69:  Warning — investigate; likely declining COP or ΔT drift
    0–49:   Critical — anomalous operating point; schedule inspection

Usage:
    from src.scorer import Scorer
    scorer = Scorer()
    scorer.fit(X_train)
    results = scorer.score(X_new, building_ids=ids)
    scorer.save("models/")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Health score thresholds
# ---------------------------------------------------------------------------

THRESHOLDS = {
    "healthy":  (90, 100),
    "monitor":  (70, 89),
    "warning":  (50, 69),
    "critical": (0,  49),
}

def score_to_tier(score: float) -> str:
    if score >= 90: return "healthy"
    if score >= 70: return "monitor"
    if score >= 50: return "warning"
    return "critical"


# ---------------------------------------------------------------------------
# Scorer class
# ---------------------------------------------------------------------------

class Scorer:
    """
    HVAC unit health scorer: Isolation Forest + LOF + 0–100 health gauge.

    Args:
        contamination: expected fraction of anomalous operating points (default 0.05)
            Set based on industry rule of thumb: ~5% of readings are genuinely
            anomalous. Validated against physical outliers in EDA.
        n_estimators: number of trees in Isolation Forest (default 200)
        use_lof: also train a LOF model for comparison (default True)
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 200,
        use_lof: bool = True,
    ):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.use_lof = use_lof

        self.scaler = StandardScaler()
        self.iforest = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=42,
            n_jobs=-1,
        )
        self.lof: Optional[LocalOutlierFactor] = (
            LocalOutlierFactor(n_neighbors=20, contamination=contamination, novelty=True)
            if use_lof else None
        )
        self.feature_names: list[str] = []
        self._unit_score_stats: dict[str, dict] = {}  # per-unit score normalization
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, building_ids: Optional[pd.Series] = None) -> "Scorer":
        """
        Fit the scaler, Isolation Forest, and (optionally) LOF.

        Args:
            X: feature matrix from src.features.get_feature_matrix()
            building_ids: optional Series of building_id values aligned with X
                If provided, computes per-unit score baselines for normalization.
        """
        self.feature_names = list(X.columns)
        X_scaled = self.scaler.fit_transform(X)

        self.iforest.fit(X_scaled)
        if self.lof is not None:
            self.lof.fit(X_scaled)

        # Compute raw anomaly scores on training data for per-unit normalization
        raw_scores = self.iforest.decision_function(X_scaled)  # higher = more normal

        if building_ids is not None:
            temp = pd.DataFrame({"raw_score": raw_scores, "building_id": building_ids.values})
            for bid, grp in temp.groupby("building_id"):
                self._unit_score_stats[str(bid)] = {
                    "mean": float(grp["raw_score"].mean()),
                    "std":  float(grp["raw_score"].std() + 1e-6),
                    "p5":   float(grp["raw_score"].quantile(0.05)),
                    "p95":  float(grp["raw_score"].quantile(0.95)),
                }

        self.is_fitted = True
        n_flagged = (self.iforest.predict(X_scaled) == -1).sum()
        print(f"Scorer fitted: {len(X):,} samples, {n_flagged} anomalies ({n_flagged/len(X)*100:.1f}%)")
        return self

    def _raw_to_health_score(
        self,
        raw_score: float,
        building_id: Optional[str] = None,
    ) -> float:
        """
        Convert Isolation Forest decision_function score to 0–100 health score.

        The decision_function returns higher values for more normal points.
        We invert and normalize: low anomaly score → high health score.

        If unit-level stats exist, use per-unit normalization (better for
        units with systematically different operating profiles).
        """
        if building_id and str(building_id) in self._unit_score_stats:
            stats = self._unit_score_stats[str(building_id)]
            # Normalize relative to unit's own range
            normalized = (raw_score - stats["p5"]) / (stats["p95"] - stats["p5"] + 1e-6)
        else:
            # Global normalization
            normalized = (raw_score + 0.5) / 0.5  # IF scores typically in [-0.5, 0.5]

        # Clip and convert to 0–100 (higher = healthier)
        health = float(np.clip(normalized * 100, 0, 100))
        return health

    def score(
        self,
        X: pd.DataFrame,
        building_ids: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Score new data. Returns a DataFrame with health scores and anomaly flags.

        Returns columns:
            building_id (if provided), health_score (0–100), health_tier,
            anomaly_flag (1 = anomalous), iforest_score, lof_flag (if fitted)
        """
        if not self.is_fitted:
            raise RuntimeError("Scorer not fitted. Call fit() first.")

        X_scaled = self.scaler.transform(X)
        raw_scores = self.iforest.decision_function(X_scaled)
        if_preds = self.iforest.predict(X_scaled)   # 1=normal, -1=anomaly

        results = []
        for i, (raw, pred) in enumerate(zip(raw_scores, if_preds)):
            bid = str(building_ids.iloc[i]) if building_ids is not None else None
            health = self._raw_to_health_score(raw, bid)
            results.append({
                "building_id": bid,
                "health_score": round(health, 1),
                "health_tier": score_to_tier(health),
                "anomaly_flag": int(pred == -1),
                "iforest_score": round(float(raw), 4),
            })

        out = pd.DataFrame(results)

        # LOF comparison (optional)
        if self.lof is not None:
            lof_preds = self.lof.predict(X_scaled)
            out["lof_flag"] = (lof_preds == -1).astype(int)
            out["if_lof_agree"] = (out["anomaly_flag"] == out["lof_flag"]).astype(int)

        return out

    def score_single(
        self,
        x: dict,
        building_id: Optional[str] = None,
    ) -> dict:
        """
        Score a single data point (dict of feature_name → value).
        Used by the FastAPI /score endpoint.
        """
        row = pd.DataFrame([x])[self.feature_names].fillna(0)
        result_df = self.score(row, pd.Series([building_id]) if building_id else None)
        return result_df.iloc[0].to_dict()

    # ---------------------------------------------------------------------------
    # SHAP explanations
    # ---------------------------------------------------------------------------

    def explain(self, X: pd.DataFrame, max_display: int = 6) -> pd.DataFrame:
        """
        Compute SHAP values for the Isolation Forest model.

        Returns a DataFrame of shape (n_samples, n_features) with SHAP values.
        Positive SHAP = feature pushes toward anomaly (lowers health score).
        Negative SHAP = feature pushes toward normal (raises health score).
        """
        import shap
        X_scaled = self.scaler.transform(X)
        explainer = shap.TreeExplainer(self.iforest)
        shap_values = explainer.shap_values(X_scaled)
        return pd.DataFrame(shap_values, columns=self.feature_names, index=X.index)

    def top_shap_factors(
        self,
        x: dict,
        top_n: int = 5,
    ) -> list[dict]:
        """
        Return the top-n SHAP factors for a single prediction.
        Used by the FastAPI /score endpoint for explainability.
        """
        row = pd.DataFrame([x])[self.feature_names].fillna(0)
        shap_df = self.explain(row)
        row_shap = shap_df.iloc[0].abs().sort_values(ascending=False)
        factors = []
        for feat in row_shap.index[:top_n]:
            val = shap_df.iloc[0][feat]
            factors.append({
                "feature": feat,
                "shap_value": round(float(val), 4),
                "direction": "worsens_health" if val > 0 else "improves_health",
                "feature_value": round(float(x.get(feat, 0)), 4),
            })
        return factors

    # ---------------------------------------------------------------------------
    # Persistence
    # ---------------------------------------------------------------------------

    def save(self, model_dir: str = "models/") -> None:
        path = Path(model_dir)
        path.mkdir(exist_ok=True)
        joblib.dump(self.scaler,  path / "isolation_forest_scaler.joblib")
        joblib.dump(self.iforest, path / "isolation_forest.joblib")
        if self.lof is not None:
            joblib.dump(self.lof, path / "lof_model.joblib")
        meta = {
            "feature_names": self.feature_names,
            "contamination": self.contamination,
            "n_estimators": self.n_estimators,
            "use_lof": self.use_lof,
            "unit_score_stats": self._unit_score_stats,
            "thresholds": THRESHOLDS,
        }
        (path / "scorer_meta.json").write_text(__import__("json").dumps(meta, indent=2))
        print(f"Scorer saved to {path}")

    @classmethod
    def load(cls, model_dir: str = "models/") -> "Scorer":
        path = Path(model_dir)
        meta = __import__("json").loads((path / "scorer_meta.json").read_text())
        scorer = cls(
            contamination=meta["contamination"],
            n_estimators=meta["n_estimators"],
            use_lof=meta["use_lof"],
        )
        scorer.scaler         = joblib.load(path / "isolation_forest_scaler.joblib")
        scorer.iforest        = joblib.load(path / "isolation_forest.joblib")
        scorer.feature_names  = meta["feature_names"]
        scorer._unit_score_stats = meta.get("unit_score_stats", {})
        if meta["use_lof"] and (path / "lof_model.joblib").exists():
            scorer.lof = joblib.load(path / "lof_model.joblib")
        scorer.is_fitted = True
        return scorer
