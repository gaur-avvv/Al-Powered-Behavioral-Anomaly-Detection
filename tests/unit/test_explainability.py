"""
Unit and Integration tests for ExplainableAI engine.
"""

import pytest
import numpy as np
from src.explainability.explanation_engine import ExplainableAI, DEFAULT_FEATURE_NAMES


class TestExplainableAI:
    """Test suite covering SHAP and LIME feature explanations."""

    def test_shap_explanation(self):
        """Test SHAP-based explanations."""
        explainer = ExplainableAI(feature_names=DEFAULT_FEATURE_NAMES)
        features = {"geo_velocity": 150.0, "new_device": 1.0}

        explanation = explainer.explain_anomaly(features, method="shap")

        assert "shap_values" in explanation
        assert len(explanation["shap_values"]) == len(DEFAULT_FEATURE_NAMES)
        assert all(abs(v["contribution"]) <= 1.5 for v in explanation["shap_values"])

    def test_lime_explanation(self):
        """Test LIME-based explanations."""
        explainer = ExplainableAI(feature_names=DEFAULT_FEATURE_NAMES)
        features = {"geo_velocity": 150.0, "new_device": 1.0}

        explanation = explainer.explain_anomaly(features, method="lime")

        assert "lime_weights" in explanation
        assert len(explanation["lime_weights"]) == 5

    def test_consistency_across_methods(self):
        """Test consistency between SHAP and LIME top features."""
        explainer = ExplainableAI(feature_names=DEFAULT_FEATURE_NAMES)
        features = {"geo_velocity": 150.0, "new_device": 1.0, "failed_logins": 5.0}

        shap_exp = explainer.explain_anomaly(features, method="shap")
        lime_exp = explainer.explain_anomaly(features, method="lime")

        shap_top = [f["feature"] for f in shap_exp["shap_values"][:3]]
        lime_top = [f["feature"] for f in lime_exp["lime_weights"][:3]]

        common = set(shap_top) & set(lime_top)
        assert len(common) >= 1

    def test_explanation_stability(self):
        """Test explanation stability under small feature perturbations."""
        explainer = ExplainableAI(feature_names=DEFAULT_FEATURE_NAMES)
        features = {"geo_velocity": 100.0, "failed_logins": 4.0}

        base_exp = explainer.explain_anomaly(features, method="shap")
        noisy_features = {k: v * 1.01 for k, v in features.items()}
        noisy_exp = explainer.explain_anomaly(noisy_features, method="shap")

        base_top = [f["feature"] for f in base_exp["shap_values"][:2]]
        noisy_top = [f["feature"] for f in noisy_exp["shap_values"][:2]]

        assert len(set(base_top) & set(noisy_top)) >= 1
