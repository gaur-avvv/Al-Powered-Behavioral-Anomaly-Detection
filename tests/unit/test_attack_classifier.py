"""
Unit tests for AttackClassifier.
"""

import pytest
from src.models.attack_classifier import AttackClassifier


class TestAttackClassifier:
    """Test suite for attack category classification."""

    def test_classification_accuracy(self):
        """Test primary category assignment for failed logins."""
        classifier = AttackClassifier("models/lightgbm_classifier.pkl")

        features = {"failed_logins": 10.0, "geo_velocity": 200.0}
        res = classifier.classify_anomaly(features)

        assert res["primary_category"] in classifier.categories
        assert res["confidence"] > 0.0

    def test_top_k_categories(self):
        """Test top-K category retrieval structure."""
        classifier = AttackClassifier("models/lightgbm_classifier.pkl")

        features = {"new_device": 1.0, "admin_calls": 5.0}
        result = classifier.classify_anomaly(features)

        assert len(result["top_categories"]) == 3
        assert all(0.0 <= cat["confidence"] <= 1.0 for cat in result["top_categories"])
