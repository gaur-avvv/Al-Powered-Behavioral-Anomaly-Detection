"""
Unit tests for PerformanceReport generator.
"""

import pytest
from src.report.report_generator import PerformanceReport


class TestPerformanceReport:
    """Test suite covering operational performance report metrics, assumptions, and limitations."""

    def test_report_generation(self):
        """Test full report generation dictionary structure."""
        report_gen = PerformanceReport({
            'autoencoder': 'models/autoencoder.onnx',
            'classifier': 'models/lightgbm_classifier.pkl'
        })

        report = report_gen.generate_full_report()

        assert "report_date" in report
        assert "metrics" in report
        assert "model_performance" in report
        assert "assumptions" in report

        metrics = report["metrics"]["detection_metrics"]
        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"] <= 1.0
        assert 0.0 <= metrics["f1_score"] <= 1.0

    def test_assumptions_validation(self):
        """Test assumptions list contains required domain constraints."""
        report_gen = PerformanceReport({
            'autoencoder': 'models/autoencoder.onnx',
            'classifier': 'models/lightgbm_classifier.pkl'
        })

        assumptions = report_gen._get_assumptions()

        assert len(assumptions) == 6
        assert all(isinstance(a, str) for a in assumptions)
        assert "Entity behavior is stationary" in assumptions
        assert "No malicious insiders" in assumptions

    def test_limitations_identification(self):
        """Test limitations list contains required system edge cases."""
        report_gen = PerformanceReport({
            'autoencoder': 'models/autoencoder.onnx',
            'classifier': 'models/lightgbm_classifier.pkl'
        })

        limitations = report_gen._list_limitations()

        assert len(limitations) == 8
        assert "Cold start problem" in limitations
        assert "Concept drift" in limitations
        assert "High false positives" in limitations
