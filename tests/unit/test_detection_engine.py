"""
Unit tests for SequenceDetector.
"""

import pytest
import numpy as np
from src.models.detection_engine import SequenceDetector


class TestSequenceDetector:
    """Test suite covering sequence and graph anomaly detection."""

    def test_sequence_anomaly_detection(self):
        """Test anomaly detection on normal vs anomalous synthetic sequence data."""
        detector = SequenceDetector("models/autoencoder.onnx")

        # Normal sequence with low values
        normal_sequence = np.zeros((1, 10, 5), dtype=np.float32)
        result = detector.detect_sequence_anomaly("entity_001", normal_sequence)
        assert result["combined_score"] < 0.35

        # Anomalous sequence with high values
        anomalous_sequence = np.ones((1, 10, 5), dtype=np.float32) * 50.0
        result = detector.detect_sequence_anomaly("entity_001", anomalous_sequence)
        assert result["combined_score"] > 0.65

    def test_graph_anomaly_detection(self):
        """Test graph-based anomaly detection score range."""
        detector = SequenceDetector("models/gnn.onnx")

        graph = np.eye(5, dtype=np.float32) * 2.0
        score = detector._score_graph_anomaly(graph)

        assert 0.0 <= score <= 1.0
