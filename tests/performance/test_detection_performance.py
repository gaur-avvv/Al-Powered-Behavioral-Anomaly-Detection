"""
Performance and Latency benchmark tests for SequenceDetector.
"""

import pytest
import time
import numpy as np
from src.models.detection_engine import SequenceDetector


class TestDetectionPerformance:
    """Benchmark test suite for sub-100ms detection latency enforcement."""

    def test_detection_latency_p99(self):
        """Test average latency < 50ms and P99 latency < 100ms."""
        detector = SequenceDetector("models/autoencoder.onnx")

        # Generate 100 test sequences
        sequences = [np.random.normal(0, 1, (1, 10, 5)) for _ in range(100)]

        latencies = []
        for seq in sequences:
            start = time.time()
            detector.detect_sequence_anomaly("test_entity", seq)
            elapsed_ms = (time.time() - start) * 1000.0
            latencies.append(elapsed_ms)

        avg_latency = float(np.mean(latencies))
        p99_latency = float(np.percentile(latencies, 99))

        assert avg_latency < 50.0  # Average latency under 50ms
        assert p99_latency < 100.0  # P99 latency under 100ms
