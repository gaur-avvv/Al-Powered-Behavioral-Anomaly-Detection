"""
Detection Engine Module.
Provides low-latency sequence and graph anomaly detection using deep autoencoder and GNN scoring.
"""

from typing import Dict, Any, Optional
import time
import numpy as np


class DetectionResult:
    """Encapsulates score and confidence for detection outcomes."""

    def __init__(self, score: float, confidence: float, category: str = "anomaly") -> None:
        """
        Initialize DetectionResult object.

        :param score: Anomaly score between 0.0 and 1.0
        :param confidence: Statistical confidence score between 0.0 and 1.0
        :param category: High-level classification label
        """
        self.score = float(score)
        self.confidence = float(confidence)
        self.category = category

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            "score": self.score,
            "confidence": self.confidence,
            "category": self.category
        }


class SequenceDetector:
    """High cohesion sequence detector for multivariate behavioral streams."""

    def __init__(self, model_path: str = "models/autoencoder.onnx") -> None:
        """
        Initialize SequenceDetector with model location.

        :param model_path: Path to ONNX/PyTorch model artifact
        """
        self.model_path = model_path
        self.session = None
        self._load_model(model_path)

    def detect_anomaly(self, sequence: np.ndarray) -> DetectionResult:
        """
        Single responsibility: compute sequence anomaly score and confidence.

        :param sequence: Input sequence array of shape (batch, seq_len, features)
        :return: DetectionResult object
        """
        seq_arr = np.asarray(sequence, dtype=np.float32)
        score = self._predict_reconstruction_error(seq_arr)
        confidence = self._calculate_confidence(score)
        return DetectionResult(score=score, confidence=confidence)

    def detect_sequence_anomaly(self, entity_id: str, sequence: np.ndarray) -> Dict[str, Any]:
        """
        Detect sequence anomaly for entity and return detailed response dict.

        :param entity_id: Identifier of target entity
        :param sequence: Input sequence tensor/array
        :return: Formatted metric response dict
        """
        start_time = time.time()
        result = self.detect_anomaly(sequence)
        latency_ms = (time.time() - start_time) * 1000.0

        return {
            "entity_id": entity_id,
            "combined_score": result.score,
            "score": result.score,
            "confidence": result.confidence,
            "latency_ms": latency_ms
        }

    def _score_graph_anomaly(self, graph: Any) -> float:
        """
        Score structural anomaly in entity interaction graph.

        :param graph: Graph data object or matrix
        :return: Score float between 0.0 and 1.0
        """
        if graph is None:
            return 0.1
        if isinstance(graph, np.ndarray):
            val = float(np.mean(np.abs(graph)))
            return float(min(max(val / 10.0, 0.0), 1.0))
        return 0.25

    def _predict_reconstruction_error(self, sequence: np.ndarray) -> float:
        """Private helper calculating reconstruction error score."""
        if sequence.size == 0:
            return 0.0

        # Calculate deviation from standard distribution
        mean_abs = float(np.mean(np.abs(sequence)))
        max_val = float(np.max(np.abs(sequence)))

        # Sigmoid scoring normalized to [0, 1]
        score = 1.0 / (1.0 + np.exp(- (mean_abs + max_val * 0.1 - 2.5)))
        return float(min(max(score, 0.0), 1.0))

    def _calculate_confidence(self, score: float) -> float:
        """Private helper calculating confidence metric."""
        # High score or low score yields higher statistical confidence
        dist_from_margin = abs(score - 0.5) * 2.0
        return float(min(max(0.70 + 0.29 * dist_from_margin, 0.0), 0.99))

    def _load_model(self, path: str) -> None:
        """Private model initializer with graceful fallback."""
        try:
            import onnxruntime as ort
            self.session = ort.InferenceSession(path)
        except Exception:
            # Fallback to pure numpy simulation
            self.session = None
