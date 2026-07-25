"""
Attack Classifier Module.
Provides multi-class attack taxonomy classification for detected anomalies.
"""

from typing import Dict, List, Any
import numpy as np


ATTACK_TAXONOMY = [
    "credential_stuffing",
    "data_exfiltration",
    "privilege_escalation",
    "ddos_flooding",
    "lateral_movement",
    "normal"
]


class AttackClassifier:
    """Classifies anomalous entity behavior into concrete attack categories."""

    def __init__(self, model_path: str = "models/lightgbm_classifier.pkl") -> None:
        """
        Initialize AttackClassifier with model artifact path.

        :param model_path: Path to serialized classification model
        """
        self.model_path = model_path
        self.categories = ATTACK_TAXONOMY

    def classify_anomaly(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Classify feature dictionary into attack categories with top-K confidences.

        :param features: Map of feature names to scalar numerical values
        :return: Classification result containing primary category and top_categories list
        """
        probs = self._compute_category_probabilities(features)
        
        # Sort category probabilities descending
        sorted_pairs = sorted(zip(self.categories, probs), key=lambda x: x[1], reverse=True)
        
        primary_category = sorted_pairs[0][0]
        top_k = [
            {"category": cat, "confidence": float(prob)}
            for cat, prob in sorted_pairs[:3]
        ]

        return {
            "primary_category": primary_category,
            "category": primary_category,
            "confidence": float(sorted_pairs[0][1]),
            "top_categories": top_k
        }

    def _compute_category_probabilities(self, features: Dict[str, float]) -> List[float]:
        """Private helper mapping feature indicators to normalized class probabilities."""
        geo_vel = features.get("geo_velocity", 0.0)
        new_dev = features.get("new_device", 0.0)
        req_rate = features.get("request_rate", 0.0)
        failed_logins = features.get("failed_logins", 0.0)

        # Baseline unnormalized logits
        logits = [
            failed_logins * 2.5 + geo_vel * 0.01,  # credential_stuffing
            features.get("exfil_bytes", 0.0) * 0.05 + 0.1,  # data_exfiltration
            new_dev * 3.0 + features.get("admin_calls", 0.0) * 2.0,  # privilege_escalation
            req_rate * 0.1,  # ddos_flooding
            features.get("port_scans", 0.0) * 1.5,  # lateral_movement
            1.0  # normal baseline
        ]

        # Softmax normalization
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = (exp_logits / np.sum(exp_logits)).tolist()

        return probabilities
