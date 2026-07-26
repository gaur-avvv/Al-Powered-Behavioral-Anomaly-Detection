"""
Attack Classifier Module.
Provides multi-class attack taxonomy classification for 8 behavioral cyber attack vectors
defined in the UEBA problem statement:
  1. Brute Force
  2. Impossible Travel
  3. Credential Stuffing
  4. Lateral Movement
  5. Device Spoofing
  6. Low-and-Slow Exfiltration
  7. Insider Drift
  8. Credential Misuse
  9. Normal Access Baseline
"""

from typing import Dict, List, Any
import numpy as np


ATTACK_TAXONOMY = [
    "brute_force",
    "impossible_travel",
    "credential_stuffing",
    "lateral_movement",
    "device_spoofing",
    "low_and_slow_exfiltration",
    "insider_drift",
    "credential_misuse",
    "normal"
]


class AttackClassifier:
    """Classifies anomalous entity behavior into concrete attack categories matching problem statement."""

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
        """Private helper mapping feature indicators to normalized Softmax class probabilities."""
        geo_vel = features.get("geo_velocity", features.get("geo_distance", 0.0))
        dev_change = features.get("device_fingerprint_change", features.get("new_device", 0.0))
        req_rate = features.get("request_rate", features.get("resource_access_frequency", 0.0))
        failed_logins = features.get("failed_logins", 0.0)
        session_dur_dev = features.get("session_duration_deviation", features.get("session_duration", 0.0))
        cmd_entropy = features.get("command_sequence_entropy", 0.0)
        prev_interval = features.get("previous_login_interval", 0.0)
        unusual_resource = features.get("unusual_resource_access", 0.0)

        # Unnormalized logits mapped to 8 UEBA behavioral attack classes
        logits = [
            # 1. brute_force
            failed_logins * 4.0 + (2.0 if prev_interval < 0.05 else 0.0),

            # 2. impossible_travel
            geo_vel * 5.0 + (3.0 if geo_vel > 0.8 else 0.0),

            # 3. credential_stuffing
            failed_logins * 2.5 + req_rate * 0.02 + dev_change * 1.5,

            # 4. lateral_movement
            features.get("port_scans", 0.0) * 3.0 + unusual_resource * 3.5,

            # 5. device_spoofing
            dev_change * 5.0 + (2.0 if features.get("auth_method_change", 0.0) > 0 else 0.0),

            # 6. low_and_slow_exfiltration
            features.get("exfil_bytes", 0.0) * 0.05 + (3.0 if (session_dur_dev > 0 and req_rate < 20.0 and failed_logins == 0) else 0.0),

            # 7. insider_drift
            cmd_entropy * 4.0 + features.get("admin_calls", 0.0) * 3.0,

            # 8. credential_misuse
            (2.5 if prev_interval > 0.9 else 0.0) + unusual_resource * 2.0 + (2.0 if features.get("auth_method_change", 0.0) > 0 else 0.0),

            # 9. normal
            1.0
        ]

        # Softmax normalization
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = (exp_logits / np.sum(exp_logits)).tolist()

        return probabilities
