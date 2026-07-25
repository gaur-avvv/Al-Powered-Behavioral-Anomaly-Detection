"""
Attack Classifier Module.
Provides multi-class attack taxonomy classification for detected anomalies,
mapping synthetic attack vectors directly to explicit threat taxonomy categories,
and tracking autonomous LLM plugin / agent access anomalies.
"""

from typing import Dict, List, Any
import numpy as np


ATTACK_TAXONOMY = [
    "credential_stuffing",   # Core Auth: Brute Force & Credential Stuffing
    "data_exfiltration",     # Exfiltration: Low-and-slow & bulk data export
    "privilege_escalation",  # Escalation: Insider drift & admin tool usage
    "ddos_flooding",         # Flooding: High-frequency API request bursts
    "lateral_movement",      # Lateral: Unusual SSH / resource scan paths
    "llm_agent_anomaly",     # Autonomous LLM plugin & enterprise agent anomaly
    "normal"                 # Benign baseline
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
        """Private helper mapping feature indicators to normalized Softmax class probabilities."""
        geo_vel = features.get("geo_velocity", 0.0)
        new_dev = features.get("new_device", 0.0)
        req_rate = features.get("request_rate", 0.0)
        failed_logins = features.get("failed_logins", 0.0)
        session_dur = features.get("session_duration", 0.0)
        is_llm = features.get("llm_agent_flag", 0.0)

        # Explicit mapping of feature signals to threat taxonomy logits
        logits = [
            # 1. credential_stuffing (Brute Force & Credential Stuffing)
            failed_logins * 3.5 + (1.0 if req_rate > 150.0 else 0.0) * 2.0,

            # 2. data_exfiltration (Low-and-slow & Data Exfiltration)
            features.get("exfil_bytes", 0.0) * 0.05 + (1.0 if (session_dur > 0 and session_dur < 10.0 and req_rate < 30.0 and failed_logins == 0) else 0.0) * 4.0,

            # 3. privilege_escalation (Insider Drift & Admin Escalation)
            new_dev * 2.0 + features.get("admin_calls", 0.0) * 3.0 + (1.0 if (session_dur > 40.0 and req_rate > 80.0) else 0.0) * 2.5,

            # 4. ddos_flooding (High-Frequency Request Flooding)
            (req_rate / 50.0) * 2.0,

            # 5. lateral_movement (Unusual Resource Sequence / Port Scan)
            features.get("port_scans", 0.0) * 2.0 + (1.0 if (geo_vel > 50.0 and req_rate > 180.0) else 0.0) * 3.0,

            # 6. llm_agent_anomaly (Enterprise Agent / LLM Plugin Anomaly Track)
            is_llm * 5.0 + features.get("prompt_injection_score", 0.0) * 4.0,

            # 7. normal (Benign baseline)
            1.0
        ]

        # Softmax normalization: P(Y = k | x) = exp(w_k^T x + b_k) / sum_j exp(w_j^T x + b_j)
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = (exp_logits / np.sum(exp_logits)).tolist()

        return probabilities
