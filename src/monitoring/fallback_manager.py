"""
Tiered Fallback Architecture & Mitigation Engine Module.

Implements a 4-level resilience and mitigation framework:
  Level 1: Deterministic Rule Fallback (Fail-Safe Decoupling)
  Level 2: Peer-Group Hierarchical Fallback (Cold-Start Resolution)
  Level 3: Dynamic Structural Load Shedding (Telemetry Protection)
  Level 4: Circuit-Breaker Cooldown (Drift Retraining Dampening)
"""

import time
import logging
import threading
from typing import Dict, Any, Optional

logger = logging.getLogger("TieredFallbackEngine")


class FailSafeRuleEngine:
    """
    Level 1 Fallback Strategy: Deterministic Rule Fallback (Fail-Safe Decoupling).
    Evaluates stateless heuristic security rules if PyTorch deep learning models crash.
    """

    def execute_fail_safe_rules(self, raw_log: Dict[str, Any], error_reason: str) -> Dict[str, Any]:
        """
        Executes stateless fallback rules when primary neural inference fails.

        :param raw_log: Inbound raw telemetry log dictionary
        :param error_reason: Diagnostic string explaining neural model crash
        :return: Fallback risk prediction payload
        """
        logger.warning(f"🛡️ [FAIL-SAFE TRIGGERED] Neural inference failed ({error_reason}). Routing via stateless rules.")

        auth_method = str(raw_log.get("auth_method", "")).lower()
        session_dur = float(raw_log.get("session_duration", 0.0))
        cmds = raw_log.get("command_sequence", [])
        if not isinstance(cmds, list):
            cmds = []

        is_failed_auth = (auth_method == "password" and session_dur == 0.0)
        is_suspicious_cmd = any("rm" in str(c) or "sudo" in str(c) for c in cmds)

        if is_failed_auth:
            return {
                "score": 0.85,
                "confidence": 0.95,
                "category": "credential_stuffing",
                "routing_path": "Level-1-Fail-Safe-Rules",
                "feature_attributions": {"fail_safe_trigger": "High-velocity authentication failure"}
            }

        if is_suspicious_cmd:
            return {
                "score": 0.90,
                "confidence": 0.98,
                "category": "privilege_escalation",
                "routing_path": "Level-1-Fail-Safe-Rules",
                "feature_attributions": {"fail_safe_trigger": "Privileged destructive command execution"}
            }

        return {
            "score": 0.05,
            "confidence": 0.80,
            "category": "normal",
            "routing_path": "Level-1-Fail-Safe-Rules",
            "feature_attributions": {"fail_safe_trigger": "System baseline override"}
        }


class ColdStartManager:
    """
    Level 2 Fallback Strategy: Peer-Group Hierarchical Fallback (Cold-Start Resolution).
    Evaluates novel entities with zero sequence history against group behavioral baselines.
    """

    def __init__(self) -> None:
        self.peer_group_baselines = {
            "user": {"allowed_hours": set(range(7, 19)), "max_session_duration": 480.0},
            "service_account": {"allowed_hours": set(range(0, 24)), "max_session_duration": 5.0},
            "edge_device": {"allowed_hours": set(range(0, 24)), "max_session_duration": 1440.0}
        }

    def resolve_cold_start_risk(
        self,
        entity_id: str,
        entity_type: str,
        current_hour: int,
        session_duration: float
    ) -> Dict[str, Any]:
        """
        Evaluates an unknown entity against aggregate peer group limits.

        :param entity_id: Novel entity identifier
        :param entity_type: Categorical entity type string
        :param current_hour: Ingest hour (0 - 23)
        :param session_duration: Connection length in minutes
        :return: Cold-start peer group risk assessment
        """
        etype = str(entity_type).lower()
        group = self.peer_group_baselines.get(etype, self.peer_group_baselines["user"])

        is_hour_viol = current_hour not in group["allowed_hours"]
        is_dur_viol = session_duration > group["max_session_duration"]

        if is_hour_viol or is_dur_viol:
            score = 0.70
            category = "privilege_escalation" if is_hour_viol else "data_exfiltration"
        else:
            score = 0.10
            category = "normal"

        return {
            "score": score,
            "confidence": 0.82,
            "category": category,
            "is_cold_start": True,
            "routing_path": "Level-2-Peer-Group-Baseline"
        }


class LoadSheddingManager:
    """
    Level 3 Fallback Strategy: Dynamic Structural Load Shedding (Telemetry Protection).
    Monitors queue depth; disables heavy SHAP explainability loops and downsamples benign traffic.
    """

    def __init__(self, queue_threshold: int = 5000, sample_rate: float = 0.10) -> None:
        self.queue_threshold = queue_threshold
        self.sample_rate = sample_rate
        self.explainability_active = True

    def check_load_shedding(self, current_queue_size: int) -> bool:
        """
        Evaluate queue size against safety threshold and toggle SHAP explainability.

        :param current_queue_size: Current depth of INGESTION_QUEUE
        :return: True if load shedding active (SHAP disabled)
        """
        if current_queue_size > self.queue_threshold:
            if self.explainability_active:
                logger.warning(
                    f"⚠️ [LOAD SHEDDING ACTIVE] Ingestion queue depth ({current_queue_size}) > {self.queue_threshold}. "
                    "Disabling SHAP explainability compute loops to preserve throughput."
                )
                self.explainability_active = False
            return True

        if not self.explainability_active and current_queue_size < (self.queue_threshold // 2):
            logger.info("✅ [LOAD SHEDDING CLEARED] Ingestion queue normalized. Re-enabling SHAP explainability.")
            self.explainability_active = True

        return False

    def should_sample_benign_log(self, raw_score: float) -> bool:
        """Determines whether benign log should be downsampled during load shedding."""
        if not self.explainability_active and raw_score < 0.30:
            import random
            return random.random() > self.sample_rate  # Drop 90% of benign logs
        return False


class RetrainingCircuitBreaker:
    """
    Level 4 Fallback Strategy: Circuit-Breaker for Retraining Loops (Drift Dampening).
    Locks background retraining triggers for a cooldown window to prevent thrashing.
    """

    def __init__(self, cooldown_seconds: float = 3600.0) -> None:
        self.cooldown_seconds = cooldown_seconds
        self.last_retrain_time = 0.0
        self._lock = threading.Lock()

    def can_trigger_retrain(self) -> bool:
        """
        Check if retraining loop is permitted under circuit breaker cooldown lock.

        :return: True if retrain allowed, False if in cooldown
        """
        with self._lock:
            now = time.time()
            elapsed = now - self.last_retrain_time
            if elapsed < self.cooldown_seconds:
                logger.info(f"🔒 [CIRCUIT BREAKER LOCK] Retraining cooldown active ({self.cooldown_seconds - elapsed:.1f}s remaining). Trigger suppressed.")
                return False
            return True

    def record_retrain_completion(self) -> None:
        """Record timestamp of completed background retraining sequence."""
        with self._lock:
            self.last_retrain_time = time.time()
            logger.info("⏱️ [CIRCUIT BREAKER TIMER RESET] Retraining completed. Cooldown lock engaged.")


if __name__ == "__main__":
    rule_engine = FailSafeRuleEngine()
    cold_start = ColdStartManager()
    load_shedder = LoadSheddingManager(queue_threshold=100)
    circuit_breaker = RetrainingCircuitBreaker(cooldown_seconds=10.0)

    print("--- 1. Testing Fail-Safe Rule Fallback ---")
    res1 = rule_engine.execute_fail_safe_rules({"auth_method": "password", "session_duration": 0.0}, "CUDA OOM")
    print(res1)

    print("\n--- 2. Testing Peer Group Cold Start ---")
    res2 = cold_start.resolve_cold_start_risk("E_999", "user", current_hour=3, session_duration=600.0)
    print(res2)

    print("\n--- 3. Testing Load Shedding ---")
    shed = load_shedder.check_load_shedding(150)
    print(f"Load shedding active: {shed}, Explainability active: {load_shedder.explainability_active}")

    print("\n--- 4. Testing Circuit Breaker ---")
    print(f"Can trigger retrain: {circuit_breaker.can_trigger_retrain()}")
    circuit_breaker.record_retrain_completion()
    print(f"Can trigger retrain right after: {circuit_breaker.can_trigger_retrain()}")
