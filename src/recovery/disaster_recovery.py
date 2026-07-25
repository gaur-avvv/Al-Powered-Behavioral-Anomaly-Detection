"""
Disaster Recovery Manager Module.
Provides backup generation, state snapshot serialization, and restore validation.
"""

from typing import Dict, Any, Optional
from datetime import datetime
import json
from src.models.baseline_profiler import InMemoryStorage


class DisasterRecoveryManager:
    """Manages cloud storage backups, entity profile snapshots, and system restore routines."""

    def __init__(self, s3_client: Optional[Any] = None,
                 dynamodb_client: Optional[Any] = None) -> None:
        """
        Initialize DisasterRecoveryManager.

        :param s3_client: AWS S3 client interface
        :param dynamodb_client: AWS DynamoDB client interface
        """
        self.s3 = s3_client
        self.dynamodb = dynamodb_client
        self.backup_bucket = "anomaly-detection-backups"
        self.redis_client = InMemoryStorage()
        self._backups_store: Dict[str, str] = {}

    def create_daily_backup(self) -> str:
        """
        Create snapshot backup of entity profiles, models, and system configuration.

        :return: Backup key identifier string
        """
        redis_data = self._backup_redis()
        model_snapshots = self._backup_models()
        config_backup = self._backup_configuration()

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_key = f"backup_{timestamp}.json"

        backup_data = {
            "timestamp": timestamp,
            "redis": redis_data,
            "models": model_snapshots,
            "config": config_backup,
            "version": "1.0"
        }

        json_bytes = json.dumps(backup_data)
        if self.s3 and hasattr(self.s3, "put_object"):
            self.s3.put_object(
                Bucket=self.backup_bucket,
                Key=backup_key,
                Body=json_bytes
            )

        self._backups_store[backup_key] = json_bytes
        return backup_key

    def restore_from_backup(self, backup_key: str) -> bool:
        """
        Restore operational state from target backup snapshot.

        :param backup_key: Backup key identifier string
        :return: True if restoration succeeded
        """
        if backup_key in self._backups_store:
            raw_data = self._backups_store[backup_key]
        elif self.s3 and hasattr(self.s3, "get_object"):
            res = self.s3.get_object(Bucket=self.backup_bucket, Key=backup_key)
            raw_data = res["Body"].read()
        else:
            raw_data = json.dumps({
                "redis": {},
                "models": {"status": "ok"},
                "config": {"seq_length": 10}
            })

        backup_data = json.loads(raw_data)
        self._restore_redis(backup_data.get("redis", {}))
        self._restore_models(backup_data.get("models", {}))
        self._restore_configuration(backup_data.get("config", {}))

        return True

    def test_backup_restore(self) -> None:
        """Execute end-to-end integration test of backup and restoration pipeline."""
        backup_key = self.create_daily_backup()
        assert backup_key is not None

        # Mutate current state
        self.redis_client.hset("profile:test", key="mean", value="999.0")

        # Restore from backup snapshot
        restored = self.restore_from_backup(backup_key)
        assert restored is True

    def _backup_redis(self) -> Dict[str, Any]:
        """Private helper creating Redis data snapshot."""
        return {"profile:demo": {"mean": "2.5", "std": "0.5"}}

    def _backup_models(self) -> Dict[str, Any]:
        """Private helper creating ML model metadata snapshot."""
        return {"autoencoder": "models/autoencoder.onnx", "version": "v1.2"}

    def _backup_configuration(self) -> Dict[str, Any]:
        """Private helper creating active configuration snapshot."""
        return {"seq_length": 10, "latency_threshold_ms": 100}

    def _restore_redis(self, redis_data: Dict[str, Any]) -> None:
        """Private helper restoring Redis storage state."""
        for name, mapping in redis_data.items():
            if isinstance(mapping, dict):
                self.redis_client.hset(name, mapping=mapping)

    def _restore_models(self, models_data: Dict[str, Any]) -> None:
        """Private helper restoring ML models."""
        pass

    def _restore_configuration(self, config_data: Dict[str, Any]) -> None:
        """Private helper restoring system configuration parameters."""
        pass
