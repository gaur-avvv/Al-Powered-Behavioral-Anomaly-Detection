"""
Entity Baseline Profiler Module.
Provides statistical and sequence profiling per entity with Redis in-memory storage fallback.
"""

from typing import Dict, List, Any, Optional
import numpy as np


class InMemoryStorage:
    """In-memory storage fallback for Redis when external server is unavailable."""

    def __init__(self) -> None:
        """Initialize in-memory storage dictionary."""
        self._store: Dict[str, Any] = {}
        self._zsets: Dict[str, Dict[str, float]] = {}

    def hset(self, name: str, key: Optional[str] = None, value: Optional[str] = None,
             mapping: Optional[Dict[str, Any]] = None) -> None:
        """Set hash fields in-memory."""
        if name not in self._store:
            self._store[name] = {}
        if mapping:
            self._store[name].update(mapping)
        elif key is not None:
            self._store[name][key] = value

    def hgetall(self, name: str) -> Dict[str, Any]:
        """Get all fields of a hash."""
        return self._store.get(name, {})

    def zadd(self, name: str, mapping: Dict[str, float]) -> None:
        """Add elements with scores to sorted set."""
        if name not in self._zsets:
            self._zsets[name] = {}
        self._zsets[name].update(mapping)

    def zrange(self, name: str, start: int, end: int,
               desc: bool = False, withscores: bool = False) -> List[Any]:
        """Get range of elements in sorted set."""
        zset = self._zsets.get(name, {})
        sorted_items = sorted(zset.items(), key=lambda x: x[1], reverse=desc)
        if end == -1 or end >= len(sorted_items):
            sliced = sorted_items[start:]
        else:
            sliced = sorted_items[start:end + 1]
        if withscores:
            return sliced
        return [item[0] for item in sliced]

    def set(self, name: str, value: Any) -> None:
        """Set simple key-value pair."""
        self._store[name] = value

    def get(self, name: str) -> Optional[Any]:
        """Get simple key-value pair."""
        return self._store.get(name)

    def ping(self) -> bool:
        """Health check ping."""
        return True


class EntityBaselineProfiler:
    """High cohesion baseline profiler managing per-entity statistical behavior."""

    def __init__(self, seq_length: int = 10, redis_host: str = "redis") -> None:
        """
        Initialize profiler with sequence length and storage client.

        :param seq_length: Length of sequence history window
        :param redis_host: Redis host address
        """
        self.seq_length = seq_length
        self.redis_client = InMemoryStorage()
        self._profiles: Dict[str, Dict[str, Any]] = {}

    def create_profile(self, entity_id: str, data: np.ndarray) -> Dict[str, Any]:
        """
        Create statistical profile for an entity.

        :param entity_id: Unique entity identifier
        :param data: 1D or 2D numpy array of historical observations
        :return: Created profile dictionary
        """
        data_arr = np.asarray(data, dtype=np.float64)
        profile = self._create_statistical_profile(data_arr)
        profile["entity_id"] = entity_id
        profile["history"] = data_arr.tolist()
        profile["age_hours"] = 0.0
        
        self._profiles[entity_id] = profile
        self.redis_client.hset(f"profile:{entity_id}", mapping={
            "mean": str(profile["mean"]),
            "std": str(profile["std"]),
            "entity_id": entity_id
        })
        return profile

    def update_profile(self, entity_id: str, observation: float) -> None:
        """
        Incrementally update profile with a new numerical observation.

        :param entity_id: Unique entity identifier
        :param observation: New observed numerical scalar
        """
        if entity_id not in self._profiles:
            self.create_profile(entity_id, np.array([observation]))
            return

        self._incremental_update(entity_id, float(observation))

    def get_entity_profile(self, entity_id: str) -> Dict[str, Any]:
        """
        Retrieve existing entity profile.

        :param entity_id: Unique entity identifier
        :return: Profile dictionary or default stats
        """
        if entity_id in self._profiles:
            return self._profiles[entity_id]
        
        # Check Redis storage
        stored = self.redis_client.hgetall(f"profile:{entity_id}")
        if stored:
            return {
                "entity_id": entity_id,
                "mean": float(stored.get("mean", 0.0)),
                "std": float(stored.get("std", 1.0)),
                "percentiles": {"p25": 0.0, "p50": 0.0, "p75": 0.0},
                "history": [],
                "age_hours": 0.0
            }
        
        # Default fallback
        return {
            "entity_id": entity_id,
            "mean": 0.0,
            "std": 1.0,
            "percentiles": {"p25": 0.0, "p50": 0.0, "p75": 0.0},
            "history": [],
            "age_hours": 0.0
        }

    def _create_statistical_profile(self, data: np.ndarray) -> Dict[str, Any]:
        """Private helper to compute statistical summary metrics."""
        mean_val = float(np.mean(data))
        std_val = float(np.std(data)) if np.std(data) > 1e-6 else 1.0
        p25, p50, p75 = np.percentile(data, [25, 50, 75]).tolist()

        return {
            "mean": mean_val,
            "std": std_val,
            "percentiles": {"p25": p25, "p50": p50, "p75": p75}
        }

    def _incremental_update(self, entity_id: str, observation: float) -> None:
        """Private helper for Welford's incremental online mean and std updates."""
        prof = self._profiles[entity_id]
        history = prof.get("history", [])
        history.append(observation)
        if len(history) > self.seq_length * 10:
            history = history[-self.seq_length * 10:]

        prof["history"] = history
        arr = np.array(history, dtype=np.float64)
        updated_stats = self._create_statistical_profile(arr)
        prof["mean"] = updated_stats["mean"]
        prof["std"] = updated_stats["std"]
        prof["percentiles"] = updated_stats["percentiles"]
        prof["age_hours"] = prof.get("age_hours", 0.0) + 0.1
