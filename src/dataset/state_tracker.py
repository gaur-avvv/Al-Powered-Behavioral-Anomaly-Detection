"""
Stateful Sequence Rolling & Telemetry Preprocessing Tracker Module.

Thread-safe state manager that processes real-time event logs, maps categorical tokens,
tracks geospatial velocities using the Haversine formula, and constructs chronological sequence
buffers per entity to feed deep learning sequential models.
"""

import collections
import logging
import threading
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger("StreamingStateTracker")


class StreamingStateTracker:
    """
    Thread-safe state manager that processes real-time event logs, maps categorical
    tokens, tracks geo-velocities, and constructs chronological sequence buffers
    per individual entity to feed deep learning sequential models.
    """

    def __init__(self, sequence_length: int = 10, feature_dim: int = 6) -> None:
        """
        Initialize StreamingStateTracker.

        :param sequence_length: Sliding window length for Bi-LSTM sequence input (default: 10)
        :param feature_dim: Number of continuous feature channels (default: 6)
        """
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim

        # Thread-safe dictionary holding historical deques per active entity_id
        self.state_registry: Dict[str, collections.deque] = {}

        # Tracks last known geo-coordinate and timestamp per entity for velocity math
        self.geo_registry: Dict[str, Tuple[Tuple[float, float], datetime]] = {}

        # Thread isolation lock to guarantee memory consistency across parallel FastAPI workers
        self._lock = threading.Lock()

        # Static vocab mappings to normalize categorical features into numeric structures
        self.auth_map = {"password": 0.1, "token": 0.5, "certificate": 0.9}
        self.type_map = {"user": 0.2, "service_account": 0.6, "edge_device": 1.0}

    def _calculate_haversine_velocity(
        self,
        entity_id: str,
        new_coords: List[float],
        new_time: datetime
    ) -> float:
        """
        Calculates travel velocity (km/hour) between consecutive connection footprints
        to detect impossible travel vectors. Returns 0.0 if first event.

        :param entity_id: Target entity identifier
        :param new_coords: [latitude, longitude] float pair
        :param new_time: Event timestamp datetime
        :return: Normalized travel velocity scalar in [0.0, 1.0]
        """
        if not new_coords or len(new_coords) < 2:
            return 0.0

        if entity_id not in self.geo_registry:
            self.geo_registry[entity_id] = ((new_coords[0], new_coords[1]), new_time)
            return 0.0

        (old_lat, old_lon), old_time = self.geo_registry[entity_id]

        # Update cache values immediately
        self.geo_registry[entity_id] = ((new_coords[0], new_coords[1]), new_time)

        # Delta time calculation in hours
        time_delta_hours = (new_time - old_time).total_seconds() / 3600.0
        if time_delta_hours <= 0:
            return 0.0

        # Haversine distance calculations
        lat1, lon1, lat2, lon2 = map(
            np.radians, [old_lat, old_lon, new_coords[0], new_coords[1]]
        )
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        c = 2.0 * np.arcsin(np.sqrt(a))
        distance_km = 6371.0 * c  # Earth radius multiplier constant in kilometers

        velocity_kmh = distance_km / time_delta_hours
        return float(min(2000.0, velocity_kmh) / 2000.0)  # Normalized ceiling bound

    def process_and_roll_log(self, raw_log: Dict[str, Any]) -> Tuple[List[List[float]], bool]:
        """
        Transforms incoming structured JSON log events into normalized numerical feature vectors.
        Appends the event to the entity state pool and extracts a padded sliding window sequence.

        :param raw_log: Dictionary containing log telemetry fields
        :return: Tuple[List[List[float]], bool]: (The numeric sequence matrix, is_ready_for_inference)
        """
        entity_id = str(raw_log.get("entity_id", "unknown"))

        # Parse text timestamp string into timezone-naive datetime object
        ts_str = str(raw_log.get("timestamp", ""))
        try:
            log_time = datetime.strptime(ts_str.replace("Z", ""), "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            log_time = datetime.utcnow()

        # 1. Feature Extraction & Normalization
        norm_hour = log_time.hour / 24.0
        auth_method = str(raw_log.get("auth_method", "password")).lower()
        norm_auth = self.auth_map.get(auth_method, 0.1)

        entity_type = str(raw_log.get("entity_type", "user")).lower()
        norm_type = self.type_map.get(entity_type, 0.2)

        session_dur = float(raw_log.get("session_duration", 0.0))
        norm_duration = float(np.log1p(max(0.0, session_dur)) / 10.0)

        geo_loc = raw_log.get("geo_location", [0.0, 0.0])
        if not isinstance(geo_loc, list) or len(geo_loc) < 2:
            geo_loc = [0.0, 0.0]
        norm_velocity = self._calculate_haversine_velocity(entity_id, geo_loc, log_time)

        cmds = raw_log.get("command_sequence", [])
        if not isinstance(cmds, list):
            cmds = []
        norm_cmds = float(min(20.0, len(cmds)) / 20.0)

        # Build clean numerical vector frame
        feature_vector = [norm_hour, norm_auth, norm_type, norm_duration, norm_velocity, norm_cmds]

        # 2. Thread-Safe State Rolling Layer
        with self._lock:
            if entity_id not in self.state_registry:
                # Maxlen drops expired chronological metrics naturally off the left side of the deque
                self.state_registry[entity_id] = collections.deque(maxlen=self.sequence_length)

            self.state_registry[entity_id].append(feature_vector)

            current_window = list(self.state_registry[entity_id])
            window_size = len(current_window)

        # 3. Handle Sequence Structural Continuity (Zero-Padding if history is insufficient)
        if window_size < self.sequence_length:
            padding_count = self.sequence_length - window_size
            padding = [[0.0] * self.feature_dim for _ in range(padding_count)]
            complete_sequence = padding + current_window
            is_ready_for_deep_inference = False
        else:
            complete_sequence = current_window
            is_ready_for_deep_inference = True

        return complete_sequence, is_ready_for_deep_inference


if __name__ == "__main__":
    tracker = StreamingStateTracker(sequence_length=3, feature_dim=6)

    log_t0 = {
        "entity_id": "E_0987",
        "entity_type": "user",
        "timestamp": "2026-07-26T12:00:00Z",
        "geo_location": [12.9165, 79.1325],
        "auth_method": "token",
        "session_duration": 45.0,
        "command_sequence": ["ls", "cd /etc"]
    }

    log_t1_impossible = {
        "entity_id": "E_0987",
        "entity_type": "user",
        "timestamp": "2026-07-26T12:15:00Z",
        "geo_location": [-12.9165, -79.1325],
        "auth_method": "password",
        "session_duration": 0.0,
        "command_sequence": []
    }

    print("--- Frame Ingestion Step 1 (Cold Start Baseline) ---")
    seq, ready = tracker.process_and_roll_log(log_t0)
    print(f"Is Ready For Deep Model Path: {ready}")
    print(f"Padded Extracted Tensor Matrix Input:\n{np.array(seq)}")

    print("\n--- Frame Ingestion Step 2 (Evaluating Velocity Multipliers) ---")
    seq, ready = tracker.process_and_roll_log(log_t1_impossible)
    print(f"Is Ready For Deep Model Path: {ready}")
    print(f"Rolled Extracted Tensor Matrix Input:\n{np.array(seq)}")
