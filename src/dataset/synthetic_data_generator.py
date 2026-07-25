"""
Synthetic Access Log Data Generator with documented behavioral assumptions and
injected attack taxonomy for AI-Powered Behavioral Anomaly Detection.

Deliverable #1: Generates per-entity behavioral profiles and injects
attack patterns at controlled rates (0.5-3% of sessions) with
ground-truth labels retained for evaluation.

Simulates the following behavioral patterns:
  - Normal baseline: Per-entity habitual access patterns
  - Brute force: Rapid repeated failed-auth attempts
  - Impossible travel: Geographically distant logins in implausible time gaps
  - Credential stuffing: Many entity_ids, few source_ips, high failure rate
  - Lateral movement: Unusual sequence/breadth of resource access
  - Device spoofing: Mismatched device fingerprints
  - Low-and-slow exfiltration: Gradual off-hours small resource access
  - Insider drift: Slowly expanding privilege footprint (edge case)
  - Credential misuse: Valid credentials used from suspicious contexts
"""

import os
import random
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GEO_LOCATIONS = [
    "US-East", "US-West", "EU-West", "EU-Central", "AP-South",
    "AP-East", "SA-East", "AF-South", "ME-Central", "CA-Central"
]

RESOURCES = [
    "/api/users", "/api/admin", "/api/payments", "/api/reports",
    "/api/settings", "/api/logs", "/api/data-export", "/api/configs",
    "/api/deploy", "/api/health", "/api/tokens", "/api/audit",
    "/db/read", "/db/write", "/db/admin", "/storage/download",
    "/storage/upload", "/storage/delete", "/network/scan",
    "/network/firewall", "/network/vpn", "/ssh/connect"
]

AUTH_METHODS = ["password", "token", "certificate", "biometric", "sso"]

ENTITY_TYPES = ["user", "service_account", "edge_device"]

OS_FINGERPRINTS = [
    "Windows-11/22H2", "macOS-14.2/ARM64", "Ubuntu-22.04/x86_64",
    "RHEL-9.3/x86_64", "iOS-17.2/ARM64", "Android-14/ARM64",
    "FW-EdgeOS-3.1", "IoT-RTOS-2.4", "ChromeOS-120/x86_64"
]

COMMAND_SEQUENCES_NORMAL = [
    "login,view,logout",
    "login,read,read,logout",
    "login,search,view,logout",
    "login,upload,verify,logout",
    "connect,heartbeat,disconnect"
]

COMMAND_SEQUENCES_SUSPICIOUS = [
    "login,escalate,read,read,read,download,logout",
    "login,scan,enumerate,exploit,pivot,logout",
    "login,download,download,download,download,logout",
    "connect,scan,bruteforce,escalate,exfiltrate",
    "login,modify_config,disable_logging,read,logout"
]


def _generate_ip() -> str:
    """Generate a random IPv4 address."""
    return f"{random.randint(1, 223)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"


def _generate_mac() -> str:
    """Generate a random MAC address."""
    return ":".join(f"{random.randint(0, 255):02x}" for _ in range(6))


def _entity_hash_seed(entity_id: str) -> int:
    """Deterministic seed from entity_id for consistent profiles."""
    return int(hashlib.md5(entity_id.encode()).hexdigest()[:8], 16)


class EntityProfile:
    """
    Per-entity behavioral baseline profile.

    Each entity has habitual patterns:
    - Preferred login hours (mean +/- std)
    - Home geo-location (primary + occasional secondary)
    - Typical resources accessed (subset of all resources)
    - Preferred auth method
    - Typical session duration (mean +/- std)
    - Device fingerprint (OS, MAC)
    """

    def __init__(self, entity_id: str, entity_type: str, seed: int) -> None:
        """
        Initialize entity behavioral profile.

        :param entity_id: Unique entity identifier
        :param entity_type: Entity type (user/service_account/edge_device)
        :param seed: Deterministic random seed for reproducibility
        """
        rng = random.Random(seed)

        self.entity_id = entity_id
        self.entity_type = entity_type

        # Temporal patterns
        self.login_hour_mean = rng.gauss(10, 3)
        self.login_hour_std = rng.uniform(0.5, 2.0)

        # Geographic patterns
        self.home_geo = rng.choice(GEO_LOCATIONS)
        self.secondary_geo = rng.choice(
            [g for g in GEO_LOCATIONS if g != self.home_geo]
        )
        self.secondary_geo_probability = rng.uniform(0.02, 0.10)

        # Resource access patterns
        n_resources = rng.randint(3, 8)
        self.typical_resources = rng.sample(RESOURCES, n_resources)

        # Auth patterns
        self.primary_auth = rng.choice(AUTH_METHODS)
        self.alt_auth_probability = rng.uniform(0.01, 0.05)

        # Session duration patterns (minutes)
        self.session_mean = rng.uniform(5, 120)
        self.session_std = self.session_mean * rng.uniform(0.1, 0.3)

        # Device fingerprint
        self.os_fingerprint = rng.choice(OS_FINGERPRINTS)
        self.mac_address = _generate_mac()
        self.home_ip = _generate_ip()

        # Command sequence patterns
        self.typical_commands = rng.sample(
            COMMAND_SEQUENCES_NORMAL,
            min(3, len(COMMAND_SEQUENCES_NORMAL))
        )


class SyntheticDataGenerator:
    """
    Synthetic Access Log Data Generator.

    Generates per-entity behavioral profiles, produces normal access log
    events, and injects attack patterns at controlled rates with
    ground-truth labels for evaluation.

    Behavioral Assumptions:
    1. Each entity has stable habitual patterns (hours, geo, resources)
    2. Normal variation follows Gaussian noise around the baseline
    3. Attack events deviate systematically from baseline profiles
    4. Attack injection rate is configurable (default 2% of total events)
    5. Multiple attack types are injected at balanced rates

    Attack Taxonomy:
    - brute_force: 20+ failed auths in <5 minute window
    - impossible_travel: >5000km geo jump in <60 minutes
    - credential_stuffing: Many entities from same IP, high fail rate
    - lateral_movement: Access to 5+ unusual resources in one session
    - device_spoofing: Mismatched OS/MAC from entity history
    - low_and_slow_exfiltration: Off-hours small downloads over days
    - insider_drift: Gradually expanding resource access scope
    - credential_misuse: Valid creds from unusual geo/time
    """

    def __init__(
        self,
        n_entities: int = 200,
        n_events: int = 10000,
        anomaly_rate: float = 0.02,
        start_date: str = "2024-01-01",
        end_date: str = "2024-03-31",
        seed: int = 42
    ) -> None:
        """
        Initialize SyntheticDataGenerator.

        :param n_entities: Number of unique entities to simulate
        :param n_events: Total number of access log events
        :param anomaly_rate: Fraction of events that are anomalous (0.005-0.03)
        :param start_date: Simulation start date
        :param end_date: Simulation end date
        :param seed: Global random seed for reproducibility
        """
        self.n_entities = n_entities
        self.n_events = n_events
        self.anomaly_rate = anomaly_rate
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.seed = seed

        random.seed(seed)
        np.random.seed(seed)

        self.profiles: Dict[str, EntityProfile] = {}
        self._create_entity_profiles()

    def _create_entity_profiles(self) -> None:
        """Create per-entity behavioral baseline profiles."""
        for i in range(self.n_entities):
            entity_type = random.choice(ENTITY_TYPES)
            prefix = {"user": "USR", "service_account": "SVC", "edge_device": "DEV"}
            entity_id = f"{prefix[entity_type]}-{i:04d}"
            seed = _entity_hash_seed(entity_id)
            self.profiles[entity_id] = EntityProfile(entity_id, entity_type, seed)

    def _generate_normal_event(self, profile: EntityProfile) -> Dict[str, Any]:
        """
        Generate a single normal access event following entity baseline.

        :param profile: Entity behavioral profile
        :return: Access log event dictionary
        """
        # Timestamp: within habitual hours with Gaussian noise
        days_range = (self.end_date - self.start_date).days
        day_offset = random.randint(0, days_range)
        hour = max(0, min(23, int(np.random.normal(profile.login_hour_mean, profile.login_hour_std))))
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        ts = self.start_date + timedelta(days=day_offset, hours=hour, minutes=minute, seconds=second)

        # Geo-location: primary with small chance of secondary
        geo = (
            profile.secondary_geo
            if random.random() < profile.secondary_geo_probability
            else profile.home_geo
        )

        # Source IP: home IP with small variations
        source_ip = profile.home_ip

        # Resource accessed: from typical set
        resource = random.choice(profile.typical_resources)

        # Auth method: primary with small alt chance
        auth = (
            random.choice(AUTH_METHODS)
            if random.random() < profile.alt_auth_probability
            else profile.primary_auth
        )

        # Session duration: Gaussian around baseline
        session_dur = max(0.5, np.random.normal(profile.session_mean, profile.session_std))

        # Command sequence
        cmd_seq = random.choice(profile.typical_commands)

        # Device fingerprint
        device_fp = f"{profile.os_fingerprint}|{profile.mac_address}"

        return {
            "entity_id": profile.entity_id,
            "entity_type": profile.entity_type,
            "timestamp": ts.isoformat(),
            "source_ip": source_ip,
            "geo_location": geo,
            "resource_accessed": resource,
            "auth_method": auth,
            "session_duration": round(session_dur, 2),
            "command_sequence": cmd_seq,
            "device_fingerprint": device_fp,
            "label": "normal"
        }

    def _inject_brute_force(self, profile: EntityProfile) -> Dict[str, Any]:
        """Inject brute force attack: rapid repeated failed-auth attempts."""
        event = self._generate_normal_event(profile)
        event["label"] = "brute_force"
        event["auth_method"] = "password"
        event["session_duration"] = round(random.uniform(0.1, 0.5), 2)
        event["command_sequence"] = "login_fail," * random.randint(15, 40) + "login_fail"
        event["source_ip"] = _generate_ip()
        return event

    def _inject_impossible_travel(self, profile: EntityProfile) -> Dict[str, Any]:
        """Inject impossible travel: distant geo in implausible time gap."""
        event = self._generate_normal_event(profile)
        event["label"] = "impossible_travel"
        distant_geos = [g for g in GEO_LOCATIONS if g != profile.home_geo]
        event["geo_location"] = random.choice(distant_geos)
        event["source_ip"] = _generate_ip()
        return event

    def _inject_credential_stuffing(self, profile: EntityProfile) -> Dict[str, Any]:
        """Inject credential stuffing: many entities, few IPs, high failure."""
        event = self._generate_normal_event(profile)
        event["label"] = "credential_stuffing"
        event["source_ip"] = f"10.0.{random.randint(1, 3)}.{random.randint(1, 254)}"
        event["auth_method"] = "password"
        event["session_duration"] = round(random.uniform(0.05, 0.3), 2)
        event["command_sequence"] = "login_fail,login_fail,login_fail"
        return event

    def _inject_lateral_movement(self, profile: EntityProfile) -> Dict[str, Any]:
        """Inject lateral movement: unusual breadth of resource access."""
        event = self._generate_normal_event(profile)
        event["label"] = "lateral_movement"
        unusual_resources = [r for r in RESOURCES if r not in profile.typical_resources]
        if unusual_resources:
            event["resource_accessed"] = random.choice(unusual_resources)
        event["command_sequence"] = random.choice(COMMAND_SEQUENCES_SUSPICIOUS)
        event["session_duration"] = round(random.uniform(30, 300), 2)
        return event

    def _inject_device_spoofing(self, profile: EntityProfile) -> Dict[str, Any]:
        """Inject device spoofing: mismatched fingerprint from history."""
        event = self._generate_normal_event(profile)
        event["label"] = "device_spoofing"
        fake_os = random.choice([o for o in OS_FINGERPRINTS if o != profile.os_fingerprint])
        fake_mac = _generate_mac()
        event["device_fingerprint"] = f"{fake_os}|{fake_mac}"
        return event

    def _inject_low_and_slow(self, profile: EntityProfile) -> Dict[str, Any]:
        """Inject low-and-slow exfiltration: off-hours gradual downloads."""
        event = self._generate_normal_event(profile)
        event["label"] = "low_and_slow_exfiltration"
        event["resource_accessed"] = random.choice(
            ["/storage/download", "/api/data-export", "/db/read"]
        )
        # Off-hours: 1am - 5am
        ts = datetime.fromisoformat(event["timestamp"])
        ts = ts.replace(hour=random.randint(1, 4))
        event["timestamp"] = ts.isoformat()
        event["session_duration"] = round(random.uniform(1, 10), 2)
        event["command_sequence"] = "login,download,logout"
        return event

    def _inject_insider_drift(self, profile: EntityProfile) -> Dict[str, Any]:
        """Inject insider drift: gradually expanding privilege footprint."""
        event = self._generate_normal_event(profile)
        event["label"] = "insider_drift"
        admin_resources = ["/api/admin", "/db/admin", "/api/configs", "/network/firewall"]
        event["resource_accessed"] = random.choice(admin_resources)
        event["command_sequence"] = "login,escalate,view,modify,logout"
        return event

    def _inject_credential_misuse(self, profile: EntityProfile) -> Dict[str, Any]:
        """Inject credential misuse: valid creds from suspicious context."""
        event = self._generate_normal_event(profile)
        event["label"] = "credential_misuse"
        event["geo_location"] = random.choice(
            [g for g in GEO_LOCATIONS if g != profile.home_geo]
        )
        event["source_ip"] = _generate_ip()
        # Unusual hour
        ts = datetime.fromisoformat(event["timestamp"])
        ts = ts.replace(hour=random.randint(0, 4))
        event["timestamp"] = ts.isoformat()
        return event

    def generate(self) -> pd.DataFrame:
        """
        Generate the complete synthetic access log dataset.

        :return: DataFrame with n_events rows and ground-truth labels
        """
        n_anomaly = int(self.n_events * self.anomaly_rate)
        n_normal = self.n_events - n_anomaly

        events: List[Dict[str, Any]] = []

        # Generate normal events
        entity_ids = list(self.profiles.keys())
        for _ in range(n_normal):
            eid = random.choice(entity_ids)
            events.append(self._generate_normal_event(self.profiles[eid]))

        # Inject anomalies at balanced rates across attack types
        attack_injectors = [
            self._inject_brute_force,
            self._inject_impossible_travel,
            self._inject_credential_stuffing,
            self._inject_lateral_movement,
            self._inject_device_spoofing,
            self._inject_low_and_slow,
            self._inject_insider_drift,
            self._inject_credential_misuse,
        ]

        per_type = n_anomaly // len(attack_injectors)
        remainder = n_anomaly % len(attack_injectors)

        for idx, injector in enumerate(attack_injectors):
            count = per_type + (1 if idx < remainder else 0)
            for _ in range(count):
                eid = random.choice(entity_ids)
                events.append(injector(self.profiles[eid]))

        # Shuffle to avoid temporal clustering of attack types
        random.shuffle(events)

        df = pd.DataFrame(events)
        df = df.sort_values("timestamp").reset_index(drop=True)

        return df

    def save(self, output_path: str) -> str:
        """
        Generate and save dataset to CSV.

        :param output_path: Output file path
        :return: Absolute path to saved file
        """
        df = self.generate()
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        df.to_csv(output_path, index=False)
        return os.path.abspath(output_path)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Return generator configuration and dataset statistics.

        :return: Statistics dictionary
        """
        return {
            "n_entities": self.n_entities,
            "n_events": self.n_events,
            "anomaly_rate": self.anomaly_rate,
            "date_range": f"{self.start_date.date()} to {self.end_date.date()}",
            "entity_types": list(set(p.entity_type for p in self.profiles.values())),
            "attack_types": [
                "brute_force", "impossible_travel", "credential_stuffing",
                "lateral_movement", "device_spoofing", "low_and_slow_exfiltration",
                "insider_drift", "credential_misuse"
            ],
            "n_attack_types": 8,
            "geo_locations": GEO_LOCATIONS,
            "auth_methods": AUTH_METHODS,
            "seed": self.seed
        }


def main() -> None:
    """CLI entrypoint for synthetic data generation."""
    generator = SyntheticDataGenerator(
        n_entities=200,
        n_events=10000,
        anomaly_rate=0.02,
        start_date="2024-01-01",
        end_date="2024-03-31",
        seed=42
    )

    output_path = os.path.join("src", "dataset", "synthetic_access_logs_10000.csv")
    saved = generator.save(output_path)
    print(f"Dataset saved to: {saved}")

    stats = generator.get_statistics()
    print(f"Total events: {stats['n_events']}")
    print(f"Anomaly rate: {stats['anomaly_rate'] * 100:.1f}%")
    print(f"Attack types: {stats['n_attack_types']}")
    print(f"Entity count: {stats['n_entities']}")
    print(f"Date range: {stats['date_range']}")


if __name__ == "__main__":
    main()
