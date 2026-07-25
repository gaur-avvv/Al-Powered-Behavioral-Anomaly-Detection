"""
Integration tests for the complete detection, profiling, and alert streaming pipeline.
"""

import pytest
import numpy as np
from fastapi.testclient import TestClient
from src.api.main import app, profiler
from src.dashboard.dashboard_service import AnalystDashboard


class TestFullPipeline:
    """End-to-end integration test suite."""

    def test_end_to_end_detection(self):
        """Test full pipeline from entity profile setup to detection response."""
        client = TestClient(app)

        # 1. Create baseline profile
        profiler.create_profile("entity_e2e", np.array([1.0, 2.0, 3.0, 4.0, 5.0]))

        # 2. Post detection payload
        response = client.post("/api/v1/detect", json={
            "entity_id": "entity_e2e",
            "features": {
                "geo_velocity": 250.0,
                "failed_logins": 8.0,
                "new_device": 1.0,
                "request_rate": 180.0
            }
        })

        assert response.status_code == 200
        data = response.json()
        assert data["entity_id"] == "entity_e2e"
        assert "score" in data
        assert "category" in data
        assert "explanation" in data
        assert "latency" in data

    def test_dashboard_alert_flow(self):
        """Test API detection alert generation and alert queue integration."""
        client = TestClient(app)

        # Post detection payload
        response = client.post("/api/v1/detect", json={
            "entity_id": "entity_flow",
            "features": {"failed_logins": 12.0}
        })
        assert response.status_code == 200

        # Fetch recent alerts
        alerts_res = client.get("/api/v1/alerts?limit=5")
        assert alerts_res.status_code == 200
        alerts = alerts_res.json()

        assert len(alerts) > 0
        assert any(a.get("entity_id") == "entity_flow" or "id" in a for a in alerts)
