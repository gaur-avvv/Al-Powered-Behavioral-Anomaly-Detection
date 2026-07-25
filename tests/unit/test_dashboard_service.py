"""
Unit tests for AnalystDashboard service.
"""

import pytest
import asyncio
from unittest.mock import Mock
from src.dashboard.dashboard_service import AnalystDashboard


class TestDashboardService:
    """Test suite covering WebSocket connections and risk ranking."""

    def test_dashboard_connection(self):
        """Test WebSocket connection registration."""
        dashboard = AnalystDashboard()

        mock_ws = Mock()
        mock_ws.accept = Mock()

        asyncio.run(dashboard.connect(mock_ws, "analyst_001"))

        assert "analyst_001" in dashboard.connections

    def test_dashboard_updates(self):
        """Test broadcast update send trigger."""
        dashboard = AnalystDashboard()

        mock_ws = Mock()
        mock_ws.send_text = Mock()

        dashboard.connections["analyst_001"] = mock_ws
        asyncio.run(dashboard.broadcast_alert({"id": "alert_1", "score": 0.9}))

        assert mock_ws.send_text.called

    def test_alert_queue_ranking(self):
        """Test alert queue ranking by risk score."""
        dashboard = AnalystDashboard()

        alerts = [
            {"id": "1", "score": 0.9, "entity_id": "e1"},
            {"id": "2", "score": 0.7, "entity_id": "e2"},
            {"id": "3", "score": 0.95, "entity_id": "e3"}
        ]

        for alert in alerts:
            dashboard.redis_client.hset(f"alert:{alert['id']}", mapping=alert)
            dashboard.redis_client.zadd("recent_alerts", {alert["id"]: alert["score"]})

        ranked = dashboard._get_recent_alerts(limit=10)

        assert ranked[0]["id"] == "3"  # Highest score first
        assert ranked[1]["id"] == "1"
        assert ranked[2]["id"] == "2"
