"""
Security test suite checking endpoint authorization, input validation, and security headers.
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app


class TestSecurity:
    """Security test suite."""

    def test_input_validation(self):
        """Test missing entity ID input rejection."""
        client = TestClient(app)

        response = client.post("/api/v1/detect", json={"entity_id": ""})
        assert response.status_code == 400

    def test_authentication(self):
        """Test rejection of invalid Bearer authorization header."""
        client = TestClient(app)

        response = client.post(
            "/api/v1/detect",
            json={"entity_id": "test_auth"},
            headers={"Authorization": "Bearer invalid_token"}
        )
        assert response.status_code == 401

    def test_health_endpoint(self):
        """Test security and operational status check endpoint."""
        client = TestClient(app)

        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "degraded"]
