"""
Analyst Dashboard Service Module.
Manages WebSocket analyst connections, real-time alert broadcasts, and risk-ranked alert queues.
"""

from typing import Dict, List, Any, Optional
import json
import asyncio
from datetime import datetime
from src.models.baseline_profiler import InMemoryStorage


class AnalystDashboard:
    """Manages WebSocket stream connections and prioritized analyst alert feeds."""

    def __init__(self) -> None:
        """Initialize AnalystDashboard service."""
        self.connections: Dict[str, Any] = {}
        self.redis_client = InMemoryStorage()

    async def connect(self, websocket: Any, analyst_id: str) -> None:
        """
        Accept and register new analyst WebSocket connection.

        :param websocket: FastAPI/Starlette WebSocket object
        :param analyst_id: Unique analyst session identifier
        """
        if hasattr(websocket, "accept"):
            res = websocket.accept()
            if asyncio.iscoroutine(res):
                await res
        self.connections[analyst_id] = websocket

    def disconnect(self, analyst_id: str) -> None:
        """
        Remove active analyst WebSocket connection.

        :param analyst_id: Unique analyst session identifier
        """
        if analyst_id in self.connections:
            del self.connections[analyst_id]

    async def broadcast_alert(self, alert: Dict[str, Any]) -> None:
        """
        Broadcast alert payload to all connected dashboard analyst sessions.

        :param alert: Alert details dictionary
        """
        message = json.dumps({
            "timestamp": datetime.utcnow().isoformat(),
            "alert": alert,
            "alerts": [alert]
        })
        disconnected = []
        for analyst_id, ws in self.connections.items():
            try:
                if hasattr(ws, "send_text"):
                    res = ws.send_text(message)
                    if asyncio.iscoroutine(res):
                        await res
            except Exception:
                disconnected.append(analyst_id)

        for analyst_id in disconnected:
            self.disconnect(analyst_id)

    def _get_recent_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Fetch top risk-score ranked alerts from storage.

        :param limit: Maximum number of alerts to return
        :return: Sorted list of alert dictionaries (highest risk score first)
        """
        ranked_items = self.redis_client.zrange("recent_alerts", 0, limit - 1,
                                                 desc=True, withscores=True)
        results = []
        for item in ranked_items:
            alert_id = item[0] if isinstance(item, tuple) else item
            alert_data = self.redis_client.hgetall(f"alert:{alert_id}")
            if alert_data:
                score_val = float(alert_data.get("score", 0.0))
                alert_data["score"] = score_val
                results.append(alert_data)
            else:
                results.append({
                    "id": str(alert_id),
                    "score": float(item[1]) if isinstance(item, tuple) else 0.5,
                    "entity_id": f"entity_{alert_id}"
                })

        # Ensure sorted descending by score
        results = sorted(results, key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return results[:limit]
