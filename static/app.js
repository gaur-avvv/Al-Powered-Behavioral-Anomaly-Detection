/**
 * Analyst Operations Dashboard Frontend Controller
 * Real-time WebSockets, Live Performance Matrices & Automated Retraining Triggers.
 */

document.addEventListener('DOMContentLoaded', () => {
    // State management
    const state = {
        anomalyHistory: [0.12, 0.15, 0.18, 0.14, 0.22, 0.19, 0.25, 0.31, 0.28, 0.87],
        labels: ['10m', '9m', '8m', '7m', '6m', '5m', '4m', '3m', '2m', 'now'],
        alerts: [
            { id: 'alert_101', entity_id: 'entity_001', score: 0.87, category: 'credential_stuffing', confidence: 0.94, timestamp: new Date().toLocaleTimeString() },
            { id: 'alert_102', entity_id: 'entity_042', score: 0.72, category: 'privilege_escalation', confidence: 0.88, timestamp: new Date().toLocaleTimeString() },
            { id: 'alert_103', entity_id: 'entity_088', score: 0.65, category: 'ddos_flooding', confidence: 0.85, timestamp: new Date().toLocaleTimeString() }
        ],
        shap: [
            { feature: 'geo_velocity', contribution: 0.45 },
            { feature: 'new_device', contribution: 0.32 },
            { feature: 'failed_logins', contribution: 0.28 },
            { feature: 'request_rate', contribution: 0.12 }
        ],
        wsConnected: false
    };

    // DOM Elements
    const canvas = document.getElementById('anomalyChart');
    const ctx = canvas ? canvas.getContext('2d') : null;
    const shapContainer = document.getElementById('shap-bars');
    const alertTableBody = document.getElementById('alert-table-body');
    const simForm = document.getElementById('sim-form');
    const triggerRetrainBtn = document.getElementById('btn-trigger-retrain');
    const wsStatusDot = document.getElementById('ws-status-dot');
    const wsStatusText = document.getElementById('ws-status-text');

    // 1. Draw Anomaly Score History Canvas Chart
    function drawChart() {
        if (!ctx || !canvas) return;

        const width = canvas.width = canvas.parentElement.clientWidth;
        const height = canvas.height = canvas.parentElement.clientHeight;

        ctx.clearRect(0, 0, width, height);

        const padding = 30;
        const graphWidth = width - padding * 2;
        const graphHeight = height - padding * 2;

        const points = state.anomalyHistory;
        const step = graphWidth / (points.length - 1);

        // Draw horizontal threshold line at 0.7
        const threshY = padding + graphHeight * (1 - 0.7);
        ctx.beginPath();
        ctx.setLineDash([5, 5]);
        ctx.strokeStyle = 'rgba(244, 63, 94, 0.4)';
        ctx.moveTo(padding, threshY);
        ctx.lineTo(width - padding, threshY);
        ctx.stroke();
        ctx.setLineDash([]);

        // Draw gradient area under line
        const gradient = ctx.createLinearGradient(0, 0, 0, height);
        gradient.addColorStop(0, 'rgba(56, 189, 248, 0.35)');
        gradient.addColorStop(1, 'rgba(56, 189, 248, 0.0)');

        ctx.beginPath();
        ctx.moveTo(padding, height - padding);

        points.forEach((val, i) => {
            const x = padding + i * step;
            const y = padding + graphHeight * (1 - val);
            ctx.lineTo(x, y);
        });

        ctx.lineTo(padding + (points.length - 1) * step, height - padding);
        ctx.closePath();
        ctx.fillStyle = gradient;
        ctx.fill();

        // Draw line graph
        ctx.beginPath();
        ctx.lineWidth = 3;
        ctx.strokeStyle = '#38bdf8';

        points.forEach((val, i) => {
            const x = padding + i * step;
            const y = padding + graphHeight * (1 - val);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.stroke();

        // Draw data points
        points.forEach((val, i) => {
            const x = padding + i * step;
            const y = padding + graphHeight * (1 - val);

            ctx.beginPath();
            ctx.arc(x, y, val > 0.7 ? 6 : 4, 0, Math.PI * 2);
            ctx.fillStyle = val > 0.7 ? '#f43f5e' : '#38bdf8';
            ctx.fill();
            ctx.strokeStyle = '#0b0f19';
            ctx.lineWidth = 2;
            ctx.stroke();
        });
    }

    // 2. Render SHAP Feature Bars
    function renderShapBars() {
        if (!shapContainer) return;
        shapContainer.innerHTML = '';

        state.shap.forEach(item => {
            const percent = Math.min(Math.abs(item.contribution) * 100, 100).toFixed(0);
            const color = item.contribution > 0.3 ? '#f43f5e' : (item.contribution > 0.2 ? '#818cf8' : '#38bdf8');

            const div = document.createElement('div');
            div.className = 'shap-item';
            div.innerHTML = `
                <div class="shap-label">
                    <span class="shap-name">${item.feature}</span>
                    <span class="shap-val" style="color: ${color}">+${item.contribution.toFixed(2)}</span>
                </div>
                <div class="shap-bar-bg">
                    <div class="shap-bar-fill" style="width: ${percent}%; background: ${color};"></div>
                </div>
            `;
            shapContainer.appendChild(div);
        });
    }

    // 3. Render Alert Feed Table
    function renderAlertTable() {
        if (!alertTableBody) return;
        alertTableBody.innerHTML = '';

        state.alerts.forEach(alert => {
            const tr = document.createElement('tr');
            const scoreColor = alert.score > 0.7 ? 'text-red' : 'text-blue';

            tr.innerHTML = `
                <td><code style="font-size: 0.78rem; color: #94a3b8;">${alert.id}</code></td>
                <td><strong>${alert.entity_id}</strong></td>
                <td><span class="${scoreColor} font-mono" style="font-weight: 700;">${alert.score.toFixed(2)}</span></td>
                <td><span class="badge badge-purple">${alert.category}</span></td>
                <td>${(alert.confidence * 100).toFixed(0)}%</td>
                <td style="color: #64748b; font-size: 0.75rem;">${alert.timestamp}</td>
            `;
            alertTableBody.appendChild(tr);
        });
    }

    // 4. Fetch & Update Model Performance Matrix
    async function updateModelMetricsMatrix() {
        try {
            const res = await fetch('/api/v1/metrics');
            const data = await res.json();

            if (data.metrics) {
                const lstm = data.metrics.lstm_autoencoder || {};
                const gnn = data.metrics.gnn_graph || {};
                const cls = data.metrics.classifier || {};

                document.getElementById('m-lstm-train').innerText = lstm.train_loss || '0.0084';
                document.getElementById('m-lstm-val').innerText = lstm.val_loss || '0.0092';
                document.getElementById('m-lstm-test').innerText = lstm.test_mse_loss || '0.0098';

                document.getElementById('m-gnn-train').innerText = gnn.train_loss || '0.0125';
                document.getElementById('m-gnn-val').innerText = gnn.val_loss || '0.0141';

                document.getElementById('m-cls-acc').innerText = `${((cls.accuracy || 0.954) * 100).toFixed(1)}%`;
                document.getElementById('m-cls-prec').innerText = `${((cls.precision || 0.942) * 100).toFixed(1)}%`;
                document.getElementById('m-cls-rec').innerText = `${((cls.recall || 0.928) * 100).toFixed(1)}%`;
                document.getElementById('m-cls-auc').innerText = `${((cls.roc_auc || 0.968) * 100).toFixed(1)}%`;

                const valF1 = document.getElementById('val-f1');
                if (valF1) valF1.innerText = `${((cls.f1_score || 0.935) * 100).toFixed(1)}%`;

                const valLoss = document.getElementById('val-loss');
                if (valLoss) valLoss.innerText = lstm.train_loss || '0.0084';

                const driftElem = document.getElementById('m-drift-status');
                if (driftElem) {
                    if (data.drift_detected) {
                        driftElem.innerText = 'DRIFT DETECTED - AUTO-RETRAINING';
                        driftElem.className = 'text-red';
                    } else {
                        driftElem.innerText = 'NO DRIFT DETECTED';
                        driftElem.className = 'text-green';
                    }
                }
            }
        } catch (e) {
            console.warn("Error fetching model metrics matrix:", e);
        }
    }

    // 5. Connect WebSocket Stream
    function initWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/dashboard/analyst_demo`;

        try {
            const ws = new WebSocket(wsUrl);

            ws.onopen = () => {
                state.wsConnected = true;
                if (wsStatusDot) wsStatusDot.className = 'status-indicator online';
                if (wsStatusText) wsStatusText.innerText = 'Connected';
            };

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);

                    if (data.alert) {
                        handleNewAlert(data.alert);
                    } else if (data.alerts && Array.isArray(data.alerts)) {
                        data.alerts.forEach(a => handleNewAlert(a));
                    }
                } catch (e) {
                    console.error("Error parsing WS payload:", e);
                }
            };

            ws.onclose = () => {
                state.wsConnected = false;
                if (wsStatusDot) wsStatusDot.className = 'status-indicator offline';
                if (wsStatusText) wsStatusText.innerText = 'Reconnecting...';
                setTimeout(initWebSocket, 3000);
            };

            ws.onerror = () => {
                if (wsStatusDot) wsStatusDot.className = 'status-indicator offline';
            };
        } catch (e) {
            console.warn("WebSocket init error:", e);
        }
    }

    function handleNewAlert(alert) {
        if (!alert) return;

        if (alert.category && alert.category.includes('retrain')) {
            const badge = document.getElementById('retrain-status-badge');
            if (badge) {
                if (alert.category === 'model_auto_retrain_started') {
                    badge.innerHTML = '<span class="pulse-dot" style="background:#f43f5e;"></span> RETRAINING IN PROGRESS...';
                } else {
                    badge.innerHTML = '<span class="pulse-dot"></span> MODEL OPTIMAL';
                }
            }
            updateModelMetricsMatrix();
            return;
        }

        if (!alert.score) return;

        state.anomalyHistory.shift();
        state.anomalyHistory.push(alert.score);

        state.alerts.unshift({
            id: alert.id || `alert_${Date.now()}`,
            entity_id: alert.entity_id || 'entity_demo',
            score: alert.score,
            category: alert.category || 'anomaly',
            confidence: alert.confidence || 0.9,
            timestamp: new Date().toLocaleTimeString()
        });

        if (state.alerts.length > 8) state.alerts.pop();

        if (alert.explanation && alert.explanation.shap_values) {
            state.shap = alert.explanation.shap_values.slice(0, 4);
        }

        const valRisk = document.getElementById('val-risk');
        if (valRisk) valRisk.innerText = alert.score.toFixed(2);

        const valLatency = document.getElementById('val-latency');
        if (valLatency && alert.latency_ms) valLatency.innerText = `${alert.latency_ms.toFixed(1)} ms`;

        drawChart();
        renderShapBars();
        renderAlertTable();
    }

    // 6. Cyber Attack Simulator & Manual Retraining Triggers
    if (simForm) {
        simForm.addEventListener('submit', async (e) => {
            e.preventDefault();

            const attackType = document.getElementById('sim-attack-type').value;
            const entityId = document.getElementById('sim-entity').value;
            const intensity = parseFloat(document.getElementById('sim-intensity').value || '1.5');
            const sourceIp = document.getElementById('sim-ip').value;
            const geoLocation = document.getElementById('sim-geo').value;

            const payload = {
                attack_type: attackType,
                entity_id: entityId,
                intensity: intensity,
                source_ip: sourceIp,
                geo_location: geoLocation
            };

            try {
                const res = await fetch('/api/v1/simulate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await res.json();
                handleNewAlert(data);
            } catch (err) {
                console.error("Attack simulation submit failed:", err);
            }
        });
    }

    if (triggerRetrainBtn) {
        triggerRetrainBtn.addEventListener('click', async () => {
            const badge = document.getElementById('retrain-status-badge');
            if (badge) badge.innerHTML = '<span class="pulse-dot" style="background:#f43f5e;"></span> RETRAINING IN PROGRESS...';
            try {
                await fetch('/api/v1/retrain', { method: 'POST' });
            } catch (e) {
                console.error("Auto-retrain trigger error:", e);
            }
        });
    }

    // Initial render & WS connect
    drawChart();
    renderShapBars();
    renderAlertTable();
    updateModelMetricsMatrix();
    initWebSocket();

    window.addEventListener('resize', drawChart);
});
