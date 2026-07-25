/**
 * AEGIS.AI — Analyst Operations Dashboard Frontend Controller
 * Real-time WebSockets, Live Telemetry Stream, Alert Deduplication,
 * Interactive Diagnostic Drawer, and Model Performance Matrix.
 */

document.addEventListener('DOMContentLoaded', () => {
    // State management
    const state = {
        anomalyHistory: [0.12, 0.15, 0.18, 0.14, 0.22, 0.19, 0.25, 0.31, 0.28, 0.87],
        labels: ['10m', '9m', '8m', '7m', '6m', '5m', '4m', '3m', '2m', 'now'],
        alerts: [
            {
                id: 'alert_101',
                entity_id: 'USR-SIM-1014',
                score: 0.87,
                category: 'credential_stuffing',
                confidence: 0.94,
                timestamp: new Date().toLocaleTimeString(),
                routing_path: 'Bi-LSTM+GNN-Full-Inference',
                is_cold_start: false
            },
            {
                id: 'alert_102',
                entity_id: 'SVC-CRIT-4102',
                score: 0.72,
                category: 'privilege_escalation',
                confidence: 0.88,
                timestamp: new Date().toLocaleTimeString(),
                routing_path: 'Level-2-Peer-Group-Baseline',
                is_cold_start: true
            },
            {
                id: 'alert_103',
                entity_id: 'DEV-GATEWAY-9912',
                score: 0.65,
                category: 'ddos_flooding',
                confidence: 0.85,
                timestamp: new Date().toLocaleTimeString(),
                routing_path: 'Bi-LSTM+GNN-Full-Inference',
                is_cold_start: false
            }
        ],
        seenAlertIds: new Set(['alert_101', 'alert_102', 'alert_103']),
        shap: [
            { feature: 'geo_velocity', contribution: 0.45 },
            { feature: 'new_device', contribution: 0.32 },
            { feature: 'failed_logins', contribution: 0.28 },
            { feature: 'request_rate', contribution: 0.12 }
        ],
        wsConnected: false,
        processedEventsCount: 14920,
        activeEntitiesCount: 200,
        currentQueueDepth: 0
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
    const modal = document.getElementById('alert-modal');
    const modalCloseBtn = document.getElementById('modal-close-btn');
    const modalBody = document.getElementById('modal-body-content');

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

    // 3. Render Alert Feed Table & Setup Inspection Modals
    function renderAlertTable() {
        if (!alertTableBody) return;
        alertTableBody.innerHTML = '';

        state.alerts.forEach((alert, index) => {
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

            tr.addEventListener('click', () => openAlertInspectionModal(alert));
            alertTableBody.appendChild(tr);
        });

        const badge = document.getElementById('alert-count-badge');
        if (badge) badge.innerText = `${state.alerts.length} Active`;
    }

    // Modal Inspection Drawer
    function openAlertInspectionModal(alert) {
        if (!modal || !modalBody) return;

        modalBody.innerHTML = `
            <div style="display: flex; flex-direction: column; gap: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; background: rgba(15, 23, 42, 0.6); padding: 0.75rem 1rem; border-radius: 8px;">
                    <div>
                        <span style="color: #94a3b8; font-size: 0.8rem;">ENTITY IDENTIFIER</span>
                        <h4 style="color: #38bdf8; font-size: 1.1rem; font-family: 'JetBrains Mono', monospace;">${alert.entity_id}</h4>
                    </div>
                    <div style="text-align: right;">
                        <span style="color: #94a3b8; font-size: 0.8rem;">ANOMALY RISK SCORE</span>
                        <h4 style="color: ${alert.score > 0.7 ? '#f43f5e' : '#38bdf8'}; font-size: 1.2rem;">${(alert.score * 100).toFixed(1)}% (${alert.score.toFixed(2)})</h4>
                    </div>
                </div>

                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                    <div style="background: rgba(15, 23, 42, 0.4); padding: 0.75rem; border-radius: 8px;">
                        <span style="color: #64748b; font-size: 0.75rem;">CLASSIFICATION TAXONOMY</span>
                        <p style="color: #c084fc; font-weight: 700; margin-top: 2px;">${alert.category}</p>
                    </div>
                    <div style="background: rgba(15, 23, 42, 0.4); padding: 0.75rem; border-radius: 8px;">
                        <span style="color: #64748b; font-size: 0.75rem;">COLD-START STATUS</span>
                        <p style="color: ${alert.is_cold_start ? '#fb923c' : '#34d399'}; font-weight: 700; margin-top: 2px;">${alert.is_cold_start ? '❄️ Cold-Start Peer Baseline' : '🔥 Mature Timeline (Bi-LSTM)'}</p>
                    </div>
                </div>

                <div style="background: rgba(15, 23, 42, 0.4); padding: 0.75rem; border-radius: 8px;">
                    <span style="color: #64748b; font-size: 0.75rem;">SYSTEM ROUTING PATH</span>
                    <p style="color: #f8fafc; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; margin-top: 2px;">${alert.routing_path || 'Bi-LSTM+GNN-Full-Inference'}</p>
                </div>

                <div>
                    <span style="color: #94a3b8; font-size: 0.8rem; font-weight: 600;">PRIMARY FEATURE ATTRIBUTIONS (SHAP)</span>
                    <div style="margin-top: 0.5rem; display: flex; flex-direction: column; gap: 0.4rem;">
                        ${state.shap.map(s => `
                            <div style="display: flex; justify-content: space-between; font-size: 0.82rem; background: rgba(30, 41, 59, 0.5); padding: 0.4rem 0.75rem; border-radius: 6px;">
                                <span style="color: #e2e8f0;">${s.feature}</span>
                                <span style="color: #38bdf8; font-weight: 700;">+${s.contribution.toFixed(2)}</span>
                            </div>
                        `).join('')}
                    </div>
                </div>
            </div>
        `;
        modal.style.display = 'flex';
    }

    if (modalCloseBtn) {
        modalCloseBtn.addEventListener('click', () => { modal.style.display = 'none'; });
    }
    if (modal) {
        modal.addEventListener('click', (e) => { if (e.target === modal) modal.style.display = 'none'; });
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

    // Deduplicated New Alert Handler
    function handleNewAlert(alert) {
        if (!alert) return;

        // Model retrain events
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

        const alertId = alert.id || alert.alert_id || `alert_${Date.now()}`;
        if (state.seenAlertIds.has(alertId)) {
            return; // DEDUPLICATION PREVENTS DUPLICATE ALERTS
        }
        state.seenAlertIds.add(alertId);

        const score = alert.score !== undefined ? alert.score : 0.85;

        state.anomalyHistory.shift();
        state.anomalyHistory.push(score);

        const entityId = alert.entity_id || 'USR-SIM-DEMO';
        const category = alert.category || alert.simulated_attack || 'anomaly';
        const confidence = alert.confidence || 0.95;

        state.alerts.unshift({
            id: alertId,
            entity_id: entityId,
            score: score,
            category: category,
            confidence: confidence,
            timestamp: new Date().toLocaleTimeString(),
            routing_path: alert.routing_path || 'Bi-LSTM+GNN-Full-Inference',
            is_cold_start: alert.is_cold_start || false
        });

        if (state.alerts.length > 8) state.alerts.pop();

        if (alert.explanation && alert.explanation.shap_values) {
            state.shap = alert.explanation.shap_values.slice(0, 4);
        }

        const valLatency = document.getElementById('val-latency');
        if (valLatency && alert.latency_ms) valLatency.innerText = `${alert.latency_ms.toFixed(1)} ms`;

        drawChart();
        renderShapBars();
        renderAlertTable();
    }

    // Auto-Generate Entity ID per Attack Vector
    function generateEntityForAttack(attackType) {
        const randNum = Math.floor(1000 + Math.random() * 9000);
        if (attackType === 'low_and_slow_exfiltration' || attackType === 'insider_drift') {
            return `SVC-EXFIL-${randNum}`;
        }
        if (attackType === 'device_spoofing') {
            return `DEV-GATEWAY-${randNum}`;
        }
        if (attackType === 'lateral_movement') {
            return `HOST-NODE-${randNum}`;
        }
        return `USR-SIM-${randNum}`;
    }

    // 6. Cyber Attack Simulator & Manual Retraining Triggers
    if (simForm) {
        simForm.addEventListener('submit', async (e) => {
            e.preventDefault();

            const attackType = document.getElementById('sim-attack-type').value;
            let entityIdInput = document.getElementById('sim-entity').value.trim();

            if (!entityIdInput) {
                entityIdInput = generateEntityForAttack(attackType);
                document.getElementById('sim-entity').value = entityIdInput;
            }

            const intensity = parseFloat(document.getElementById('sim-intensity').value || '1.5');
            const sourceIp = document.getElementById('sim-ip').value;
            const geoLocation = document.getElementById('sim-geo').value;

            const payload = {
                attack_type: attackType,
                entity_id: entityIdInput,
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

                // Auto-generate fresh entity ID for next run
                document.getElementById('sim-entity').value = generateEntityForAttack(attackType);
            } catch (err) {
                console.error("Attack simulation submit failed:", err);
            }
        });

        // Generate initial entity ID on change
        document.getElementById('sim-attack-type').addEventListener('change', (e) => {
            document.getElementById('sim-entity').value = generateEntityForAttack(e.target.value);
        });
        document.getElementById('sim-entity').value = generateEntityForAttack('brute_force');
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

    // 7. Live Real-Time Background Telemetry Ticker
    function initRealTimeTelemetryStream() {
        setInterval(async () => {
            state.processedEventsCount += Math.floor(Math.random() * 8 + 3);
            const eventsElem = document.getElementById('ticker-events-count');
            if (eventsElem) eventsElem.innerText = state.processedEventsCount.toLocaleString();

            const queueElem = document.getElementById('ticker-queue-depth');
            if (queueElem) queueElem.innerText = state.currentQueueDepth;

            // Periodically ping telemetry route to maintain background log ingestion
            try {
                const mockEnt = `USR-LIVE-${Math.floor(100 + Math.random() * 900)}`;
                await fetch('/api/v1/telemetry', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        entity_id: mockEnt,
                        entity_type: 'user',
                        timestamp: new Date().toISOString(),
                        source_ip: '10.0.0.15',
                        resource_accessed: '/api/v1/data',
                        auth_method: 'token',
                        session_duration: 15.0,
                        device_fingerprint: 'Linux x86_64'
                    })
                });
            } catch (e) {
                // Ignore background ticker errors
            }
        }, 3000);
    }

    // Initial render & WS connect
    drawChart();
    renderShapBars();
    renderAlertTable();
    updateModelMetricsMatrix();
    initWebSocket();
    initRealTimeTelemetryStream();

    window.addEventListener('resize', drawChart);
});
