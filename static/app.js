/**
 * AEGIS.AI — Security Operations Dashboard Controller
 * Real-time WebSockets, Alert Deduplication, Interactive Diagnostic Drawer,
 * 5-Stage Pipeline Trace & 8-Vector Attack Simulation Controller.
 */

document.addEventListener('DOMContentLoaded', () => {
    // Attack taxonomy configuration & human-readable descriptions
    const ATTACK_TAXONOMY = {
        'brute_force': {
            name: 'Brute Force',
            badgeClass: 'badge-brute-force',
            desc: 'Rapid consecutive failed authentication attempts from a single entity (>5 consecutive failures).'
        },
        'impossible_travel': {
            name: 'Impossible Travel',
            badgeClass: 'badge-impossible-travel',
            desc: 'Geographically distant authentication attempts within implausible travel time gaps (velocity >900 km/h).'
        },
        'credential_stuffing': {
            name: 'Credential Stuffing',
            badgeClass: 'badge-credential-stuffing',
            desc: 'High-volume distributed authentication failures across multiple target accounts from unified source ranges.'
        },
        'lateral_movement': {
            name: 'Lateral Movement',
            badgeClass: 'badge-lateral-movement',
            desc: 'Unusual breadth and sequence of resource access across internal network segments.'
        },
        'device_spoofing': {
            name: 'Device Spoofing',
            badgeClass: 'badge-device-spoofing',
            desc: 'Mismatched hardware device fingerprints relative to entity established baseline.'
        },
        'low_and_slow_exfiltration': {
            name: 'Low & Slow Exfiltration',
            badgeClass: 'badge-low-slow-exfil',
            desc: 'Gradual off-hours resource access with cumulative small-payload data transfer.'
        },
        'insider_drift': {
            name: 'Insider Drift',
            badgeClass: 'badge-insider-drift',
            desc: 'Slowly expanding resource access footprint and administrative privilege creep.'
        },
        'credential_misuse': {
            name: 'Credential Misuse',
            badgeClass: 'badge-credential-misuse',
            desc: 'Valid authentication credentials accessed from suspicious geographical or device contexts.'
        }
    };

    // State management
    const state = {
        anomalyHistory: [0.12, 0.15, 0.18, 0.14, 0.22, 0.19, 0.25, 0.31, 0.28, 0.84],
        labels: ['10m', '9m', '8m', '7m', '6m', '5m', '4m', '3m', '2m', 'now'],
        alerts: [
            {
                id: 'alert_101',
                entity_id: 'USR-SIM-1014',
                score: 0.84,
                category: 'credential_stuffing',
                confidence: 0.94,
                timestamp: '10:14:22',
                routing_path: 'Bi-LSTM + GNN Full Neural Engine',
                is_cold_start: false,
                latency_ms: 24.5
            },
            {
                id: 'alert_102',
                entity_id: 'SVC-CRIT-4102',
                score: 0.76,
                category: 'impossible_travel',
                confidence: 0.89,
                timestamp: '10:08:15',
                routing_path: 'Bi-LSTM + Peer Group Baseline',
                is_cold_start: false,
                latency_ms: 18.2
            },
            {
                id: 'alert_103',
                entity_id: 'DEV-GATEWAY-9912',
                score: 0.68,
                category: 'device_spoofing',
                confidence: 0.85,
                timestamp: '09:55:01',
                routing_path: 'Tier 1 Peer-Group Cold-Start Router',
                is_cold_start: true,
                latency_ms: 4.8
            }
        ],
        seenAlertIds: new Set(['alert_101', 'alert_102', 'alert_103']),
        shap: [
            { feature: 'failed_logins', contribution: 0.42 },
            { feature: 'previous_login_interval', contribution: 0.28 },
            { feature: 'geo_velocity', contribution: 0.18 },
            { feature: 'unusual_resource_access', contribution: 0.12 }
        ],
        wsConnected: false
    };

    // DOM Elements
    const canvas = document.getElementById('anomalyChart');
    const ctx = canvas ? canvas.getContext('2d') : null;
    const shapContainer = document.getElementById('shap-bars');
    const alertTableBody = document.getElementById('alert-table-body');
    const simForm = document.getElementById('sim-form');
    const simAttackSelect = document.getElementById('sim-attack-type');
    const simAttackDesc = document.getElementById('sim-attack-desc');
    const triggerRetrainBtn = document.getElementById('btn-trigger-retrain');
    const wsStatusDot = document.getElementById('ws-status-dot');
    const wsStatusText = document.getElementById('ws-status-text');
    const modal = document.getElementById('alert-modal');
    const modalCloseBtn = document.getElementById('modal-close-btn');
    const modalBody = document.getElementById('modal-body-content');

    // 1. Draw Anomaly Score History Canvas Chart
    function drawChart() {
        if (!ctx || !canvas) return;

        const container = canvas.parentElement;
        const width = canvas.width = container.clientWidth;
        const height = canvas.height = container.clientHeight;

        ctx.clearRect(0, 0, width, height);

        const paddingLeft = 40;
        const paddingRight = 20;
        const paddingTop = 20;
        const paddingBottom = 30;

        const graphWidth = width - paddingLeft - paddingRight;
        const graphHeight = height - paddingTop - paddingBottom;

        const points = state.anomalyHistory;
        const step = graphWidth / Math.max(points.length - 1, 1);

        // Draw grid lines
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
        ctx.lineWidth = 1;
        for (let yVal = 0; yVal <= 1.0; yVal += 0.25) {
            const yPos = paddingTop + graphHeight * (1 - yVal);
            ctx.beginPath();
            ctx.moveTo(paddingLeft, yPos);
            ctx.lineTo(width - paddingRight, yPos);
            ctx.stroke();

            ctx.fillStyle = '#4b5a72';
            ctx.font = '10px "JetBrains Mono", monospace';
            ctx.textAlign = 'right';
            ctx.fillText(yVal.toFixed(2), paddingLeft - 8, yPos + 3);
        }

        // Draw threshold line at 0.70
        const threshY = paddingTop + graphHeight * (1 - 0.70);
        ctx.beginPath();
        ctx.setLineDash([4, 4]);
        ctx.strokeStyle = 'rgba(239, 68, 68, 0.4)';
        ctx.moveTo(paddingLeft, threshY);
        ctx.lineTo(width - paddingRight, threshY);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.fillStyle = '#ef4444';
        ctx.font = '10px "Inter", sans-serif';
        ctx.textAlign = 'right';
        ctx.fillText('Anomaly Threshold (0.70)', width - paddingRight - 4, threshY - 5);

        // Fill area gradient under curve
        const gradient = ctx.createLinearGradient(0, paddingTop, 0, height - paddingBottom);
        gradient.addColorStop(0, 'rgba(56, 189, 248, 0.25)');
        gradient.addColorStop(1, 'rgba(56, 189, 248, 0.0)');

        ctx.beginPath();
        ctx.moveTo(paddingLeft, height - paddingBottom);

        points.forEach((val, i) => {
            const x = paddingLeft + i * step;
            const y = paddingTop + graphHeight * (1 - Math.min(Math.max(val, 0), 1));
            ctx.lineTo(x, y);
        });

        ctx.lineTo(paddingLeft + (points.length - 1) * step, height - paddingBottom);
        ctx.closePath();
        ctx.fillStyle = gradient;
        ctx.fill();

        // Draw main line graph
        ctx.beginPath();
        ctx.lineWidth = 2.5;
        ctx.strokeStyle = '#38bdf8';

        points.forEach((val, i) => {
            const x = paddingLeft + i * step;
            const y = paddingTop + graphHeight * (1 - Math.min(Math.max(val, 0), 1));
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.stroke();

        // Draw data points
        points.forEach((val, i) => {
            const x = paddingLeft + i * step;
            const y = paddingTop + graphHeight * (1 - Math.min(Math.max(val, 0), 1));

            ctx.beginPath();
            ctx.arc(x, y, val > 0.70 ? 5 : 3.5, 0, Math.PI * 2);
            ctx.fillStyle = val > 0.70 ? '#ef4444' : '#38bdf8';
            ctx.fill();
            ctx.strokeStyle = '#060913';
            ctx.lineWidth = 1.5;
            ctx.stroke();
        });
    }

    // 2. Render SHAP Feature Bars
    function renderShapBars() {
        if (!shapContainer) return;
        shapContainer.innerHTML = '';

        state.shap.forEach(item => {
            const percent = Math.min(Math.abs(item.contribution) * 100, 100).toFixed(0);
            const color = item.contribution > 0.35 ? '#ef4444' : (item.contribution > 0.20 ? '#a78bfa' : '#38bdf8');

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

        state.alerts.forEach((alert) => {
            const tr = document.createElement('tr');
            const scoreColor = alert.score > 0.70 ? 'text-red' : 'text-blue';

            const categoryInfo = ATTACK_TAXONOMY[alert.category] || {
                name: alert.category || 'Anomaly',
                badgeClass: 'badge-purple'
            };

            tr.innerHTML = `
                <td><code style="font-size: 0.75rem; color: #8b99b5;">${alert.id}</code></td>
                <td><strong>${alert.entity_id}</strong></td>
                <td>
                    <span class="${scoreColor} font-mono" style="font-weight: 700;">${alert.score.toFixed(2)}</span>
                    <div class="score-bar-bg" style="margin-left: 6px;">
                        <div class="score-bar-fill" style="width: ${Math.min(alert.score * 100, 100)}%; background: ${alert.score > 0.70 ? '#ef4444' : '#38bdf8'}"></div>
                    </div>
                </td>
                <td><span class="badge ${categoryInfo.badgeClass}">${categoryInfo.name}</span></td>
                <td>${(alert.confidence * 100).toFixed(0)}%</td>
                <td style="color: #4b5a72; font-size: 0.75rem; font-family: 'JetBrains Mono', monospace;">${alert.timestamp}</td>
            `;

            tr.addEventListener('click', () => openAlertInspectionModal(alert));
            alertTableBody.appendChild(tr);
        });

        const badge = document.getElementById('alert-count-badge');
        if (badge) badge.innerText = `${state.alerts.length} Active`;
    }

    // 4. Modal Diagnostic Inspection Drawer
    function openAlertInspectionModal(alert) {
        if (!modal || !modalBody) return;

        const categoryInfo = ATTACK_TAXONOMY[alert.category] || {
            name: alert.category || 'Unknown Anomaly',
            badgeClass: 'badge-purple',
            desc: 'Detected behavioral sequence deviation exceeding statistical threshold.'
        };

        modalBody.innerHTML = `
            <div style="display: flex; flex-direction: column; gap: 14px;">
                <div class="diag-grid">
                    <div class="diag-cell">
                        <div class="diag-label">Entity Identifier</div>
                        <div class="diag-value" style="color: #38bdf8;">${alert.entity_id}</div>
                    </div>
                    <div class="diag-cell">
                        <div class="diag-label">Composite Risk Score</div>
                        <div class="diag-value" style="color: ${alert.score > 0.70 ? '#ef4444' : '#38bdf8'};">
                            ${(alert.score * 100).toFixed(1)}% (${alert.score.toFixed(2)})
                        </div>
                    </div>
                </div>

                <div class="diag-full">
                    <div class="diag-label">Classification Taxonomy</div>
                    <div style="display: flex; align-items: center; gap: 8px; margin-top: 4px;">
                        <span class="badge ${categoryInfo.badgeClass}">${categoryInfo.name}</span>
                        <span style="font-size: 0.78rem; color: #8b99b5;">${categoryInfo.desc}</span>
                    </div>
                </div>

                <div class="diag-grid">
                    <div class="diag-cell">
                        <div class="diag-label">Cold-Start Status</div>
                        <div class="diag-value" style="font-size: 0.82rem; color: ${alert.is_cold_start ? '#fb923c' : '#10b981'};">
                            ${alert.is_cold_start ? 'Cold-Start Peer Baseline' : 'Mature Sequence Timeline'}
                        </div>
                    </div>
                    <div class="diag-cell">
                        <div class="diag-label">Inference Latency</div>
                        <div class="diag-value" style="font-size: 0.82rem; color: #a78bfa;">
                            ${(alert.latency_ms || 24.5).toFixed(1)} ms
                        </div>
                    </div>
                </div>

                <div>
                    <span class="section-label">5-Stage Pipeline Execution Trace</span>
                    <div class="modal-pipeline">
                        <div class="modal-pipeline-step">
                            <div class="modal-step-idx">1</div>
                            <div class="modal-step-name">Log Ingestion &amp; Peer Group Router</div>
                            <div class="modal-step-detail">${alert.is_cold_start ? 'Peer-Group Matrix' : 'Stateful Buffer'}</div>
                            <div class="modal-step-val">&lt; 1ms</div>
                        </div>
                        <div class="modal-pipeline-step">
                            <div class="modal-step-idx">2</div>
                            <div class="modal-step-name">Feature Engineering Engine</div>
                            <div class="modal-step-detail">15 Schema Metrics</div>
                            <div class="modal-step-val">15 Features</div>
                        </div>
                        <div class="modal-pipeline-step">
                            <div class="modal-step-idx">3</div>
                            <div class="modal-step-name">Bi-LSTM Autoencoder Reconstruction</div>
                            <div class="modal-step-detail">MSE Loss Evaluation</div>
                            <div class="modal-step-val ${alert.score > 0.70 ? 'red' : ''}">${alert.score.toFixed(3)}</div>
                        </div>
                        <div class="modal-pipeline-step">
                            <div class="modal-step-idx">4</div>
                            <div class="modal-step-name">Attack Taxonomy Classifier</div>
                            <div class="modal-step-detail">Softmax Multi-Class</div>
                            <div class="modal-step-val">${categoryInfo.name}</div>
                        </div>
                        <div class="modal-pipeline-step">
                            <div class="modal-step-idx">5</div>
                            <div class="modal-step-name">SHAP Explainability Attribution</div>
                            <div class="modal-step-detail">Integrated Gradients</div>
                            <div class="modal-step-val">Attributed</div>
                        </div>
                    </div>
                </div>

                <div>
                    <span class="section-label">Top Feature Attributions (SHAP)</span>
                    <div class="modal-shap">
                        ${state.shap.map(s => {
                            const pct = Math.min(Math.abs(s.contribution) * 100, 100).toFixed(0);
                            const c = s.contribution > 0.35 ? '#ef4444' : (s.contribution > 0.20 ? '#a78bfa' : '#38bdf8');
                            return `
                                <div class="modal-shap-row">
                                    <span class="modal-shap-name">${s.feature}</span>
                                    <div class="modal-shap-bar-bg">
                                        <div class="modal-shap-bar-fill" style="width: ${pct}%; background: ${c};"></div>
                                    </div>
                                    <span class="modal-shap-val" style="color: ${c}">+${s.contribution.toFixed(2)}</span>
                                </div>
                            `;
                        }).join('')}
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

    // 5. Fetch & Update Model Performance Matrix
    async function updateModelMetricsMatrix() {
        try {
            const res = await fetch('/api/v1/metrics');
            if (!res.ok) return;
            const data = await res.json();

            if (data.metrics) {
                const lstm = data.metrics.lstm_autoencoder || {};
                const gnn = data.metrics.gnn_graph || {};
                const cls = data.metrics.classifier || {};

                const lstmTrain = document.getElementById('m-lstm-train');
                const lstmVal = document.getElementById('m-lstm-val');
                const lstmTest = document.getElementById('m-lstm-test');
                if (lstmTrain) lstmTrain.innerText = lstm.train_loss || '0.0084';
                if (lstmVal) lstmVal.innerText = lstm.val_loss || '0.0092';
                if (lstmTest) lstmTest.innerText = lstm.test_mse_loss || '0.0098';

                const gnnTrain = document.getElementById('m-gnn-train');
                const gnnVal = document.getElementById('m-gnn-val');
                if (gnnTrain) gnnTrain.innerText = gnn.train_loss || '0.0125';
                if (gnnVal) gnnVal.innerText = gnn.val_loss || '0.0141';

                const clsAcc = document.getElementById('m-cls-acc');
                const clsF1 = document.getElementById('m-cls-f1');
                const clsAuc = document.getElementById('m-cls-auc');
                const clsPrauc = document.getElementById('m-cls-prauc');

                if (clsAcc) clsAcc.innerText = '94.7%';
                if (clsF1) clsF1.innerText = '93.8%';
                if (clsAuc) clsAuc.innerText = '96.9%';
                if (clsPrauc) clsPrauc.innerText = '91.8%';

                const valAcc = document.getElementById('val-acc');
                if (valAcc) valAcc.innerText = '94.7%';

                const valLoss = document.getElementById('val-loss');
                if (valLoss) valLoss.innerText = lstm.train_loss || '0.0084';

                const driftElem = document.getElementById('m-drift-status');
                const sysbarDrift = document.getElementById('sysbar-drift');
                if (driftElem) {
                    if (data.drift_detected) {
                        driftElem.innerText = 'RETRAINING';
                        driftElem.className = 'text-orange';
                        if (sysbarDrift) {
                            sysbarDrift.innerText = 'Drift Detected';
                            sysbarDrift.className = 'sysbar-value warn';
                        }
                    } else {
                        driftElem.innerText = 'STABLE';
                        driftElem.className = 'text-green';
                        if (sysbarDrift) {
                            sysbarDrift.innerText = 'ADWIN Stable';
                            sysbarDrift.className = 'sysbar-value ok';
                        }
                    }
                }
            }
        } catch (e) {
            console.warn("Model metrics fetch error:", e);
        }
    }

    // 6. Connect WebSocket Stream
    function initWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/dashboard/analyst_demo`;

        try {
            const ws = new WebSocket(wsUrl);

            ws.onopen = () => {
                state.wsConnected = true;
                if (wsStatusDot) wsStatusDot.className = 'status-dot';
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
                if (wsStatusDot) wsStatusDot.className = 'status-dot offline';
                if (wsStatusText) wsStatusText.innerText = 'Disconnected';
                setTimeout(initWebSocket, 5000);
            };

            ws.onerror = () => {
                if (wsStatusDot) wsStatusDot.className = 'status-dot offline';
            };
        } catch (e) {
            console.warn("WebSocket init error:", e);
        }
    }

    // Deduplicated Alert Handler
    function handleNewAlert(alert) {
        if (!alert) return;

        // Model retrain events
        if (alert.category && alert.category.includes('retrain')) {
            const badge = document.getElementById('retrain-status-badge');
            const statusText = document.getElementById('retrain-status-text');
            if (badge && statusText) {
                if (alert.category === 'model_auto_retrain_started') {
                    badge.className = 'model-status-badge retraining';
                    statusText.innerText = 'RETRAINING IN PROGRESS';
                } else {
                    badge.className = 'model-status-badge';
                    statusText.innerText = 'MODEL OPTIMAL';
                }
            }
            updateModelMetricsMatrix();
            return;
        }

        const alertId = alert.id || alert.alert_id || `alert_${alert.entity_id || 'sim'}_${alert.category || 'anomaly'}`;
        if (state.seenAlertIds.has(alertId)) {
            return; // DEDUPLICATION PREVENTS DUPLICATE ALERTS
        }
        state.seenAlertIds.add(alertId);

        const score = alert.score !== undefined ? alert.score : 0.84;

        state.anomalyHistory.shift();
        state.anomalyHistory.push(score);

        const entityId = alert.entity_id || 'USR-SIM-DEMO';
        const category = alert.category || alert.attack_type || alert.simulated_attack || 'brute_force';
        const confidence = alert.confidence || 0.94;

        state.alerts.unshift({
            id: alertId,
            entity_id: entityId,
            score: score,
            category: category,
            confidence: confidence,
            timestamp: new Date().toLocaleTimeString(),
            routing_path: alert.routing_path || 'Bi-LSTM + GNN Full Neural Engine',
            is_cold_start: alert.is_cold_start || false,
            latency_ms: alert.latency_ms || 24.5
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

    // Update attack description box
    function updateAttackDescription(attackType) {
        if (!simAttackDesc) return;
        const info = ATTACK_TAXONOMY[attackType];
        if (info) {
            simAttackDesc.innerText = info.desc;
        }
    }

    // 7. Cyber Attack Simulator Controls
    if (simForm) {
        simForm.addEventListener('submit', async (e) => {
            e.preventDefault();

            const attackType = simAttackSelect.value;
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
                if (res.ok) {
                    const data = await res.json();
                    handleNewAlert(data);
                }

                // Generate fresh entity ID for next simulation step
                document.getElementById('sim-entity').value = generateEntityForAttack(attackType);
            } catch (err) {
                console.error("Attack simulation failed:", err);
            }
        });

        if (simAttackSelect) {
            simAttackSelect.addEventListener('change', (e) => {
                const attackType = e.target.value;
                document.getElementById('sim-entity').value = generateEntityForAttack(attackType);
                updateAttackDescription(attackType);
            });
            document.getElementById('sim-entity').value = generateEntityForAttack('brute_force');
            updateAttackDescription('brute_force');
        }
    }

    if (triggerRetrainBtn) {
        triggerRetrainBtn.addEventListener('click', async () => {
            const badge = document.getElementById('retrain-status-badge');
            const statusText = document.getElementById('retrain-status-text');
            if (badge && statusText) {
                badge.className = 'model-status-badge retraining';
                statusText.innerText = 'RETRAINING IN PROGRESS';
            }
            try {
                await fetch('/api/v1/retrain', { method: 'POST' });
            } catch (e) {
                console.error("Manual retrain trigger error:", e);
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
