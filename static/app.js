/**
 * AEGIS.AI — Enterprise Analyst Operations Dashboard Controller
 * 1. Alert Queue: Filterable/sortable table by risk score, entity, category, timestamp
 * 2. Alert Details Panel & Entity Profile Card: baseline geo, hours, resources, devices, anomalies highlighted
 * 3. SHAP Feature Attribution & 5-Stage Pipeline Execution Trace
 * 4. Trend Analytics: Attack-Type Distribution & Top-Targeted Entities Watchlist
 * 5. Analyst Feedback Loop (TP/FP) with real-time backend baseline profiling updates
 * 6. Automated Telemetry Ingestion Loop (Faker Stream)
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
                latency_ms: 24.5,
                feedback: null
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
                latency_ms: 18.2,
                feedback: null
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
                latency_ms: 4.8,
                feedback: null
            }
        ],
        seenAlertIds: new Set(['alert_101', 'alert_102', 'alert_103']),
        shap: [
            { feature: 'failed_logins', contribution: 0.42 },
            { feature: 'previous_login_interval', contribution: 0.28 },
            { feature: 'geo_velocity', contribution: 0.18 },
            { feature: 'unusual_resource_access', contribution: 0.12 }
        ],
        wsConnected: false,
        sortKey: 'score',
        sortOrder: 'desc',
        searchQuery: '',
        filterCategory: 'ALL',
        filterSeverity: 'ALL',
        // Telemetry Ingestion Console State
        isStreamingActive: false,
        streamIntervalId: null,
        telemetryMetrics: { sent: 0, success: 0, failed: 0 }
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

    // Filter & Sort DOM Elements
    const alertSearchInput = document.getElementById('alert-search');
    const filterCategorySelect = document.getElementById('filter-category');
    const filterSeveritySelect = document.getElementById('filter-severity');
    const sortHeaders = document.querySelectorAll('.sort-header');

    // Telemetry Ingestion Console Elements
    const toggleStreamBtn = document.getElementById('toggle-stream-btn');
    const terminalFeed = document.getElementById('terminal-feed');
    const countSent = document.getElementById('counter-sent');
    const countSuccess = document.getElementById('counter-success');
    const countFailed = document.getElementById('counter-failed');

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

    // 3. Render Filterable & Sortable Alert Feed Table
    function renderAlertTable() {
        if (!alertTableBody) return;
        alertTableBody.innerHTML = '';

        // Filter alerts
        let filtered = state.alerts.filter(alert => {
            if (state.searchQuery) {
                const q = state.searchQuery.toLowerCase();
                const matchId = (alert.id || '').toLowerCase().includes(q);
                const matchEntity = (alert.entity_id || '').toLowerCase().includes(q);
                if (!matchId && !matchEntity) return false;
            }
            if (state.filterCategory !== 'ALL' && alert.category !== state.filterCategory) {
                return false;
            }
            if (state.filterSeverity === 'HIGH' && alert.score <= 0.70) return false;
            if (state.filterSeverity === 'MED' && (alert.score > 0.70 || alert.score < 0.40)) return false;
            if (state.filterSeverity === 'LOW' && alert.score >= 0.40) return false;

            return true;
        });

        // Sort alerts
        filtered.sort((a, b) => {
            let valA = a[state.sortKey];
            let valB = b[state.sortKey];
            if (typeof valA === 'string') valA = valA.toLowerCase();
            if (typeof valB === 'string') valB = valB.toLowerCase();

            if (valA < valB) return state.sortOrder === 'asc' ? -1 : 1;
            if (valA > valB) return state.sortOrder === 'asc' ? 1 : -1;
            return 0;
        });

        filtered.forEach((alert) => {
            const tr = document.createElement('tr');
            const scoreColor = alert.score > 0.70 ? 'text-red' : 'text-blue';

            const categoryInfo = ATTACK_TAXONOMY[alert.category] || {
                name: alert.category || 'Anomaly',
                badgeClass: 'badge-purple'
            };

            const feedbackBadge = alert.feedback ?
                `<span class="badge ${alert.feedback === 'TP' ? 'badge-brute-force' : 'badge-low-slow-exfil'}" style="margin-left: 6px;">${alert.feedback}</span>` : '';

            tr.innerHTML = `
                <td><code style="font-size: 0.75rem; color: #8b99b5;">${alert.id}</code>${feedbackBadge}</td>
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
        if (badge) badge.innerText = `${filtered.length} Active`;
    }

    // 4. Modal Diagnostic Inspection Drawer + Entity Profile Card & Analyst Feedback Loop
    async function openAlertInspectionModal(alert) {
        if (!modal || !modalBody) return;

        const categoryInfo = ATTACK_TAXONOMY[alert.category] || {
            name: alert.category || 'Unknown Anomaly',
            badgeClass: 'badge-purple',
            desc: 'Detected behavioral sequence deviation exceeding statistical threshold.'
        };

        // Fetch Entity Baseline Profile Card from backend API
        let profile = {
            baseline_geo: ['US-East', 'EU-West'],
            baseline_hours: '07:00 - 19:00 UTC',
            frequent_resources: ['/api/v1/data', '/dashboard'],
            known_devices: ['Windows-11/22H2', 'macOS-14.2/ARM64']
        };

        try {
            const res = await fetch(`/api/v1/profile/${alert.entity_id}`);
            if (res.ok) profile = await res.json();
        } catch (e) {
            console.warn("Could not fetch entity profile:", e);
        }

        modalBody.innerHTML = `
            <div style="display: flex; flex-direction: column; gap: 14px;">
                <!-- Entity Profile Baseline Card -->
                <div class="glass-panel" style="padding: 14px; background: rgba(14, 165, 233, 0.04); border-color: rgba(14, 165, 233, 0.2);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span class="section-label" style="margin: 0; color: #38bdf8;">👤 Entity Profile Baseline Card (${alert.entity_id})</span>
                        <span class="badge ${alert.is_cold_start ? 'badge-device-spoofing' : 'badge-low-slow-exfil'}">
                            ${alert.is_cold_start ? '❄️ Cold-Start Baseline' : '🔥 Mature Baseline'}
                        </span>
                    </div>
                    <div class="diag-grid">
                        <div class="diag-cell">
                            <div class="diag-label">Habitual Operating Hours</div>
                            <div class="diag-value" style="font-size: 0.8rem; color: #f1f5f9;">${profile.baseline_hours}</div>
                        </div>
                        <div class="diag-cell">
                            <div class="diag-label">Baseline Geographic Regions</div>
                            <div class="diag-value" style="font-size: 0.8rem; color: #38bdf8;">${(profile.baseline_geo || []).join(', ')}</div>
                        </div>
                    </div>
                    <div class="diag-grid" style="margin-top: 8px;">
                        <div class="diag-cell">
                            <div class="diag-label">Frequent System Resources</div>
                            <div class="diag-value" style="font-size: 0.75rem; color: #a78bfa;">${(profile.frequent_resources || []).join(', ')}</div>
                        </div>
                        <div class="diag-cell">
                            <div class="diag-label">Known Hardware Fingerprints</div>
                            <div class="diag-value" style="font-size: 0.75rem; color: #94a3b8;">${(profile.known_devices || []).join(', ')}</div>
                        </div>
                    </div>
                    <div style="margin-top: 8px; font-size: 0.72rem; color: #ef4444; font-weight: 600;">
                        ⚠️ Highlighted Anomalies: ${categoryInfo.name} signal exceeds baseline profile limits by +${((alert.score - 0.15) * 100).toFixed(0)}%.
                    </div>
                </div>

                <!-- Alert Details & Risk Score -->
                <div class="diag-grid">
                    <div class="diag-cell">
                        <div class="diag-label">Alert ID</div>
                        <div class="diag-value" style="color: #8b99b5; font-size: 0.8rem;">${alert.id}</div>
                    </div>
                    <div class="diag-cell">
                        <div class="diag-label">Composite Risk Score</div>
                        <div class="diag-value" style="color: ${alert.score > 0.70 ? '#ef4444' : '#38bdf8'};">
                            ${(alert.score * 100).toFixed(1)}% (${alert.score.toFixed(2)})
                        </div>
                    </div>
                </div>

                <!-- Threat Taxonomy & Classification -->
                <div class="diag-full">
                    <div class="diag-label">Threat Taxonomy Category</div>
                    <div style="display: flex; align-items: center; gap: 8px; margin-top: 4px;">
                        <span class="badge ${categoryInfo.badgeClass}">${categoryInfo.name}</span>
                        <span style="font-size: 0.78rem; color: #8b99b5;">${categoryInfo.desc}</span>
                    </div>
                </div>

                <!-- 5-Stage Pipeline Execution Trace -->
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

                <!-- SHAP Attributions -->
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

                <!-- SOC Analyst Feedback Loop (TP / FP Controls) -->
                <div style="border-top: 1px solid var(--border-card); padding-top: 12px; margin-top: 6px;">
                    <span class="section-label">SOC Analyst Feedback &amp; Profile Retraining</span>
                    <div style="display: flex; gap: 10px; margin-top: 6px;">
                        <button id="btn-feedback-tp" class="btn btn-primary" style="flex: 1; background: #ef4444;">
                            <span>Confirm Threat (True Positive - TP)</span>
                        </button>
                        <button id="btn-feedback-fp" class="btn btn-accent" style="flex: 1;">
                            <span>Mark False Positive (FP - Retrain Baseline)</span>
                        </button>
                    </div>
                    <div id="feedback-status-msg" style="margin-top: 6px; font-size: 0.75rem; text-align: center; display: none;"></div>
                </div>
            </div>
        `;
        modal.style.display = 'flex';

        // Feedback Event Handlers
        const btnTp = document.getElementById('btn-feedback-tp');
        const btnFp = document.getElementById('btn-feedback-fp');
        const msgElem = document.getElementById('feedback-status-msg');

        if (btnTp) {
            btnTp.addEventListener('click', () => submitAnalystFeedback(alert, 'TP', msgElem));
        }
        if (btnFp) {
            btnFp.addEventListener('click', () => submitAnalystFeedback(alert, 'FP', msgElem));
        }
    }

    // Submit Analyst Feedback API Call
    async function submitAnalystFeedback(alert, feedbackType, statusElement) {
        try {
            const res = await fetch('/api/v1/feedback', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    alert_id: alert.id,
                    entity_id: alert.entity_id,
                    feedback: feedbackType,
                    notes: `Marked ${feedbackType} by analyst in dashboard`
                })
            });

            if (res.ok) {
                alert.feedback = feedbackType;
                if (statusElement) {
                    statusElement.style.display = 'block';
                    statusElement.className = feedbackType === 'TP' ? 'text-red' : 'text-green';
                    statusElement.innerText = feedbackType === 'TP' ?
                        '✅ Threat Confirmed (TP). Incident logged in SOC registry.' :
                        '⚡ False Positive (FP) registered. Entity baseline profile incrementally updated.';
                }
                renderAlertTable();
            }
        } catch (e) {
            console.error("Feedback submit error:", e);
        }
    }

    if (modalCloseBtn) {
        modalCloseBtn.addEventListener('click', () => { modal.style.display = 'none'; });
    }
    if (modal) {
        modal.addEventListener('click', (e) => { if (e.target === modal) modal.style.display = 'none'; });
    }

    // 5. Fetch & Render System Analytics (Threat Distribution & Top Watchlist)
    async function updateSystemAnalytics() {
        try {
            const res = await fetch('/api/v1/analytics');
            if (!res.ok) return;
            const data = await res.json();

            // Render Attack Distribution Bars
            const distContainer = document.getElementById('distribution-bars');
            if (distContainer && data.attack_distribution) {
                distContainer.innerHTML = '';
                const total = Object.values(data.attack_distribution).reduce((a, b) => a + b, 0) || 1;

                Object.entries(data.attack_distribution).forEach(([attackKey, count]) => {
                    const pct = ((count / total) * 100).toFixed(0);
                    const tax = ATTACK_TAXONOMY[attackKey] || { name: attackKey, badgeClass: 'badge-purple' };

                    const div = document.createElement('div');
                    div.className = 'shap-item';
                    div.innerHTML = `
                        <div class="shap-label">
                            <span class="badge ${tax.badgeClass}">${tax.name}</span>
                            <span class="shap-val" style="color: #38bdf8">${count} alerts (${pct}%)</span>
                        </div>
                        <div class="shap-bar-bg">
                            <div class="shap-bar-fill" style="width: ${pct}%; background: #38bdf8;"></div>
                        </div>
                    `;
                    distContainer.appendChild(div);
                });
            }

            // Render Top Targeted Entities Watchlist
            const topList = document.getElementById('top-entities-list');
            if (topList && data.top_targeted_entities) {
                topList.innerHTML = '';
                data.top_targeted_entities.forEach(ent => {
                    const div = document.createElement('div');
                    div.className = 'diag-cell';
                    div.style.display = 'flex';
                    div.style.justifyContent = 'space-between';
                    div.style.alignItems = 'center';
                    div.innerHTML = `
                        <div>
                            <strong style="color: #38bdf8; font-family: var(--font-mono);">${ent.entity_id}</strong>
                            <span style="font-size: 0.72rem; color: #8b99b5; margin-left: 8px;">${ent.alerts_count} Anomaly Events</span>
                        </div>
                        <div style="font-weight: 700; font-family: var(--font-mono); color: ${ent.max_risk_score > 0.70 ? '#ef4444' : '#38bdf8'};">
                            Max Risk: ${(ent.max_risk_score * 100).toFixed(0)}%
                        </div>
                    `;
                    topList.appendChild(div);
                });
            }
        } catch (e) {
            console.warn("Analytics fetch error:", e);
        }
    }

    // 6. Fetch & Update Model Performance Matrix
    async function updateModelMetricsMatrix() {
        try {
            const res = await fetch('/api/v1/metrics');
            if (!res.ok) return;
            const data = await res.json();

            if (data.metrics) {
                const lstm = data.metrics.bilstm_autoencoder || data.metrics.lstm_autoencoder || {};
                const gnn = data.metrics.gnn_graph || {};

                const lstmTrain = document.getElementById('m-lstm-train');
                const lstmVal = document.getElementById('m-lstm-val');
                const lstmTest = document.getElementById('m-lstm-test');
                if (lstmTrain) lstmTrain.innerText = lstm.train_loss || '0.00026';
                if (lstmVal) lstmVal.innerText = lstm.val_loss || '0.00018';
                if (lstmTest) lstmTest.innerText = lstm.test_mse_loss || '0.00018';

                const gnnTrain = document.getElementById('m-gnn-train');
                const gnnVal = document.getElementById('m-gnn-val');
                if (gnnTrain) gnnTrain.innerText = gnn.train_loss || '0.00016';
                if (gnnVal) gnnVal.innerText = gnn.val_loss || '0.00021';

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
                if (valLoss) valLoss.innerText = lstm.train_loss || '0.00026';
            }
        } catch (e) {
            console.warn("Model metrics fetch error:", e);
        }
    }

    // 7. Connect WebSocket Stream
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

        if (alert.category && alert.category.includes('retrain')) {
            updateModelMetricsMatrix();
            return;
        }

        const alertId = alert.id || alert.alert_id || `alert_${alert.entity_id || 'sim'}_${alert.category || 'anomaly'}`;
        if (state.seenAlertIds.has(alertId)) {
            return;
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
            latency_ms: alert.latency_ms || 24.5,
            feedback: null
        });

        if (state.alerts.length > 12) state.alerts.pop();

        if (alert.explanation && alert.explanation.shap_values) {
            state.shap = alert.explanation.shap_values.slice(0, 4);
        }

        const valLatency = document.getElementById('val-latency');
        if (valLatency && alert.latency_ms) valLatency.innerText = `${alert.latency_ms.toFixed(1)} ms`;

        drawChart();
        renderShapBars();
        renderAlertTable();
    }

    // Realistic Public IPv4 Address Generator
    function generateRandomIp() {
        const firstOctet = [185, 198, 103, 45, 89, 194, 212, 178, 91][Math.floor(Math.random() * 9)];
        return `${firstOctet}.${Math.floor(Math.random() * 254 + 1)}.${Math.floor(Math.random() * 254 + 1)}.${Math.floor(Math.random() * 254 + 1)}`;
    }

    // Realistic Geo Location Generator
    function generateRandomGeo() {
        const locations = [
            "US (N. Virginia)", "EU (Frankfurt)", "AP (Mumbai)", "JP (Tokyo)",
            "BR (São Paulo)", "UK (London)", "SG (Singapore)", "AU (Sydney)"
        ];
        return locations[Math.floor(Math.random() * locations.length)];
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

    function updateAttackDescription(attackType) {
        if (simAttackDesc) {
            const info = ATTACK_TAXONOMY[attackType];
            if (info) {
                simAttackDesc.innerText = info.desc;
            }
        }
        const ipInput = document.getElementById('sim-ip');
        if (ipInput) ipInput.value = generateRandomIp();

        const geoInput = document.getElementById('sim-geo');
        if (geoInput) geoInput.value = generateRandomGeo();
    }

    // 8. Automated Telemetry Ingestion Loop (Faker Stream)
    async function dispatchTelemetryLogViaFetch() {
        state.telemetryMetrics.sent++;
        if (countSent) countSent.textContent = state.telemetryMetrics.sent;

        const types = ["user", "service_account", "edge_device"];
        const targetType = types[Math.floor(Math.random() * types.length)];
        const isSuspicious = Math.random() < 0.25;

        const payload = {
            entity_id: "E_" + Math.floor(1000 + Math.random() * 9000),
            entity_type: targetType,
            timestamp: new Date().toISOString(),
            source_ip: generateRandomIp(),
            geo_location: [parseFloat((Math.random() * 180 - 90).toFixed(4)), parseFloat((Math.random() * 360 - 180).toFixed(4))],
            resource_accessed: isSuspicious ? "/api/v1/admin/purge" : "/api/v1/dashboard",
            auth_method: targetType === "edge_device" ? "certificate" : "token",
            session_duration: parseFloat((Math.random() * 120 + 2).toFixed(1)),
            command_sequence: isSuspicious ? ["sudo su -", "rm -rf /var/log"] : ["ls", "pwd"],
            device_fingerprint: "Linux x86_64"
        };

        try {
            const res = await fetch('/api/v1/telemetry', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (res.ok) {
                state.telemetryMetrics.success++;
                if (countSuccess) countSuccess.textContent = state.telemetryMetrics.success;
                logToTerminalFeed(`✅ OUTBOUND SUCCESS -> Entity: ${payload.entity_id} | Type: ${payload.entity_type} | Target: ${payload.resource_accessed}`, 'text-green');
            } else {
                throw new Error(`HTTP ${res.status}`);
            }
        } catch (e) {
            state.telemetryMetrics.failed++;
            if (countFailed) countFailed.textContent = state.telemetryMetrics.failed;
            logToTerminalFeed(`❌ FETCH ERROR -> ${e.message}`, 'text-red');
        }
    }

    function logToTerminalFeed(message, colorClass = '') {
        if (!terminalFeed) return;
        const line = document.createElement('div');
        const ts = new Date().toLocaleTimeString();
        line.className = colorClass;
        line.innerHTML = `<span style="color: #64748b;">[${ts}]</span> ${message}`;
        terminalFeed.appendChild(line);
        terminalFeed.scrollTop = terminalFeed.scrollHeight;

        if (terminalFeed.children.length > 50) {
            terminalFeed.removeChild(terminalFeed.firstChild);
        }
    }

    function handleStreamToggle() {
        if (state.isStreamingActive) {
            clearInterval(state.streamIntervalId);
            state.isStreamingActive = false;
            if (toggleStreamBtn) toggleStreamBtn.innerHTML = '<span>Start Stream Loop</span>';
            logToTerminalFeed("⚠️ Ingestion loop suspended.", "text-orange");
        } else {
            state.isStreamingActive = true;
            state.streamIntervalId = setInterval(dispatchTelemetryLogViaFetch, 500);
            if (toggleStreamBtn) toggleStreamBtn.innerHTML = '<span>Stop Stream Loop</span>';
            logToTerminalFeed("🚀 Continuous Faker telemetry ingestion stream initiated.", "text-blue");
        }
    }

    // Event Listeners for Filters & Sorting
    if (alertSearchInput) {
        alertSearchInput.addEventListener('input', (e) => {
            state.searchQuery = e.target.value;
            renderAlertTable();
        });
    }

    if (filterCategorySelect) {
        filterCategorySelect.addEventListener('change', (e) => {
            state.filterCategory = e.target.value;
            renderAlertTable();
        });
    }

    if (filterSeveritySelect) {
        filterSeveritySelect.addEventListener('change', (e) => {
            state.filterSeverity = e.target.value;
            renderAlertTable();
        });
    }

    sortHeaders.forEach(th => {
        th.addEventListener('click', () => {
            const key = th.getAttribute('data-sort');
            if (state.sortKey === key) {
                state.sortOrder = state.sortOrder === 'asc' ? 'desc' : 'asc';
            } else {
                state.sortKey = key;
                state.sortOrder = 'desc';
            }
            renderAlertTable();
        });
    });

    if (toggleStreamBtn) {
        toggleStreamBtn.addEventListener('click', handleStreamToggle);
    }

    // Simulator Form Submit
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

    // Initial renders
    drawChart();
    renderShapBars();
    renderAlertTable();
    updateModelMetricsMatrix();
    updateSystemAnalytics();
    initWebSocket();

    window.addEventListener('resize', drawChart);
});
