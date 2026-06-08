/**
 * ElectraGuard — Main Application Controller
 */

// Global state
let engine = null;
let chartManager = null;
let currentResults = [];
let currentSummary = null;
let currentPage = 1;
const PAGE_SIZE = 25;

// DOM Ready
document.addEventListener('DOMContentLoaded', () => {
    engine = new TheftDetectionEngine();
    chartManager = new ChartManager();
    window.chartManager = chartManager;

    initUploadZone();
    initNavigation();
    initSearch();
    initGuideButton();
});

/* =========================
   Guide Button (Hero)
   ========================= */
function initGuideButton() {
    const btnGuide = document.getElementById('btnGuide');
    if (btnGuide) {
        btnGuide.addEventListener('click', (e) => {
            e.stopPropagation();
            // Show dashboard section (which contains the guide panel)
            document.getElementById('heroSection').style.display = 'none';
            document.getElementById('dashboardSection').style.display = 'block';

            // Activate guide nav tab
            document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
            document.querySelector('.nav-link[data-section="guide"]')?.classList.add('active');

            // Show guide panel, hide others
            document.querySelectorAll('.section-panel').forEach(p => p.style.display = 'none');
            document.getElementById('section-guide').style.display = 'block';
        });
    }
}

/* =========================
   File Upload
   ========================= */
function initUploadZone() {
    const zone = document.getElementById('uploadZone');
    const fileInput = document.getElementById('fileInput');
    const btnSample = document.getElementById('btnSample');

    // Click to upload
    zone.addEventListener('click', () => fileInput.click());

    // File selected
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // Drag & drop
    zone.addEventListener('dragover', (e) => {
        e.preventDefault();
        zone.classList.add('drag-over');
    });

    zone.addEventListener('dragleave', () => {
        zone.classList.remove('drag-over');
    });

    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    // Sample data
    btnSample.addEventListener('click', (e) => {
        e.stopPropagation();
        loadSampleData();
    });
}

function handleFile(file) {
    const validTypes = [
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'application/vnd.ms-excel',
        'text/csv',
        'application/csv'
    ];

    const ext = file.name.split('.').pop().toLowerCase();
    if (!['xlsx', 'xls', 'csv'].includes(ext)) {
        showToast('error', 'Invalid file type. Please upload .xlsx, .xls, or .csv');
        return;
    }

    showProgress(true);
    updateProgress(10, 'Reading file...');

    const reader = new FileReader();
    reader.onload = (e) => {
        try {
            updateProgress(30, 'Parsing spreadsheet...');
            const data = new Uint8Array(e.target.result);
            const workbook = XLSX.read(data, { type: 'array' });

            updateProgress(50, 'Extracting data...');
            const firstSheet = workbook.SheetNames[0];
            const worksheet = workbook.Sheets[firstSheet];
            const jsonData = XLSX.utils.sheet_to_json(worksheet);

            if (jsonData.length === 0) {
                showToast('error', 'The file appears to be empty.');
                showProgress(false);
                return;
            }

            updateProgress(70, 'Running detection algorithms...');
            processAndRender(jsonData, file.name);

        } catch (err) {
            console.error(err);
            showToast('error', 'Failed to parse file. Ensure it is a valid spreadsheet.');
            showProgress(false);
        }
    };

    reader.onerror = () => {
        showToast('error', 'Failed to read file.');
        showProgress(false);
    };

    reader.readAsArrayBuffer(file);
}

function loadSampleData() {
    showProgress(true);
    updateProgress(20, 'Generating sample data with 30-day time-series...');

    setTimeout(() => {
        const sampleData = TheftDetectionEngine.generateSampleData(150);
        updateProgress(40, 'Running detection algorithms...');
        processAndRender(sampleData, 'Sample Data (150 Consumers)');
    }, 400);
}

async function processAndRender(jsonData, sourceName) {
    const headers = Object.keys(jsonData[0]);

    updateProgress(45, 'Processing records...');
    engine.processData(jsonData, headers);
    engine.calculateStats();

    // ──────── LSTM Training Phase (Kocaman & Tümen method) ────────
    let lstmSuccess = false;
    if (engine.useLSTM && typeof tf !== 'undefined') {
        updateProgress(50, 'Initializing LSTM deep learning engine...');
        showToast('info', 'Training LSTM model (Kocaman & Tümen method). This may take a moment...');

        try {
            lstmSuccess = await engine.runLSTMDetection((progress, message) => {
                // Map LSTM progress (0-100) to our progress bar (50-90)
                const mappedProgress = 50 + Math.round(progress * 0.4);
                updateProgress(mappedProgress, `🧠 LSTM: ${message}`);
            });

            if (lstmSuccess) {
                showToast('success', `LSTM model trained! 5-fold CV accuracy: ${(engine.lstmMetrics.accuracy * 100).toFixed(1)}%`);
            } else {
                showToast('warning', 'LSTM training skipped — insufficient time-series data. Using statistical methods only.');
            }
        } catch (err) {
            console.error('LSTM error:', err);
            showToast('warning', 'LSTM training encountered an error. Falling back to statistical methods.');
        }
    } else if (!engine.useLSTM) {
        showToast('info', 'No time-series columns detected. Using statistical detection only.');
    }

    updateProgress(92, 'Calculating hybrid risk scores...');
    currentResults = engine.calculateRiskScores();
    currentSummary = engine.getSummary();
    window.currentResults = currentResults;

    updateProgress(96, 'Building visualizations...');

    setTimeout(() => {
        updateProgress(100, 'Complete!');

        setTimeout(() => {
            // Switch to dashboard
            document.getElementById('heroSection').style.display = 'none';
            document.getElementById('dashboardSection').style.display = 'block';

            // Update nav
            const navStatus = document.getElementById('navStatus');
            const lstmBadge = lstmSuccess ? ' · LSTM ✓' : '';
            navStatus.innerHTML = `
                <div class="status-dot online"></div>
                <span>${sourceName} — ${currentResults.length} records${lstmBadge}</span>
            `;

            // Update alert badge
            const alerts = engine.getAlerts();
            const badge = document.getElementById('alertBadge');
            if (alerts.length > 0) {
                badge.style.display = 'inline-flex';
                badge.textContent = alerts.length;
            }

            // Render everything
            renderDashboard();
            const lstmMsg = lstmSuccess ? ' (LSTM + Statistical hybrid)' : ' (Statistical methods)';
            showToast('success', `Analysis complete${lstmMsg}: ${currentSummary.suspicious} suspicious cases out of ${currentSummary.total} consumers.`);
            showProgress(false);
        }, 300);
    }, 200);
}

function renderDashboard() {
    // Update stat cards
    animateCounter('totalConsumers', currentSummary.total);
    animateCounter('suspiciousCount', currentSummary.suspicious);
    document.getElementById('theftRate').textContent = currentSummary.theftRate + '%';
    document.getElementById('totalLoss').textContent = Number(currentSummary.estimatedLoss).toLocaleString() + ' kWh';
    document.getElementById('avgConsumption').textContent = Number(currentSummary.avgConsumption).toLocaleString();

    // Render charts
    chartManager.renderAll(currentResults, currentSummary);

    // Render data table
    renderTable(currentResults);

    // Render alerts
    renderAlerts();

    // Render analytics report
    renderAnalyticsReport();
}

/* =========================
   Navigation
   ========================= */
function initNavigation() {
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const section = link.dataset.section;

            // Update active nav
            document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
            link.classList.add('active');

            // Show/hide panels
            document.querySelectorAll('.section-panel').forEach(p => p.style.display = 'none');
            const panel = document.getElementById(`section-${section}`);
            if (panel) panel.style.display = 'block';
        });
    });
}

/* =========================
   Data Table
   ========================= */
function renderTable(data) {
    const thead = document.getElementById('tableHeader');
    const tbody = document.getElementById('tableBody');

    // Columns to show
    const hasLSTM = data.some(d => d.lstmProb !== null && d.lstmProb !== undefined);
    const columns = [
        { key: 'id', label: 'Consumer ID' },
        { key: 'name', label: 'Name' },
        { key: 'region', label: 'Region' },
        { key: 'category', label: 'Category' },
        { key: 'consumption', label: 'Consumption (kWh)' },
        { key: 'billing', label: 'Billing' },
        ...(hasLSTM ? [{ key: 'lstmProb', label: 'LSTM Prob.' }] : []),
        { key: 'riskScore', label: 'Risk Score' },
        { key: 'riskLevel', label: 'Risk Level' },
        { key: 'flags', label: 'Flags' }
    ];

    // Header
    thead.innerHTML = columns.map(c => `<th>${c.label}</th>`).join('');

    // Paginate
    const totalPages = Math.ceil(data.length / PAGE_SIZE);
    const start = (currentPage - 1) * PAGE_SIZE;
    const pageData = data.slice(start, start + PAGE_SIZE);

    // Rows
    tbody.innerHTML = pageData.map(row => {
        const rowClass = row.riskLevel === 'critical' ? 'row-critical' : row.riskLevel === 'high' ? 'row-high' : '';
        return `<tr class="${rowClass}">
            <td><code style="color: var(--accent-cyan); font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;">${row.id}</code></td>
            <td>${row.name}</td>
            <td>${row.region}</td>
            <td>${row.category}</td>
            <td>${row.consumption.toLocaleString()}</td>
            <td>${row.billing ? '₹' + row.billing.toLocaleString() : '—'}</td>
            ${hasLSTM ? `<td>${row.lstmProb !== null && row.lstmProb !== undefined ? `<span style="font-family:'JetBrains Mono',monospace; font-size:0.8rem; color:${row.lstmProb >= 0.7 ? '#ff6b6b' : row.lstmProb >= 0.4 ? '#ffd666' : '#5de0b5'};">${(row.lstmProb * 100).toFixed(1)}%</span>` : '—'}</td>` : ''}
            <td>
                <div style="display:flex; align-items:center; gap:8px;">
                    <div style="width:40px; height:4px; border-radius:2px; background:rgba(255,255,255,0.05); overflow:hidden;">
                        <div style="width:${row.riskScore}%; height:100%; border-radius:2px; background:${getRiskColor(row.riskLevel)};"></div>
                    </div>
                    <span style="font-family:'JetBrains Mono',monospace; font-size:0.8rem; color:${getRiskColor(row.riskLevel)};">${row.riskScore}</span>
                </div>
            </td>
            <td><span class="risk-badge ${row.riskLevel}"><span class="risk-dot ${row.riskLevel}"></span>${row.riskLevel}</span></td>
            <td style="max-width:250px; white-space:normal; font-size:0.78rem; color:var(--text-secondary);">${row.flags.length > 0 ? row.flags.join('; ') : '—'}</td>
        </tr>`;
    }).join('');

    // Pagination
    renderPagination(totalPages, data);
}

function renderPagination(totalPages, data) {
    const pagination = document.getElementById('pagination');
    if (totalPages <= 1) {
        pagination.innerHTML = '';
        return;
    }

    let html = '';
    html += `<button class="page-btn" ${currentPage === 1 ? 'disabled' : ''} onclick="goToPage(${currentPage - 1}, 'filtered')">‹</button>`;
    
    const maxVisible = 7;
    let startPage = Math.max(1, currentPage - Math.floor(maxVisible / 2));
    let endPage = Math.min(totalPages, startPage + maxVisible - 1);
    if (endPage - startPage < maxVisible - 1) startPage = Math.max(1, endPage - maxVisible + 1);

    if (startPage > 1) {
        html += `<button class="page-btn" onclick="goToPage(1, 'filtered')">1</button>`;
        if (startPage > 2) html += `<span style="color:var(--text-muted);">…</span>`;
    }

    for (let i = startPage; i <= endPage; i++) {
        html += `<button class="page-btn ${i === currentPage ? 'active' : ''}" onclick="goToPage(${i}, 'filtered')">${i}</button>`;
    }

    if (endPage < totalPages) {
        if (endPage < totalPages - 1) html += `<span style="color:var(--text-muted);">…</span>`;
        html += `<button class="page-btn" onclick="goToPage(${totalPages}, 'filtered')">${totalPages}</button>`;
    }

    html += `<button class="page-btn" ${currentPage === totalPages ? 'disabled' : ''} onclick="goToPage(${currentPage + 1}, 'filtered')">›</button>`;

    pagination.innerHTML = html;

    // Store filtered data for pagination
    window._filteredTableData = data;
}

function goToPage(page) {
    currentPage = page;
    const data = window._filteredTableData || currentResults;
    renderTable(data);
    // Scroll to top of table
    document.querySelector('.table-wrapper')?.scrollTo({ top: 0, behavior: 'smooth' });
}

/* =========================
   Search & Filter
   ========================= */
function initSearch() {
    const searchInput = document.getElementById('searchInput');
    const riskFilter = document.getElementById('riskFilter');

    let debounceTimer;
    searchInput?.addEventListener('input', () => {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => applyFilters(), 250);
    });

    riskFilter?.addEventListener('change', () => applyFilters());
}

function applyFilters() {
    const query = document.getElementById('searchInput')?.value.toLowerCase().trim() || '';
    const riskLevel = document.getElementById('riskFilter')?.value || 'all';

    let filtered = currentResults;

    if (query) {
        filtered = filtered.filter(r =>
            r.id.toLowerCase().includes(query) ||
            r.name.toLowerCase().includes(query) ||
            r.region.toLowerCase().includes(query) ||
            r.category.toLowerCase().includes(query)
        );
    }

    if (riskLevel !== 'all') {
        filtered = filtered.filter(r => r.riskLevel === riskLevel);
    }

    currentPage = 1;
    renderTable(filtered);
}

/* =========================
   Alerts Panel
   ========================= */
function renderAlerts() {
    const alerts = engine.getAlerts();
    const container = document.getElementById('alertsList');
    const summary = document.getElementById('alertsSummary');

    // Summary chips
    const criticalCount = alerts.filter(a => a.riskLevel === 'critical').length;
    const highCount = alerts.filter(a => a.riskLevel === 'high').length;
    const mediumCount = alerts.filter(a => a.riskLevel === 'medium').length;

    summary.innerHTML = `
        ${criticalCount > 0 ? `<span class="summary-chip critical">🔴 ${criticalCount} Critical</span>` : ''}
        ${highCount > 0 ? `<span class="summary-chip high">🟠 ${highCount} High</span>` : ''}
        ${mediumCount > 0 ? `<span class="summary-chip medium">🟡 ${mediumCount} Medium</span>` : ''}
    `;

    // Alert items
    container.innerHTML = alerts.map((alert, idx) => `
        <div class="alert-item ${alert.riskLevel}" style="animation-delay: ${idx * 0.05}s">
            <div class="alert-icon">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
                    <line x1="12" y1="9" x2="12" y2="13"/>
                    <line x1="12" y1="17" x2="12.01" y2="17"/>
                </svg>
            </div>
            <div class="alert-body">
                <div class="alert-title">${alert.id} — ${alert.name}</div>
                <div class="alert-desc">${alert.flags.join('. ')}.</div>
                <div class="alert-meta">
                    <span>📍 ${alert.region}</span>
                    <span>⚡ ${alert.consumption.toLocaleString()} kWh</span>
                    <span>🎯 Score: ${alert.riskScore}/100</span>
                    <span>📂 ${alert.category}</span>
                </div>
            </div>
            <span class="risk-badge ${alert.riskLevel}"><span class="risk-dot ${alert.riskLevel}"></span>${alert.riskLevel}</span>
        </div>
    `).join('');

    if (alerts.length === 0) {
        container.innerHTML = `
            <div style="text-align:center; padding:60px 20px; color:var(--text-muted);">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" style="margin-bottom:16px; color:var(--accent-green);">
                    <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
                    <polyline points="22 4 12 14.01 9 11.01"/>
                </svg>
                <h3 style="font-size:1.1rem; margin-bottom:4px;">All Clear</h3>
                <p>No suspicious activity detected in the uploaded data.</p>
            </div>
        `;
    }
}

/* =========================
   Analytics Report
   ========================= */
function renderAnalyticsReport() {
    const report = document.getElementById('reportContent');
    const s = currentSummary;
    const stats = s.stats;

    report.innerHTML = `
        <div class="report-section">
            <h4>📊 Dataset Overview</h4>
            <p>Analyzed <span class="report-metric">${s.total}</span> consumer records. 
            Mean consumption: <span class="report-metric">${stats.mean.toFixed(1)} kWh</span>, 
            Median: <span class="report-metric">${stats.median.toFixed(1)} kWh</span>, 
            Std Dev: <span class="report-metric">${stats.stdDev.toFixed(1)}</span>.
            Range: ${stats.min.toFixed(0)} – ${stats.max.toFixed(0)} kWh.</p>
        </div>

        <div class="report-section">
            <h4>🔍 Detection Results</h4>
            <p>
                <span class="report-metric" style="background:rgba(239,68,68,0.12); color:#ff6b6b;">${s.critical} Critical</span>
                <span class="report-metric" style="background:rgba(249,115,22,0.12); color:#ffb347;">${s.high} High</span>
                <span class="report-metric" style="background:rgba(245,158,11,0.1); color:#ffd666;">${s.medium} Medium</span>
                <span class="report-metric" style="background:rgba(59,130,246,0.1); color:#6da9ff;">${s.low} Low</span>
                <span class="report-metric" style="background:rgba(16,185,129,0.08); color:#5de0b5;">${s.normal} Normal</span>
            </p>
            <p style="margin-top:8px;">Estimated theft rate: <span class="report-metric" style="background:rgba(239,68,68,0.12); color:#ff6b6b;">${s.theftRate}%</span> 
            with an estimated loss of <span class="report-metric">${Number(s.estimatedLoss).toLocaleString()} kWh</span>.</p>
        </div>

        <div class="report-section">
            <h4>🛡️ Detection Methods Applied</h4>
            <ul style="list-style:none; padding:0;">
                <li style="padding:4px 0;">${s.lstmEnabled ? '✅' : '⬜'} <strong>LSTM Deep Learning</strong> — Kocaman & Tümen (2020), LSTM(128), 5-fold CV ${s.lstmEnabled ? '<span class="report-metric" style="background:rgba(168,85,247,0.12); color:#c084fc;">Active</span>' : '<span style="color:var(--text-muted);">(No time-series data)</span>'}</li>
                <li style="padding:4px 0;">✅ Z-Score Analysis (threshold: ±2σ)</li>
                <li style="padding:4px 0;">✅ IQR Outlier Detection (1.5× multiplier)</li>
                <li style="padding:4px 0;">✅ Consumption-to-Billing Ratio Analysis</li>
                <li style="padding:4px 0;">✅ Sanctioned vs Actual Load Comparison</li>
                <li style="padding:4px 0;">✅ Consumption Change Detection (>50% drop)</li>
                <li style="padding:4px 0;">✅ Meter Status Anomaly Detection</li>
                <li style="padding:4px 0;">✅ Hybrid Risk Scoring: ${s.lstmEnabled ? 'LSTM (40%) + Statistical (60%)' : 'Statistical only (100%)'}</li>
            </ul>
        </div>

        <div class="report-section">
            <h4>💡 Recommendations</h4>
            <p>${s.critical > 0 ? `<strong style="color:#ff6b6b;">Immediate Action Required:</strong> ${s.critical} consumers flagged as critical risk. Recommend physical meter inspection and field verification for these accounts.` : 'No critical cases found.'}</p>
            ${s.high > 0 ? `<p style="margin-top:6px;"><strong style="color:#ffb347;">High Priority:</strong> ${s.high} high-risk consumers should be scheduled for audit within the next billing cycle.</p>` : ''}
            ${s.medium > 0 ? `<p style="margin-top:6px;"><strong style="color:#ffd666;">Monitor:</strong> ${s.medium} medium-risk consumers should be monitored for patterns across future billing periods.</p>` : ''}
        </div>
    `;

    // Render LSTM Model Performance panel if available
    renderLSTMReport();
}

function renderLSTMReport() {
    const panel = document.getElementById('lstmPanel');
    const content = document.getElementById('lstmReportContent');
    if (!panel || !content) return;

    if (!currentSummary.lstmEnabled || !currentSummary.lstmMetrics) {
        panel.style.display = 'none';
        return;
    }

    panel.style.display = 'block';
    const m = currentSummary.lstmMetrics;
    const info = currentSummary.lstmModelInfo;

    content.innerHTML = `
        <div class="report-section">
            <h4>📐 Model Architecture (per Kocaman & Tümen, Sadhana 2020)</h4>
            <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap:12px; margin-top:10px;">
                ${info ? info.layers.map(l => `
                    <div style="padding:10px 14px; border-radius:8px; background:rgba(168,85,247,0.06); border:1px solid rgba(168,85,247,0.12);">
                        <div style="font-size:0.75rem; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.5px;">${l.type}</div>
                        <div style="font-size:0.9rem; font-weight:600; color:var(--text-primary);">${l.name}</div>
                        <div style="font-size:0.78rem; color:var(--text-secondary);">${l.params.toLocaleString()} params</div>
                    </div>
                `).join('') : ''}
            </div>
            ${info ? `<p style="margin-top:10px;">Total parameters: <span class="report-metric" style="background:rgba(168,85,247,0.12); color:#c084fc;">${info.totalParams.toLocaleString()}</span> · Sequence length: <span class="report-metric">${info.sequenceLength}</span> · Epochs: <span class="report-metric">${info.epochs}</span> · Batch size: <span class="report-metric">${info.batchSize}</span> · Threshold: <span class="report-metric">${info.threshold.toFixed(3)}</span></p>` : ''}
        </div>

        <div class="report-section">
            <h4>📊 5-Fold Cross-Validation Metrics</h4>
            <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:12px; margin-top:10px;">
                <div style="text-align:center; padding:16px; border-radius:10px; background:rgba(16,185,129,0.06); border:1px solid rgba(16,185,129,0.12);">
                    <div style="font-size:1.6rem; font-weight:700; color:#10b981; font-family:'JetBrains Mono',monospace;">${(m.accuracy * 100).toFixed(1)}%</div>
                    <div style="font-size:0.75rem; color:var(--text-secondary); text-transform:uppercase; letter-spacing:0.5px;">Accuracy</div>
                </div>
                <div style="text-align:center; padding:16px; border-radius:10px; background:rgba(0,212,255,0.06); border:1px solid rgba(0,212,255,0.12);">
                    <div style="font-size:1.6rem; font-weight:700; color:#00d4ff; font-family:'JetBrains Mono',monospace;">${(m.precision * 100).toFixed(1)}%</div>
                    <div style="font-size:0.75rem; color:var(--text-secondary); text-transform:uppercase; letter-spacing:0.5px;">Precision</div>
                </div>
                <div style="text-align:center; padding:16px; border-radius:10px; background:rgba(168,85,247,0.06); border:1px solid rgba(168,85,247,0.12);">
                    <div style="font-size:1.6rem; font-weight:700; color:#a855f7; font-family:'JetBrains Mono',monospace;">${(m.recall * 100).toFixed(1)}%</div>
                    <div style="font-size:0.75rem; color:var(--text-secondary); text-transform:uppercase; letter-spacing:0.5px;">Recall</div>
                </div>
                <div style="text-align:center; padding:16px; border-radius:10px; background:rgba(245,158,11,0.06); border:1px solid rgba(245,158,11,0.12);">
                    <div style="font-size:1.6rem; font-weight:700; color:#f59e0b; font-family:'JetBrains Mono',monospace;">${(m.f1 * 100).toFixed(1)}%</div>
                    <div style="font-size:0.75rem; color:var(--text-secondary); text-transform:uppercase; letter-spacing:0.5px;">F1 Score</div>
                </div>
            </div>
            ${m.foldDetails ? `
            <div style="margin-top:14px;">
                <table style="width:100%; font-size:0.8rem; border-collapse:collapse;">
                    <tr style="color:var(--text-muted); text-transform:uppercase; font-size:0.7rem; letter-spacing:0.5px;">
                        <th style="text-align:left; padding:6px 10px; border-bottom:1px solid var(--border-color);">Fold</th>
                        <th style="text-align:center; padding:6px 10px; border-bottom:1px solid var(--border-color);">Accuracy</th>
                        <th style="text-align:center; padding:6px 10px; border-bottom:1px solid var(--border-color);">Precision</th>
                        <th style="text-align:center; padding:6px 10px; border-bottom:1px solid var(--border-color);">Recall</th>
                        <th style="text-align:center; padding:6px 10px; border-bottom:1px solid var(--border-color);">F1</th>
                    </tr>
                    ${m.foldDetails.map((f, i) => `
                        <tr style="color:var(--text-secondary);">
                            <td style="padding:6px 10px; border-bottom:1px solid rgba(0,212,255,0.04);">Fold ${i + 1}</td>
                            <td style="text-align:center; padding:6px 10px; border-bottom:1px solid rgba(0,212,255,0.04); font-family:'JetBrains Mono',monospace;">${(f.accuracy * 100).toFixed(1)}%</td>
                            <td style="text-align:center; padding:6px 10px; border-bottom:1px solid rgba(0,212,255,0.04); font-family:'JetBrains Mono',monospace;">${(f.precision * 100).toFixed(1)}%</td>
                            <td style="text-align:center; padding:6px 10px; border-bottom:1px solid rgba(0,212,255,0.04); font-family:'JetBrains Mono',monospace;">${(f.recall * 100).toFixed(1)}%</td>
                            <td style="text-align:center; padding:6px 10px; border-bottom:1px solid rgba(0,212,255,0.04); font-family:'JetBrains Mono',monospace;">${(f.f1 * 100).toFixed(1)}%</td>
                        </tr>
                    `).join('')}
                </table>
            </div>` : ''}
        </div>

        <div class="report-section">
            <h4>📄 Reference</h4>
            <p style="font-style:italic; color:var(--text-secondary);">
                Kocaman, B. & Tümen, V. (2020). "Detection of electricity theft using data processing and LSTM method in distribution systems." 
                <em>Sadhana</em>, 45, 286. 
                <a href="https://www.ias.ac.in/article/fulltext/sadh/045/0286" target="_blank" style="color:var(--accent-cyan); text-decoration:none;">DOI Link ↗</a>
            </p>
        </div>
    `;
}

/* =========================
   Export
   ========================= */
function exportResults() {
    if (!currentResults || currentResults.length === 0) {
        showToast('warning', 'No data to export.');
        return;
    }

    const csvRows = [];
    const hasLSTM = currentResults.some(r => r.lstmProb !== null && r.lstmProb !== undefined);
    const headers = ['Consumer ID', 'Name', 'Region', 'Category', 'Consumption (kWh)', 'Billing', ...(hasLSTM ? ['LSTM Probability'] : []), 'Risk Score', 'Risk Level', 'Flags'];
    csvRows.push(headers.join(','));

    currentResults.forEach(r => {
        csvRows.push([
            r.id,
            `"${r.name}"`,
            `"${r.region}"`,
            r.category,
            r.consumption,
            r.billing,
            ...(hasLSTM ? [r.lstmProb !== null ? (r.lstmProb * 100).toFixed(1) + '%' : 'N/A'] : []),
            r.riskScore,
            r.riskLevel,
            `"${r.flags.join('; ')}"`
        ].join(','));
    });

    const blob = new Blob([csvRows.join('\n')], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `electraguard_results_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);

    showToast('success', 'Results exported as CSV successfully.');
}

/* =========================
   Reset / Upload New
   ========================= */
function resetApp() {
    engine = new TheftDetectionEngine();
    chartManager.destroyAll();
    currentResults = [];
    currentSummary = null;
    currentPage = 1;

    document.getElementById('dashboardSection').style.display = 'none';
    document.getElementById('heroSection').style.display = 'flex';

    const navStatus = document.getElementById('navStatus');
    navStatus.innerHTML = '<div class="status-dot offline"></div><span>No Data Loaded</span>';

    document.getElementById('alertBadge').style.display = 'none';

    // Reset file input
    document.getElementById('fileInput').value = '';

    // Reset nav
    document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
    document.querySelector('.nav-link[data-section="dashboard"]')?.classList.add('active');
    document.querySelectorAll('.section-panel').forEach(p => p.style.display = 'none');
    document.getElementById('section-dashboard').style.display = 'block';
}

/* =========================
   Utility Functions
   ========================= */
function showProgress(show) {
    const inner = document.querySelector('.upload-zone-inner');
    const progress = document.getElementById('uploadProgress');
    if (show) {
        inner.style.display = 'none';
        progress.style.display = 'block';
    } else {
        inner.style.display = 'block';
        progress.style.display = 'none';
        document.getElementById('progressFill').style.width = '0%';
    }
}

function updateProgress(percent, text) {
    document.getElementById('progressFill').style.width = percent + '%';
    document.getElementById('progressText').textContent = text;
}

function animateCounter(elementId, target) {
    const el = document.getElementById(elementId);
    if (!el) return;
    const duration = 800;
    const start = performance.now();

    function tick(now) {
        const elapsed = now - start;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3); // ease out cubic
        el.textContent = Math.round(eased * target);
        if (progress < 1) requestAnimationFrame(tick);
    }

    requestAnimationFrame(tick);
}

function getRiskColor(level) {
    switch (level) {
        case 'critical': return '#ef4444';
        case 'high': return '#f97316';
        case 'medium': return '#f59e0b';
        case 'low': return '#3b82f6';
        default: return '#10b981';
    }
}

function showToast(type, message) {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;

    const icons = {
        success: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
        error: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>',
        warning: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
        info: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>'
    };

    toast.innerHTML = `<span class="toast-icon">${icons[type] || icons.info}</span><span>${message}</span>`;
    container.appendChild(toast);

    setTimeout(() => {
        toast.classList.add('toast-exit');
        setTimeout(() => toast.remove(), 300);
    }, 4500);
}
