/**
 * ElectraGuard — Hybrid Detection Engine
 * 
 * Combines:
 *   A) LSTM Deep Learning (Kocaman & Tümen, 2020 — Sadhana 45:286)
 *      - Data cleaning → Min-Max normalization → LSTM(128) → 5-fold CV
 *   B) Statistical Analysis (Z-Score, IQR, billing ratio, load analysis)
 *   C) Multi-Factor Composite Risk Scoring
 * 
 * The LSTM provides a learned theft probability per consumer.
 * Statistical methods provide interpretable flags.
 * Both are fused into a final composite risk score.
 */

class TheftDetectionEngine {
    constructor() {
        this.data = [];
        this.results = [];
        this.stats = {};
        this.columnMap = {};
        this.lstmEngine = null;
        this.lstmResults = new Map(); // consumerId → theft probability
        this.lstmMetrics = null;
        this.lstmModelInfo = null;
        this.useLSTM = false;
    }

    /**
     * Intelligently map spreadsheet columns to expected fields.
     * Supports flexible column naming.
     */
    mapColumns(headers) {
        const mappings = {
            id: ['consumer_id', 'consumer id', 'id', 'cust_id', 'customer_id', 'customer id', 'meter_id', 'meter id', 'account', 'acc_no', 'account_no', 'sr', 'sr_no', 'serial', 's.no', 'sno'],
            name: ['name', 'consumer_name', 'consumer name', 'customer_name', 'customer name', 'cust_name'],
            region: ['region', 'area', 'zone', 'district', 'location', 'address', 'city', 'sector', 'feeder', 'subdivision'],
            consumption: ['consumption', 'consumption_kwh', 'consumption kwh', 'kwh', 'units', 'units_consumed', 'units consumed', 'energy', 'energy_kwh', 'actual_consumption', 'meter_reading', 'reading', 'usage', 'load'],
            billing: ['billing', 'billing_amount', 'billing amount', 'bill', 'bill_amount', 'bill amount', 'amount', 'charge', 'charges', 'total_bill', 'billed_amount', 'revenue'],
            sanctioned_load: ['sanctioned_load', 'sanctioned load', 'sanc_load', 'load_sanctioned', 'contract_demand', 'contract demand', 'connected_load', 'connected load', 'max_demand', 'max demand'],
            actual_load: ['actual_load', 'actual load', 'measured_load', 'measured load', 'current_load', 'peak_load', 'peak load', 'demand'],
            meter_status: ['meter_status', 'meter status', 'status', 'meter_condition', 'meter condition', 'defective', 'faulty'],
            category: ['category', 'type', 'consumer_type', 'consumer type', 'tariff', 'tariff_type', 'connection_type', 'connection type'],
            date: ['date', 'reading_date', 'reading date', 'month', 'period', 'billing_date', 'billing date', 'bill_date'],
            previous_consumption: ['previous_consumption', 'prev_consumption', 'previous consumption', 'last_consumption', 'last consumption', 'prev_reading', 'prev_units'],
            flag: ['flag', 'label', 'theft', 'is_theft', 'theft_flag', 'fraud', 'anomaly', 'class', 'target']
        };

        const map = {};
        const normalizedHeaders = headers.map(h => h.toString().toLowerCase().trim().replace(/[^a-z0-9_ ]/g, ''));

        for (const [field, aliases] of Object.entries(mappings)) {
            const idx = normalizedHeaders.findIndex(h => aliases.includes(h));
            if (idx !== -1) {
                map[field] = headers[idx];
            }
        }

        this.columnMap = map;
        return map;
    }

    /**
     * Process raw spreadsheet data into normalized format
     */
    processData(rawData, headers) {
        this.mapColumns(headers);
        const map = this.columnMap;

        // Auto-detect numeric columns if consumption not mapped
        if (!map.consumption) {
            const numericCols = headers.filter(h => {
                const vals = rawData.slice(0, 10).map(r => parseFloat(r[h])).filter(v => !isNaN(v));
                return vals.length > 5;
            });
            if (numericCols.length > 0) {
                let maxAvg = 0;
                let bestCol = numericCols[0];
                for (const col of numericCols) {
                    const avg = rawData.reduce((s, r) => s + (parseFloat(r[col]) || 0), 0) / rawData.length;
                    if (avg > maxAvg) { maxAvg = avg; bestCol = col; }
                }
                map.consumption = bestCol;
            }
        }

        // Detect time-series columns (day_1, day_2, ... or col with dates)
        const timeSeriesCols = headers.filter(h => {
            const lower = h.toString().toLowerCase().trim();
            return /^(day|d|reading|r|t|period|p|week|w|month|m)[\s_-]?\d+$/i.test(lower)
                || /^\d{4}[-\/]\d{2}[-\/]\d{2}$/.test(lower)
                || /^\d{1,2}[-\/]\d{1,2}[-\/]\d{2,4}$/.test(lower);
        });

        this.data = rawData.map((row, idx) => {
            // Build daily consumption series from time-series columns
            let consumptionSeries = [];
            if (timeSeriesCols.length > 0) {
                consumptionSeries = timeSeriesCols.map(col => parseFloat(row[col]) || 0);
            }

            // Build label if present (for supervised LSTM mode)
            let label = undefined;
            if (map.flag) {
                const flagVal = row[map.flag];
                if (flagVal !== undefined && flagVal !== null && flagVal !== '') {
                    label = parseFloat(flagVal) >= 1 ? 1 : 0;
                }
            }

            return {
                _index: idx,
                id: row[map.id] || `C-${String(idx + 1).padStart(4, '0')}`,
                name: row[map.name] || `Consumer ${idx + 1}`,
                region: row[map.region] || 'Unknown',
                consumption: parseFloat(row[map.consumption]) || 0,
                billing: parseFloat(row[map.billing]) || 0,
                sanctionedLoad: parseFloat(row[map.sanctioned_load]) || 0,
                actualLoad: parseFloat(row[map.actual_load]) || 0,
                meterStatus: row[map.meter_status] || 'OK',
                category: row[map.category] || 'General',
                date: row[map.date] || '',
                previousConsumption: parseFloat(row[map.previous_consumption]) || 0,
                consumptionSeries,
                label,
                _raw: row
            };
        });

        // Check if LSTM is viable (need time-series data or enough features)
        this.useLSTM = timeSeriesCols.length >= 7 ||
            this.data.some(d => d.consumptionSeries.length >= 7);

        return this.data;
    }

    /**
     * Calculate basic statistics
     */
    calculateStats() {
        const consumptions = this.data.map(d => d.consumption).filter(c => c > 0);

        if (consumptions.length === 0) {
            this.stats = { mean: 0, median: 0, stdDev: 0, q1: 0, q3: 0, iqr: 0, min: 0, max: 0 };
            return this.stats;
        }

        const sorted = [...consumptions].sort((a, b) => a - b);
        const n = sorted.length;

        const mean = consumptions.reduce((a, b) => a + b, 0) / n;
        const variance = consumptions.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / n;
        const stdDev = Math.sqrt(variance);

        const median = n % 2 === 0
            ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2
            : sorted[Math.floor(n / 2)];

        const q1 = sorted[Math.floor(n * 0.25)];
        const q3 = sorted[Math.floor(n * 0.75)];
        const iqr = q3 - q1;

        this.stats = {
            mean, median, stdDev, q1, q3, iqr,
            min: sorted[0],
            max: sorted[n - 1],
            total: consumptions.reduce((a, b) => a + b, 0),
            count: n
        };

        return this.stats;
    }

    /**
     * Run the LSTM engine (Kocaman & Tümen method)
     * This is async because TensorFlow.js training is async.
     */
    async runLSTMDetection(onProgress) {
        if (!window.LSTMDetectionEngine) {
            console.warn('LSTM engine not loaded. Falling back to statistical methods.');
            return false;
        }

        // Prepare data for LSTM
        let consumers = [];

        if (this.data.some(d => d.consumptionSeries.length >= 7)) {
            // Use actual time-series columns
            consumers = this.data
                .filter(d => d.consumptionSeries.length >= 7)
                .map(d => ({
                    id: d.id,
                    consumptionSeries: d.consumptionSeries,
                    label: d.label
                }));
        } else {
            // Synthesize time-series from cross-sectional data
            // Group by consumer ID and create pseudo-series from aggregate features
            consumers = this.data.map(d => {
                // Create a pseudo time-series using available numeric features
                // This allows the LSTM to work even with single-row-per-consumer data
                const series = this.generatePseudoTimeSeries(d);
                return {
                    id: d.id,
                    consumptionSeries: series,
                    label: d.label
                };
            }).filter(c => c.consumptionSeries.length >= 7);
        }

        if (consumers.length < 10) {
            console.warn('Insufficient data for LSTM training. Need at least 10 consumers with time-series.');
            return false;
        }

        // Initialize and train LSTM
        this.lstmEngine = new LSTMDetectionEngine();

        // Auto-label using statistical pre-screening if no labels exist
        const hasLabels = consumers.some(c => c.label !== undefined);
        if (!hasLabels) {
            this.autoLabelForLSTM(consumers);
        }

        try {
            this.lstmMetrics = await this.lstmEngine.trainModel(consumers, onProgress);

            // Run predictions for all consumers
            for (const consumer of consumers) {
                const prob = await this.lstmEngine.predict(consumer.consumptionSeries);
                if (prob !== null) {
                    this.lstmResults.set(consumer.id, prob);
                }
            }

            this.lstmModelInfo = this.lstmEngine.getModelInfo();
            return true;

        } catch (err) {
            console.error('LSTM training failed:', err);
            return false;
        }
    }

    /**
     * Generate pseudo time-series from a single consumer record
     * Uses consumption, billing, load, and derived features to create
     * a feature vector that LSTM can process.
     */
    generatePseudoTimeSeries(consumer) {
        const { mean, stdDev } = this.stats;
        const c = consumer.consumption;
        const b = consumer.billing;
        const sLoad = consumer.sanctionedLoad;
        const aLoad = consumer.actualLoad;
        const prev = consumer.previousConsumption;

        // Construct a 10-feature pseudo-series
        const features = [
            c > 0 ? c / (mean || 1) : 0,                                    // normalized consumption
            stdDev > 0 ? (c - mean) / stdDev : 0,                           // z-score
            b > 0 && c > 0 ? c / b : 0,                                     // consumption/billing ratio
            sLoad > 0 ? aLoad / sLoad : 0,                                  // load ratio
            prev > 0 ? (c - prev) / prev : 0,                               // consumption change
            c > 0 ? Math.log(c + 1) / Math.log((mean || 1) + 1) : 0,       // log-normalized
            this.stats.iqr > 0 ? (c - this.stats.q1) / this.stats.iqr : 0, // IQR position
            c > 0 ? Math.min(c / (this.stats.max || 1), 1) : 0,            // max-normalized
            prev > 0 ? c / prev : 0,                                        // current/previous ratio
            c <= 0 ? 1 : 0                                                   // zero-flag
        ];

        return features;
    }

    /**
     * Auto-label consumers for LSTM training using statistical heuristics
     * Per the paper: uses pre-processing to identify likely theft cases
     */
    autoLabelForLSTM(consumers) {
        const { mean, stdDev, q1, iqr } = this.stats;
        const lowerBound = q1 - 1.5 * iqr;

        for (const consumer of consumers) {
            const series = consumer.consumptionSeries;
            const avgSeries = series.reduce((a, b) => a + b, 0) / series.length;

            // Heuristic labeling
            let suspicious = false;

            // Very low average consumption
            if (mean > 0 && avgSeries < mean * 0.15) suspicious = true;
            // High variance in consumption (tampering pattern)
            const seriesStd = Math.sqrt(series.reduce((s, v) => s + Math.pow(v - avgSeries, 2), 0) / series.length);
            if (avgSeries > 0 && seriesStd / avgSeries > 1.5) suspicious = true;
            // Below IQR lower bound
            if (avgSeries < lowerBound && avgSeries > 0) suspicious = true;
            // Multiple zero readings
            const zeroCount = series.filter(v => v <= 0).length;
            if (zeroCount > series.length * 0.3) suspicious = true;

            consumer.label = suspicious ? 1 : 0;
        }

        // Ensure at least ~10% positive labels for training balance
        const positiveCount = consumers.filter(c => c.label === 1).length;
        const targetPositive = Math.max(Math.ceil(consumers.length * 0.1), 5);

        if (positiveCount < targetPositive) {
            // Sort by average consumption ascending and label lowest as suspicious
            const sortedByAvg = [...consumers]
                .filter(c => c.label === 0)
                .sort((a, b) => {
                    const avgA = a.consumptionSeries.reduce((s, v) => s + v, 0) / a.consumptionSeries.length;
                    const avgB = b.consumptionSeries.reduce((s, v) => s + v, 0) / b.consumptionSeries.length;
                    return avgA - avgB;
                });

            const needed = targetPositive - positiveCount;
            for (let i = 0; i < Math.min(needed, sortedByAvg.length); i++) {
                sortedByAvg[i].label = 1;
            }
        }
    }

    // =========================================
    // Statistical Detection Methods
    // =========================================

    /**
     * Z-Score anomaly detection
     */
    zScoreAnalysis(threshold = 2) {
        const { mean, stdDev } = this.stats;
        if (stdDev === 0) return [];

        return this.data.map(d => {
            const zScore = (d.consumption - mean) / stdDev;
            return {
                ...d,
                zScore: parseFloat(zScore.toFixed(3)),
                isZAnomaly: Math.abs(zScore) > threshold || (d.consumption > 0 && zScore < -threshold)
            };
        });
    }

    /**
     * IQR-based outlier detection
     */
    iqrAnalysis(multiplier = 1.5) {
        const { q1, q3, iqr } = this.stats;
        const lowerBound = q1 - multiplier * iqr;
        const upperBound = q3 + multiplier * iqr;

        return this.data.map(d => ({
            ...d,
            isIQRAnomaly: d.consumption < lowerBound || d.consumption > upperBound,
            iqrLower: parseFloat(lowerBound.toFixed(2)),
            iqrUpper: parseFloat(upperBound.toFixed(2))
        }));
    }

    /**
     * Consumption-to-billing ratio analysis
     */
    billingAnalysis() {
        if (!this.columnMap.billing) return this.data.map(d => ({ ...d, billingAnomaly: false, billingRatio: 0 }));

        const ratios = this.data
            .filter(d => d.billing > 0 && d.consumption > 0)
            .map(d => d.consumption / d.billing);

        if (ratios.length === 0) return this.data.map(d => ({ ...d, billingAnomaly: false, billingRatio: 0 }));

        const avgRatio = ratios.reduce((a, b) => a + b, 0) / ratios.length;
        const ratioStd = Math.sqrt(ratios.reduce((s, r) => s + Math.pow(r - avgRatio, 2), 0) / ratios.length);

        return this.data.map(d => {
            const ratio = d.billing > 0 ? d.consumption / d.billing : 0;
            return {
                ...d,
                billingRatio: parseFloat(ratio.toFixed(3)),
                billingAnomaly: ratioStd > 0 && Math.abs(ratio - avgRatio) > 2 * ratioStd
            };
        });
    }

    /**
     * Load analysis — compare sanctioned vs actual
     */
    loadAnalysis() {
        if (!this.columnMap.sanctioned_load) return this.data.map(d => ({ ...d, loadAnomaly: false }));

        return this.data.map(d => ({
            ...d,
            loadAnomaly: d.sanctionedLoad > 0 && d.actualLoad > d.sanctionedLoad * 1.2,
            loadRatio: d.sanctionedLoad > 0 ? parseFloat((d.actualLoad / d.sanctionedLoad).toFixed(2)) : 0
        }));
    }

    /**
     * Consumption change analysis
     */
    consumptionChangeAnalysis() {
        if (!this.columnMap.previous_consumption) return this.data.map(d => ({ ...d, suddenDrop: false, changePercent: 0 }));

        return this.data.map(d => {
            const change = d.previousConsumption > 0
                ? ((d.consumption - d.previousConsumption) / d.previousConsumption) * 100
                : 0;
            return {
                ...d,
                changePercent: parseFloat(change.toFixed(1)),
                suddenDrop: change < -50
            };
        });
    }

    // =========================================
    // Hybrid Risk Scoring (LSTM + Statistical)
    // =========================================

    /**
     * Multi-factor risk scoring (0–100)
     * Fuses LSTM deep learning probability with statistical flags.
     * 
     * Score composition:
     *   - LSTM theft probability: 40 points (when available)
     *   - Z-Score anomaly:        15 points (25 if no LSTM)
     *   - IQR outlier:            10 points (20 if no LSTM)
     *   - Low consumption:        10 points (20 if no LSTM)
     *   - Billing mismatch:       10 points (15 if no LSTM)
     *   - Load violation:          5 points (10 if no LSTM)
     *   - Sudden drop:             5 points (10 if no LSTM)
     *   - Zero usage:              5 points (15 if no LSTM)
     *   - Meter issues:            (statistical bonus)
     */
    calculateRiskScores() {
        const zResults = this.zScoreAnalysis();
        const iqrResults = this.iqrAnalysis();
        const billingResults = this.billingAnalysis();
        const loadResults = this.loadAnalysis();
        const changeResults = this.consumptionChangeAnalysis();
        const { mean } = this.stats;
        const hasLSTM = this.lstmResults.size > 0;

        this.results = this.data.map((d, i) => {
            let riskScore = 0;
            const flags = [];
            let lstmProb = null;

            // ──────── LSTM Score Component (40 pts max) ────────
            if (hasLSTM && this.lstmResults.has(d.id)) {
                lstmProb = this.lstmResults.get(d.id);
                const lstmPoints = Math.round(lstmProb * 40);
                riskScore += lstmPoints;
                if (lstmProb >= 0.7) {
                    flags.push(`LSTM: High theft probability (${(lstmProb * 100).toFixed(1)}%)`);
                } else if (lstmProb >= 0.4) {
                    flags.push(`LSTM: Moderate theft probability (${(lstmProb * 100).toFixed(1)}%)`);
                }
            }

            // Weight multiplier — if no LSTM, statistical methods get more weight
            const w = hasLSTM ? 1 : 1.67;

            // ──────── Factor 1: Z-Score anomaly ────────
            const zData = zResults[i];
            if (zData.isZAnomaly) {
                riskScore += Math.round(15 * w);
                flags.push(`Z-Score anomaly (z=${zData.zScore})`);
            } else if (Math.abs(zData.zScore) > 1.5) {
                riskScore += Math.round(6 * w);
            }

            // ──────── Factor 2: IQR outlier ────────
            if (iqrResults[i].isIQRAnomaly) {
                riskScore += Math.round(10 * w);
                flags.push('IQR outlier');
            }

            // ──────── Factor 3: Very low consumption ────────
            if (d.consumption > 0 && d.consumption < mean * 0.2) {
                riskScore += Math.round(10 * w);
                flags.push(`Unusually low consumption (${d.consumption.toFixed(0)} kWh vs avg ${mean.toFixed(0)} kWh)`);
            } else if (d.consumption > 0 && d.consumption < mean * 0.4) {
                riskScore += Math.round(5 * w);
            }

            // ──────── Factor 4: Billing mismatch ────────
            if (billingResults[i].billingAnomaly) {
                riskScore += Math.round(10 * w);
                flags.push(`Billing ratio anomaly (ratio=${billingResults[i].billingRatio})`);
            }

            // ──────── Factor 5: Load exceeds sanctioned ────────
            if (loadResults[i].loadAnomaly) {
                riskScore += Math.round(5 * w);
                flags.push(`Load exceeds sanctioned (${loadResults[i].loadRatio}x)`);
            }

            // ──────── Factor 6: Sudden consumption drop ────────
            if (changeResults[i].suddenDrop) {
                riskScore += Math.round(5 * w);
                flags.push(`Sudden drop (${changeResults[i].changePercent}%)`);
            }

            // ──────── Factor 7: Zero consumption ────────
            if (d.consumption <= 0) {
                riskScore += Math.round(5 * w);
                flags.push('Zero consumption recorded');
            }

            // ──────── Factor 8: Meter status issues ────────
            const meterIssues = ['faulty', 'defective', 'dead', 'stuck', 'tampered', 'bypassed', 'error'];
            if (meterIssues.some(issue => d.meterStatus.toLowerCase().includes(issue))) {
                riskScore += 10;
                flags.push(`Meter issue: ${d.meterStatus}`);
            }

            // Cap at 100
            riskScore = Math.min(100, riskScore);

            // Determine risk level
            let riskLevel;
            if (riskScore >= 75) riskLevel = 'critical';
            else if (riskScore >= 55) riskLevel = 'high';
            else if (riskScore >= 35) riskLevel = 'medium';
            else if (riskScore >= 15) riskLevel = 'low';
            else riskLevel = 'normal';

            return {
                ...d,
                zScore: zData.zScore,
                riskScore,
                riskLevel,
                flags,
                lstmProb,
                billingRatio: billingResults[i].billingRatio,
                changePercent: changeResults[i].changePercent || 0
            };
        });

        // Sort by risk score descending
        this.results.sort((a, b) => b.riskScore - a.riskScore);

        return this.results;
    }

    /**
     * Get summary statistics for the dashboard
     */
    getSummary() {
        const total = this.results.length;
        const critical = this.results.filter(r => r.riskLevel === 'critical').length;
        const high = this.results.filter(r => r.riskLevel === 'high').length;
        const medium = this.results.filter(r => r.riskLevel === 'medium').length;
        const low = this.results.filter(r => r.riskLevel === 'low').length;
        const normal = this.results.filter(r => r.riskLevel === 'normal').length;
        const suspicious = critical + high;

        const totalConsumption = this.results.reduce((s, r) => s + r.consumption, 0);
        const avgConsumption = total > 0 ? totalConsumption / total : 0;

        const estimatedLoss = this.results
            .filter(r => r.riskLevel === 'critical' || r.riskLevel === 'high')
            .reduce((sum, r) => sum + Math.max(0, this.stats.mean - r.consumption), 0);

        const theftRate = total > 0 ? ((suspicious / total) * 100).toFixed(1) : '0.0';

        return {
            total,
            critical,
            high,
            medium,
            low,
            normal,
            suspicious,
            totalConsumption: totalConsumption.toFixed(0),
            avgConsumption: avgConsumption.toFixed(1),
            estimatedLoss: estimatedLoss.toFixed(0),
            theftRate,
            stats: this.stats,
            lstmEnabled: this.lstmResults.size > 0,
            lstmMetrics: this.lstmMetrics,
            lstmModelInfo: this.lstmModelInfo
        };
    }

    /**
     * Generate alert items for the alerts panel
     */
    getAlerts() {
        return this.results
            .filter(r => r.riskLevel === 'critical' || r.riskLevel === 'high' || r.riskLevel === 'medium')
            .map(r => ({
                id: r.id,
                name: r.name,
                region: r.region,
                riskLevel: r.riskLevel,
                riskScore: r.riskScore,
                consumption: r.consumption,
                flags: r.flags,
                category: r.category,
                lstmProb: r.lstmProb
            }));
    }

    /**
     * Generate sample data for demonstration
     * Includes time-series columns (day_1 through day_30) to enable LSTM
     */
    static generateSampleData(count = 150) {
        const regions = ['North Zone', 'South Zone', 'East Zone', 'West Zone', 'Central', 'Industrial Area', 'Residential-A', 'Residential-B'];
        const categories = ['Domestic', 'Commercial', 'Industrial', 'Agricultural'];
        const statuses = ['OK', 'OK', 'OK', 'OK', 'OK', 'OK', 'OK', 'Faulty', 'Defective', 'Stuck'];
        const months = ['Jan 2025', 'Feb 2025', 'Mar 2025'];
        const DAYS = 30; // 30 days of daily readings

        const data = [];
        for (let i = 0; i < count; i++) {
            const category = categories[Math.floor(Math.random() * categories.length)];
            let baseDailyConsumption;

            switch (category) {
                case 'Industrial':
                    baseDailyConsumption = 30 + Math.random() * 60;
                    break;
                case 'Commercial':
                    baseDailyConsumption = 10 + Math.random() * 30;
                    break;
                case 'Agricultural':
                    baseDailyConsumption = 8 + Math.random() * 15;
                    break;
                default:
                    baseDailyConsumption = 3 + Math.random() * 12;
            }

            // Generate daily consumption time-series
            const isThief = Math.random() < 0.12; // ~12% theft rate
            const dailyReadings = {};
            let totalConsumption = 0;

            for (let day = 1; day <= DAYS; day++) {
                let reading;
                if (isThief) {
                    // Theft patterns (per paper — irregular, sudden drops, near-zero)
                    const theftType = Math.random();
                    if (theftType < 0.3) {
                        // Type 1: Consistently very low
                        reading = baseDailyConsumption * (0.05 + Math.random() * 0.15);
                    } else if (theftType < 0.6) {
                        // Type 2: Normal with sudden drops to near-zero
                        reading = day % 5 === 0 ? baseDailyConsumption * 0.02 : baseDailyConsumption * (0.8 + Math.random() * 0.4);
                    } else if (theftType < 0.8) {
                        // Type 3: Erratic — high variance
                        reading = baseDailyConsumption * (Math.random() * 3);
                    } else {
                        // Type 4: Gradual decline (meter tampering)
                        reading = baseDailyConsumption * Math.max(0, 1 - (day / DAYS) * 0.8) * (0.8 + Math.random() * 0.4);
                    }
                } else {
                    // Normal consumption: stable with natural daily variation
                    const weekendFactor = (day % 7 === 0 || day % 7 === 6) ? 1.15 : 1.0;
                    reading = baseDailyConsumption * weekendFactor * (0.8 + Math.random() * 0.4);
                }
                reading = Math.max(0, Math.round(reading * 100) / 100);
                dailyReadings[`day_${day}`] = reading;
                totalConsumption += reading;
            }

            const consumption = Math.round(totalConsumption);
            const billing = Math.round(consumption * (5 + Math.random() * 3));
            const sanctionedLoad = Math.round(baseDailyConsumption * 0.5 * (8 + Math.random() * 4)) / 10;
            let actualLoad = sanctionedLoad * (0.6 + Math.random() * 0.4);
            const meterStatus = statuses[Math.floor(Math.random() * statuses.length)];
            const prevConsumption = Math.round(baseDailyConsumption * DAYS * (0.85 + Math.random() * 0.3));

            if (isThief && Math.random() < 0.3) {
                actualLoad = sanctionedLoad * (1.5 + Math.random() * 1.5);
            }

            data.push({
                'Consumer ID': `EG-${String(i + 1001).padStart(5, '0')}`,
                'Name': `Consumer ${i + 1}`,
                'Region': regions[Math.floor(Math.random() * regions.length)],
                'Category': category,
                'Consumption (kWh)': consumption,
                'Billing Amount': billing,
                'Sanctioned Load': parseFloat(sanctionedLoad.toFixed(1)),
                'Actual Load': parseFloat(actualLoad.toFixed(1)),
                'Meter Status': meterStatus,
                'Previous Consumption': prevConsumption,
                'Date': months[Math.floor(Math.random() * months.length)],
                'Flag': isThief ? 1 : 0,
                ...dailyReadings
            });
        }

        return data;
    }
}

// Export for use
window.TheftDetectionEngine = TheftDetectionEngine;
