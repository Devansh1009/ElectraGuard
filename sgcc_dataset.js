/**
 * ElectraGuard — SGCC-Style Dataset Generator
 * 
 * Generates a scientifically accurate electricity consumption dataset modeled
 * after the SGCC (State Grid Corporation of China) benchmark dataset used in:
 *   Kocaman, B. & Tümen, V. (2020). Sadhana, 45, 286.
 * 
 * SGCC Original:
 *   - 42,372 consumers, 1,035 days (Jan 2014 – Oct 2016)
 *   - ~15% theft rate (imbalanced)
 *   - 6 known theft attack types
 * 
 * This generator creates a browser-friendly subset with realistic:
 *   - Seasonal consumption patterns
 *   - Weekly cycles (weekday/weekend)
 *   - Gaussian noise
 *   - 6 theft attack types per the paper
 *   - Configurable size (default: 500 consumers × 60 days)
 */

class SGCCDatasetGenerator {

    /**
     * Generate a complete SGCC-style dataset
     * @param {number} numConsumers - Number of consumers (default 500)
     * @param {number} numDays - Number of daily readings (default 60)
     * @param {number} theftRate - Fraction of consumers who are thieves (default 0.15)
     * @returns {Object} { data: Array<Object>, stats: Object }
     */
    static generate(numConsumers = 500, numDays = 60, theftRate = 0.15) {
        const consumers = [];
        const numThieves = Math.round(numConsumers * theftRate);
        const theftIndices = new Set();

        // Randomly select which consumers are thieves
        while (theftIndices.size < numThieves) {
            theftIndices.add(Math.floor(Math.random() * numConsumers));
        }

        // Consumer categories with base consumption profiles (kWh/day)
        const profiles = [
            { category: 'Residential-Low',    base: 3,  variance: 1.5,  weight: 0.30 },
            { category: 'Residential-Medium',  base: 8,  variance: 3,    weight: 0.25 },
            { category: 'Residential-High',    base: 18, variance: 5,    weight: 0.10 },
            { category: 'Commercial-Small',    base: 25, variance: 8,    weight: 0.12 },
            { category: 'Commercial-Large',    base: 60, variance: 15,   weight: 0.08 },
            { category: 'Industrial',          base: 120, variance: 30,  weight: 0.08 },
            { category: 'Agricultural',        base: 15, variance: 7,    weight: 0.07 }
        ];

        // Build cumulative weights for category selection
        const cumWeights = [];
        let cumSum = 0;
        for (const p of profiles) {
            cumSum += p.weight;
            cumWeights.push(cumSum);
        }

        const regions = [
            'District-A', 'District-B', 'District-C', 'District-D',
            'District-E', 'District-F', 'Zone-North', 'Zone-South',
            'Zone-East', 'Zone-West', 'Industrial-Park', 'Rural-Sector'
        ];

        for (let i = 0; i < numConsumers; i++) {
            const isThief = theftIndices.has(i);

            // Select consumer category based on weighted distribution
            const rand = Math.random();
            const profileIdx = cumWeights.findIndex(w => rand <= w);
            const profile = profiles[Math.max(0, profileIdx)];

            // Generate base daily consumption series
            const dailySeries = this._generateNormalSeries(
                profile.base,
                profile.variance,
                numDays
            );

            // Apply seasonal pattern (sinusoidal — summer peak)
            const seasonalSeries = this._applySeasonalPattern(dailySeries, numDays);

            // Apply weekly pattern (higher on weekdays for commercial/industrial)
            const weeklySeries = this._applyWeeklyPattern(
                seasonalSeries,
                profile.category,
                numDays
            );

            let finalSeries;
            let attackType = 'none';

            if (isThief) {
                // Apply one of 6 theft attack types from the SGCC paper
                const attack = this._applyTheftAttack(weeklySeries, numDays);
                finalSeries = attack.series;
                attackType = attack.type;
            } else {
                finalSeries = weeklySeries;
            }

            // Ensure non-negative values
            finalSeries = finalSeries.map(v => Math.max(0, Math.round(v * 100) / 100));

            // Build row
            const totalConsumption = finalSeries.reduce((s, v) => s + v, 0);
            const avgDaily = totalConsumption / numDays;
            const billingRate = profile.category.includes('Industrial') ? 7.5 :
                               profile.category.includes('Commercial') ? 8.5 :
                               profile.category.includes('Agricultural') ? 3.5 : 6.0;
            const sanctionedLoad = Math.round(profile.base * 0.15 * 10) / 10;
            const actualLoad = isThief && Math.random() < 0.25
                ? sanctionedLoad * (1.3 + Math.random() * 0.7)
                : sanctionedLoad * (0.5 + Math.random() * 0.4);

            const meterStatuses = ['OK', 'OK', 'OK', 'OK', 'OK', 'OK'];
            if (isThief && Math.random() < 0.2) meterStatuses.push('Tampered', 'Faulty', 'Bypassed');
            else meterStatuses.push('OK', 'OK');

            const row = {
                'Consumer ID': `SGCC-${String(i + 1).padStart(5, '0')}`,
                'Name': `Consumer ${i + 1}`,
                'Region': regions[Math.floor(Math.random() * regions.length)],
                'Category': profile.category,
            };

            // Add daily readings
            for (let d = 0; d < numDays; d++) {
                row[`day_${d + 1}`] = finalSeries[d];
            }

            // Add aggregate columns
            row['Consumption (kWh)'] = Math.round(totalConsumption);
            row['Billing Amount'] = Math.round(totalConsumption * billingRate);
            row['Sanctioned Load'] = sanctionedLoad;
            row['Actual Load'] = Math.round(actualLoad * 10) / 10;
            row['Meter Status'] = meterStatuses[Math.floor(Math.random() * meterStatuses.length)];
            row['Previous Consumption'] = Math.round(totalConsumption * (0.85 + Math.random() * 0.3));
            row['Date'] = '2025-01';
            row['Flag'] = isThief ? 1 : 0;

            // Metadata (not used for detection, but useful for analysis)
            row['_attackType'] = attackType;

            consumers.push(row);
        }

        // Dataset statistics
        const stats = {
            totalConsumers: numConsumers,
            totalThieves: numThieves,
            theftRate: (theftRate * 100).toFixed(1) + '%',
            numDays,
            attackTypes: this._getAttackTypeDistribution(consumers),
            categoryDistribution: this._getCategoryDistribution(consumers)
        };

        return { data: consumers, stats };
    }

    /**
     * Generate a Gaussian (normal) random series
     */
    static _generateNormalSeries(mean, stdDev, length) {
        const series = [];
        for (let i = 0; i < length; i++) {
            // Box-Muller transform for normal distribution
            const u1 = Math.random();
            const u2 = Math.random();
            const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
            series.push(mean + z * stdDev);
        }
        return series;
    }

    /**
     * Apply seasonal sinusoidal pattern (summer peak)
     * Per SGCC: consumption peaks in summer months
     */
    static _applySeasonalPattern(series, numDays) {
        return series.map((v, i) => {
            // Sinusoidal with peak at ~day 180 (summer) in a year cycle
            const seasonFactor = 1 + 0.2 * Math.sin(2 * Math.PI * (i / 365 - 0.25));
            return v * seasonFactor;
        });
    }

    /**
     * Apply weekly cycle
     * Commercial/Industrial: higher on weekdays
     * Residential: slightly higher on weekends
     */
    static _applyWeeklyPattern(series, category, numDays) {
        const isCommercial = category.includes('Commercial') || category.includes('Industrial');
        return series.map((v, i) => {
            const dayOfWeek = i % 7;
            const isWeekend = dayOfWeek === 5 || dayOfWeek === 6;
            if (isCommercial) {
                return isWeekend ? v * 0.4 : v; // Commercial: low on weekends
            } else {
                return isWeekend ? v * 1.15 : v; // Residential: slight increase
            }
        });
    }

    /**
     * Apply one of 6 theft attack types from the SGCC paper
     * 
     * Types:
     *   1. Constant reduction — multiply all readings by α ∈ [0.1, 0.5]
     *   2. Random factor — multiply by random α(t) ∈ [0.1, 0.8] each day
     *   3. Time-based — reduce to near-zero during certain periods
     *   4. Mean-based — report the mean value every day (hides variation)
     *   5. Reverse pattern — swap high/low days
     *   6. Gradual decline — consumption slowly drops to near-zero
     */
    static _applyTheftAttack(series, numDays) {
        const attackIdx = Math.floor(Math.random() * 6);
        const attacked = [...series];
        let type;

        switch (attackIdx) {
            case 0: {
                // Type 1: Constant reduction f(x) = α·x, α ∈ [0.1, 0.5]
                type = 'Type-1: Constant Reduction';
                const alpha = 0.1 + Math.random() * 0.4;
                for (let i = 0; i < numDays; i++) {
                    attacked[i] = series[i] * alpha;
                }
                break;
            }
            case 1: {
                // Type 2: Random factor f(x) = α(t)·x, α(t) random each day
                type = 'Type-2: Random Factor';
                for (let i = 0; i < numDays; i++) {
                    const alphaT = 0.1 + Math.random() * 0.7;
                    attacked[i] = series[i] * alphaT;
                }
                break;
            }
            case 2: {
                // Type 3: Time-based — report 0 during certain periods
                type = 'Type-3: Time-Based Zero';
                const blockStart = Math.floor(Math.random() * (numDays * 0.5));
                const blockLen = Math.floor(numDays * (0.3 + Math.random() * 0.4));
                for (let i = 0; i < numDays; i++) {
                    if (i >= blockStart && i < blockStart + blockLen) {
                        attacked[i] = series[i] * (Math.random() * 0.05); // near zero
                    }
                }
                break;
            }
            case 3: {
                // Type 4: Mean-based — report the mean value every day
                type = 'Type-4: Mean Flattening';
                const mean = series.reduce((a, b) => a + b, 0) / numDays;
                const reducedMean = mean * (0.2 + Math.random() * 0.3);
                for (let i = 0; i < numDays; i++) {
                    attacked[i] = reducedMean + (Math.random() - 0.5) * 0.5;
                }
                break;
            }
            case 4: {
                // Type 5: Reverse pattern — swap max/min periods
                type = 'Type-5: Reverse Pattern';
                const sorted = [...series].sort((a, b) => a - b);
                const reverseSorted = [...sorted].reverse();
                // Map: original high values get low replacements
                const indices = series.map((v, i) => ({ v, i })).sort((a, b) => a.v - b.v);
                for (let j = 0; j < indices.length; j++) {
                    attacked[indices[j].i] = reverseSorted[j] * (0.3 + Math.random() * 0.3);
                }
                break;
            }
            case 5: {
                // Type 6: Gradual decline (meter tampering over time)
                type = 'Type-6: Gradual Decline';
                for (let i = 0; i < numDays; i++) {
                    const decayFactor = Math.max(0.02, 1 - (i / numDays) * (0.7 + Math.random() * 0.3));
                    attacked[i] = series[i] * decayFactor;
                }
                break;
            }
        }

        return { series: attacked, type };
    }

    /**
     * Get attack type distribution
     */
    static _getAttackTypeDistribution(consumers) {
        const dist = {};
        consumers.filter(c => c['Flag'] === 1).forEach(c => {
            const type = c['_attackType'];
            dist[type] = (dist[type] || 0) + 1;
        });
        return dist;
    }

    /**
     * Get category distribution
     */
    static _getCategoryDistribution(consumers) {
        const dist = {};
        consumers.forEach(c => {
            const cat = c['Category'];
            dist[cat] = (dist[cat] || 0) + 1;
        });
        return dist;
    }

    /**
     * Convert dataset to XLSX Blob for download
     */
    static toXLSX(data) {
        // Remove internal _attackType column for the export
        const cleanData = data.map(row => {
            const clean = { ...row };
            delete clean['_attackType'];
            return clean;
        });

        const worksheet = XLSX.utils.json_to_sheet(cleanData);
        const workbook = XLSX.utils.book_new();
        XLSX.utils.book_append_sheet(workbook, worksheet, 'Electricity Data');
        const xlsxData = XLSX.write(workbook, { bookType: 'xlsx', type: 'array' });
        return new Blob([xlsxData], {
            type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        });
    }
}

window.SGCCDatasetGenerator = SGCCDatasetGenerator;
