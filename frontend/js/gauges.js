// --- gauges.js ---

// Declare gauge objects in the global scope
let lossGauge, speedGauge;

function setupGauges() {
    const commonOpts = {
        angle: 0,
        lineWidth: 0.2,
        radiusScale: 0.9,
        pointer: { length: 0.6, strokeWidth: 0.035, color: '#abb2bf' },
        limitMax: false, limitMin: false,
        colorStart: '#61afef', colorStop: '#61afef',
        strokeColor: '#3b4048', generateGradient: true, highDpiSupport: true
    };

    // --- Loss Gauge ---
    const lossTarget = document.getElementById('loss-gauge');
    lossGauge = new Gauge(lossTarget).setOptions({
        ...commonOpts,
        angle: -0.2,
        staticLabels: { font: "12px sans-serif", labels: [0, 2, 5, 10], color: "#abb2bf" },
        staticZones: [
            {strokeStyle: "#98c379", min: 0, max: 2},    // Green (Good)
            {strokeStyle: "#e5c07b", min: 2, max: 5},    // Yellow (Okay)
            {strokeStyle: "#e06c75", min: 5, max: 10}    // Red (High)
        ]
    });
    lossGauge.maxValue = 10;
    lossGauge.setMinValue(0);
    lossGauge.set(0);

    // --- Speed Gauge ---
    const speedTarget = document.getElementById('speed-gauge');
    speedGauge = new Gauge(speedTarget).setOptions({ ...commonOpts, angle: -0.2 });
    speedGauge.maxValue = 50; 
    speedGauge.setMinValue(0);
    speedGauge.set(0);

    console.log("Easy Bake Gauges are online.");
}

// --- NEW: Handle clean JSON telemetry directly ---
function updateDashboardFromTelemetry(data) {
    // Data structure matches trainer.py payload: 
    // {"loss": float, "speed": float, "epoch": int, ...}
    
    if (data.loss !== undefined && lossGauge) {
        lossGauge.set(data.loss);
    }
    
    if (data.speed !== undefined && speedGauge) {
        speedGauge.set(data.speed);
    }
}

// Keep this as a fallback, but Telemetry is preferred
function updateDashboardFromLog(logLine) {
    if (!logLine || typeof logLine !== 'string') return;
    const match = logLine.match(/(\d+\.\d+)it\/s, Loss=(\d+\.\d+)/);
    if (match) {
        if (speedGauge) speedGauge.set(parseFloat(match[1]));
        if (lossGauge) lossGauge.set(parseFloat(match[2]));
    }
}

document.addEventListener('DOMContentLoaded', setupGauges);