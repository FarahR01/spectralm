/**
 * views/health/main.js
 * SpectraLM health dashboard — live updates + animations
 */

const POLL_INTERVAL_MS = 5000;

// DOM refs
const statusBadge = document.getElementById('status-badge');
const modelBadge = document.getElementById('model-badge');
const deviceValue = document.getElementById('device-value');
const epochValue = document.getElementById('epoch-value');
const ecrValue = document.getElementById('ecr-value');
const checkpointValue = document.getElementById('checkpoint-value');
const uptimeMetric = document.getElementById('uptime-metric');
const requestsMetric = document.getElementById('requests-metric');
const errorsMetric = document.getElementById('errors-metric');
const errorRateMetric = document.getElementById('error-rate-metric');
const serverStatus = document.getElementById('server-status');
const lastUpdated = document.getElementById('last-updated');

let previousRequests = 0;

// Format uptime
function formatUptime(seconds) {
    seconds = Math.floor(seconds);
    if (seconds < 60) return `${seconds}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    return `${h}h ${m}m`;
}

// Animate number changes
function animateNumber(el, fromVal, toVal, duration = 300) {
    if (!el || fromVal === toVal) return;
    const start = performance.now();
    const diff = toVal - fromVal;

    function step(now) {
        const elapsed = now - start;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
        el.textContent = Math.round(fromVal + diff * eased).toLocaleString();
        if (progress < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
}

// Flash effect
function flash(el) {
    if (!el) return;
    el.style.animation = 'none';
    setTimeout(() => {
        el.style.animation = 'badgePulse 0.3s ease-out';
    }, 10);
}

// Update dashboard
async function updateHealth() {
    try {
        const res = await fetch('/health', { headers: { 'Accept': 'application/json' } });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();

        // Status badge
        const statusClass = data.model_loaded ? 'ok' : 'degraded';
        const statusText = data.model_loaded ? 'OK' : 'DEGRADED';
        statusBadge.textContent = statusText;
        statusBadge.className = `status-badge ${statusClass}`;

        // Model loaded badge
        const modelLoaded = data.model_loaded;
        modelBadge.textContent = modelLoaded ? '✓ Loaded' : '✗ Not Loaded';
        modelBadge.className = `badge ${modelLoaded ? 'success' : 'danger'}`;
        flash(modelBadge);

        // Device
        deviceValue.textContent = data.device;

        // Epoch
        epochValue.textContent = data.loaded_epoch;

        // ECR
        ecrValue.textContent = data.val_ecr > 0 ? data.val_ecr.toFixed(4) : '—';

        // Checkpoint
        checkpointValue.textContent = data.checkpoint || '—';
        checkpointValue.title = data.checkpoint || '';

        // Uptime
        uptimeMetric.textContent = formatUptime(data.uptime_sec);

        // Requests with animation
        if (data.requests_served !== previousRequests) {
            animateNumber(requestsMetric, previousRequests, data.requests_served);
            previousRequests = data.requests_served;
        }

        // Errors
        errorsMetric.textContent = data.error_count;

        // Error rate with color coding
        const errorRate = data.error_rate;
        const errorRatePercent = (errorRate * 100).toFixed(2);
        errorRateMetric.textContent = `${errorRatePercent}%`;

        if (errorRate === 0) {
            errorRateMetric.className = 'metric-value error-rate low';
        } else if (errorRate < 0.05) {
            errorRateMetric.className = 'metric-value error-rate low';
        } else if (errorRate < 0.1) {
            errorRateMetric.className = 'metric-value error-rate medium';
        } else {
            errorRateMetric.className = 'metric-value error-rate high';
        }

        // Server status
        serverStatus.textContent = data.status.toUpperCase();

        // Last updated
        const now = new Date();
        lastUpdated.textContent = `Last updated: ${now.toLocaleTimeString()}`;

    } catch (err) {
        console.error('Health check failed:', err);
        statusBadge.textContent = 'UNREACHABLE';
        statusBadge.className = 'status-badge degraded';
        lastUpdated.textContent = `Last updated: — (error: ${err.message})`;
    }
}

// Initial update and polling
updateHealth();
setInterval(updateHealth, POLL_INTERVAL_MS);
