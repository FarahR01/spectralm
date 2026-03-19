/**
 * static/main.js
 * SpectraLM landing page — live metrics + animations
 *
 * What this does:
 *   1. Polls GET /health every 5 seconds and updates metrics in place
 *   2. Animates the request counter when it increments
 *   3. Pulses the status dot while the model is online
 *   4. Shows a "last updated" timestamp
 *   5. Animates metric cards on first load
 */

const POLL_INTERVAL_MS = 5000;

// ── DOM refs ──────────────────────────────────────────────────────────────────
const statusDot   = document.querySelector('.status-dot');
const statusLabel = document.querySelector('.status-label');
const uptimeEl    = document.getElementById('metric-uptime');
const requestsEl  = document.getElementById('metric-requests');
const ecrEl       = document.getElementById('metric-ecr');
const lastUpdEl   = document.getElementById('last-updated');

// ── Animate metric cards on load ──────────────────────────────────────────────
document.querySelectorAll('.metric').forEach((card, i) => {
    card.style.opacity = '0';
    card.style.transform = 'translateY(8px)';
    card.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
    setTimeout(() => {
        card.style.opacity = '1';
        card.style.transform = 'translateY(0)';
    }, 80 + i * 60);
});

document.querySelectorAll('.link-row').forEach((row, i) => {
    row.style.opacity = '0';
    row.style.transform = 'translateX(-6px)';
    row.style.transition = 'opacity 0.25s ease, transform 0.25s ease, border-color 0.15s';
    setTimeout(() => {
        row.style.opacity = '1';
        row.style.transform = 'translateX(0)';
    }, 300 + i * 50);
});

// ── Counter animation ─────────────────────────────────────────────────────────
let previousRequests = parseInt(requestsEl?.textContent || '0', 10);

function animateCounter(el, fromVal, toVal, duration = 400) {
    if (!el || fromVal === toVal) return;
    const start = performance.now();
    const diff  = toVal - fromVal;

    function step(now) {
        const elapsed  = now - start;
        const progress = Math.min(elapsed / duration, 1);
        const eased    = 1 - Math.pow(1 - progress, 3); // ease-out cubic
        el.textContent = Math.round(fromVal + diff * eased).toLocaleString();
        if (progress < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
}

// ── Flash effect when a value changes ────────────────────────────────────────
function flash(el, color = '#1D9E75') {
    if (!el) return;
    el.style.transition = 'color 0.15s';
    el.style.color = color;
    setTimeout(() => {
        el.style.color = '';
        el.style.transition = 'color 0.6s';
    }, 300);
}

// ── Status dot pulse animation ────────────────────────────────────────────────
function startPulse() {
    if (!statusDot) return;
    statusDot.style.animation = 'pulse 2s ease-in-out infinite';
}

function stopPulse() {
    if (!statusDot) return;
    statusDot.style.animation = 'none';
}

// Inject the pulse keyframe once
const style = document.createElement('style');
style.textContent = `
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50%       { opacity: 0.5; transform: scale(1.3); }
    }
    .metric-val { transition: color 0.6s; }
`;
document.head.appendChild(style);

// ── Format uptime ─────────────────────────────────────────────────────────────
function formatUptime(seconds) {
    seconds = Math.floor(seconds);
    if (seconds < 60)   return `${seconds}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    return `${h}h ${m}m`;
}

// ── Poll /health ──────────────────────────────────────────────────────────────
async function pollHealth() {
    try {
        const res  = await fetch('/health');
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();

        // Update uptime
        if (uptimeEl) {
            uptimeEl.textContent = formatUptime(data.uptime_sec);
        }

        // Animate request counter if it changed
        if (requestsEl) {
            const newRequests = data.requests_served;
            if (newRequests !== previousRequests) {
                animateCounter(requestsEl, previousRequests, newRequests);
                flash(requestsEl);
                previousRequests = newRequests;
            }
        }

        // Update ECR if somehow it changed (checkpoint hot-reload)
        if (ecrEl && data.val_ecr > 0) {
            const newEcr = data.val_ecr.toFixed(4);
            if (ecrEl.textContent !== newEcr) {
                ecrEl.textContent = newEcr;
                flash(ecrEl);
            }
        }

        // Update status dot and label
        const online = data.model_loaded;
        if (statusDot) {
            statusDot.style.background = online ? '#1D9E75' : '#E24B4A';
        }
        if (statusLabel) {
            statusLabel.style.color = online ? '#1D9E75' : '#E24B4A';
            statusLabel.textContent = `model ${online ? 'online' : 'offline'}`;
        }

        if (online) startPulse(); else stopPulse();

        // Last updated timestamp
        if (lastUpdEl) {
            const now = new Date();
            lastUpdEl.textContent = `updated ${now.toLocaleTimeString()}`;
        }

    } catch (err) {
        // Server unreachable — show degraded state
        if (statusDot)   statusDot.style.background = '#BA7517';
        if (statusLabel) {
            statusLabel.style.color   = '#BA7517';
            statusLabel.textContent   = 'polling…';
        }
    }
}

// ── Start polling ─────────────────────────────────────────────────────────────
startPulse();
pollHealth();
setInterval(pollHealth, POLL_INTERVAL_MS);