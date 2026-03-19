/**
 * views/model-info/main.js
 * SpectraLM model card — fetches and displays model metadata
 */

// Fetch and display model info
async function loadModelInfo() {
    try {
        const res = await fetch('/model/info', { 
            headers: { 'Accept': 'application/json' } 
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();

        // Overview
        document.getElementById('model-name').textContent = data.model_name || '—';
        document.getElementById('model-version').textContent = data.version || '—';
        document.getElementById('num-parameters').textContent = 
            data.num_parameters ? data.num_parameters.toLocaleString() : '—';

        // Architecture
        const archContainer = document.getElementById('architecture');
        archContainer.innerHTML = Object.entries(data.architecture || {})
            .map(([key, value]) => `
                <div class="detail-card">
                    <div class="detail-card-title">${formatKey(key)}</div>
                    <div class="detail-card-value">${value}</div>
                </div>
            `).join('');

        // Physics Constraints
        const physicsContainer = document.getElementById('physics-constraints');
        physicsContainer.innerHTML = Object.entries(data.physics_constraints || {})
            .map(([key, value]) => `
                <div class="detail-card">
                    <div class="detail-card-title">${formatKey(key)}</div>
                    <div class="detail-card-value">${value}</div>
                </div>
            `).join('');

        // Training Info
        const trainingContainer = document.getElementById('training-info');
        trainingContainer.innerHTML = Object.entries(data.training_info || {})
            .map(([key, value]) => `
                <div class="detail-card">
                    <div class="detail-card-title">${formatKey(key)}</div>
                    <div class="detail-card-value">${value}</div>
                </div>
            `).join('');

        // Input Format
        const inputContainer = document.getElementById('input-format');
        inputContainer.innerHTML = Object.entries(data.input_format || {})
            .map(([key, value]) => `
                <div class="detail-card">
                    <div class="detail-card-title">${formatKey(key)}</div>
                    <div class="detail-card-value">${value}</div>
                </div>
            `).join('');

        // Output Format
        const outputContainer = document.getElementById('output-format');
        outputContainer.innerHTML = Object.entries(data.output_format || {})
            .map(([key, value]) => `
                <div class="detail-card">
                    <div class="detail-card-title">${formatKey(key)}</div>
                    <div class="detail-card-value">${value}</div>
                </div>
            `).join('');

        // Known Limitations
        const limitationsContainer = document.getElementById('limitations');
        limitationsContainer.innerHTML = (data.known_limitations || [])
            .map(limitation => `<li>${limitation}</li>`)
            .join('');

        // Last updated
        const now = new Date();
        document.getElementById('last-updated').textContent = 
            `Loaded: ${now.toLocaleTimeString()}`;

    } catch (err) {
        console.error('Failed to load model info:', err);
        document.getElementById('model-name').textContent = 'Error loading model info';
        document.getElementById('last-updated').textContent = 
            `Error: ${err.message}`;
    }
}

// Format keys to be human-readable
function formatKey(key) {
    return key
        .replace(/_/g, ' ')
        .split(' ')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
}

// Load on page load
document.addEventListener('DOMContentLoaded', loadModelInfo);
