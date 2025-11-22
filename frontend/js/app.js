document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('train-form');
    const logOutput = document.getElementById('log-output');
    const toggleBtn = document.getElementById('toggle-advanced');
    const advancedPanel = document.getElementById('advanced-options');

    // --- Toggle Logic ---
    toggleBtn.addEventListener('click', () => {
        const isHidden = advancedPanel.classList.contains('hidden');
        if (isHidden) {
            advancedPanel.classList.remove('hidden');
            toggleBtn.textContent = '▲ Hide Advanced';
        } else {
            advancedPanel.classList.add('hidden');
            toggleBtn.textContent = '▼ Advanced Settings';
        }
    });

    // --- WebSocket Connection ---
    const socket = io('http://127.0.0.1:5555');

    socket.on('connect', () => {
        console.log("Socket connected!");
        logOutput.textContent += '\n[DASHBOARD] Connected to backend.';
    });

    socket.on('log_message', (msg) => {
        logOutput.textContent += msg.data;
        logOutput.scrollTop = logOutput.scrollHeight;
    });

    socket.on('training_update', (data) => {
        console.log("Telemetry Received:", data);
        if (typeof updateDashboardFromTelemetry === 'function') {
            updateDashboardFromTelemetry(data);
        }
    });

    socket.on('process_finished', (msg) => {
        logOutput.textContent += `\n\n[DASHBOARD] BUILD COMPLETE! Process ${msg.pid} finished with code ${msg.return_code}.`;
        logOutput.scrollTop = logOutput.scrollHeight;
        if (speedGauge) speedGauge.set(0);
    });

    // --- Form Submission ---
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        logOutput.textContent = '[DASHBOARD] Forging AI... Sending build request...';

        if (lossGauge) lossGauge.set(0);
        if (speedGauge) speedGauge.set(0);

        // Grab standard values
        const config = {
            curriculum_dir: document.getElementById('curriculum_dir').value,
            output_name: document.getElementById('output_name').value,
            epochs: parseInt(document.getElementById('epochs').value, 10),
            arch_type: document.getElementById('arch_type').value, 
            // Grab Advanced Values (with fallbacks just in case)
            batch_size: parseInt(document.getElementById('batch_size').value, 10) || 32,
            learning_rate: parseFloat(document.getElementById('learning_rate').value) || 0.0001,
            max_seq_length: parseInt(document.getElementById('max_seq_length').value, 10) || 256,
            d_model: parseInt(document.getElementById('d_model').value, 10) || 512,
            nhead: parseInt(document.getElementById('nhead').value, 10) || 8,
            num_encoder_layers: parseInt(document.getElementById('num_encoder_layers').value, 10) || 6,
            num_decoder_layers: parseInt(document.getElementById('num_decoder_layers').value, 10) || 6,
            dim_feedforward: parseInt(document.getElementById('dim_feedforward').value, 10) || 2048
        };

        try {
            const response = await fetch('http://127.0.0.1:5555/api/start-training', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config),
            });
            const result = await response.json();
            logOutput.textContent += `\n[DASHBOARD] ${result.message}`;
        } catch (error) {
            logOutput.textContent += `\n[DASHBOARD] FATAL: Backend connection failed. ${error}`;
        }
    });
});