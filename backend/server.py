import eventlet
# CRITICAL: Monkey patch must be the very first thing called.
# This makes standard library networking/threading compatible with Eventlet.
eventlet.monkey_patch()

import os
import sys
from flask import Flask, jsonify, request, render_template
from flask_socketio import SocketIO
from flask_cors import CORS

# --- PATHING CONFIGURATION ---
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
APP_ROOT = os.path.dirname(BACKEND_DIR)
FRONTEND_DIR = os.path.join(APP_ROOT, 'frontend')
PROJECT_ROOT = os.path.dirname(APP_ROOT) 

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from .process_manager import ProcessManager

# --- FLASK CONFIGURATION ---
app = Flask(__name__,
            template_folder=FRONTEND_DIR,
            static_folder=FRONTEND_DIR)
CORS(app)

# Init SocketIO with eventlet
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*")
process_manager = ProcessManager(socketio)

print("--- HCTS Easy Bake AI Backend Initialized ---")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/<path:path>')
def serve_static_files(path):
    return app.send_static_file(path)

@app.route('/api/start-training', methods=['POST'])
def start_training():
    config = request.json
    print(f"Received build request: {config}")
    
    build_engine_script = os.path.join('backend', 'engine', 'build_engine.py')
    output_dir = os.path.join('builds', config.get('output_name', 'default_build'))

    # Construct the command with all advanced parameters
    # We use .get() with defaults matching the UI defaults to be safe
    command = [
        sys.executable, 
        '-u', 
        build_engine_script,
        '--curriculum-dir', config.get('curriculum_dir', 'curriculum'),
        '--output-dir', output_dir,
        '--arch-type', config.get('arch_type', 'standard'), # <--- NEW ARGUMENT
        '--epochs', str(config.get('epochs', 30)),
        '--batch-size', str(config.get('batch_size', 32)),
        '--learning-rate', str(config.get('learning_rate', 1e-4)),
        '--max-seq-length', str(config.get('max_seq_length', 256)),
        '--d-model', str(config.get('d_model', 512)),
        '--nhead', str(config.get('nhead', 8)),
        '--num-encoder-layers', str(config.get('num_encoder_layers', 6)),
        '--num-decoder-layers', str(config.get('num_decoder_layers', 6)),
        '--dim-feedforward', str(config.get('dim_feedforward', 2048)),
        '--callback-url', 'http://127.0.0.1:5555/api/telemetry'
    ]
    
    try:
        pid = process_manager.start_process(command)
        return jsonify({"status": "success", "message": f"Build process started with PID {pid}", "pid": pid})
    except Exception as e:
        print(f"Error starting process: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/telemetry', methods=['POST'])
def receive_telemetry():
    data = request.json
    # This emit will now work because of monkey_patch
    socketio.emit('training_update', data)
    return jsonify({"status": "received"})

@socketio.on('connect')
def connect():
    print("Client connected.")

@socketio.on('disconnect')
def disconnect():
    print("Client disconnected.")

def run_server():
    print("Starting Flask-SocketIO server on http://127.0.0.1:5555")
    socketio.run(app, host='127.0.0.1', port=5555, debug=True)