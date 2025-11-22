import subprocess
# import threading  <-- Removed: Standard threading conflicts with Eventlet
import time

class ProcessManager:
    def __init__(self, socketio):
        self.socketio = socketio
        self.processes = {} 

    def _stream_output(self, process, pid):
        """Reads a process's unified output and emits it over WebSocket."""
        # Iterate over stdout. Because we used '-u' and bufsize=1, this happens in real-time.
        for line in iter(process.stdout.readline, ''):
            if line:
                self.socketio.emit('log_message', {'pid': pid, 'data': line})
                # Required yield for Eventlet to process the emit
                self.socketio.sleep(0)
        
        process.stdout.close()
        process.wait()
        
        status = 'success' if process.returncode == 0 else 'error'
        self.socketio.emit('process_finished', {'pid': pid, 'return_code': process.returncode, 'status': status})
            
        if pid in self.processes:
            del self.processes[pid]

    def start_process(self, command):
        print(f"Starting process with command: {' '.join(command)}")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1, 
            universal_newlines=True
        )
        
        pid = process.pid
        self.processes[pid] = process

        # FIX: Use SocketIO's native background task spawner
        # This creates a green thread compatible with Eventlet
        self.socketio.start_background_task(target=self._stream_output, process=process, pid=pid)
        
        return pid

    def get_active_processes(self):
        return list(self.processes.keys())