import os
import sys

# This script is the definitive entry point for the application.
# Its purpose is to establish a stable execution context.

def main():
    """Sets up the environment and launches the backend server."""
    
    # 1. Get the absolute path of this script (main.py).
    script_path = os.path.abspath(__file__)

    # 2. The directory containing this script is our Application Root.
    app_root = os.path.dirname(script_path)

    # 3. CRITICAL: Change the current working directory to the Application Root.
    # This forces all subsequent file operations to be relative to a known, stable base.
    os.chdir(app_root)

    # 4. Add the Application Root to the Python path to ensure imports work reliably.
    if app_root not in sys.path:
        sys.path.insert(0, app_root)
        
    print(f"--- Launching Easy Bake AI from root: {os.getcwd()} ---")
    
    # 5. Now that the context is stable, import and run the server.
    from backend.server import run_server
    run_server()

if __name__ == '__main__':
    main()