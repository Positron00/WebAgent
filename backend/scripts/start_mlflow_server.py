#!/usr/bin/env python
"""
Start MLflow Server with Correct Artifact Location
=================================================

This script starts the MLflow tracking server with the correct artifact location
set to backend/mlruns to ensure all experiments use the right directory.

Usage:
    python start_mlflow_server.py [--port PORT]

Options:
    --port PORT    Port to run the MLflow server on (default: 5000)
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Add the project root to the Python path
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import required modules
from backend.app.core.config import settings

def main():
    parser = argparse.ArgumentParser(description="Start MLflow server with correct artifact location")
    parser.add_argument("--port", type=int, default=5000, help="Port to run MLflow server on")
    args = parser.parse_args()
    
    # Get the mlruns path (without file:// prefix)
    mlruns_path = settings.MLFLOW_ARTIFACT_LOCATION
    if mlruns_path.startswith("file://"):
        mlruns_path = mlruns_path[7:]
    
    # Ensure the directory exists
    os.makedirs(mlruns_path, exist_ok=True)
    
    # Construct MLflow command with proper artifact location
    mlflow_cmd = [
        "mlflow", "server",
        "--backend-store-uri", "sqlite:///backend/mlflow.db",  # Use SQLite for storage
        "--default-artifact-root", settings.MLFLOW_DEFAULT_ARTIFACT_ROOT,
        "--host", "0.0.0.0",
        "--port", str(args.port)
    ]
    
    print(f"Starting MLflow server with artifact root: {settings.MLFLOW_DEFAULT_ARTIFACT_ROOT}")
    print(f"MLflow tracking URI: http://localhost:{args.port}")
    print(f"Command: {' '.join(mlflow_cmd)}")
    
    # Set the environment variable as well
    os.environ["MLFLOW_DEFAULT_ARTIFACT_ROOT"] = settings.MLFLOW_DEFAULT_ARTIFACT_ROOT
    
    # Execute the command
    subprocess.run(mlflow_cmd)

if __name__ == "__main__":
    main() 