"""
MLflow Quickstart Guide for Self-Hosted LLM Service
===================================================

This script demonstrates the basic usage of MLflow features with the WebAgent self-hosted LLM service.
It's designed as a quick introduction to help you get started with experiment tracking.

Usage:
    python mlflow_quickstart.py

Requirements:
    - Running MLflow server (default: http://localhost:5000)
    - Running self-hosted LLM service (default: http://localhost:8080)
"""
import requests
import mlflow
import time
import json
import os
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# Configuration
LLM_SERVICE_URL = "http://localhost:8080"
MLFLOW_URL = "http://localhost:5000"
EXPERIMENT_NAME = "mlflow-quickstart"

def print_section(title):
    """Print a section title."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")

def check_services():
    """Check if MLflow and LLM services are running."""
    print_section("Checking services")
    
    # Check MLflow
    try:
        mlflow_response = requests.get(f"{MLFLOW_URL}/api/2.0/mlflow/experiments/list")
        print(f"✅ MLflow server is running at {MLFLOW_URL}")
    except Exception as e:
        print(f"❌ MLflow server not reachable at {MLFLOW_URL}: {str(e)}")
        print(f"   Start MLflow with: mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow-artifacts --host 0.0.0.0 --port 5000")
        return False
    
    # Check LLM service
    try:
        llm_response = requests.get(f"{LLM_SERVICE_URL}/health")
        if llm_response.status_code == 200:
            data = llm_response.json()
            print(f"✅ LLM service is running at {LLM_SERVICE_URL}")
            print(f"   Model: {data.get('model', 'unknown')}")
            print(f"   Device: {data.get('device', 'unknown')}")
            
            # Check MLflow status in LLM service
            mlflow_status = data.get('mlflow_status', 'unknown')
            if mlflow_status == 'connected':
                print(f"✅ LLM service is connected to MLflow")
            else:
                print(f"⚠️ LLM service MLflow status: {mlflow_status}")
                print(f"   Start the LLM service with MLflow parameters:")
                print(f"   --mlflow_tracking_uri {MLFLOW_URL} --mlflow_experiment_name {EXPERIMENT_NAME}")
        else:
            print(f"❌ LLM service returned status code {llm_response.status_code}")
            return False
    except Exception as e:
        print(f"❌ LLM service not reachable at {LLM_SERVICE_URL}: {str(e)}")
        return False
    
    return True

def explore_mlflow_api():
    """Explore the MLflow API endpoints in the LLM service."""
    print_section("Exploring MLflow API endpoints")
    
    # Get metrics
    try:
        response = requests.get(f"{LLM_SERVICE_URL}/metrics")
        data = response.json()
        print("MLflow configuration from metrics endpoint:")
        if 'mlflow' in data:
            print(f"   Tracking URI: {data['mlflow']['tracking_uri']}")
            print(f"   Experiment name: {data['mlflow']['experiment_name']}")
        else:
            print("   MLflow configuration not found in metrics")
    except Exception as e:
        print(f"Error accessing metrics endpoint: {str(e)}")
    
    # List experiments
    try:
        response = requests.get(f"{LLM_SERVICE_URL}/experiments")
        data = response.json()
        print("\nExperiments available through LLM service:")
        if 'experiments' in data and data['experiments']:
            for exp in data['experiments']:
                print(f"   ID: {exp['experiment_id']}, Name: {exp['name']}")
        else:
            print("   No experiments found or endpoint not available")
    except Exception as e:
        print(f"Error listing experiments: {str(e)}")
    
    return True

def simple_experiment():
    """Run a simple experiment with different prompts."""
    print_section("Running a simple experiment")
    
    # Set up MLflow
    mlflow.set_tracking_uri(MLFLOW_URL)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    prompts = [
        "Explain quantum computing in 2 sentences.",
        "Write a short poem about AI.",
        "List three benefits of experiment tracking.",
        "Describe the concept of hyperparameter tuning briefly."
    ]
    
    results = []
    
    for i, prompt in enumerate(prompts):
        print(f"\nTesting prompt {i+1}/{len(prompts)}: \"{prompt}\"")
        
        # Prepare request with tracking enabled
        request_data = {
            "messages": [
                {"role": "system", "content": "You are a helpful, concise assistant."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 150,
            "temperature": 0.7,
            "track_experiment": True,
            "experiment_name": EXPERIMENT_NAME,
            "run_name": f"prompt-test-{i+1}",
            "tags": {
                "example": "quickstart",
                "prompt_type": "short_response"
            }
        }
        
        # Send request
        start_time = time.time()
        response = requests.post(
            f"{LLM_SERVICE_URL}/v1/chat/completions",
            json=request_data
        )
        request_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            response_text = result["choices"][0]["message"]["content"]
            
            # Print response summary
            print(f"Response (truncated to 60 chars): \"{response_text[:60]}...\"")
            print(f"Generation time: {result['performance']['generation_time_seconds']:.2f}s")
            print(f"Tokens per second: {result['performance']['tokens_per_second']:.2f}")
            
            # Store result details
            if "experiment_tracking" in result:
                run_id = result["experiment_tracking"]["run_id"]
                print(f"MLflow run ID: {run_id}")
                
                results.append({
                    "prompt": prompt,
                    "response": response_text,
                    "tokens": result["usage"]["total_tokens"],
                    "generation_time": result["performance"]["generation_time_seconds"],
                    "tokens_per_second": result["performance"]["tokens_per_second"],
                    "run_id": run_id
                })
        else:
            print(f"Error: {response.status_code} - {response.text}")
    
    return results

def visualize_results(results):
    """Create and save basic visualizations of the experiment results."""
    print_section("Visualizing results")
    
    if not results:
        print("No results to visualize.")
        return
    
    # Create a DataFrame
    df = pd.DataFrame(results)
    
    # Create visualizations
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot generation time
    prompt_labels = [f"Prompt {i+1}" for i in range(len(df))]
    ax1.bar(prompt_labels, df["generation_time"])
    ax1.set_title("Generation Time by Prompt")
    ax1.set_ylabel("Seconds")
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot tokens per second
    ax2.bar(prompt_labels, df["tokens_per_second"])
    ax2.set_title("Tokens per Second by Prompt")
    ax2.set_ylabel("Tokens/second")
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save the figure
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_file = f"quickstart_results_{timestamp}.png"
    plt.savefig(output_file)
    print(f"Results visualization saved to: {output_file}")
    
    # Log the visualization and data to MLflow
    try:
        with mlflow.start_run(run_name=f"quickstart-summary-{timestamp}"):
            # Log summary metrics
            mlflow.log_metric("avg_generation_time", df["generation_time"].mean())
            mlflow.log_metric("avg_tokens_per_second", df["tokens_per_second"].mean())
            mlflow.log_metric("total_prompts", len(df))
            
            # Log artifacts
            mlflow.log_artifact(output_file)
            
            # Save and log results CSV
            csv_file = f"quickstart_results_{timestamp}.csv"
            df.to_csv(csv_file, index=False)
            mlflow.log_artifact(csv_file)
            
            # Clean up local files
            os.remove(csv_file)
            
            run_id = mlflow.active_run().info.run_id
            print(f"Results logged to MLflow run: {run_id}")
            print(f"View in MLflow UI: {MLFLOW_URL}/#/experiments/{mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id}/runs/{run_id}")
    except Exception as e:
        print(f"Error logging to MLflow: {str(e)}")
    
    return True

def update_mlflow_config():
    """Demonstrate updating MLflow configuration via API."""
    print_section("Updating MLflow configuration")
    
    # Create a new experiment name
    new_experiment = f"{EXPERIMENT_NAME}-updated"
    
    print(f"Updating MLflow configuration to use experiment: {new_experiment}")
    
    try:
        response = requests.post(
            f"{LLM_SERVICE_URL}/mlflow-config",
            json={
                "experiment_name": new_experiment
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ MLflow configuration updated:")
            print(f"   Tracking URI: {result.get('tracking_uri')}")
            print(f"   Experiment name: {result.get('experiment_name')}")
            print("\nNote: Any new requests with track_experiment=true will now use this experiment by default.")
        else:
            print(f"❌ Error updating MLflow configuration: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    return True

def main():
    """Run the MLflow quickstart example."""
    print_section("MLflow Quickstart for Self-Hosted LLM Service")
    print("This example demonstrates the basic usage of MLflow with the self-hosted LLM service.")
    
    if not check_services():
        print("\n❌ Services check failed. Please ensure both MLflow and LLM services are running.")
        return
    
    explore_mlflow_api()
    
    results = simple_experiment()
    
    if results:
        visualize_results(results)
    
    update_mlflow_config()
    
    print_section("Quickstart Complete")
    print(f"You can view all experiments in the MLflow UI: {MLFLOW_URL}")
    print("\nNext steps:")
    print(" 1. Explore more advanced examples:")
    print("    - hyperparameter_tuning.py for optimizing model parameters")
    print("    - model_fine_tuning.py for comparing model performance")
    print(" 2. Check out the MLflow documentation: https://mlflow.org/docs/latest/index.html")
    print(" 3. Read the WebAgent MLflow integration guide: backend/app/services/MLFLOW_INTEGRATION.md")

if __name__ == "__main__":
    main() 