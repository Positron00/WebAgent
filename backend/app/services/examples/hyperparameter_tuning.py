"""
Example script for hyperparameter tuning with the self-hosted LLM service.
This script tests various temperature and top_p settings and logs results to MLflow.

Usage:
    python hyperparameter_tuning.py --base-url http://localhost:8080
"""
import argparse
import json
import time
import requests
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def test_parameter_combination(base_url, prompt, temperature, top_p, experiment_name):
    """Test a specific combination of temperature and top_p."""
    # Prepare request
    request_data = {
        "messages": [
            {"role": "system", "content": "You are a helpful, concise assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 200,
        "temperature": temperature,
        "top_p": top_p,
        "track_experiment": True,
        "experiment_name": experiment_name,
        "run_name": f"temp-{temperature}-top_p-{top_p}",
        "tags": {
            "parameter_tuning": "true",
            "prompt": prompt
        }
    }
    
    # Send request
    start_time = time.time()
    response = requests.post(
        f"{base_url}/v1/chat/completions",
        json=request_data
    )
    request_time = time.time() - start_time
    
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
        return None
    
    result = response.json()
    
    # Print summary
    print(f"Temperature: {temperature}, Top-p: {top_p}")
    print(f"Generation time: {result['performance']['generation_time_seconds']:.2f}s")
    print(f"Tokens per second: {result['performance']['tokens_per_second']:.2f}")
    print(f"Total tokens: {result['usage']['total_tokens']}")
    print("-" * 50)
    
    # Return the run ID and performance metrics
    if "experiment_tracking" in result:
        return {
            "run_id": result["experiment_tracking"]["run_id"],
            "temperature": temperature,
            "top_p": top_p,
            "total_tokens": result["usage"]["total_tokens"],
            "generation_time": result["performance"]["generation_time_seconds"],
            "tokens_per_second": result["performance"]["tokens_per_second"],
            "request_time": request_time,
            "content": result["choices"][0]["message"]["content"]
        }
    return None

def optimize_parameters(base_url, prompt="Explain quantum computing in simple terms", experiment_name="parameter-tuning"):
    """Run hyperparameter optimization for temperature and top_p."""
    # Set MLflow tracking URI to match the server
    mlflow.set_tracking_uri(base_url.replace("8080", "5000"))
    
    # Create or get experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)
    
    # Define parameter grid
    temperatures = [0.1, 0.3, 0.5, 0.7, 0.9]
    top_ps = [0.5, 0.7, 0.9, 1.0]
    
    # Storage for results
    results = []
    
    # Run grid search
    print(f"Starting parameter tuning with prompt: '{prompt}'")
    print(f"Testing {len(temperatures) * len(top_ps)} combinations...")
    
    for temp in temperatures:
        for top_p in top_ps:
            print(f"Testing temperature={temp}, top_p={top_p}")
            result = test_parameter_combination(base_url, prompt, temp, top_p, experiment_name)
            if result:
                results.append(result)
            # Sleep to avoid overloading the service
            time.sleep(1)
    
    # Convert results to DataFrame for analysis
    if results:
        df = pd.DataFrame(results)
        
        # Find best parameters for generation speed
        best_speed = df.loc[df["tokens_per_second"].idxmax()]
        print("\nBest parameters for generation speed:")
        print(f"Temperature: {best_speed['temperature']}")
        print(f"Top-p: {best_speed['top_p']}")
        print(f"Tokens per second: {best_speed['tokens_per_second']:.2f}")
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Heatmap for tokens per second
        plt.subplot(2, 1, 1)
        heatmap_data = df.pivot_table(
            values="tokens_per_second", 
            index="temperature", 
            columns="top_p"
        )
        plt.imshow(heatmap_data, cmap="viridis")
        plt.colorbar(label="Tokens per second")
        plt.title("Tokens per Second by Temperature and Top-p")
        plt.xlabel("Top-p")
        plt.ylabel("Temperature")
        # Add x and y tick labels
        plt.xticks(range(len(heatmap_data.columns)), heatmap_data.columns)
        plt.yticks(range(len(heatmap_data.index)), heatmap_data.index)
        
        # Add values to heatmap cells
        for i in range(len(heatmap_data.index)):
            for j in range(len(heatmap_data.columns)):
                plt.text(j, i, f"{heatmap_data.iloc[i, j]:.1f}", 
                         ha="center", va="center", color="white")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        plt.tight_layout()
        
        # Save plot to file
        plot_file = f"parameter_tuning_{timestamp}.png"
        plt.savefig(plot_file)
        print(f"Plot saved to {plot_file}")
        
        # Save results to CSV
        csv_file = f"parameter_tuning_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        print(f"Results saved to {csv_file}")
        
        # Log top parameters to MLflow
        with mlflow.start_run(run_name=f"parameter-tuning-summary-{timestamp}"):
            mlflow.log_param("best_temperature", best_speed["temperature"])
            mlflow.log_param("best_top_p", best_speed["top_p"])
            mlflow.log_metric("best_tokens_per_second", best_speed["tokens_per_second"])
            mlflow.log_artifact(plot_file)
            mlflow.log_artifact(csv_file)
        
        return df
    else:
        print("No valid results obtained from parameter tuning.")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run hyperparameter tuning for self-hosted LLM")
    parser.add_argument("--base-url", type=str, default="http://localhost:8080", 
                        help="Base URL of the self-hosted LLM service")
    parser.add_argument("--prompt", type=str, 
                        default="Explain quantum computing in simple terms", 
                        help="Prompt to use for testing")
    parser.add_argument("--experiment-name", type=str, 
                        default="parameter-tuning", 
                        help="MLflow experiment name")
    
    args = parser.parse_args()
    
    results = optimize_parameters(
        base_url=args.base_url, 
        prompt=args.prompt,
        experiment_name=args.experiment_name
    ) 