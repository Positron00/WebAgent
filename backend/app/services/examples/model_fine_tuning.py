"""
Example script for tracking model fine-tuning with MLflow.
This script demonstrates how to use MLflow to track fine-tuning jobs and compare model performance.

Usage:
    python model_fine_tuning.py --base-url http://localhost:8080 --model-path /path/to/model
"""
import argparse
import os
import json
import time
import requests
import subprocess
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def start_llm_service(model_path, port, mlflow_tracking_uri, experiment_name, quantize=None):
    """Start the self-hosted LLM service with a specific model."""
    cmd = [
        "python", "backend/app/services/self_hosted_llm_service.py",
        "--model_path", model_path,
        "--port", str(port),
        "--mlflow_tracking_uri", mlflow_tracking_uri,
        "--mlflow_experiment_name", experiment_name
    ]
    
    if quantize == "8bit":
        cmd.append("--load_in_8bit")
    elif quantize == "4bit":
        cmd.append("--load_in_4bit")
    
    print(f"Starting LLM service: {' '.join(cmd)}")
    
    # Use subprocess.Popen to run the command in the background
    process = subprocess.Popen(cmd)
    
    # Wait for service to start
    print(f"Waiting for service to start on port {port}...")
    is_ready = False
    max_retries = 30
    retries = 0
    
    while not is_ready and retries < max_retries:
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=2)
            if response.status_code == 200 and response.json().get("status") == "ready":
                is_ready = True
                print(f"Service is ready on port {port}")
            else:
                print(f"Service not ready yet, status: {response.json().get('status', 'unknown')}")
        except:
            print("Waiting for service to start...")
        
        retries += 1
        time.sleep(5)
    
    if not is_ready:
        print(f"Service failed to start on port {port}")
        process.kill()
        return None
    
    return process

def stop_llm_service(process):
    """Stop the LLM service."""
    if process:
        process.terminate()
        time.sleep(5)
        if process.poll() is None:  # If process hasn't terminated
            process.kill()
        print("LLM service stopped")

def run_evaluation(base_url, test_set, run_name=None):
    """Run evaluation on a test set and log results to MLflow."""
    results = []
    
    # Start MLflow run if run_name is provided
    run_id = None
    if run_name:
        run = mlflow.start_run(run_name=run_name)
        run_id = run.info.run_id
    
    # Get health check to log model details
    try:
        health_response = requests.get(f"{base_url}/health")
        health_data = health_response.json()
        
        if run_id:
            mlflow.log_param("model_name", health_data.get("model", "unknown"))
            mlflow.log_param("device", health_data.get("device", "unknown"))
            if "memory_allocated_gb" in health_data:
                mlflow.log_metric("memory_allocated_gb", float(health_data.get("memory_allocated_gb", 0)))
    except:
        print("Warning: Could not retrieve model information from health check")
    
    # Process each test case
    total_tokens = 0
    total_time = 0
    success_count = 0
    
    for i, test_case in enumerate(test_set):
        prompt = test_case.get("prompt", "")
        expected = test_case.get("expected", "")
        system = test_case.get("system", "You are a helpful assistant.")
        
        # Prepare request
        request_data = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 500,
            "temperature": 0.7,
            "track_experiment": True,
            "run_name": f"eval-case-{i}" if not run_id else None,
            "tags": {
                "evaluation": "true",
                "case_id": str(i)
            }
        }
        
        # Run inference
        start_time = time.time()
        try:
            response = requests.post(
                f"{base_url}/v1/chat/completions",
                json=request_data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result["choices"][0]["message"]["content"]
                
                # Calculate metrics
                success = True
                tokens_per_second = result["performance"]["tokens_per_second"]
                generation_time = result["performance"]["generation_time_seconds"]
                total_tokens += result["usage"]["total_tokens"]
                total_time += generation_time
                success_count += 1
                
                # Simple exact match evaluation
                exact_match = expected in generated_text
                
                # Add to results
                results.append({
                    "case_id": i,
                    "success": success,
                    "tokens_per_second": tokens_per_second,
                    "generation_time": generation_time,
                    "total_tokens": result["usage"]["total_tokens"],
                    "exact_match": exact_match,
                    "prompt": prompt,
                    "generated": generated_text,
                    "expected": expected
                })
                
                print(f"Case {i}: Success, {tokens_per_second:.2f} tokens/sec, Exact match: {exact_match}")
            else:
                print(f"Case {i}: API Error {response.status_code} - {response.text}")
                results.append({
                    "case_id": i,
                    "success": False,
                    "error": f"API Error {response.status_code}",
                    "prompt": prompt,
                    "expected": expected
                })
        except Exception as e:
            print(f"Case {i}: Exception - {str(e)}")
            results.append({
                "case_id": i,
                "success": False,
                "error": str(e),
                "prompt": prompt,
                "expected": expected
            })
        
        # Don't overwhelm the service
        time.sleep(1)
    
    # Calculate aggregated metrics
    avg_tokens_per_second = 0
    exact_match_rate = 0
    
    if success_count > 0:
        avg_tokens_per_second = total_tokens / total_time if total_time > 0 else 0
        exact_match_rate = sum(1 for r in results if r.get("exact_match", False)) / len(test_set)
    
    metrics = {
        "total_test_cases": len(test_set),
        "successful_cases": success_count,
        "success_rate": success_count / len(test_set) if len(test_set) > 0 else 0,
        "avg_tokens_per_second": avg_tokens_per_second,
        "exact_match_rate": exact_match_rate,
        "total_tokens": total_tokens,
        "total_generation_time": total_time
    }
    
    # Log metrics to MLflow if run_id is provided
    if run_id:
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        
        # Save results as JSON artifact
        results_file = f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, "w") as f:
            json.dump({"results": results, "metrics": metrics}, f, indent=2)
        
        mlflow.log_artifact(results_file)
        os.remove(results_file)
        
        # End MLflow run
        mlflow.end_run()
    
    return results, metrics

def compare_models(model_paths, test_set, mlflow_tracking_uri="http://localhost:5000", experiment_name="model-comparison"):
    """Compare multiple models using the same test set."""
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    # Create or get experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)
    
    base_port = 8100
    all_metrics = []
    
    for i, model_config in enumerate(model_paths):
        if isinstance(model_config, str):
            model_path = model_config
            model_name = os.path.basename(model_path)
            quantize = None
        else:
            model_path = model_config["path"]
            model_name = model_config.get("name", os.path.basename(model_path))
            quantize = model_config.get("quantize")
        
        port = base_port + i
        
        print(f"\n===== Testing model: {model_name} =====")
        
        # Start the LLM service for this model
        process = start_llm_service(
            model_path=model_path,
            port=port,
            mlflow_tracking_uri=mlflow_tracking_uri,
            experiment_name=experiment_name,
            quantize=quantize
        )
        
        if not process:
            print(f"Skipping model {model_name} due to service startup failure")
            continue
        
        try:
            # Run evaluation
            _, metrics = run_evaluation(
                base_url=f"http://localhost:{port}",
                test_set=test_set,
                run_name=f"eval-{model_name}"
            )
            
            # Add model name to metrics
            metrics["model_name"] = model_name
            metrics["model_path"] = model_path
            metrics["quantize"] = quantize
            
            all_metrics.append(metrics)
        finally:
            # Stop the service
            stop_llm_service(process)
    
    # Create comparison visualization
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        
        # Create bar charts
        plt.figure(figsize=(15, 10))
        
        metrics_to_plot = [
            "avg_tokens_per_second", 
            "exact_match_rate", 
            "success_rate"
        ]
        
        for i, metric in enumerate(metrics_to_plot):
            plt.subplot(len(metrics_to_plot), 1, i+1)
            plt.bar(df["model_name"], df[metric])
            plt.title(f"Comparison of {metric.replace('_', ' ').title()}")
            plt.xlabel("Model")
            plt.xticks(rotation=45)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add values on top of bars
            for j, v in enumerate(df[metric]):
                plt.text(j, v, f"{v:.3f}", ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save comparison chart
        chart_file = f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(chart_file)
        print(f"Comparison chart saved to {chart_file}")
        
        # Log final comparison as a separate run
        with mlflow.start_run(run_name="model-comparison-summary"):
            best_model = df.loc[df["exact_match_rate"].idxmax()]
            mlflow.log_param("best_model", best_model["model_name"])
            mlflow.log_param("models_compared", ", ".join(df["model_name"]))
            mlflow.log_metric("best_exact_match_rate", best_model["exact_match_rate"])
            mlflow.log_metric("best_tokens_per_second", best_model["avg_tokens_per_second"])
            mlflow.log_artifact(chart_file)
            
            # Save comparison data
            df.to_csv("model_comparison.csv", index=False)
            mlflow.log_artifact("model_comparison.csv")
            os.remove("model_comparison.csv")
            
        # Clean up
        os.remove(chart_file)
        
        print("\n===== Model Comparison Results =====")
        print(df.to_string())
        print(f"\nBest model based on exact match rate: {best_model['model_name']}")
        
        return df
    else:
        print("No valid results for comparison")
        return None
    
def load_test_set(file_path=None):
    """Load or create a test set for evaluation."""
    if file_path and os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    
    # Create a simple test set
    return [
        {
            "prompt": "What is the capital of France?",
            "expected": "Paris",
            "system": "You are a helpful assistant. Keep responses short and to the point."
        },
        {
            "prompt": "Calculate 15 * 24",
            "expected": "360",
            "system": "You are a helpful assistant. Keep responses short and to the point."
        },
        {
            "prompt": "Explain how photosynthesis works in one sentence.",
            "expected": "plants use sunlight",
            "system": "You are a helpful assistant. Keep responses short and to the point."
        },
        {
            "prompt": "Write a haiku about mountains.",
            "expected": "mountain",
            "system": "You are a helpful assistant. Write a haiku (three lines with 5-7-5 syllable pattern)."
        },
        {
            "prompt": "What programming language is often used for data science?",
            "expected": "Python",
            "system": "You are a helpful assistant. Keep responses short and to the point."
        }
    ]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track model fine-tuning with MLflow")
    parser.add_argument("--base-url", type=str, default="http://localhost:8080", 
                        help="Base URL of the self-hosted LLM service")
    parser.add_argument("--mlflow-uri", type=str, default="http://localhost:5000", 
                        help="MLflow tracking URI")
    parser.add_argument("--experiment-name", type=str, default="model-fine-tuning", 
                        help="MLflow experiment name")
    parser.add_argument("--model-path", type=str, 
                        help="Path to the model for evaluation")
    parser.add_argument("--test-set", type=str,
                        help="Path to JSON file with test cases")
    parser.add_argument("--compare", action="store_true",
                        help="Compare multiple models")
    parser.add_argument("--model-paths", type=str, nargs="+",
                        help="Paths to models for comparison (use with --compare)")
    
    args = parser.parse_args()
    
    # Load test set
    test_set = load_test_set(args.test_set)
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment_name)
    
    if args.compare and args.model_paths:
        # Compare multiple models
        model_configs = []
        for path in args.model_paths:
            if ":" in path:
                # Format: path:name:quantize
                parts = path.split(":")
                model_path = parts[0]
                name = parts[1] if len(parts) > 1 else os.path.basename(model_path)
                quantize = parts[2] if len(parts) > 2 else None
                model_configs.append({
                    "path": model_path,
                    "name": name,
                    "quantize": quantize
                })
            else:
                model_configs.append(path)
        
        compare_models(
            model_paths=model_configs,
            test_set=test_set,
            mlflow_tracking_uri=args.mlflow_uri,
            experiment_name=args.experiment_name
        )
    elif args.model_path:
        # Evaluate a single model
        process = start_llm_service(
            model_path=args.model_path,
            port=8080,
            mlflow_tracking_uri=args.mlflow_uri,
            experiment_name=args.experiment_name
        )
        
        if process:
            try:
                run_evaluation(
                    base_url=args.base_url,
                    test_set=test_set,
                    run_name=f"eval-{os.path.basename(args.model_path)}"
                )
            finally:
                stop_llm_service(process)
    else:
        # Just run evaluation against an existing service
        run_evaluation(
            base_url=args.base_url,
            test_set=test_set,
            run_name="evaluation-existing-service"
        ) 