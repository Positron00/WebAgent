# MLflow Integration for Self-Hosted LLM Service

This guide explains how to use MLflow with the WebAgent self-hosted LLM service to track experiments, compare model performance, and optimize hyperparameters.

## Overview

The self-hosted LLM service now integrates with MLflow to provide:

1. **Experiment Tracking**: Log parameters, metrics, and artifacts for each model run
2. **Model Comparison**: Compare different models and parameter settings
3. **Hyperparameter Optimization**: Find optimal settings for your models
4. **Performance Monitoring**: Track inference time, token generation speed, and memory usage
5. **Artifact Storage**: Save prompts, responses, and model metadata

## Setup

### 1. Install MLflow

MLflow is included in the `requirements-llm.txt` file, but you can also install it separately:

```bash
pip install mlflow scikit-learn
```

### 2. Start MLflow Server

For local development, start the MLflow tracking server:

```bash
# Start MLflow server with SQLite backend and local artifact storage
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow-artifacts --host 0.0.0.0 --port 5000
```

For production deployments, use more robust backends:

```bash
# PostgreSQL backend with S3 artifact storage
mlflow server \
  --backend-store-uri postgresql://username:password@localhost/mlflow \
  --default-artifact-root s3://mybucket/mlflow-artifacts \
  --host 0.0.0.0 \
  --port 5000
```

### 3. Configure the Self-Hosted LLM Service

Start the service with MLflow configuration:

```bash
python backend/app/services/self_hosted_llm_service.py \
  --model_path "/path/to/model" \
  --mlflow_tracking_uri "http://localhost:5000" \
  --mlflow_experiment_name "llm-experiments"
```

Or use environment variables:

```bash
export MLFLOW_TRACKING_URI="http://localhost:5000"
export MLFLOW_EXPERIMENT_NAME="llm-experiments"
python backend/app/services/self_hosted_llm_service.py --model_path "/path/to/model"
```

## Tracking Experiments

### Automatic Tracking

The service automatically logs basic model information:
- Model parameters (architecture, size, etc.)
- Device information (GPU/CPU)
- Memory usage

This happens when the service starts and creates a "model_load" run.

### Request-Level Tracking

To track individual requests, include the `track_experiment` flag in your API requests:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Tell me about MLflow."}
  ],
  "max_tokens": 1024,
  "temperature": 0.7,
  "track_experiment": true,
  "experiment_name": "temperature-comparison",
  "run_name": "temp-0.7-test",
  "tags": {
    "model_type": "Llama-3-8B",
    "purpose": "documentation"
  }
}
```

Parameters:
- `track_experiment` (boolean): Enable MLflow tracking for this request
- `experiment_name` (string, optional): Custom experiment name
- `run_name` (string, optional): Custom run name
- `tags` (object, optional): Custom tags for the experiment

### Logged Information

For each tracked request, MLflow logs:

#### Parameters
- `max_tokens`: Maximum tokens to generate
- `temperature`: Temperature value
- `top_p`: Top-p sampling value
- `frequency_penalty`: Frequency penalty value
- `presence_penalty`: Presence penalty value

#### Metrics
- `input_tokens`: Number of tokens in the prompt
- `output_tokens`: Number of tokens generated
- `total_tokens`: Total token count
- `generation_time_seconds`: Time taken to generate the response
- `tokens_per_second`: Generation speed

#### Artifacts
- `prompts/prompt-{request_id}.txt`: The formatted prompt
- `responses/response-{request_id}.txt`: The generated response

## Hyperparameter Optimization

### Temperature Comparison

Test different temperature settings to find the optimal value:

```python
import requests
import time

# Base request
base_request = {
    "messages": [
        {"role": "system", "content": "You are a helpful, concise assistant."},
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
    "max_tokens": 1024,
    "track_experiment": True,
    "experiment_name": "temperature-tuning",
}

# Test different temperatures
for temp in [0.1, 0.3, 0.5, 0.7, 0.9]:
    request = {**base_request, 
               "temperature": temp,
               "run_name": f"temp-{temp}",
               "tags": {"parameter": "temperature"}}
    
    response = requests.post(
        "http://localhost:8080/v1/chat/completions",
        json=request
    )
    
    print(f"Temperature {temp}: {response.status_code}")
    if response.status_code == 200:
        print(f"Run ID: {response.json().get('experiment_tracking', {}).get('run_id')}")
    
    # Wait to not overload the system
    time.sleep(2)
```

### Sampling Strategy Comparison

Compare top-p vs. temperature sampling:

```python
# Test different sampling strategies
strategies = [
    {"temperature": 0.7, "top_p": 1.0, "name": "pure-temperature"},
    {"temperature": 0.0, "top_p": 0.9, "name": "pure-top-p"},
    {"temperature": 0.5, "top_p": 0.9, "name": "combined"}
]

for strategy in strategies:
    request = {**base_request, 
               "temperature": strategy["temperature"],
               "top_p": strategy["top_p"],
               "run_name": strategy["name"],
               "tags": {"parameter": "sampling_strategy"}}
    
    response = requests.post(
        "http://localhost:8080/v1/chat/completions",
        json=request
    )
    
    print(f"Strategy {strategy['name']}: {response.status_code}")
    # Wait to not overload the system
    time.sleep(2)
```

## Analyzing Results

### Viewing Experiments in UI

Access the MLflow UI at http://localhost:5000 to view:
- List of experiments
- Run comparisons
- Parameter and metric visualizations
- Tracked artifacts

### Using the MLflow API

Query experiment results programmatically:

```python
import mlflow
from mlflow.tracking import MlflowClient

# Set the tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Get the experiment
experiment = mlflow.get_experiment_by_name("temperature-tuning")
experiment_id = experiment.experiment_id

# Search for runs in the experiment
runs = mlflow.search_runs(experiment_ids=[experiment_id])

# Display metrics for each run
print("Temperature vs. Performance:")
for _, run in runs.iterrows():
    temp = run.get("params.temperature", "unknown")
    tokens_per_sec = run.get("metrics.tokens_per_second", 0)
    total_tokens = run.get("metrics.total_tokens", 0)
    
    print(f"Temperature: {temp}, " 
          f"Speed: {tokens_per_sec:.2f} tokens/sec, "
          f"Tokens: {total_tokens}")
```

### Finding Optimal Parameters

Calculate the best parameters based on your criteria:

```python
# Find the run with the best token generation speed
best_speed_run = runs.loc[runs["metrics.tokens_per_second"].idxmax()]
print(f"Best speed: Temperature = {best_speed_run['params.temperature']}, "
      f"Speed = {best_speed_run['metrics.tokens_per_second']:.2f} tokens/sec")

# Get the run details
client = MlflowClient()
run_details = client.get_run(best_speed_run.run_id)

# Access the artifacts
artifacts = client.list_artifacts(best_speed_run.run_id)
for artifact in artifacts:
    print(f"Artifact: {artifact.path}")
```

## Service Endpoints for MLflow

The service provides endpoints to interact with MLflow:

### Get MLflow Configuration

```bash
curl http://localhost:8080/metrics
```

Response includes MLflow information:
```json
{
  "mlflow": {
    "tracking_uri": "http://localhost:5000",
    "experiment_name": "llm-experiments"
  }
}
```

### List Experiments

```bash
curl http://localhost:8080/experiments
```

Response:
```json
{
  "experiments": [
    {
      "experiment_id": "1",
      "name": "llm-experiments",
      "artifact_location": "./mlflow-artifacts/1",
      "lifecycle_stage": "active"
    },
    {
      "experiment_id": "2",
      "name": "temperature-tuning",
      "artifact_location": "./mlflow-artifacts/2",
      "lifecycle_stage": "active"
    }
  ],
  "tracking_uri": "http://localhost:5000"
}
```

### List Runs for an Experiment

```bash
curl http://localhost:8080/experiments/1/runs
```

Response includes runs with metrics, parameters, and tags.

### Update MLflow Configuration

```bash
curl -X POST http://localhost:8080/mlflow-config \
  -H "Content-Type: application/json" \
  -d '{"tracking_uri": "http://new-server:5000", "experiment_name": "new-experiment"}'
```

Response:
```json
{
  "status": "success",
  "tracking_uri": "http://new-server:5000",
  "experiment_name": "new-experiment"
}
```

## Tips for Effective Model Tuning

1. **Consistent Prompts**: Use the same prompts when comparing different parameters
2. **Isolated Variables**: Change only one parameter at a time
3. **Multiple Runs**: Use multiple runs for each setting to account for variability
4. **Quantitative Metrics**: Focus on objective metrics like tokens per second
5. **Custom Metrics**: Add custom metrics to evaluate response quality:

```python
import requests
import mlflow
import time

# Set up MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("model-quality")

# Create response
response = requests.post(
    "http://localhost:8080/v1/chat/completions",
    json={
        "messages": [
            {"role": "user", "content": "Explain quantum entanglement"}
        ],
        "temperature": 0.7,
        "track_experiment": True,
        "run_name": "quality-test"
    }
)

# Get the generated text and MLflow run ID
result = response.json()
text = result["choices"][0]["message"]["content"]
run_id = result["experiment_tracking"]["run_id"]

# Evaluate the response quality (custom metric)
# This could be a call to another model for evaluation
response_length = len(text.split())
science_terms = sum(1 for term in ["quantum", "particle", "physics", "state"] 
                     if term.lower() in text.lower())
readability_score = calculate_readability(text)  # Your own function

# Log additional quality metrics to the same run
with mlflow.start_run(run_id=run_id):
    mlflow.log_metrics({
        "response_length": response_length,
        "science_terms": science_terms,
        "readability_score": readability_score
    })
```

## Advanced Usage

### Docker Deployment

Include MLflow in your Docker setup:

```dockerfile
# Add to your existing Dockerfile
ENV MLFLOW_TRACKING_URI=http://mlflow-server:5000
ENV MLFLOW_EXPERIMENT_NAME=production-llm

# Start the service with MLflow config
CMD ["python", "self_hosted_llm_service.py", "--model_path", "${MODEL_PATH}", "--mlflow_tracking_uri", "${MLFLOW_TRACKING_URI}", "--mlflow_experiment_name", "${MLFLOW_EXPERIMENT_NAME}"]
```

### A/B Testing Fine-Tuned Models

If you're fine-tuning models, use MLflow to compare them:

```python
# Test two different fine-tuned models
models = [
    {"path": "/models/base-model", "name": "base"},
    {"path": "/models/fine-tuned-model", "name": "fine-tuned"}
]

# Start each model on a different port
for i, model in enumerate(models):
    port = 8080 + i
    os.system(f"""
    python self_hosted_llm_service.py \
      --model_path "{model['path']}" \
      --port {port} \
      --mlflow_tracking_uri "http://localhost:5000" \
      --mlflow_experiment_name "model-comparison" &
    """)
    
    # Wait for model to load
    time.sleep(30)
    
    # Send test requests to each model
    response = requests.post(
        f"http://localhost:{port}/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Explain fine-tuning"}],
            "track_experiment": True,
            "run_name": f"model-{model['name']}",
            "tags": {"model_version": model['name']}
        }
    )
```

### Automated Hyperparameter Optimization

For systematic hyperparameter tuning:

```python
from skopt import gp_minimize
from skopt.space import Real

# Define the optimization function
def evaluate_temperature(temperature):
    response = requests.post(
        "http://localhost:8080/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Write a short poem"}],
            "temperature": float(temperature[0]),
            "track_experiment": True,
            "experiment_name": "auto-optimization",
            "run_name": f"temp-{temperature[0]:.2f}"
        }
    )
    
    if response.status_code != 200:
        return 100  # Penalty for failure
    
    result = response.json()
    # Use tokens per second as our optimization metric (negative for minimization)
    return -result.get("performance", {}).get("tokens_per_second", 0)

# Run Bayesian optimization
search_result = gp_minimize(
    evaluate_temperature,
    [Real(0.1, 1.0, name='temperature')],
    n_calls=10,
    random_state=42
)

# Get the best parameter
best_temperature = search_result.x[0]
print(f"Best temperature: {best_temperature:.2f}")
```

## Performance Monitoring in Production

Set up monitoring dashboards with MLflow metrics:

1. **Create a monitoring script**:

```python
import time
import mlflow
import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Set up MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("production-monitoring")

while True:
    # Get metrics
    response = requests.get("http://localhost:8080/metrics")
    metrics = response.json()
    
    # Get a sample request performance
    gen_response = requests.post(
        "http://localhost:8080/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 50,
            "track_experiment": True,
            "experiment_name": "production-monitoring",
            "run_name": f"health-check-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        }
    )
    
    perf_data = gen_response.json().get("performance", {})
    
    # Log to MLflow
    with mlflow.start_run():
        mlflow.log_metrics({
            "memory_allocated_gb": float(metrics.get("gpu_memory_allocated_gb", 0)),
            "tokens_per_second": perf_data.get("tokens_per_second", 0),
            "latency_seconds": perf_data.get("generation_time_seconds", 0)
        })
    
    # Sleep for 5 minutes
    time.sleep(300)
```

2. **Create performance dashboards using MLflow data**:

```python
# Get the monitoring data
runs = mlflow.search_runs(experiment_ids=["3"])  # production-monitoring
df = pd.DataFrame({
    "timestamp": runs["start_time"],
    "memory_gb": runs["metrics.memory_allocated_gb"],
    "tokens_per_sec": runs["metrics.tokens_per_second"],
    "latency": runs["metrics.latency_seconds"]
})

# Plot the data
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

ax1.plot(df["timestamp"], df["memory_gb"])
ax1.set_title("GPU Memory Usage")
ax1.set_ylabel("GB")

ax2.plot(df["timestamp"], df["tokens_per_sec"])
ax2.set_title("Generation Speed")
ax2.set_ylabel("Tokens/sec")

ax3.plot(df["timestamp"], df["latency"])
ax3.set_title("Response Latency")
ax3.set_ylabel("Seconds")

plt.tight_layout()
plt.savefig("performance_dashboard.png")
```

## Troubleshooting

### MLflow Server Not Accessible

If the LLM service can't connect to MLflow:

1. Check if MLflow server is running:
   ```bash
   curl http://localhost:5000
   ```

2. Verify network connectivity between services
3. Check for firewall issues
4. Ensure tracking URI is correct

### Missing Data in MLflow

If metrics or parameters are missing:

1. Check API request format - ensure `track_experiment` is set to `true`
2. Verify the response contains experiment tracking info
3. Check for exceptions in the server logs

### Performance Issues

If tracking slows down model inference:

1. Use MLflow's asynchronous logging when possible
2. Reduce the frequency of tracking in production
3. Only track important requests
4. Use a separate MLflow instance for production and development

### "Run Already Exists" Errors

If you see errors about runs already existing:

1. Ensure you're calling `mlflow.end_run()` properly
2. Check for zombie runs: `mlflow runs list --status RUNNING`
3. Kill any stale runs: `mlflow runs delete -id RUN_ID` 