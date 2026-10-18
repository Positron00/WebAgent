# ML Observability Demo

> **Since 3.10.0** the backend installs only the MLflow client (`mlflow-skinny`, 2.x). To run the tracking server or the UI described here, install the full package: `pip install -r backend/requirements-demos.txt` (it adds `mlflow` 2.x). The backend itself only logs runs, and only when `MLFLOW_TRACKING_URI` is set. If the tracking store can't be used, it turns tracking off and keeps running.


This demo showcases the ML observability features of the WebAgent platform, including metrics collection, drift detection, and performance monitoring.

## Features Demonstrated

- **Model Performance Tracking**: Monitors latency, accuracy, and token usage
- **Drift Detection**: Detects distribution shifts in model inputs and outputs
- **Categorical Drift**: Identifies changes in categorical distributions (e.g., document types)
- **Embedding Drift**: Detects shifts in embedding vector distributions
- **Performance Degradation**: Monitors for model performance degradation over time
- **Visualization**: Generates plots to visualize metrics and drift
- **Alert Generation**: Demonstrates when model retraining would be triggered
- **MLflow Integration**: Tracks experiments, metrics, and artifacts in MLflow
- **Grafana Dashboards**: Generates and optionally uploads Grafana dashboards

## Requirements

- Python 3.8+
- NumPy
- Matplotlib
- Pandas
- MLflow (optional, for experiment tracking)
- Grafana (optional, for dashboard visualization)

These dependencies should already be installed if you have the WebAgent platform installed.

## Running the Demo

From the project root directory:

```bash
python backend/demos/ml_observability_demo.py
```

### Command Line Options

- `--output-dir PATH`: Specify a custom output directory for metrics and plots (default: `backend/demos/results`)
- `--skip-plots`: Skip generating plots (useful for headless environments)
- `--mlflow-tracking-uri URI`: Specify MLflow tracking URI (default: from MLFLOW_TRACKING_URI env var)
- `--grafana-url URL`: Specify Grafana server URL (default: from GRAFANA_URL env var)
- `--grafana-token TOKEN`: Specify Grafana service account token (default: from GRAFANA_SERVICE_ACCOUNT_TOKEN env var)
- `--grafana-api-key KEY`: DEPRECATED - Use `--grafana-token` instead

### MLflow Integration

To enable MLflow tracking:

1. Install MLflow:
   ```bash
   pip install mlflow
   ```

2. Start the MLflow tracking server:
   ```bash
   mlflow server --host 127.0.0.1 --port 8080
   ```

3. Run the demo with the MLflow tracking URI:
   ```bash
   python backend/demos/ml_observability_demo.py --mlflow-tracking-uri http://127.0.0.1:8080
   ```

4. Open the MLflow UI at http://127.0.0.1:8080 to view experiments, metrics, parameters, and artifacts.

### Grafana Integration

To enable Grafana dashboard generation and upload:

1. Install a Grafana server or use an existing one.

2. Create a service account and token:
   - Navigate to Grafana's Admin section → Users and access → Service accounts
   - Click "Add service account"
   - Give it a name and role (Editor or Admin)
   - After creating the service account, add a token
   - Copy the service account token value

3. Run the demo with Grafana configuration:
   ```bash
   python backend/demos/ml_observability_demo.py --grafana-url http://localhost:3000 --grafana-token YOUR_SERVICE_ACCOUNT_TOKEN
   ```

> Note: Grafana API keys are deprecated as of 2024. If you're using an older Grafana instance, you can still use the `--grafana-api-key` parameter, but it's recommended to migrate to service account tokens.

## Output

The demo will:

1. Simulate normal model operations
2. Gradually introduce drift in various metrics
3. Detect when drift exceeds thresholds
4. Generate visualization plots
5. Export metrics to JSON
6. Create Grafana dashboards (if configured)
7. Log all metrics and artifacts to MLflow (if configured)
8. Print a summary of findings

### Generated Files

- `observability_metrics.json`: All collected metrics and drift statistics
- `model_performance.png`: Plot of model latency and accuracy over time
- `drift_error_rate.png`: Plot of drift scores and error rates
- `token_usage.png`: Plot of token usage over time
- `embedding_drift.png`: Visualization of embedding drift detection
- `category_drift.png`: Visualization of categorical drift detection
- `model_performance_dashboard.json`: Grafana dashboard for model performance metrics
- `model_drift_dashboard.json`: Grafana dashboard for drift detection metrics

## Integrating with Your Models

This demo illustrates how to use the following components in your own applications:

1. **Metrics Registry**: Track model performance, pipeline efficiency, and data quality
2. **Drift Detectors**: Monitor for distributional shifts that require attention
3. **Visualization**: Create custom plots for your specific metrics
4. **MLflow Tracking**: Log metrics, parameters, and artifacts to MLflow
5. **Grafana Dashboards**: Generate custom Grafana dashboards

For implementation details, see the following modules:
- `backend/app/core/metrics_registry.py` - Core metrics collection
- `backend/app/core/drift_detection.py` - Drift detection algorithms
- `backend/app/core/mlflow_integration.py` - MLflow integration
- `backend/app/core/grafana_integration.py` - Grafana dashboard generation

## Sample Observability Implementation

Here's a minimal example of how to integrate observability in your own services:

```python
from backend.app.core.metrics_registry import get_model_metrics_registry
from backend.app.core.drift_detection import create_numerical_drift_detector
from backend.app.core.mlflow_integration import get_mlflow_manager, log_drift_detection

# Get the model metrics registry
model_registry = get_model_metrics_registry()

# Initialize MLflow (optional)
mlflow_manager = get_mlflow_manager()
if mlflow_manager.initialized:
    run_id = mlflow_manager.start_run(experiment_name="ModelMonitoring")

# Register your model
model_registry.register_model(
    model_id="my-model",
    model_type="classification",
    task_type="sentiment-analysis",
    description="Sentiment analysis model"
)

# Create a drift detector for a key metric
accuracy_detector = create_numerical_drift_detector(
    name="sentiment_accuracy",
    window_size=100,
    threshold=0.1
)

# In your prediction loop:
def predict(input_text):
    start_time = time.time()
    try:
        # Make prediction
        result = model.predict(input_text)
        
        # Record prediction metrics
        latency = time.time() - start_time
        model_registry.record_prediction(
            model_id="my-model",
            latency=latency,
            success=True,
            confidence=result.confidence,
            metadata={"input_length": len(input_text)}
        )
        
        # Check for drift
        drift_score, drift_detected = accuracy_detector.add_current_sample(result.accuracy)
        if drift_detected:
            logger.warning(f"Accuracy drift detected! Score: {drift_score:.4f}")
            
            # Log drift detection to MLflow
            if mlflow_manager.initialized:
                log_drift_detection(
                    detector_id="sentiment_accuracy",
                    scores={"accuracy": drift_score},
                    detected=True
                )
            
        # Log metrics to MLflow
        if mlflow_manager.initialized:
            mlflow_manager.log_metrics({
                "latency": latency,
                "accuracy": result.accuracy,
                "confidence": result.confidence,
                "drift_score": drift_score
            })
            
        return result
    except Exception as e:
        model_registry.record_prediction(
            model_id="my-model",
            latency=time.time() - start_time,
            success=False,
            metadata={"error": str(e)}
        )
        raise 