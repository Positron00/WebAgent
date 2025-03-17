# ML Observability Demo

This demo showcases the ML observability features of the WebAgent platform, including metrics collection, drift detection, and performance monitoring.

## Features Demonstrated

- **Model Performance Tracking**: Monitors latency, accuracy, and token usage
- **Drift Detection**: Detects distribution shifts in model inputs and outputs
- **Categorical Drift**: Identifies changes in categorical distributions (e.g., document types)
- **Embedding Drift**: Detects shifts in embedding vector distributions
- **Performance Degradation**: Monitors for model performance degradation over time
- **Visualization**: Generates plots to visualize metrics and drift
- **Alert Generation**: Demonstrates when model retraining would be triggered

## Requirements

- Python 3.8+
- NumPy
- Matplotlib
- Pandas

These dependencies should already be installed if you have the WebAgent platform installed.

## Running the Demo

From the project root directory:

```bash
python backend/demos/ml_observability_demo.py
```

### Command Line Options

- `--output-dir PATH`: Specify a custom output directory for metrics and plots (default: `backend/demos/results`)
- `--skip-plots`: Skip generating plots (useful for headless environments)

## Output

The demo will:

1. Simulate normal model operations
2. Gradually introduce drift in various metrics
3. Detect when drift exceeds thresholds
4. Generate visualization plots
5. Export metrics to JSON
6. Print a summary of findings

### Generated Files

- `observability_metrics.json`: All collected metrics and drift statistics
- `model_performance.png`: Plot of model latency and accuracy over time
- `drift_error_rate.png`: Plot of drift scores and error rates
- `token_usage.png`: Plot of token usage over time
- `embedding_drift.png`: Visualization of embedding drift detection
- `category_drift.png`: Visualization of categorical drift detection

## Integrating with Your Models

This demo illustrates how to use the following components in your own applications:

1. **Metrics Registry**: Track model performance, pipeline efficiency, and data quality
2. **Drift Detectors**: Monitor for distributional shifts that require attention
3. **Visualization**: Create custom plots for your specific metrics

For implementation details, see the `backend/app/core/metrics_registry.py` and `backend/app/core/drift_detection.py` modules.

## Sample Observability Implementation

Here's a minimal example of how to integrate observability in your own services:

```python
from backend.app.core.metrics_registry import get_model_metrics_registry
from backend.app.core.drift_detection import create_numerical_drift_detector

# Get the model metrics registry
model_registry = get_model_metrics_registry()

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
            
        return result
    except Exception as e:
        model_registry.record_prediction(
            model_id="my-model",
            latency=time.time() - start_time,
            success=False,
            metadata={"error": str(e)}
        )
        raise 