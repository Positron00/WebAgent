# MLflow Integration in WebAgent

This document describes how to use MLflow with WebAgent, ensuring that all experiments and artifacts are stored in the correct directory structure.

## Important: MLflow Directory Location

WebAgent uses MLflow for experiment tracking, model versioning, and artifact storage. All MLflow artifacts should be stored in the `backend/mlruns` directory, not in the project root.

## Starting the MLflow Server

Use the provided script to start the MLflow server with the correct artifact location:

```bash
# Start the MLflow server on the default port (5000)
python backend/scripts/start_mlflow_server.py

# Or specify a custom port
python backend/scripts/start_mlflow_server.py --port 8080
```

This script ensures that:
1. The correct `backend/mlruns` directory is used
2. The SQLite database is stored in `backend/mlflow.db`
3. All environment variables are properly configured

## Checking and Fixing MLflow Directory Issues

If you have existing experiments that are using the incorrect directory, you can use the provided tool to check and fix them:

```bash
# Check for experiments with incorrect locations
python backend/scripts/fix_mlflow_locations.py

# Delete experiments with incorrect locations (WARNING: this will delete all runs and artifacts for these experiments)
python backend/scripts/fix_mlflow_locations.py --delete-incorrect
```

## Using MLflow in Your Code

When working with MLflow in your code, make sure to use the provided utilities:

```python
from backend.app.core.mlflow_utils import get_or_create_experiment

# Get or create an experiment with the correct artifact location
experiment_id = get_or_create_experiment("my_experiment_name")

# Now run MLflow operations as normal
with mlflow.start_run(experiment_id=experiment_id):
    # Log metrics, parameters, etc.
    mlflow.log_metric("accuracy", 0.95)
```

Or use the `log_to_mlflow` function from `observability.py`:

```python
from backend.app.core.observability import log_to_mlflow

# Log metrics, parameters, etc.
log_to_mlflow(
    metrics={"accuracy": 0.95},
    params={"model_type": "gpt-4"},
    experiment_name="my_experiment"
)
```

## Testing MLflow Configuration

You can test if your MLflow configuration is working correctly with:

```bash
python backend/scripts/test_mlflow_location.py
```

This script will create a test experiment and verify that it's using the correct `backend/mlruns` directory.

## Common Issues

1. **MLflow experiments are created in the project root**: Make sure to use the provided utilities and start the MLflow server with the correct configuration.

2. **Can't find MLflow experiments**: Check that your `MLFLOW_TRACKING_URI` is set correctly in your `.env.local` file.

3. **Permission errors**: Ensure that the `backend/mlruns` directory is writable by the current user.

## Environment Variables

Configure these variables in your `.env.local` file:

```
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_DEFAULT_ARTIFACT_ROOT=file:///path/to/WebAgent/backend/mlruns
```

Replace `/path/to/WebAgent` with your actual WebAgent project path. 