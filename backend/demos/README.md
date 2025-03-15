# WebAgent Demos

This directory contains demonstration scripts for various WebAgent features and capabilities.

## Fine-Tuning Service Demo

The `ft_demo.py` script demonstrates how to use the FineTuneService to fine-tune models and track experiments with MLflow.

### Prerequisites

Before running the demo, make sure you have:

1. Installed the required dependencies:
   ```bash
   pip install mlflow torch transformers
   ```

2. Set up MLflow (optional, will use a local SQLite database by default):
   ```bash
   # Set custom MLflow tracking URI if needed
   export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
   
   # Start MLflow UI (optional, for viewing results)
   mlflow ui --port 5000
   ```

### Running the Demo

To run the fine-tuning demo:

```bash
# Basic usage
python -m backend.demos.ft_demo --model-id gpt2-small --dataset summarization --mode lora8

# Advanced usage with custom parameters
python -m backend.demos.ft_demo \
    --model-id gpt2-small \
    --dataset squad \
    --mode lora16 \
    --epochs 2 \
    --learning-rate 2e-5 \
    --batch-size 8 \
    --grad-accum 4 \
    --save-model \
    --run-name "gpt2-squad-lora16-demo"
```

### Command-Line Arguments

The demo script supports the following arguments:

| Argument | Description | Default |
|----------|-------------|---------|
| `--model-id` | ID of the model to fine-tune | gpt2-small |
| `--model-path` | Path to the model | models/{model_id} |
| `--dataset` | Dataset to use (squad or summarization) | summarization |
| `--mode` | Fine-tuning mode | lora8 |
| `--epochs` | Number of training epochs | 1 |
| `--learning-rate` | Learning rate for optimization | 1e-4 |
| `--batch-size` | Batch size for training | 4 |
| `--grad-accum` | Gradient accumulation steps | 2 |
| `--device` | Device to use (cpu, cuda, mps) | auto-detect |
| `--max-tokens` | Maximum number of tokens for generation | 100 |
| `--temperature` | Temperature for generation | 0.7 |
| `--save-model` | Save the fine-tuned model | False |
| `--save-name` | Name to save the fine-tuned model as | {model_id}_{mode}_{dataset} |
| `--run-name` | Name for the MLflow run | auto-generated |

### Viewing Results

After running the demo, you can view the results in the MLflow UI:

```bash
mlflow ui --port 5000
```

Then open your browser to http://localhost:5000 to see the experiment tracking information, including:
- Training parameters
- Loss and accuracy metrics
- Model artifacts
- Evaluation results 