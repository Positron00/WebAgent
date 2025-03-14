# Self-Hosted LLM Service for WebAgent

This document provides instructions for setting up and using the self-hosted LLM service with WebAgent.

## Overview

The self-hosted LLM service allows you to run your own local language model instead of relying on cloud-based services like OpenAI or Together AI. This can be useful for:

- Handling sensitive data that shouldn't leave your system
- Working in environments without internet access
- Reducing API costs
- Using custom fine-tuned models

## Requirements

- Python 3.9+
- PyTorch (with CUDA support recommended for larger models)
- transformers
- Flask
- pydantic
- A language model compatible with Hugging Face's transformers library

## Installation

1. Install the required packages:

```bash
pip install torch transformers flask pydantic requests accelerate bitsandbytes
```

2. Download a model from Hugging Face (or use your own):

```bash
# Example: Download Llama-3-8B
git clone https://huggingface.co/meta-llama/Llama-3-8B-Instruct
```

## Configuration

### 1. Update YAML Configuration

Edit your environment YAML file (e.g., `backend/config/dev.yaml`) to use the self-hosted provider:

```yaml
llm:
  provider: "self"
  temperature: 0.7
  timeout_seconds: 60
  max_tokens: 4096
  retry_attempts: 3
  batch_size: 10
  self_hosted_url: "http://localhost:8080"  # Update if running on a different port/host
  model_path: "/path/to/your/model"
```

### 2. Environment Variables

You can also configure the service using environment variables:

```bash
# Set the provider
export WEBAGENT_LLM_PROVIDER=self

# Set the URL for the self-hosted service
export SELF_HOSTED_LLM_URL=http://localhost:8080
```

## Starting the Service

Run the self-hosted LLM service:

### Basic Usage

```bash
python backend/app/services/self_hosted_llm_service.py --model_path "/path/to/your/model" --port 8080
```

### With Model Quantization (Reduced Memory Usage)

For larger models that may not fit in GPU memory, you can use quantization:

```bash
# 8-bit quantization (uses approximately half the memory)
python backend/app/services/self_hosted_llm_service.py --model_path "/path/to/your/model" --port 8080 --load_in_8bit

# 4-bit quantization (uses approximately quarter the memory)
python backend/app/services/self_hosted_llm_service.py --model_path "/path/to/your/model" --port 8080 --load_in_4bit
```

### Available Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model_path` | Path to the Hugging Face model | Required |
| `--port` | Port to run the service on | 8080 |
| `--host` | Host to bind the server to | 0.0.0.0 |
| `--load_in_8bit` | Load model in 8-bit precision (CUDA only) | False |
| `--load_in_4bit` | Load model in 4-bit precision (CUDA only) | False |

## Using with WebAgent

Once the self-hosted LLM service is running, you can start the WebAgent backend with the "self" provider:

```bash
cd backend
python -m app.main
```

The WebAgent backend will automatically connect to your self-hosted LLM service.

## API Endpoints

### Chat Completions

```
POST /v1/chat/completions
```

Request body:
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, how are you?"}
  ],
  "max_tokens": 1024,
  "temperature": 0.7,
  "top_p": 0.9,
  "frequency_penalty": 0.0,
  "presence_penalty": 0.0,
  "timeout": 60
}
```

### Health Check

```
GET /health
```

Expected response if the model is loaded:
```json
{
  "status": "ready",
  "model": "your-model-name",
  "device": "cuda",
  "cuda_available": true,
  "cuda_device_count": 1,
  "memory_allocated_gb": "4.21",
  "memory_reserved_gb": "5.00"
}
```

### Metrics

```
GET /metrics
```

Returns detailed metrics about the service:
```json
{
  "service": "self-hosted-llm",
  "model_loaded": true,
  "model_name": "Llama-3-8B-Instruct",
  "device": "cuda",
  "gpu_memory_allocated_gb": "4.21",
  "gpu_memory_reserved_gb": "5.00",
  "gpu_name": "NVIDIA GeForce RTX 3090"
}
```

## Advanced Features

### Memory Management

The service implements automatic memory management:
- Garbage collection after model loading and text generation
- GPU memory cache clearing when using CUDA
- Graceful shutdown to properly unload models
- OOM detection and helpful error messages

### Request Timeouts

All requests have a configurable timeout:
- Default timeout: 60 seconds
- Maximum timeout: 300 seconds
- Client can specify a custom timeout in request body
- Request will abort with a 408 status code if it exceeds timeout

### Model Quantization Options

Reduce memory usage with quantization options:
- 8-bit quantization: Reduces memory by ~50%
- 4-bit quantization: Reduces memory by ~75%
- Uses bitsandbytes library for efficient quantization
- Automatically selects appropriate parameters based on hardware

### Error Handling

Improved error handling for cleaner operation:
- Detailed error messages with more specific HTTP status codes
- Better handling of out-of-memory conditions with recovery
- Validation of request parameters with meaningful feedback
- Context length checking to prevent out-of-memory errors
- Graceful shutdown with proper resource cleanup

## Fallback Behavior

If the self-hosted LLM service is unavailable, WebAgent will attempt to fall back to:
1. Together AI (if configured)
2. OpenAI (if configured)

This ensures the application remains functional even if the self-hosted service isn't available.

## Troubleshooting

### Common Issues

1. **Out of memory errors**:
   - Use `--load_in_8bit` or `--load_in_4bit` to quantize the model
   - Try a smaller model (e.g., 7B instead of 13B)
   - Reduce the `max_tokens` parameter in requests
   - Add more VRAM/RAM to your system

2. **Slow responses**:
   - Ensure you're using GPU acceleration if available
   - Consider using a smaller model
   - Try quantization, which can sometimes be faster
   - Reduce the `max_tokens` parameter in requests

3. **Connection refused**:
   - Check if the service is running
   - Verify the port isn't blocked by firewall
   - Ensure the host is correctly configured

4. **Model loading errors**:
   - Verify the model path is correct
   - Check if the model is compatible with Hugging Face transformers
   - Ensure you have enough disk space for model files
   - Check for required files (.bin, config.json, etc.)

5. **Request timeout errors**:
   - Increase the `timeout` parameter in the request
   - Reduce the `max_tokens` parameter to generate shorter responses
   - Consider using model quantization for faster inference

### Checking Service Status

To check if the service is running properly, use the health endpoint:

```bash
curl http://localhost:8080/health
```

For more detailed metrics:

```bash
curl http://localhost:8080/metrics
```

### Logs

The service logs detailed information to stdout. Check the logs for:
- Model loading status and timing
- Request processing details
- Error messages with stack traces
- Memory usage statistics

## Optimizing Performance

1. **Hardware Recommendations**:
   - GPU with 8GB+ VRAM for 7B models with 8-bit quantization
   - GPU with 16GB+ VRAM for 13B models with 8-bit quantization
   - GPU with 24GB+ VRAM for running 70B models with 4-bit quantization

2. **Model Size Tradeoffs**:
   - 7B models: Fast, lower memory usage, less powerful
   - 13B models: Good balance of performance and speed
   - 70B models: Most powerful, but significantly slower and higher resource usage

3. **Quantization Impact**:
   - FP16 (default): Best quality, highest memory usage
   - 8-bit: Good quality, ~50% memory reduction
   - 4-bit: Acceptable quality, ~75% memory reduction 