"""
Self-hosted LLM service for WebAgent.

This standalone script serves a local LLM model via a Flask API,
making it compatible with the WebAgent LLM service framework.

Usage:
    python self_hosted_llm_service.py --model_path "/path/to/model" --port 8080

Requirements:
    - torch
    - transformers
    - flask
    - pydantic
    - mlflow
"""
import argparse
import logging
import os
import time
import gc
import signal
import traceback
import uuid
from functools import wraps
from typing import List, Dict, Any, Optional

import torch
import mlflow
from flask import Flask, request, jsonify, Response, current_app
from pydantic import BaseModel, Field, ValidationError
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread, Event

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("self-hosted-llm")

# Initialize Flask app
app = Flask(__name__)

# Global variables for model and tokenizer
model = None
tokenizer = None
device = None
model_name = None
generation_thread = None
shutdown_event = Event()

# Timeout handling (in seconds)
DEFAULT_REQUEST_TIMEOUT = 60
MAX_REQUEST_TIMEOUT = 300

# MLflow configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "self-hosted-llm")
mlflow_run_id = None

# Pydantic models for request validation
class Message(BaseModel):
    role: str
    content: str

class GenerationRequest(BaseModel):
    messages: List[Message]
    max_tokens: int = Field(default=1024, ge=1, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    frequency_penalty: float = Field(default=0.0, ge=0.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=0.0, le=2.0)
    stream: bool = Field(default=False)
    timeout: Optional[int] = Field(default=None, ge=1, le=MAX_REQUEST_TIMEOUT)
    track_experiment: bool = Field(default=False)
    experiment_name: Optional[str] = Field(default=None)
    run_name: Optional[str] = Field(default=None)
    tags: Optional[Dict[str, str]] = Field(default=None)

class GenerationResponse(BaseModel):
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]

def setup_mlflow(tracking_uri: str, experiment_name: str):
    """Set up MLflow tracking."""
    try:
        mlflow.set_tracking_uri(tracking_uri)
        
        # Create the experiment if it doesn't exist
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name)
        
        mlflow.set_experiment(experiment_name)
        
        logger.info(f"MLflow tracking set up with URI: {tracking_uri}, experiment: {experiment_name}")
    except Exception as e:
        logger.error(f"Failed to set up MLflow: {e}")
        logger.error(traceback.format_exc())

def start_mlflow_run(run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
    """Start a new MLflow run and return the run ID."""
    global mlflow_run_id
    
    try:
        # Start a new MLflow run
        run = mlflow.start_run(run_name=run_name)
        mlflow_run_id = run.info.run_id
        
        # Set tags if provided
        if tags:
            mlflow.set_tags(tags)
        
        # Always add some default tags
        mlflow.set_tag("model_name", model_name)
        mlflow.set_tag("device", device)
        mlflow.set_tag("cuda_available", torch.cuda.is_available())
        
        if torch.cuda.is_available():
            mlflow.set_tag("gpu_name", torch.cuda.get_device_name(0))
            mlflow.set_tag("gpu_memory_gb", f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f}")
        
        logger.info(f"Started MLflow run: {mlflow_run_id}")
        return mlflow_run_id
    except Exception as e:
        logger.error(f"Failed to start MLflow run: {e}")
        logger.error(traceback.format_exc())
        return None

def log_model_parameters():
    """Log model parameters to MLflow."""
    if model is None or not mlflow_run_id:
        return
    
    try:
        # Log model configuration parameters
        config_dict = model.config.to_dict()
        for key, value in config_dict.items():
            # Skip complex objects that can't be serialized properly
            if isinstance(value, (str, int, float, bool)) and not key.startswith("_"):
                mlflow.log_param(f"model.{key}", value)
        
        # Log model size (parameters count)
        parameter_count = sum(p.numel() for p in model.parameters())
        mlflow.log_param("model.parameter_count", parameter_count)
        mlflow.log_param("model.parameter_count_millions", parameter_count / 1_000_000)
        
        logger.info(f"Logged model parameters to MLflow run: {mlflow_run_id}")
    except Exception as e:
        logger.error(f"Failed to log model parameters: {e}")
        logger.error(traceback.format_exc())

def request_timeout(seconds_or_func=None):
    """Decorator to add timeout to Flask route handlers."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            timeout = DEFAULT_REQUEST_TIMEOUT
            if request.json and "timeout" in request.json:
                timeout = min(request.json["timeout"], MAX_REQUEST_TIMEOUT)
            
            def target():
                current_app.response = func(*args, **kwargs)
                
            thread = Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout)
            
            if thread.is_alive():
                # Request timed out
                logger.warning(f"Request timed out after {timeout} seconds")
                return jsonify({
                    "error": f"Request timed out after {timeout} seconds. Consider reducing max_tokens or increasing timeout."
                }), 408
            
            return current_app.response
        return wrapper

    # Handle both @request_timeout and @request_timeout(seconds)
    if callable(seconds_or_func):
        return decorator(seconds_or_func)
    else:
        return decorator

def format_prompt(messages: List[Message]) -> str:
    """Format a list of chat messages into a prompt string."""
    formatted_prompt = ""
    
    for message in messages:
        if message.role == "system":
            formatted_prompt += f"<|system|>\n{message.content}\n"
        elif message.role == "user":
            formatted_prompt += f"<|user|>\n{message.content}\n"
        elif message.role == "assistant":
            formatted_prompt += f"<|assistant|>\n{message.content}\n"
    
    # Add final assistant marker for the response
    formatted_prompt += "<|assistant|>\n"
    
    return formatted_prompt

def load_model(model_path: str, load_in_8bit: bool = False, load_in_4bit: bool = False):
    """
    Load the model and tokenizer from the specified path.
    
    Args:
        model_path: Path to the Hugging Face model
        load_in_8bit: Whether to load model in 8-bit precision
        load_in_4bit: Whether to load model in 4-bit precision
    """
    global model, tokenizer, device, model_name
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    
    try:
        logger.info(f"Loading model from {model_path}")
        start_time = time.time()
        
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            if tokenizer.pad_token_id is None:
                logger.info("Setting pad_token_id to eos_token_id")
                tokenizer.pad_token_id = tokenizer.eos_token_id
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            raise
        
        # Load model with appropriate settings based on available hardware
        try:
            # Check memory availability for loading model
            if device == "cuda":
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                logger.info(f"Available GPU memory: {gpu_memory:.2f} GB")
            
            # Set loading configuration based on options
            quantization_config = None
            torch_dtype = torch.float16 if device == "cuda" else torch.float32
            
            if load_in_8bit and device == "cuda":
                logger.info("Loading model in 8-bit precision")
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    load_in_8bit=True
                )
            elif load_in_4bit and device == "cuda":
                logger.info("Loading model in 4-bit precision")
                from transformers import BitsAndBytesConfig
                
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    quantization_config=quantization_config
                )
            elif device == "cuda":
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch_dtype,
                    device_map="auto"
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map={"": device}
                )
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.error(traceback.format_exc())
            raise
        
        model_name = os.path.basename(model_path)
        
        # Log loading time
        elapsed_time = time.time() - start_time
        logger.info(f"Model loaded in {elapsed_time:.2f} seconds")
        
        # Force garbage collection to free memory
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
            
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.error(traceback.format_exc())
        raise

def unload_model():
    """Unload model and free up memory."""
    global model, tokenizer
    
    logger.info("Unloading model from memory")
    
    # Delete model and tokenizer
    if model is not None:
        del model
        model = None
    
    if tokenizer is not None:
        del tokenizer
        tokenizer = None
    
    # Force garbage collection
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    
    logger.info("Model unloaded successfully")

@app.route("/v1/chat/completions", methods=["POST"])
@request_timeout
def generate():
    """Handle chat completion requests."""
    if model is None or tokenizer is None:
        return jsonify({"error": "Model not loaded"}), 503
    
    # Parse request
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Missing request body"}), 400
            
        req = GenerationRequest(**data)
    except ValidationError as e:
        logger.error(f"Invalid request: {e}")
        return jsonify({"error": f"Validation error: {str(e)}"}), 400
    except Exception as e:
        logger.error(f"Error parsing request: {e}")
        return jsonify({"error": f"Error parsing request: {str(e)}"}), 400
    
    # Generate a unique request ID for tracking
    request_id = str(uuid.uuid4())
    
    # Start MLflow run if tracking is enabled
    mlflow_run = None
    if req.track_experiment:
        try:
            # Set up run name if not provided
            run_name = req.run_name or f"generation-{request_id}"
            
            # Set up tags with default request parameters
            tags = req.tags or {}
            tags.update({
                "request_id": request_id,
                "max_tokens": str(req.max_tokens),
                "temperature": str(req.temperature),
                "top_p": str(req.top_p)
            })
            
            # Start the MLflow run
            experiment_name = req.experiment_name or MLFLOW_EXPERIMENT_NAME
            mlflow.set_experiment(experiment_name)
            mlflow_run = mlflow.start_run(run_name=run_name)
            
            # Log request parameters
            mlflow.log_params({
                "max_tokens": req.max_tokens,
                "temperature": req.temperature,
                "top_p": req.top_p,
                "frequency_penalty": req.frequency_penalty,
                "presence_penalty": req.presence_penalty
            })
            
            # Set tags
            mlflow.set_tags(tags)
            
            logger.info(f"Started MLflow run: {mlflow_run.info.run_id} for request {request_id}")
        except Exception as e:
            logger.error(f"Failed to start MLflow run: {e}")
            logger.error(traceback.format_exc())
    
    # Format prompt
    try:
        prompt = format_prompt(req.messages)
        
        # Tokenize input
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        input_tokens_count = input_ids.shape[1]
        
        # Check if inputs exceed model's context length
        max_context_length = getattr(model.config, "max_position_embeddings", 4096)
        if input_tokens_count > max_context_length:
            return jsonify({
                "error": f"Input length ({input_tokens_count} tokens) exceeds model's maximum context length ({max_context_length} tokens)"
            }), 400
    except Exception as e:
        logger.error(f"Error preparing input: {e}")
        return jsonify({"error": f"Error preparing input: {str(e)}"}), 500
    
    # Generate text
    try:
        start_time = time.time()
        
        if req.stream:
            # Streamed generation not implemented in this basic version
            return jsonify({"error": "Streaming not supported in this implementation"}), 501
        else:
            # Set generation parameters
            gen_kwargs = {
                "input_ids": input_ids,
                "max_new_tokens": req.max_tokens,
                "temperature": req.temperature,
                "top_p": req.top_p,
                "do_sample": req.temperature > 0,
                "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id
            }
            
            # Add frequency and presence penalties if model supports them
            if hasattr(model.config, "frequency_penalty") and req.frequency_penalty > 0:
                gen_kwargs["frequency_penalty"] = req.frequency_penalty
            
            if hasattr(model.config, "presence_penalty") and req.presence_penalty > 0:
                gen_kwargs["presence_penalty"] = req.presence_penalty
            
            # Generate text
            with torch.no_grad():
                output = model.generate(**gen_kwargs)
            
            # Decode output
            generated_text = tokenizer.decode(output[0][input_tokens_count:], skip_special_tokens=True)
            output_tokens_count = output.shape[1] - input_tokens_count
            
            # Calculate time metrics
            elapsed_time = time.time() - start_time
            tokens_per_second = output_tokens_count / elapsed_time if elapsed_time > 0 else 0
            
            # Log metrics to MLflow if tracking is enabled
            if mlflow_run:
                try:
                    mlflow.log_metrics({
                        "input_tokens": input_tokens_count,
                        "output_tokens": output_tokens_count,
                        "total_tokens": input_tokens_count + output_tokens_count,
                        "generation_time_seconds": elapsed_time,
                        "tokens_per_second": tokens_per_second
                    })
                    
                    # Log prompt and generated text as artifacts
                    with mlflow.start_run(run_id=mlflow_run.info.run_id):
                        prompt_path = f"/tmp/prompt-{request_id}.txt"
                        response_path = f"/tmp/response-{request_id}.txt"
                        
                        with open(prompt_path, "w") as f:
                            f.write(prompt)
                        
                        with open(response_path, "w") as f:
                            f.write(generated_text)
                            
                        mlflow.log_artifact(prompt_path, "prompts")
                        mlflow.log_artifact(response_path, "responses")
                        
                        # Clean up temporary files
                        os.remove(prompt_path)
                        os.remove(response_path)
                        
                    # End the run
                    mlflow.end_run()
                    logger.info(f"MLflow tracking completed for request {request_id}")
                except Exception as e:
                    logger.error(f"Error logging to MLflow: {e}")
                    logger.error(traceback.format_exc())
            
            # Format response
            response = {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": generated_text.strip()
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": input_tokens_count,
                    "completion_tokens": output_tokens_count,
                    "total_tokens": input_tokens_count + output_tokens_count
                },
                "performance": {
                    "generation_time_seconds": elapsed_time,
                    "tokens_per_second": tokens_per_second
                }
            }
            
            # Add MLflow run info if tracking was enabled
            if mlflow_run:
                response["experiment_tracking"] = {
                    "run_id": mlflow_run.info.run_id,
                    "experiment_id": mlflow_run.info.experiment_id,
                    "tracking_uri": mlflow.get_tracking_uri()
                }
            
            logger.info(f"Generated response in {elapsed_time:.2f} seconds with {output_tokens_count} tokens ({tokens_per_second:.2f} tokens/sec)")
            
            # Force garbage collection to free memory
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
                
            return jsonify(response)
    
    except torch.cuda.OutOfMemoryError:
        logger.error("CUDA out of memory error during generation")
        gc.collect()
        torch.cuda.empty_cache()
        return jsonify({
            "error": "GPU out of memory. Try reducing max_tokens or use a smaller model."
        }), 507  # Custom status code for resource exceeded
    except Exception as e:
        logger.error(f"Error generating text: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Generation error: {str(e)}"}), 500
    finally:
        # Make sure to end the MLflow run if it was started
        if mlflow_run:
            try:
                mlflow.end_run()
            except:
                pass

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    try:
        if model is None or tokenizer is None:
            return jsonify({
                "status": "loading",
                "model": model_name or "unknown"
            }), 503
        
        # Get model memory usage if on CUDA
        memory_info = {}
        if device == "cuda":
            memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
            memory_reserved = torch.cuda.memory_reserved() / (1024**3)    # GB
            memory_info = {
                "memory_allocated_gb": f"{memory_allocated:.2f}",
                "memory_reserved_gb": f"{memory_reserved:.2f}"
            }
        
        # Get MLflow status
        mlflow_status = "not_configured"
        try:
            tracking_uri = mlflow.get_tracking_uri()
            mlflow_status = "configured"
            try:
                # Try to list experiments to verify connection
                mlflow.list_experiments()
                mlflow_status = "connected"
            except:
                mlflow_status = "error_connecting"
        except:
            pass
        
        return jsonify({
            "status": "ready",
            "model": model_name,
            "device": device,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "mlflow_status": mlflow_status,
            **memory_info
        })
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route("/metrics", methods=["GET"])
def metrics():
    """Return service metrics."""
    try:
        # Get GPU memory info if available
        gpu_info = {}
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
            memory_reserved = torch.cuda.memory_reserved() / (1024**3)    # GB
            gpu_info = {
                "gpu_memory_allocated_gb": f"{memory_allocated:.2f}",
                "gpu_memory_reserved_gb": f"{memory_reserved:.2f}",
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "N/A"
            }
        
        # Get MLflow info
        mlflow_info = {
            "tracking_uri": mlflow.get_tracking_uri(),
            "current_experiment": mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
        }
        
        return jsonify({
            "service": "self-hosted-llm",
            "model_loaded": model is not None and tokenizer is not None,
            "model_name": model_name or "none",
            "device": device or "none",
            "mlflow": {
                "tracking_uri": str(mlflow_info.get("tracking_uri")),
                "experiment_name": MLFLOW_EXPERIMENT_NAME
            },
            **gpu_info
        })
    except Exception as e:
        logger.error(f"Error in metrics: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route("/experiments", methods=["GET"])
def list_experiments():
    """List MLflow experiments."""
    try:
        experiments = mlflow.list_experiments()
        result = []
        
        for exp in experiments:
            result.append({
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "artifact_location": exp.artifact_location,
                "lifecycle_stage": exp.lifecycle_stage
            })
        
        return jsonify({
            "experiments": result,
            "tracking_uri": mlflow.get_tracking_uri()
        })
    except Exception as e:
        logger.error(f"Error listing experiments: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route("/experiments/<experiment_id>/runs", methods=["GET"])
def list_runs(experiment_id):
    """List runs for a specific experiment."""
    try:
        runs = mlflow.search_runs(experiment_ids=[experiment_id])
        
        runs_data = []
        for _, run in runs.iterrows():
            runs_data.append({
                "run_id": run.run_id,
                "status": run.status,
                "start_time": run.start_time,
                "end_time": run.end_time,
                "metrics": {k: v for k, v in run.items() if k.startswith("metrics.")},
                "params": {k: v for k, v in run.items() if k.startswith("params.")},
                "tags": {k: v for k, v in run.items() if k.startswith("tags.")}
            })
        
        return jsonify({
            "runs": runs_data,
            "experiment_id": experiment_id
        })
    except Exception as e:
        logger.error(f"Error listing runs: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route("/mlflow-config", methods=["POST"])
def update_mlflow_config():
    """Update MLflow configuration."""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Missing request body"}), 400
            
        # Update tracking URI if provided
        if "tracking_uri" in data:
            mlflow.set_tracking_uri(data["tracking_uri"])
            
        # Update experiment name if provided
        if "experiment_name" in data:
            mlflow.set_experiment(data["experiment_name"])
            global MLFLOW_EXPERIMENT_NAME
            MLFLOW_EXPERIMENT_NAME = data["experiment_name"]
            
        return jsonify({
            "status": "success",
            "tracking_uri": mlflow.get_tracking_uri(),
            "experiment_name": MLFLOW_EXPERIMENT_NAME
        })
    except Exception as e:
        logger.error(f"Error updating MLflow config: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

def signal_handler(sig, frame):
    """Handle termination signals to clean up resources."""
    logger.info(f"Received signal {sig}, shutting down")
    shutdown_event.set()
    
    # End any active MLflow run
    try:
        mlflow.end_run()
    except:
        pass
        
    unload_model()
    # Give a brief moment for cleanup before exiting
    time.sleep(1)
    exit(0)

def main():
    """Main function to parse arguments and start the service."""
    parser = argparse.ArgumentParser(description="Self-hosted LLM service")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--port", type=int, default=8080, help="Port to run the service on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the service on")
    parser.add_argument("--load_in_8bit", action="store_true", help="Load model in 8-bit precision (CUDA only)")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit precision (CUDA only)")
    parser.add_argument("--mlflow_tracking_uri", type=str, help="MLflow tracking URI")
    parser.add_argument("--mlflow_experiment_name", type=str, help="MLflow experiment name")
    args = parser.parse_args()
    
    # Set MLflow tracking URI and experiment name if provided
    global MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME
    if args.mlflow_tracking_uri:
        MLFLOW_TRACKING_URI = args.mlflow_tracking_uri
    if args.mlflow_experiment_name:
        MLFLOW_EXPERIMENT_NAME = args.mlflow_experiment_name
    
    # Set up MLflow tracking
    setup_mlflow(MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME)
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Load model
        load_model(args.model_path, args.load_in_8bit, args.load_in_4bit)
        
        # Log model parameters to MLflow
        start_mlflow_run("model_load")
        log_model_parameters()
        mlflow.end_run()
        
        # Start server
        logger.info(f"Starting server on {args.host}:{args.port}")
        app.run(host=args.host, port=args.port, threaded=True)
    except FileNotFoundError as e:
        logger.error(f"Model not found: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"Error starting service: {e}")
        logger.error(traceback.format_exc())
        exit(1)
    finally:
        # Ensure model is unloaded properly and MLflow run is ended
        try:
            mlflow.end_run()
        except:
            pass
        unload_model()

if __name__ == "__main__":
    main() 