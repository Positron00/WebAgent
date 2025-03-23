"""
LLM service for the WebAgent backend.
Provides access to different LLM providers including Together AI, Groq, OpenAI, Anthropic, Gemini, and self-hosted models.
"""
from typing import Dict, Optional, Any, List, Tuple, Callable, Type, TypeVar, cast
import logging
import os
import time
import traceback
from functools import wraps
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log

from langchain_openai import ChatOpenAI
# Keep both imports until compatibility is resolved
try:
    from langchain_together import Together
except ImportError:
    from langchain_community.llms import Together as TogetherLLM
from langchain.chat_models.base import BaseChatModel
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.schema.runnable import Runnable
from app.core.config import settings
from app.core.loadEnvYAML import get_llm_config

# Import observability tools
from app.core.observability import (
    track_model_prediction,
    MODEL_PREDICTION_COUNT,
    MODEL_LATENCY,
    record_data_drift,
    log_to_mlflow
)

logger = logging.getLogger(__name__)

# Define model constants - these should match config.py
TOGETHER_API_BASE = os.getenv("TOGETHER_API_BASE", "https://api.together.xyz/v1")
TOGETHER_DEFAULT_MODEL = os.getenv("TOGETHER_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free")
OPENAI_DEFAULT_MODEL = "gpt-4o-mini-2024-07-18"
GROQ_DEFAULT_MODEL = "meta-llama/Llama-3.2-1B"
ANTHROPIC_DEFAULT_MODEL = "claude-3-5-sonnet-20240620"
GEMINI_DEFAULT_MODEL = "gemini-1.5-flash-002"
SELF_HOSTED_DEFAULT_URL = "http://localhost:8080"

# Define input/output types for TypeVar
Input = TypeVar('Input')
Output = TypeVar('Output')

# Custom wrapper to ensure Together LLM has invoke method
class TogetherWrapper(Runnable[str, str]):
    """A wrapper for Together LLM that implements the LangChain Runnable interface."""
    
    def __init__(self, model_name: str, temperature: float, max_tokens: int, api_key: str):
        """Initialize the wrapper."""
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm = self._create_llm()
        
        # Ensure invoke method exists
        if not hasattr(self.llm, "invoke"):
            self._add_invoke_method()
    
    def _create_llm(self):
        """Create the underlying LLM instance."""
        try:
            # Try the new package first
            from langchain_together import Together
            return Together(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                together_api_key=self.api_key
            )
        except ImportError:
            # Fall back to community package
            from langchain_community.llms import Together as TogetherLLM
            return TogetherLLM(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                together_api_key=self.api_key
            )
    
    def _add_invoke_method(self):
        """Add invoke method to the LLM if missing."""
        if hasattr(self.llm, "complete"):
            # Newer versions might use complete
            def custom_invoke(prompt, **kwargs):
                return self.llm.complete(prompt, **kwargs)
            self.llm.invoke = custom_invoke
        elif hasattr(self.llm, "generate"):
            # Older versions might use generate
            def custom_invoke(prompt, **kwargs):
                result = self.llm.generate([prompt], **kwargs)
                return result.generations[0][0].text
            self.llm.invoke = custom_invoke
        elif hasattr(self.llm, "__call__"):
            # Try the call method
            self.llm.invoke = self.llm.__call__
        else:
            raise ValueError("Unable to add invoke method to Together LLM")
    
    def invoke(self, input: str, config: Optional[Dict[str, Any]] = None) -> str:
        """Invoke the LLM. This is the main method required by the Runnable interface."""
        _enforce_together_api_rate_limit()
        try:
            return self.llm.invoke(input, **(config or {}))
        except Exception as e:
            logger.error(f"Error invoking Together LLM: {str(e)}")
            raise
    
    async def ainvoke(self, input: str, config: Optional[Dict[str, Any]] = None) -> str:
        """Async invoke method required by the Runnable interface."""
        _enforce_together_api_rate_limit()
        if hasattr(self.llm, "ainvoke"):
            return await self.llm.ainvoke(input, **(config or {}))
        else:
            # Fallback to sync version if async not available
            return self.invoke(input, config)
    
    def batch(self, inputs: List[str], config: Optional[Dict[str, Any]] = None) -> List[str]:
        """Batch processing method required by the Runnable interface."""
        return [self.invoke(input, config) for input in inputs]
    
    async def abatch(self, inputs: List[str], config: Optional[Dict[str, Any]] = None) -> List[str]:
        """Async batch processing method required by the Runnable interface."""
        return [await self.ainvoke(input, config) for input in inputs]
    
    def stream(self, input: str, config: Optional[Dict[str, Any]] = None):
        """Stream processing method required by the Runnable interface."""
        if hasattr(self.llm, "stream"):
            return self.llm.stream(input, **(config or {}))
        else:
            # If streaming not supported, yield the entire result at once
            yield self.invoke(input, config)
    
    async def astream(self, input: str, config: Optional[Dict[str, Any]] = None):
        """Async stream processing method required by the Runnable interface."""
        if hasattr(self.llm, "astream"):
            async for chunk in self.llm.astream(input, **(config or {})):
                yield chunk
        else:
            # If async streaming not supported, yield the entire result at once
            yield await self.ainvoke(input, config)
            
    def __getattr__(self, name):
        """Delegate to underlying LLM for other attributes."""
        return getattr(self.llm, name)

# Define service metrics
class LLMMetrics:
    """Metrics for LLM service."""
    def __init__(self):
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_tokens = 0
        self.total_latency_seconds = 0
        self.provider_stats = {
            "openai": {"requests": 0, "tokens": 0, "errors": 0},
            "together": {"requests": 0, "tokens": 0, "errors": 0},
            "self": {"requests": 0, "tokens": 0, "errors": 0}
        }
    
    def record_request(self, provider: str, success: bool, tokens: int = 0, latency: float = 0):
        """Record metrics for a request."""
        self.total_requests += 1
        
        if success:
            self.successful_requests += 1
            self.total_tokens += tokens
            self.total_latency_seconds += latency
            
            if provider in self.provider_stats:
                self.provider_stats[provider]["requests"] += 1
                self.provider_stats[provider]["tokens"] += tokens
                
            # Track in Prometheus metrics too for real-time monitoring
            MODEL_PREDICTION_COUNT.labels(
                model_name=provider, 
                task_type="chat", 
                status="success"
            ).inc()
            
            if latency > 0:
                MODEL_LATENCY.labels(
                    model_name=provider,
                    task_type="chat"
                ).observe(latency)
        else:
            self.failed_requests += 1
            
            if provider in self.provider_stats:
                self.provider_stats[provider]["errors"] += 1
                
            # Track in Prometheus metrics
            MODEL_PREDICTION_COUNT.labels(
                model_name=provider, 
                task_type="chat", 
                status="error"
            ).inc()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics as a dictionary."""
        success_rate = 0
        if self.total_requests > 0:
            success_rate = (self.successful_requests / self.total_requests) * 100
            
        avg_latency = 0
        if self.successful_requests > 0:
            avg_latency = self.total_latency_seconds / self.successful_requests
            
        return {
            "requests": {
                "total": self.total_requests,
                "successful": self.successful_requests,
                "failed": self.failed_requests,
                "success_rate_percent": round(success_rate, 2)
            },
            "tokens": {
                "total": self.total_tokens,
                "average_per_request": round(self.total_tokens / max(1, self.successful_requests), 2)
            },
            "latency": {
                "total_seconds": round(self.total_latency_seconds, 2),
                "average_seconds": round(avg_latency, 2)
            },
            "providers": self.provider_stats
        }

# Create global metrics instance
metrics = LLMMetrics()

# Rate limiting for Together AI API
_last_together_api_call = 0
TOGETHER_RATE_LIMIT_BUFFER = 12.0  # Buffer time in seconds (higher than the 10 req/min limit)

def _enforce_together_api_rate_limit():
    """
    Enforce rate limiting for Together AI API calls to avoid hitting the rate limit.
    """
    global _last_together_api_call
    current_time = time.time()
    elapsed = current_time - _last_together_api_call
    
    if _last_together_api_call > 0 and elapsed < TOGETHER_RATE_LIMIT_BUFFER:
        sleep_time = TOGETHER_RATE_LIMIT_BUFFER - elapsed
        logger.info(f"Rate limiting: Sleeping for {sleep_time:.2f} seconds to avoid Together AI rate limit")
        time.sleep(sleep_time)
    
    _last_together_api_call = time.time()

def log_llm_request(provider: str = "unknown"):
    """Decorator to log LLM requests and record metrics."""
    def decorator(func):
        @wraps(func)
        @track_model_prediction(model_name=provider, task_type="llm_initialization")
        def wrapper(*args, **kwargs):
            start_time = time.time()
            logger.info(f"LLM request to {provider} initiated")
            
            try:
                result = func(*args, **kwargs)
                elapsed_time = time.time() - start_time
                
                # Extract token usage if available
                tokens = 0
                if hasattr(result, "llm_output") and result.llm_output:
                    if "token_usage" in result.llm_output:
                        token_usage = result.llm_output["token_usage"]
                        if "total_tokens" in token_usage:
                            tokens = token_usage["total_tokens"]
                
                logger.info(f"LLM request to {provider} completed in {elapsed_time:.2f}s with {tokens} tokens")
                metrics.record_request(provider, True, tokens, elapsed_time)
                
                # Record MLflow metrics if available
                log_to_mlflow(
                    metrics={
                        "initialization_success": 1,
                        "initialization_time": elapsed_time
                    },
                    params={
                        "provider": provider,
                        "model_name": kwargs.get("model_name", "default")
                    },
                    experiment_name="llm_service"
                )
                
                return result
            except Exception as e:
                elapsed_time = time.time() - start_time
                logger.error(f"LLM request to {provider} failed after {elapsed_time:.2f}s: {str(e)}")
                metrics.record_request(provider, False, 0, elapsed_time)
                
                # Record MLflow metrics if available
                log_to_mlflow(
                    metrics={
                        "initialization_success": 0,
                        "initialization_time": elapsed_time
                    },
                    params={
                        "provider": provider,
                        "model_name": kwargs.get("model_name", "default"),
                        "error": str(e)
                    },
                    experiment_name="llm_service"
                )
                
                raise e
        
        return wrapper
    
    return decorator

# Add a custom LangChain chat model for self-hosted LLMs
class SelfHostedChatModel(BaseChatModel):
    """Custom LangChain chat model for self-hosted LLMs."""
    
    def __init__(
        self,
        base_url: str = SELF_HOSTED_DEFAULT_URL,
        model: str = "local-model",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        timeout: int = 60,
        retry_attempts: int = 3,
        **kwargs
    ):
        """Initialize the self-hosted chat model."""
        super().__init__(**kwargs)
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        
        # Track initialization in metrics
        metrics.record_request("self", success=True)
        MODEL_PREDICTION_COUNT.labels(
            model_name="self", 
            task_type="initialization", 
            status="success"
        ).inc()
        
        # Check if the service is available
        self._check_health()
    
    def _check_health(self):
        """Check if the self-hosted LLM service is available."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            response.raise_for_status()
            status = response.json()
            if status.get("status") != "ready":
                logger.warning(f"Self-hosted LLM not ready: {status}")
            else:
                logger.info(f"Self-hosted LLM health check successful: {status}")
                
                # Log detailed metrics if available
                if "memory_allocated_gb" in status:
                    logger.info(f"Self-hosted LLM memory usage: {status['memory_allocated_gb']} GB allocated")
                    
                if "device" in status:
                    logger.info(f"Self-hosted LLM using device: {status['device']}")
        except requests.RequestException as e:
            logger.error(f"Self-hosted LLM health check failed (connection error): {str(e)}")
            raise ValueError(f"Self-hosted LLM service unavailable: {str(e)}")
        except Exception as e:
            logger.error(f"Self-hosted LLM health check failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise ValueError(f"Self-hosted LLM service unavailable: {str(e)}")
    
    def _convert_message_to_dict(self, message):
        """Convert LangChain message to API format."""
        if isinstance(message, HumanMessage):
            return {"role": "user", "content": message.content}
        elif isinstance(message, AIMessage):
            return {"role": "assistant", "content": message.content}
        elif isinstance(message, SystemMessage):
            return {"role": "system", "content": message.content}
        else:
            return {"role": "user", "content": str(message.content)}
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.RequestException, ValueError)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """Generate a response from the self-hosted LLM."""
        start_time = time.time()
        request_id = kwargs.get("request_id", "unknown")
        
        headers = {
            "Content-Type": "application/json"
        }
        
        # Convert messages to dict format
        messages_dicts = [self._convert_message_to_dict(m) for m in messages]
        
        # Construct the request
        request_data = {
            "model": self.model,
            "messages": messages_dicts,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        
        if stop:
            request_data["stop"] = stop
            
        # Optional parameters
        if "top_p" in kwargs:
            request_data["top_p"] = kwargs["top_p"]
        if "frequency_penalty" in kwargs:
            request_data["frequency_penalty"] = kwargs["frequency_penalty"]
        if "presence_penalty" in kwargs:
            request_data["presence_penalty"] = kwargs["presence_penalty"]
            
        # Log the length of the prompt for monitoring
        prompt_length = sum(len(m.get("content", "")) for m in messages_dicts)
        logger.debug(f"Request {request_id}: prompt length {prompt_length} chars")
        
        try:
            logger.debug(f"Sending request to self-hosted LLM at {self.base_url}/v1/chat/completions")
            
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers=headers,
                json=request_data,
                timeout=self.timeout
            )
            
            # Get response time for metrics
            latency = time.time() - start_time
            
            # Check for errors
            response.raise_for_status()
            response_data = response.json()
            
            # Extract token usage if available
            token_usage = 0
            usage = response_data.get("usage", {})
            if usage:
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                token_usage = prompt_tokens + completion_tokens
                logger.debug(f"Request {request_id}: used {prompt_tokens} prompt tokens, {completion_tokens} completion tokens")
                
            # Record in metrics
            metrics.record_request("self", success=True, tokens=token_usage, latency=latency)
            
            # Log to MLflow
            log_to_mlflow(
                metrics={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": token_usage,
                    "response_time": latency
                },
                experiment_name="llm_inference"
            )
            
            # Extract the assistant message
            try:
                choice = response_data["choices"][0]
                message = choice["message"]
                content = message["content"]
                
                if not content:
                    logger.warning(f"Empty response from LLM for request {request_id}")
                
                # Check for potential model drift by monitoring unusual responses
                if len(content) < 5 and prompt_length > 100:
                    record_data_drift("llm_response", 0.8)  # High drift score for unusually short responses
                    logger.warning(f"Unusually short response from LLM for request {request_id}")
                
                return AIMessage(content=content)
            except (KeyError, IndexError) as e:
                logger.error(f"Error parsing response: {e}")
                logger.error(f"Response data: {response_data}")
                raise ValueError(f"Invalid response format: {str(e)}")
                
        except requests.RequestException as e:
            error_msg = f"Error connecting to self-hosted LLM: {str(e)}"
            logger.error(error_msg)
            
            # Record in metrics
            metrics.record_request("self", success=False)
            
            # Log to MLflow
            log_to_mlflow(
                metrics={"request_failed": 1, "response_time": time.time() - start_time},
                params={"error_type": type(e).__name__, "error_message": str(e)},
                experiment_name="llm_inference"
            )
            
            raise e
    
    @property
    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return "self-hosted"

@log_llm_request(provider="openai")
def get_openai_llm(model_name: Optional[str] = None):
    """
    Get a LangChain ChatOpenAI instance.
    
    Args:
        model_name: Optional model name to use (defaults to settings.OPENAI_MODEL)
        
    Returns:
        A configured ChatOpenAI instance
    """
    model = model_name or settings.OPENAI_MODEL
    llm_config = get_llm_config()
    
    try:
        llm = ChatOpenAI(
            model=model,
            temperature=llm_config.temperature,
            timeout=llm_config.timeout_seconds,
            max_tokens=llm_config.max_tokens,
            api_key=settings.OPENAI_API_KEY,
            max_retries=llm_config.retry_attempts
        )
        return llm
    except Exception as e:
        logger.error(f"Error creating OpenAI LLM: {str(e)}")
        logger.error(traceback.format_exc())
        raise

@log_llm_request(provider="together")
def get_together_llm(model_name: Optional[str] = None):
    """
    Get a TogetherLLM instance with configuration.
    
    Args:
        model_name: Optional model name to override the default
    
    Returns:
        TogetherLLM instance or fallback to OpenAI if TogetherLLM is not available
    """
    # Get LLM configuration from settings
    llm_config = get_llm_config()
    
    # Use provided model or default to configured TOGETHER_MODEL
    model = model_name or settings.TOGETHER_MODEL
    
    together_api_key = settings.TOGETHER_API_KEY
    
    if not together_api_key:
        logger.warning("TOGETHER_API_KEY not configured in settings (.env.local)")
        logger.info("Falling back to OpenAI LLM")
        return get_openai_llm(model_name)
    
    try:
        # Apply rate limiting to avoid hitting Together API rate limits
        _enforce_together_api_rate_limit()
        
        # Use our custom wrapper to ensure invoke method is available
        wrapper = TogetherWrapper(
            model_name=model,
            temperature=llm_config.temperature,
            max_tokens=llm_config.max_tokens,
            api_key=together_api_key
        )
        
        # Test the LLM
        try:
            wrapper.invoke("test")
            logger.info(f"Successfully initialized Together LLM ({model})")
            return wrapper
        except Exception as test_error:
            logger.warning(f"Test of Together LLM failed: {str(test_error)}")
            logger.info("Falling back to OpenAI LLM")
            return get_openai_llm(model_name)
            
    except Exception as e:
        logger.warning(f"Failed to initialize Together LLM: {str(e)}")
        logger.info("Falling back to OpenAI LLM")
        return get_openai_llm(model_name)

@log_llm_request(provider="self")
def get_self_hosted_llm(model_name: Optional[str] = None):
    """
    Get a LangChain SelfHostedChatModel instance.
    
    Args:
        model_name: Optional model name to use (will be passed to the service)
        
    Returns:
        A configured SelfHostedChatModel instance
    """
    llm_config = get_llm_config()
    
    # Check for self-hosted LLM URL in environment variables or config
    base_url = os.getenv("SELF_HOSTED_LLM_URL", llm_config.self_hosted_url)
    
    try:
        llm = SelfHostedChatModel(
            base_url=base_url,
            model=model_name or "local-model",
            temperature=llm_config.temperature,
            max_tokens=llm_config.max_tokens,
            timeout=llm_config.timeout_seconds,
            retry_attempts=llm_config.retry_attempts
        )
        return llm
    except Exception as e:
        logger.error(f"Error creating self-hosted LLM: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def get_llm(model_name: Optional[str] = None):
    """
    Get a LangChain LLM instance based on the configured provider.
    
    Args:
        model_name: Optional model name to use (defaults to provider-specific default)
        
    Returns:
        A configured LLM instance
    """
    llm_config = get_llm_config()
    provider = llm_config.provider.lower()
    
    # Check if the CONFIRM_LLM_SELECTION environment variable is set
    # This allows for explicit confirmation of the provider and model
    if os.getenv("CONFIRM_LLM_SELECTION", "False").lower() == "true":
        # Get the confirmed provider and model from environment variables
        confirmed_provider = os.getenv("CONFIRMED_LLM_PROVIDER")
        confirmed_model = os.getenv("CONFIRMED_LLM_MODEL")
        
        if confirmed_provider:
            logger.info(f"Using explicitly confirmed provider: {confirmed_provider}")
            provider = confirmed_provider.lower()
        
        if confirmed_model:
            logger.info(f"Using explicitly confirmed model: {confirmed_model}")
            model_name = confirmed_model
    
    logger.info(f"Creating LLM instance with provider: {provider}")
    
    # Define fallback order
    fallback_order = []
    
    # Set up the fallback order based on the initial provider
    if provider == "together":
        fallback_order = ["together", "groq", "openai", "anthropic", "gemini", "self"]
    elif provider == "groq":
        fallback_order = ["groq", "together", "openai", "anthropic", "gemini", "self"]
    elif provider == "openai":
        fallback_order = ["openai", "together", "groq", "anthropic", "gemini", "self"]
    elif provider == "anthropic":
        fallback_order = ["anthropic", "together", "groq", "openai", "gemini", "self"]
    elif provider == "gemini":
        fallback_order = ["gemini", "together", "groq", "openai", "anthropic", "self"]
    elif provider == "self":
        fallback_order = ["self", "together", "groq", "openai", "anthropic", "gemini"]
    else:
        # Default fallback order
        fallback_order = ["together", "groq", "openai", "anthropic", "gemini", "self"]
        logger.warning(f"Unknown provider '{provider}', using default fallback order: {fallback_order}")
    
    # Track errors for detailed reporting
    provider_errors = {}
    
    # Try each provider in the fallback order
    for current_provider in fallback_order:
        try:
            logger.info(f"Attempting to initialize LLM with provider: {current_provider}")
            
            if current_provider == "together":
                return get_together_llm(model_name)
            elif current_provider == "openai":
                return get_openai_llm(model_name)
            elif current_provider == "self":
                return get_self_hosted_llm(model_name)
            
        except Exception as e:
            error_msg = str(e)
            provider_errors[current_provider] = error_msg
            logger.warning(f"Failed to initialize {current_provider} LLM: {error_msg}")
            
            # Continue to the next provider in the fallback order
            continue
    
    # If we get here, all providers failed
    error_details = "\n".join([f"- {p}: {e}" for p, e in provider_errors.items()])
    error_message = f"All LLM providers failed. Details:\n{error_details}"
    logger.error(error_message)
    
    # As a last resort, try to create a minimal mock LLM that won't crash the application
    try:
        logger.warning("Creating emergency fallback mock LLM")
        from langchain.llms.fake import FakeListLLM
        return FakeListLLM(responses=["I apologize, but all LLM providers are currently unavailable. Please try again later."])
    except Exception as mock_error:
        # If even this fails, we have no choice but to raise an exception
        logger.critical(f"Failed to create mock LLM: {str(mock_error)}")
        raise RuntimeError(f"Unable to initialize any LLM provider. {error_message}")

async def get_llm_status():
    """
    Check the status of the LLM service.
    
    Returns:
        Dict with status information
    """
    llm_config = get_llm_config()
    provider = llm_config.provider.lower()
    
    if provider == "together":
        if not settings.TOGETHER_API_KEY:
            return {"status": "unavailable", "error": "Together API key not configured in settings (.env.local)", "provider": "together"}
        api_key = settings.TOGETHER_API_KEY
    elif provider == "openai":
        if not settings.OPENAI_API_KEY:
            return {"status": "unavailable", "error": "OpenAI API key not configured in settings (.env.local)", "provider": "openai"}
        api_key = settings.OPENAI_API_KEY
    elif provider == "self":
        # Check self-hosted LLM status
        base_url = os.getenv("SELF_HOSTED_LLM_URL", llm_config.self_hosted_url)
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                status_data = response.json()
                
                # Add memory information if available
                memory_info = {}
                if "memory_allocated_gb" in status_data:
                    memory_info["memory_allocated_gb"] = status_data["memory_allocated_gb"]
                    memory_info["memory_reserved_gb"] = status_data.get("memory_reserved_gb", "unknown")
                
                return {
                    "status": "ok" if status_data.get("status") == "ready" else "loading",
                    "model": status_data.get("model", "unknown"),
                    "provider": "self-hosted",
                    "device": status_data.get("device", "unknown"),
                    **memory_info
                }
            else:
                return {"status": "error", "error": "Self-hosted LLM service returned error", "provider": "self"}
        except Exception as e:
            logger.error(f"Error checking self-hosted LLM status: {str(e)}")
            return {"status": "unavailable", "error": f"Self-hosted LLM service unavailable: {str(e)}", "provider": "self"}
    else:
        return {"status": "unavailable", "error": f"Unsupported provider: {provider}"}
    
    try:
        # Try creating an LLM instance
        llm = get_llm()
        
        # For status check, just return the provider info without making an actual API call
        # This avoids unnecessary costs for simple status checks
        return {
            "status": "ok",
            "model": llm_config.default_model,
            "provider": provider
        }
    except Exception as e:
        logger.error(f"LLM health check failed: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "status": "error",
            "error": str(e),
            "model": llm_config.default_model,
            "provider": provider
        }

async def get_llm_metrics():
    """
    Get LLM service metrics.
    
    Returns:
        Dict[str, Any]: LLM service metrics
    """
    metrics_data = metrics.get_metrics()
    
    # Add extra provider configuration information
    metrics_data["providers"] = {
        "openai": {
            "enabled": settings.LLM_PROVIDER == "openai" or settings.LLM_PROVIDER == "all",
            "api_key_set": bool(settings.OPENAI_API_KEY),
            "default_model": settings.OPENAI_MODEL_NAME
        },
        "together": {
            "enabled": settings.LLM_PROVIDER == "together" or settings.LLM_PROVIDER == "all",
            "api_key_set": bool(settings.TOGETHER_API_KEY),
            "default_model": settings.TOGETHER_MODEL_NAME or TOGETHER_DEFAULT_MODEL
        },
        "self": {
            "enabled": settings.LLM_PROVIDER == "self" or settings.LLM_PROVIDER == "all",
            "url": settings.SELF_HOSTED_LLM_URL or SELF_HOSTED_DEFAULT_URL,
            "model_path": settings.SELF_HOSTED_MODEL_PATH,
            "active": settings.SELF_HOSTED_LLM_URL is not None
        }
    }
    
    return metrics_data

def get_llm_service_status() -> Dict[str, Any]:
    """
    Get LLM service status information without making API calls.
    
    Returns:
        Dict[str, Any]: Information about LLM service configuration
    """
    try:
        # Get the LLM config
        config = get_llm_config()
        
        # Access the provider attribute directly
        llm_provider = getattr(config, "provider", "openai").lower()
        
        # Create a status dict with all provider information
        status = {
            "provider": llm_provider,
            "openai": {
                "enabled": llm_provider == "openai" or llm_provider == "all",
                "api_key_set": bool(settings.OPENAI_API_KEY),
                "default_model": getattr(config.openai, "model_name", "gpt-3.5-turbo") if hasattr(config, "openai") else "gpt-3.5-turbo"
            },
            "together": {
                "enabled": llm_provider == "together" or llm_provider == "all",
                "api_key_set": bool(settings.TOGETHER_API_KEY),
                "default_model": getattr(config.together, "model_name", TOGETHER_DEFAULT_MODEL) if hasattr(config, "together") else TOGETHER_DEFAULT_MODEL
            },
            "self": {
                "enabled": llm_provider == "self" or llm_provider == "all",
                "url": getattr(config.self_hosted, "url", SELF_HOSTED_DEFAULT_URL) if hasattr(config, "self_hosted") else SELF_HOSTED_DEFAULT_URL,
                "model_path": getattr(config.self_hosted, "model_path", "") if hasattr(config, "self_hosted") else "",
                "active": hasattr(config, "self_hosted") and bool(getattr(config.self_hosted, "url", None))
            }
        }
        
        return status
    except Exception as e:
        # Fallback to a basic status if we can't get detailed config
        logger.error(f"Error getting LLM service status: {e}")
        return {
            "provider": "unknown",
            "error": str(e),
            "openai": {"enabled": bool(settings.OPENAI_API_KEY)},
            "together": {"enabled": bool(settings.TOGETHER_API_KEY)},
            "self": {"enabled": False}
        }

def confirm_llm_selection():
    """
    Interactive function to confirm LLM provider and model selection.
    Sets environment variables based on user selection.
    
    Returns:
        Tuple[str, str]: Selected provider and model name
    """
    # Default to TogetherAI and Llama-3.3-70B
    default_provider = settings.LLM_DEFAULT_PROVIDER or "together"
    default_model = settings.LLM_DEFAULT_MODEL or TOGETHER_DEFAULT_MODEL
    
    print("\n" + "=" * 80)
    print("SELECT LLM PROVIDER")
    print("=" * 80)
    print("1. TogetherAI (default)")
    print("2. Groq")
    print("3. OpenAI")
    print("4. Anthropic")
    print("5. Gemini")
    print("6. Self-hosted LLM")
    
    provider_choice = input("\nSelect LLM provider (1-6, default=1): ").strip()
    
    provider = default_provider
    if provider_choice == "2":
        provider = "groq"
        print("\nSelected Groq as the LLM provider")
    elif provider_choice == "3":
        provider = "openai"
        print("\nSelected OpenAI as the LLM provider")
    elif provider_choice == "4":
        provider = "anthropic"
        print("\nSelected Anthropic as the LLM provider")
    elif provider_choice == "5":
        provider = "gemini"
        print("\nSelected Gemini as the LLM provider")
    elif provider_choice == "6":
        provider = "self"
        print("\nSelected self-hosted LLM as the provider")
    else:
        print("\nSelected TogetherAI as the LLM provider")
    
    # Default model based on provider
    if provider == "together":
        model = settings.TOGETHER_MODEL or TOGETHER_DEFAULT_MODEL
        print(f"\nDefault model: {model}")
        custom_model = input("Enter a different TogetherAI model name or press Enter to use default: ").strip()
        if custom_model:
            model = custom_model
    elif provider == "groq":
        model = settings.GROQ_MODEL or GROQ_DEFAULT_MODEL
        print(f"\nDefault model: {model}")
        custom_model = input("Enter a different Groq model name or press Enter to use default: ").strip()
        if custom_model:
            model = custom_model
    elif provider == "openai":
        model = settings.OPENAI_MODEL or OPENAI_DEFAULT_MODEL
        print(f"\nDefault model: {model}")
        print(f"Available OpenAI models: gpt-3.5-turbo, gpt-4-turbo, {OPENAI_DEFAULT_MODEL}")
        custom_model = input("Enter a different OpenAI model name or press Enter to use default: ").strip()
        if custom_model:
            model = custom_model
    elif provider == "anthropic":
        model = settings.ANTHROPIC_MODEL or ANTHROPIC_DEFAULT_MODEL
        print(f"\nDefault model: {model}")
        custom_model = input("Enter a different Anthropic model name or press Enter to use default: ").strip()
        if custom_model:
            model = custom_model
    elif provider == "gemini":
        model = settings.GEMINI_MODEL or GEMINI_DEFAULT_MODEL
        print(f"\nDefault model: {model}")
        custom_model = input("Enter a different Gemini model name or press Enter to use default: ").strip()
        if custom_model:
            model = custom_model
    elif provider == "self":
        model = settings.SELF_HOSTED_MODEL if hasattr(settings, "SELF_HOSTED_MODEL") else "local-model"
        print(f"\nUsing self-hosted model: {model}")
    
    # Confirm the selection
    print("\n" + "=" * 80)
    print(f"Selected Configuration:")
    print(f"Provider: {provider}")
    print(f"Model: {model}")
    print("=" * 80)
    
    confirmation = input("\nConfirm this selection? (Y/n): ")
    if confirmation.lower() not in ["y", "yes", ""]:
        print("Selection not confirmed. Please try again.")
        return confirm_llm_selection()
    
    # Set environment variables to control provider and model
    os.environ["CONFIRM_LLM_SELECTION"] = "True"
    os.environ["CONFIRMED_LLM_PROVIDER"] = provider
    os.environ["CONFIRMED_LLM_MODEL"] = model
    
    # Set provider-specific environment variables
    if provider == "together":
        os.environ["TOGETHER_MODEL"] = model
    elif provider == "groq":
        os.environ["GROQ_MODEL"] = model
    elif provider == "openai":
        os.environ["OPENAI_MODEL"] = model
    elif provider == "anthropic":
        os.environ["ANTHROPIC_MODEL"] = model
    elif provider == "gemini":
        os.environ["GEMINI_MODEL"] = model
    elif provider == "self":
        os.environ["SELF_HOSTED_MODEL"] = model
    
    # Also set the general LLM defaults
    os.environ["WEBAGENT_LLM_DEFAULT_PROVIDER"] = provider
    os.environ["WEBAGENT_LLM_DEFAULT_MODEL"] = model
    
    print(f"\nSelection confirmed. Using {provider} with model {model}")
    return provider, model 