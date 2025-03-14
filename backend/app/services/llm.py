"""
LLM service for the WebAgent backend.
Provides access to different LLM providers including OpenAI, Together AI, and self-hosted models.
"""
from typing import Dict, Optional, Any, List, Tuple
import logging
import os
import time
import traceback
from functools import wraps
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log

from langchain_openai import ChatOpenAI
from langchain_together import Together as TogetherLLM
from langchain.chat_models.base import BaseChatModel
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from app.core.config import settings
from app.core.loadEnvYAML import get_llm_config

logger = logging.getLogger(__name__)

# Define model constants
TOGETHER_DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
SELF_HOSTED_DEFAULT_URL = "http://localhost:8080"

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
        else:
            self.failed_requests += 1
            
            if provider in self.provider_stats:
                self.provider_stats[provider]["errors"] += 1
    
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

def log_llm_request(provider: str = "unknown"):
    """Decorator to log LLM requests and record metrics."""
    def decorator(func):
        @wraps(func)
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
                return result
            except Exception as e:
                elapsed_time = time.time() - start_time
                logger.error(f"LLM request to {provider} failed after {elapsed_time:.2f}s: {str(e)}")
                metrics.record_request(provider, False, 0, elapsed_time)
                raise
        
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
        # Convert LangChain messages to API format
        api_messages = [self._convert_message_to_dict(message) for message in messages]
        
        # Prepare the request payload
        payload = {
            "messages": api_messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "timeout": self.timeout,
            **kwargs
        }
        
        start_time = time.time()
        request_id = hex(int(start_time * 10000000))[2:]
        
        logger.info(f"Sending request to self-hosted LLM [id={request_id}]")
        
        # Make the API request
        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=self.timeout + 5  # Add a buffer to the timeout
            )
            
            # Log the response status
            elapsed_time = time.time() - start_time
            logger.info(f"Received response from self-hosted LLM [id={request_id}] in {elapsed_time:.2f}s (status={response.status_code})")
            
            # Raise exception for 4xx/5xx status codes
            response.raise_for_status()
            
            # Parse the response
            data = response.json()
            
            # Extract the assistant's message
            if "choices" in data and len(data["choices"]) > 0:
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                
                # Log token usage
                if "total_tokens" in usage:
                    logger.info(f"Self-hosted LLM tokens used [id={request_id}]: {usage['total_tokens']}")
                
                # Return the generated message
                return {
                    "generations": [{
                        "text": content,
                        "message": AIMessage(content=content)
                    }],
                    "llm_output": {
                        "token_usage": usage,
                        "model_name": self.model,
                        "request_id": request_id,
                        "latency_seconds": elapsed_time
                    }
                }
            else:
                raise ValueError(f"Unexpected response format from self-hosted LLM [id={request_id}]")
                
        except requests.exceptions.Timeout:
            logger.error(f"Timeout error connecting to self-hosted LLM [id={request_id}] after {self.timeout}s")
            raise ValueError(f"Self-hosted LLM request timed out after {self.timeout} seconds")
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error connecting to self-hosted LLM [id={request_id}]: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error generating text with self-hosted LLM [id={request_id}]: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    @property
    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return "self-hosted"

@log_llm_request(provider="openai")
def get_openai_llm(model_name: Optional[str] = None):
    """
    Get a LangChain ChatOpenAI instance.
    
    Args:
        model_name: Optional model name to use (defaults to settings.DEFAULT_MODEL)
        
    Returns:
        A configured ChatOpenAI instance
    """
    model = model_name or settings.DEFAULT_MODEL
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
    Get a LangChain TogetherLLM instance.
    
    Args:
        model_name: Optional model name to use (defaults to TOGETHER_DEFAULT_MODEL)
        
    Returns:
        A configured TogetherLLM instance
    """
    model = model_name or TOGETHER_DEFAULT_MODEL
    llm_config = get_llm_config()
    
    # Check for Together API key in environment variables
    together_api_key = os.getenv("TOGETHER_API_KEY")
    
    if not together_api_key:
        logger.error("TOGETHER_API_KEY environment variable not set")
        raise ValueError("TOGETHER_API_KEY not set in environment variables")
    
    try:
        # Create TogetherLLM without the parameters that cause validation errors
        llm = TogetherLLM(
            model=model,
            temperature=llm_config.temperature,
            max_tokens=llm_config.max_tokens,
            together_api_key=together_api_key,
            # Note: timeout and max_retries are no longer supported by TogetherLLM
            # and have to be handled differently in newer versions of langchain-together
        )
        
        # Log successful creation
        logger.info(f"Created TogetherLLM with model: {model}")
        return llm
    except Exception as e:
        logger.error(f"Error creating Together AI LLM: {str(e)}")
        logger.error(traceback.format_exc())
        raise

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
    
    logger.info(f"Creating LLM instance with provider: {provider}")
    
    try:
        if provider == "together":
            return get_together_llm(model_name)
        elif provider == "openai":
            return get_openai_llm(model_name)
        elif provider == "self":
            return get_self_hosted_llm(model_name)
        else:
            logger.error(f"Unsupported LLM provider: {provider}")
            raise ValueError(f"Unsupported LLM provider: {provider}")
    except ValueError as e:
        # Handle fallback logic
        if provider == "together" and "TOGETHER_API_KEY not set" in str(e):
            logger.warning("Together AI not available, falling back to OpenAI")
            return get_openai_llm(model_name)
        elif provider == "self" and "Self-hosted LLM service unavailable" in str(e):
            logger.warning("Self-hosted LLM not available, falling back to Together AI or OpenAI")
            # Try Together AI first, then fall back to OpenAI
            try:
                return get_together_llm(model_name)
            except ValueError:
                logger.warning("Together AI not available, falling back to OpenAI")
                return get_openai_llm(model_name)
        else:
            raise

async def get_llm_status():
    """
    Check the status of the LLM service.
    
    Returns:
        Dict with status information
    """
    llm_config = get_llm_config()
    provider = llm_config.provider.lower()
    
    if provider == "together":
        if not os.getenv("TOGETHER_API_KEY"):
            return {"status": "unavailable", "error": "Together API key not configured", "provider": "together"}
        api_key = os.getenv("TOGETHER_API_KEY")
    elif provider == "openai":
        if not settings.OPENAI_API_KEY:
            return {"status": "unavailable", "error": "OpenAI API key not configured", "provider": "openai"}
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
    Get metrics for the LLM service.
    
    Returns:
        Dict with metrics information
    """
    return {
        "llm_metrics": metrics.get_metrics(),
        "current_provider": get_llm_config().provider.lower()
    } 