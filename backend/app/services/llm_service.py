"""LLM Service Module

This module provides LLM services for the application.
"""

import time
import threading
from typing import Dict, List, Any, Optional, Union

from app.core.logger import logger
from app.core.observability import (
    record_model_latency,
    record_model_token_usage,
    record_model_error_rate,
    log_model_prediction,
    log_to_mlflow
)
from app.core.drift_detection import (
    create_numerical_drift_detector,
    create_embedding_drift_detector,
    create_performance_drift_detector
)

# Initialize drift detectors
latency_drift_detector = create_numerical_drift_detector("llm_latency", window_size=1000, threshold=0.05)
token_usage_drift_detector = create_numerical_drift_detector("llm_token_usage", window_size=1000, threshold=0.05)
embedding_drift_detector = create_embedding_drift_detector("llm_embeddings", window_size=500, threshold=0.3)
performance_drift_detector = create_performance_drift_detector("llm_performance", window_size=200, threshold=0.15)

# Lock for thread safety
_lock = threading.RLock()

class LLMService:
    """Service for interacting with LLMs"""
    
    def __init__(self, model_name: str = "gpt-4"):
        """
        Initialize the LLM service.
        
        Args:
            model_name: Name of the LLM model to use
        """
        self.model_name = model_name
        logger.info(f"Initialized LLM service with model: {model_name}")
        
        # Track if we're in baseline collection mode
        self.collecting_baseline = True
        self.baseline_count = 0
        self.max_baseline_samples = 100
    
    def generate_response(self, prompt: str, max_tokens: int = 1000) -> Dict[str, Any]:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Dictionary with the generated response and metadata
        """
        start_time = time.time()
        
        try:
            # Here would be the actual LLM API call
            # For now we'll simulate it
            time.sleep(0.1)  # Simulated latency
            response = f"This is a simulated response to: {prompt[:20]}..."
            
            # Simulation of token usage
            prompt_tokens = len(prompt.split()) 
            completion_tokens = len(response.split())
            total_tokens = prompt_tokens + completion_tokens
            
            # Calculate latency
            latency = time.time() - start_time
            
            # Log metrics
            with _lock:
                record_model_latency(self.model_name, latency)
                record_model_token_usage(self.model_name, prompt_tokens, completion_tokens)
                log_model_prediction(self.model_name, success=True)
                
                # Track drift for latency and token usage
                if self.collecting_baseline and self.baseline_count < self.max_baseline_samples:
                    latency_drift_detector.add_reference_sample(latency)
                    token_usage_drift_detector.add_reference_sample(total_tokens)
                    self.baseline_count += 1
                    
                    if self.baseline_count >= self.max_baseline_samples:
                        self.collecting_baseline = False
                        logger.info(f"Completed baseline collection for LLM drift detection with {self.baseline_count} samples")
                else:
                    # Add current samples and check for drift
                    latency_score, latency_drift = latency_drift_detector.add_current_sample(latency)
                    token_score, token_drift = token_usage_drift_detector.add_current_sample(total_tokens)
                    
                    # Add performance metrics
                    perf_score, perf_drift = performance_drift_detector.add_performance_metrics(
                        accuracy=0.95,  # simulated accuracy
                        latency=latency,
                        confidence=0.85  # simulated confidence
                    )
                    
                    # Log any detected drift
                    if latency_drift or token_drift or perf_drift:
                        logger.warning(f"Drift detected in LLM service - Latency: {latency_drift}, Token: {token_drift}, Performance: {perf_drift}")
                        
                        # Check if retraining is recommended
                        retrain_recommended = performance_drift_detector.should_retrain()
                        if retrain_recommended:
                            logger.warning(f"Model retraining recommended for {self.model_name} due to performance drift")
            
            return {
                "response": response,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "latency": latency
            }
            
        except Exception as e:
            # Record error
            latency = time.time() - start_time
            record_model_error_rate(self.model_name, 1)
            record_model_latency(self.model_name, latency)
            log_model_prediction(self.model_name, success=False)
            
            logger.error(f"Error in LLM service: {str(e)}")
            
            return {
                "error": str(e),
                "latency": latency
            }
    
    def get_embedding(self, text: str) -> Dict[str, Any]:
        """
        Get an embedding for the given text.
        
        Args:
            text: The text to generate an embedding for
            
        Returns:
            Dictionary with the embedding and metadata
        """
        start_time = time.time()
        
        try:
            # Here would be the actual embedding API call
            # For now we'll simulate it
            time.sleep(0.05)  # Simulated latency
            
            # Simulate an embedding vector (simplified for demonstration)
            import numpy as np
            embedding = np.random.normal(0, 1, 1536).astype(np.float32)
            
            # Calculate latency
            latency = time.time() - start_time
            
            # Log metrics
            record_model_latency(f"{self.model_name}_embedding", latency)
            log_model_prediction(f"{self.model_name}_embedding", success=True)
            
            # Track drift for embeddings
            with _lock:
                if self.collecting_baseline and self.baseline_count < self.max_baseline_samples:
                    embedding_drift_detector.add_reference_sample(embedding)
                else:
                    # Add current sample and check for drift
                    drift_score, drift_detected = embedding_drift_detector.add_current_sample(embedding)
                    
                    if drift_detected:
                        logger.warning(f"Embedding drift detected with score {drift_score:.4f}")
            
            return {
                "embedding": embedding.tolist(),
                "dimensions": embedding.shape[0],
                "latency": latency
            }
            
        except Exception as e:
            # Record error
            latency = time.time() - start_time
            record_model_error_rate(f"{self.model_name}_embedding", 1)
            record_model_latency(f"{self.model_name}_embedding", latency)
            log_model_prediction(f"{self.model_name}_embedding", success=False)
            
            logger.error(f"Error in embedding service: {str(e)}")
            
            return {
                "error": str(e),
                "latency": latency
            }
    
    def get_drift_status(self) -> Dict[str, Any]:
        """
        Get the current drift status for the LLM service.
        
        Returns:
            Dictionary with drift statistics
        """
        latency_stats = latency_drift_detector.get_stats() if latency_drift_detector else {}
        token_stats = token_usage_drift_detector.get_stats() if token_usage_drift_detector else {}
        embedding_stats = embedding_drift_detector.get_stats() if embedding_drift_detector else {}
        performance_stats = performance_drift_detector.get_stats() if performance_drift_detector else {}
        
        return {
            "model_name": self.model_name,
            "collecting_baseline": self.collecting_baseline,
            "baseline_count": self.baseline_count,
            "latency_drift": latency_stats,
            "token_usage_drift": token_stats,
            "embedding_drift": embedding_stats,
            "performance_drift": performance_stats,
            "retrain_recommended": performance_drift_detector.should_retrain() if performance_drift_detector else False
        }
    
    def reset_drift_baselines(self):
        """Reset all drift detection baselines."""
        with _lock:
            self.collecting_baseline = True
            self.baseline_count = 0
            
            # Create new detectors
            global latency_drift_detector, token_usage_drift_detector, embedding_drift_detector, performance_drift_detector
            latency_drift_detector = create_numerical_drift_detector("llm_latency", window_size=1000, threshold=0.05)
            token_usage_drift_detector = create_numerical_drift_detector("llm_token_usage", window_size=1000, threshold=0.05)
            embedding_drift_detector = create_embedding_drift_detector("llm_embeddings", window_size=500, threshold=0.3)
            performance_drift_detector = create_performance_drift_detector("llm_performance", window_size=200, threshold=0.15)
            
            logger.info(f"Reset drift baselines for LLM service {self.model_name}")

# Global instance
llm_service = LLMService() 