#!/usr/bin/env python
"""
Transformer Model Service - Implementation for transformer-based language models

This module implements a model service for transformer-based language models
using the Hugging Face Transformers library.
"""
import logging
import time
import os
from typing import Dict, List, Optional, Union

import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    StoppingCriteria, 
    StoppingCriteriaList,
    TextStreamer
)

from .base_model_service import (
    BaseModelService, 
    ModelServiceConfig, 
    CompletionResponse
)

logger = logging.getLogger(__name__)

class StopOnTokens(StoppingCriteria):
    """Stopping criteria that stops generation when specific token IDs are generated."""
    
    def __init__(self, stop_token_ids: List[int]):
        self.stop_token_ids = stop_token_ids
    
    def __call__(self, input_ids, scores, **kwargs) -> bool:
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

class TransformerModelService(BaseModelService):
    """Service for transformer-based language models"""
    
    async def load_model(self) -> None:
        """Load the model and tokenizer"""
        try:
            start_time = time.time()
            logger.info(f"Loading model from {self.config.model_path}")
            
            # Check if the model path exists
            if not os.path.exists(self.config.model_path):
                raise FileNotFoundError(f"Model path not found: {self.config.model_path}")
            
            # Determine device
            device_map = "auto"
            if self.config.device == "cpu":
                device_map = "cpu"
            
            # Check if we should use quantization
            quantization = self.config.parameters.get("quantization", None)
            load_kwargs = {}
            
            if quantization == "4bit":
                logger.info("Using 4-bit quantization")
                load_kwargs.update({
                    "load_in_4bit": True,
                    "bnb_4bit_compute_dtype": torch.float16,
                })
            elif quantization == "8bit":
                logger.info("Using 8-bit quantization")
                load_kwargs.update({"load_in_8bit": True})
                
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                device_map=device_map,
                trust_remote_code=True,
                **load_kwargs
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path,
                trust_remote_code=True
            )
            
            # Configure tokenizer padding
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = 'left'
            
            logger.info(f"Model and tokenizer loaded in {time.time() - start_time:.2f} seconds")
            
            # Get model info
            self.model_info = {
                "model_type": type(self.model).__name__,
                "tokenizer_type": type(self.tokenizer).__name__,
                "parameters": self.config.parameters
            }
            
            # Log memory usage for CUDA
            if torch.cuda.is_available() and self.config.device != "cpu":
                logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                logger.info(f"CUDA memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
                
        except Exception as e:
            logger.exception(f"Error loading model: {e}")
            raise
    
    async def generate(
        self, 
        prompt: str, 
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None
    ) -> CompletionResponse:
        """
        Generate text completion
        
        Args:
            prompt: The input prompt
            max_tokens: Maximum number of tokens to generate
            temperature: The randomness of the generation
            top_p: Nucleus sampling parameter
            stop: Stop sequences to end generation
            
        Returns:
            CompletionResponse: The generated completion
        """
        try:
            # Prepare stopping criteria
            stopping_criteria = None
            if stop:
                stop_token_ids = [
                    self.tokenizer.encode(stop_seq, add_special_tokens=False)[-1]
                    for stop_seq in stop
                ]
                stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids)])
            
            # Convert inputs to token IDs
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
            
            # Track token usage
            input_token_count = len(input_ids[0])
            
            # Move input to the same device as the model
            device = next(self.model.parameters()).device
            input_ids = input_ids.to(device)
            
            # Set generation parameters
            gen_kwargs = {
                "input_ids": input_ids,
                "max_new_tokens": max_tokens,
                "temperature": max(0.1, temperature),  # Prevent 0 temperature
                "top_p": top_p,
                "do_sample": temperature > 0.1,
                "pad_token_id": self.tokenizer.pad_token_id,
            }
            
            if stopping_criteria:
                gen_kwargs["stopping_criteria"] = stopping_criteria
            
            # Generate text
            start_time = time.time()
            with torch.no_grad():
                output = self.model.generate(**gen_kwargs)
            generation_time = time.time() - start_time
            
            # Get only the newly generated tokens
            generated_tokens = output[0][input_ids.shape[1]:]
            output_token_count = len(generated_tokens)
            
            # Convert back to text
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Determine finish reason
            finish_reason = "stop"
            if output_token_count >= max_tokens:
                finish_reason = "length"
            
            # Log performance metrics
            logger.info(f"Generated {output_token_count} tokens in {generation_time:.2f}s "
                       f"({output_token_count / generation_time:.2f} tokens/sec)")
            
            # Create response
            return CompletionResponse(
                text=generated_text,
                model=self.config.model_id,
                usage={
                    "prompt_tokens": input_token_count,
                    "completion_tokens": output_token_count,
                    "total_tokens": input_token_count + output_token_count
                },
                finish_reason=finish_reason
            )
            
        except Exception as e:
            logger.exception(f"Error during generation: {e}")
            raise
    
    def cleanup(self) -> None:
        """Clean up resources before shutdown"""
        try:
            # Free GPU memory if using CUDA
            if hasattr(self, "model") and self.model is not None:
                del self.model
                torch.cuda.empty_cache()
                logger.info("Cleaned up model resources and freed GPU memory")
        except Exception as e:
            logger.exception(f"Error during cleanup: {e}")

# Add the model service class to the services module
__all__ = ["TransformerModelService"] 