#!/usr/bin/env python
"""
Core Transformer Model Service - Service implementation for the custom Transformer model
"""
import logging
import os
import time
import torch
import json
from typing import Dict, List, Optional, Any
import asyncio

from .base_model_service import BaseModelService, ModelServiceConfig, CompletionResponse
from ..core_transformer import Transformer

logger = logging.getLogger(__name__)

class CoreTransformerModelService(BaseModelService):
    """Service implementation for the custom core Transformer model"""
    
    async def load_model(self) -> None:
        """Load the Transformer model into memory"""
        logger.info(f"Loading Core Transformer model from {self.config.model_path}")
        start_time = time.time()
        
        try:
            # Load model parameters from config
            params = self.config.parameters
            
            # Set device (CPU or GPU)
            self.device = torch.device(self.config.device)
            
            # Get model parameters from config
            src_vocab_size = params.get("src_vocab_size", 5000)
            tgt_vocab_size = params.get("tgt_vocab_size", 5000)
            d_model = params.get("d_model", 512)
            num_heads = params.get("num_heads", 8)
            num_encoder_layers = params.get("num_encoder_layers", 6)
            num_decoder_layers = params.get("num_decoder_layers", 6)
            d_ff = params.get("d_ff", 2048)
            dropout = params.get("dropout", 0.1)
            max_seq_length = params.get("max_seq_length", 5000)
            
            # Initialize the model
            self.model = Transformer(
                src_vocab_size=src_vocab_size,
                tgt_vocab_size=tgt_vocab_size,
                d_model=d_model,
                num_heads=num_heads,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                d_ff=d_ff,
                max_seq_length=max_seq_length,
                dropout=dropout
            )
            
            # Load saved model if a checkpoint exists
            model_path = os.path.join(self.config.model_path, "model.pt")
            if os.path.exists(model_path):
                logger.info(f"Loading model weights from {model_path}")
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            else:
                logger.warning(f"No model checkpoint found at {model_path}, using initialized model")
            
            # Move model to appropriate device
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            # Load tokenizer info if available
            tokenizer_path = os.path.join(self.config.model_path, "tokenizer.json")
            if os.path.exists(tokenizer_path):
                with open(tokenizer_path, 'r') as f:
                    self.tokenizer_info = json.load(f)
                logger.info(f"Loaded tokenizer information from {tokenizer_path}")
            else:
                self.tokenizer_info = {
                    "src_vocab": {},
                    "tgt_vocab": {},
                    "src_idx_to_token": {},
                    "tgt_idx_to_token": {}
                }
                logger.warning(f"No tokenizer found at {tokenizer_path}, using empty tokenizer")
            
            logger.info(f"Model loaded successfully in {time.time() - start_time:.2f}s")
            
        except Exception as e:
            logger.exception(f"Failed to load model: {e}")
            raise
    
    def _create_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """Create source mask for padding tokens"""
        # In a real implementation, this would mask padding tokens
        batch_size = src.shape[0]
        src_len = src.shape[1]
        src_mask = torch.ones((batch_size, 1, 1, src_len), device=self.device)
        return src_mask
    
    def _create_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """Create target mask for padding tokens and future tokens"""
        # Create mask for padding tokens
        batch_size = tgt.shape[0]
        tgt_len = tgt.shape[1]
        
        # Create attention mask to prevent attending to future tokens
        tgt_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=self.device))
        tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(1).expand(batch_size, 1, tgt_len, tgt_len)
        return tgt_mask
    
    def _tokens_to_text(self, tokens: List[int]) -> str:
        """Convert token IDs to text using the target vocabulary"""
        tokens = [t for t in tokens if t > 0]  # Filter out padding tokens
        text = " ".join([self.tokenizer_info["tgt_idx_to_token"].get(str(t), "[UNK]") for t in tokens])
        return text
    
    def _text_to_tokens(self, text: str) -> List[int]:
        """Convert text to token IDs using the source vocabulary"""
        # Simple tokenization by splitting on spaces (in a real implementation, this would be more sophisticated)
        words = text.split()
        tokens = [self.tokenizer_info["src_vocab"].get(word, 1) for word in words]  # 1 is usually the unknown token
        return tokens
    
    def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text"""
        # Simple implementation - in a real system this would use the actual tokenizer
        return len(text.split())
    
    async def generate(
        self, 
        prompt: str, 
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None
    ) -> CompletionResponse:
        """Generate text completion using the Transformer model"""
        start_time = time.time()
        
        try:
            # Tokenize input text
            input_tokens = self._text_to_tokens(prompt)
            
            # Prepare input tensor
            src = torch.tensor([input_tokens], dtype=torch.long, device=self.device)
            
            # Initialize target with start token (usually token ID 2)
            tgt = torch.tensor([[2]], dtype=torch.long, device=self.device)
            
            # Track the generated tokens
            generated_tokens = []
            
            # Create source mask
            src_mask = self._create_src_mask(src)
            
            # Encode the source sequence first (this can be done once)
            with torch.no_grad():
                memory = self.model.encode(src, src_mask)
            
            # Generate tokens one by one
            for i in range(max_tokens):
                # Create target mask for the current sequence
                tgt_mask = self._create_tgt_mask(tgt)
                
                # Generate next token probabilities
                with torch.no_grad():
                    output = self.model.decode(tgt, memory, src_mask, tgt_mask)
                    output = self.model.final_layer(output)
                    
                # Get the last token's prediction
                next_token_logits = output[:, -1, :].squeeze(0)
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Apply top-p sampling (nucleus sampling)
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                next_token_probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(next_token_probs, 1).item()
                
                # Add token to output
                generated_tokens.append(next_token)
                
                # Append token to target sequence
                next_token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=self.device)
                tgt = torch.cat([tgt, next_token_tensor], dim=1)
                
                # Check for stop sequences
                if stop:
                    generated_text = self._tokens_to_text(generated_tokens)
                    if any(stop_seq in generated_text for stop_seq in stop):
                        break
                
                # Check for end-of-sequence token (usually token ID 3)
                if next_token == 3:
                    break
            
            # Convert tokens to text
            generated_text = self._tokens_to_text(generated_tokens)
            
            # Count tokens
            input_token_count = self._count_tokens(prompt)
            output_token_count = self._count_tokens(generated_text)
            
            # Create usage info
            usage = {
                "prompt_tokens": input_token_count,
                "completion_tokens": output_token_count,
                "total_tokens": input_token_count + output_token_count
            }
            
            # Determine finish reason
            finish_reason = "stop"
            if len(generated_tokens) >= max_tokens:
                finish_reason = "length"
            
            # Log generation time
            logger.info(f"Generated {output_token_count} tokens in {time.time() - start_time:.2f}s")
            
            # Return the response
            return CompletionResponse(
                text=generated_text,
                model=self.config.model_id,
                usage=usage,
                finish_reason=finish_reason
            )
            
        except Exception as e:
            logger.exception(f"Error during generation: {e}")
            raise
    
    def cleanup(self) -> None:
        """Clean up resources before shutdown"""
        if hasattr(self, 'model') and self.model is not None:
            # Free up GPU memory if using CUDA
            if self.device.type == 'cuda':
                logger.info("Cleaning up CUDA memory")
                torch.cuda.empty_cache() 