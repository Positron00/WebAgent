import logging
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import copy
import numpy as np
import random
import tqdm
from typing import Dict, Any, List, Tuple
from collections import defaultdict

from .base_model_service import BaseModelService, ModelServiceConfig
from fastapi import HTTPException

logger = logging.getLogger(__name__)

# determine device to train on
def get_device():
    device_name = os.environ.get("DEVICE", "cpu")
    
    if device_name == "gpu" and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif device_name == "gpu" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    logger.info(f"Fine-tuning using device: {device}")
    return device

DEVICE = get_device()


class LoRAConv1DWrapper(nn.Module):
    # set up LoRA-augmented layer
    def __init__(self, conv1dmodule: nn.Module, lora_rank: int):
        super().__init__()

        self.base_module = conv1dmodule
        
        for param in self.base_module.parameters():
            param.requires_grad = False

        self.lora_A = nn.Parameter(torch.empty(self.base_module.weight.shape[0], lora_rank))
        nn.init.kaiming_uniform_(self.lora_A)
        self.lora_B = nn.Parameter(torch.zeros(self.base_module.weight.shape[1], lora_rank))
        self.lora_A.requires_grad = True
        self.lora_B.requires_grad = True

    def forward(self, x):
        base_output = self.base_module(x)
        lora_output = torch.matmul(torch.matmul(x, self.lora_A), self.lora_B.T)
        return base_output + lora_output


def parameters_to_fine_tune(model, mode: str):
    """
    Select the parameters in `model` that should be fine-tuned in mode `mode`.

    Args:
      model: the model we're fine-tuning
      mode: the fine-tuning mode we're using; may be 'all', 'last', 'first',
        'middle', or 'loraN' (where N is an integer)
    
    Returns:
      A list of nn.Parameters of `model` that should be fine-tuned in the given
        fine-tuning mode.
    """
    if mode == "all":
        return model.parameters()
    
    elif mode == "last":
        for param in model.parameters():
            param.requires_grad = False
        # Assuming the model has a 'transformer' attribute with blocks 'h'
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            for param in model.transformer.h[-2:].parameters():
                param.requires_grad = True
        else:
            raise HTTPException(status_code=500, detail="Model does not support fine-tuning mode 'last'")
        return filter(lambda p: p.requires_grad, model.parameters())
        
    elif mode == "first":
        for param in model.parameters():
            param.requires_grad = False

        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            for param in model.transformer.h[:2].parameters():
                param.requires_grad = True
        else:
            raise HTTPException(status_code=500, detail="Model does not support fine-tuning mode 'first'")
        
        return filter(lambda p: p.requires_grad, model.parameters())
    
    elif mode == "middle":
        for param in model.parameters():
            param.requires_grad = False

        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            transformer_blocks = model.transformer.h
            mid_index = len(transformer_blocks) // 2
            
            parameters = list(transformer_blocks[mid_index - 1].parameters()) + list(transformer_blocks[mid_index].parameters())

            for param in parameters:
                param.requires_grad = True
        else:
            raise HTTPException(status_code=500, detail="Model does not support fine-tuning mode 'middle'")

        logger.info(f"Number of parameters with gradients enabled: {sum(p.requires_grad for p in model.parameters())}")
        return filter(lambda p: p.requires_grad, model.parameters())

    elif mode.startswith('lora'):
        for param in model.parameters():
            param.requires_grad = False

        for m in model.modules():
            if isinstance(m, LoRAConv1DWrapper):
                for name, param in m.named_parameters():
                    if 'lora_A' in name or 'lora_B' in name:
                        param.requires_grad = True

        return filter(lambda p: p.requires_grad, model.parameters())

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported fine-tuning mode: {mode}")


def get_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Computes the cross-entropy loss for either sequence classification or generation.

    For generation, note different sequences within the batch have different lengths, 
      and the targets tensor includes some mask values (-100). The average loss is the 
      *average loss over all non-masked timesteps*. The prediction for what token t will 
      be is made after seeing only t - 1 tokens; that is, there is an off-by-one shift 
      needed between the logits and targets.

    Args:
      logits: a 2D [batch_size, n_classes] (for classification) or 3D
        [batch_size, sequence_length, vocab_size] (for generation) tensor
        of *UNNORMALIZED* logits
      targets: a 1D [batch_size] (for classification) or 2D [batch_size, sequence_length]
        (for generation) tensor of target indices. For the generation case, may contain
        -100 in some positions, meaning that the loss for this timestep should be ignored.
    
    Returns:
      A zero-dim tensor representing the average cross-entropy loss over all batch 
        elements (and sequence timesteps, if applicable)
    """
    if logits.dim() == 2:
        loss = F.cross_entropy(logits, targets)
    elif logits.dim() == 3:
        logits = logits[:, :-1, :].contiguous()
        targets = targets[:, 1:].contiguous()

        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1)

        loss = F.cross_entropy(logits, targets, ignore_index=-100, reduction='mean')
    else:
        raise ValueError(f'Logits should either be 2-dim (for classification) or 3-dim (for generation); got {logits.dim()}')
    
    return loss


def get_acc(logits, targets):
    """
    Computes the exact match accuracy for either sequence classification or generation. i.e.,
      the fraction of predictions for which the most likely class/token equals the target.

    For generation, need to deal with the fact that different sequences within
      the batch have different lengths, and the targets tensor includes some mask
      values (-100). The average accuracy is the *average accuracy over all non-masked batch 
        elements (and sequence timesteps, if applicable)
      Also need to handle the fact that the prediction for what token t will be is
      made after seeing only t - 1 tokens; that is, there is an off-by-one shift needed
      between the logits and targets.

    Args:
      logits: a 2D [batch_size, n_classes] (for classification) or 3D
        [batch_size, sequence_length, vocab_size] (for generation) tensor of logits
      targets: a 1D [batch_size] (for classification) or 2D [batch_size, sequence_length]
        (for generation) tensor of target indices. For the generation case, may contain
        -100 in some positions, meaning that the loss for this timestep should be ignored.
    
    Returns:
      A *scalar* representing the average exact-match accuracy over all non-masked batch 
        elements (and sequence timesteps, if applicable)
    """
    if logits.dim() == 2:
        y = torch.argmax(logits, dim=-1) == targets
        y = y.type(torch.float)
        return torch.mean(y).item()
    
    elif logits.dim() == 3:
        logits = logits[:, :-1, :].contiguous()
        targets = targets[:, 1:].contiguous()

        y = torch.argmax(logits, dim=-1)
        mask = (targets != -100).float()
        correct = (y == targets).float() * mask

        return (correct.sum() / mask.sum()).item()
    
    else:
        raise ValueError(f'Logits should either be 2-dim (for classification) or 3-dim (for generation); got {logits.dim()}')


def tokenize_gpt2_batch(tokenizer, x, y):
    """
    Implement the tokenization step for a batch of examples for GPT-2.

    Args:
        tokenizer: a GPT2Tokenizer that one can call and receive a dictionary of:
          - input_ids: a list (or tensor) of token ids
          - attention_mask: a list (or tensor) of 1s and 0s indicating which tokens
              are padding (if you requested padding and tensors from the tokenizer)
        x: a list of strings, each of which is the input for a single example
        y: a list of strings, each of which is a *target* for a single example
    
    Returns:
        A dictionary with the following keys:
            - input_ids: a tensor of shape [batch_size, sequence_length] 
                containing the token ids
            - attention_mask: a tensor of shape [batch_size, sequence_length] 
                containing 1s and 0s indicating which tokens are padding
            - labels: a tensor of shape [batch_size, sequence_length] containing
                the target token ids, with -100 for non-target tokens (i.e., the
                tokens in the input part of each example or padding tokens)
    """
    combined_sequences = [x_ + y_ for x_, y_ in zip(x, y)]
    tokenized_sequences = tokenizer(combined_sequences, return_tensors="pt", padding=True, truncation=True)

    labels = torch.full_like(tokenized_sequences['input_ids'], -100)
    tokenized_targets = tokenizer(y, return_tensors="pt", padding=True, truncation=True)
    tokenized_inputs = tokenizer(x, return_tensors="pt", padding=True, truncation=True)

    for i in range(len(labels)):
        inputTokens = tokenized_inputs['input_ids'][i]
        targetTokens = tokenized_targets['input_ids'][i]
        inputMask = tokenized_inputs['attention_mask'][i]
        targetMask = tokenized_targets['attention_mask'][i]

        lenInput = sum(inputMask)
        lenTarget = sum(targetMask)

        for j in range(lenInput, (lenInput+lenTarget)):
            if j < labels.size(1) and (j-lenInput) < targetTokens.size(0) and targetMask[j-lenInput] == 1:
                labels[i, j] = targetTokens[j-lenInput]

    tokenized_sequences['labels'] = labels.contiguous()
    
    return tokenized_sequences.to(DEVICE)


def add_prefixes(x: List[str], y: List[str], dataset: str) -> Tuple[List[str], List[str]]:
    """Add appropriate prefixes to inputs and outputs based on dataset type"""
    is_qa_dataset = dataset.lower() in ['squad', 'nq', 'triviaqa']
    
    input_prefix = '' if is_qa_dataset else ''
    label_prefix = ' In the' if is_qa_dataset else ' TL;DR:'
    label_suffix = '.' if is_qa_dataset else ''

    x = [input_prefix + x_.replace('\n', ' ') + label_prefix for x_ in x]
    y = [' ' + y_.replace('\n', ' ') + label_suffix for y_ in y]

    return x, y


class FineTuneService(BaseModelService):
    async def load_model(self) -> None:
        # Use the same loading mechanism as the base service
        await super().load_model()

    async def fine_tune(self, dataset: str, mode: str, epochs: int = 1, learning_rate: float = 1e-4, 
                        batch_size: int = 8, grad_accum: int = 4) -> Dict[str, Any]:
        """
        Fine-tune the model using the specified dataset and mode.
        
        Args:
            dataset: The name of the dataset to use for fine-tuning
            mode: The fine-tuning mode ('all', 'last', 'first', 'middle', or 'loraN')
            epochs: Number of epochs to train for
            learning_rate: Learning rate for optimization
            batch_size: Batch size for training
            grad_accum: Gradient accumulation steps
            
        Returns:
            Dictionary with fine-tuning results
        """
        if self.model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Create a copy of the model for fine-tuning
        model = copy.deepcopy(self.model)
        
        # Apply LoRA if specified in mode
        if mode.startswith('lora'):
            lora_rank = int(mode[4:])
            logger.info(f"Applying LoRA with rank {lora_rank}")
            
            if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
                for m in model.transformer.h:
                    if hasattr(m, "mlp") and hasattr(m.mlp, "c_fc"):
                        m.mlp.c_fc = LoRAConv1DWrapper(m.mlp.c_fc, lora_rank)
                        m.mlp.c_proj = LoRAConv1DWrapper(m.mlp.c_proj, lora_rank)
                    
                    if hasattr(m, "attn") and hasattr(m.attn, "c_attn"):
                        m.attn.c_attn = LoRAConv1DWrapper(m.attn.c_attn, lora_rank)
            else:
                raise HTTPException(status_code=500, detail="Model architecture not compatible with LoRA")
        
        # Move model to device
        model.to(DEVICE)
        
        # Select parameters to fine-tune based on the given mode
        params = list(parameters_to_fine_tune(model, mode))
        if not params:
            raise HTTPException(status_code=400, detail="No parameters selected for fine-tuning")
        
        # Set up optimizer
        optimizer = torch.optim.Adam(params, lr=learning_rate)
        
        # Dummy training data - in a real implementation, this would be loaded from the dataset
        # This is just for demonstration purposes
        x_train = ["This is a sample input text " + str(i) for i in range(100)]
        y_train = ["This is a sample output " + str(i) for i in range(100)]
        
        # Add prefixes based on dataset type
        x_train, y_train = add_prefixes(x_train, y_train, dataset)
        
        # Training loop
        model.train()
        total_loss = 0.0
        steps = 0
        
        # Create progress tracking
        max_steps = min(epochs * len(x_train) // batch_size, 1000)  # Cap at 1000 steps
        
        # Tokenizer should be available from the base model
        tokenizer = self.tokenizer
        if tokenizer is None:
            raise HTTPException(status_code=500, detail="Tokenizer not available")
        
        # Determine if we're doing classification or generation based on model type
        is_causal_lm = hasattr(model, "transformer") and not hasattr(model, "classifier")
        
        try:
            for step in range(max_steps):
                # Sample batch
                batch_indices = np.random.randint(0, len(x_train), batch_size)
                x_batch = [x_train[i] for i in batch_indices]
                y_batch = [y_train[i] for i in batch_indices]
                
                # Process batch based on model type
                if is_causal_lm:
                    # For generative models like GPT-2
                    batch = tokenize_gpt2_batch(tokenizer, x_batch, y_batch)
                    outputs = model(**batch, use_cache=False)
                    loss = get_loss(outputs.logits, batch['labels'])
                else:
                    # For classification models like BERT
                    inputs = tokenizer(x_batch, return_tensors='pt', padding=True, truncation=True, max_length=512).to(DEVICE)
                    labels = torch.tensor([int(y) for y in y_batch], device=DEVICE)
                    outputs = model(**inputs)
                    loss = get_loss(outputs.logits, labels)
                
                # Scale loss for gradient accumulation
                loss = loss / grad_accum
                loss.backward()
                
                # Update weights after accumulating gradients
                if (step + 1) % grad_accum == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                
                total_loss += loss.item() * grad_accum
                steps += 1
                
                # Log progress
                if step % 10 == 0:
                    logger.info(f"Step {step}/{max_steps}, Loss: {loss.item() * grad_accum:.6f}")
                
                # Early stopping based on loss
                if loss.item() < 0.01:
                    logger.info("Early stopping due to low loss")
                    break
                    
        except Exception as e:
            logger.error(f"Error during fine-tuning: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Fine-tuning failed: {str(e)}")
        
        # Set model back to evaluation mode
        model.eval()
        
        # Replace the original model with the fine-tuned one
        self.model = model
        
        # Return fine-tuning results
        return {
            "status": "fine-tuning completed",
            "dataset": dataset,
            "mode": mode,
            "epochs": epochs,
            "steps_completed": steps,
            "average_loss": total_loss / steps if steps > 0 else 0,
            "parameters_updated": sum(1 for _ in params)
        }
    
    async def evaluate(self, test_inputs: List[str]) -> Dict[str, Any]:
        """
        Evaluate the fine-tuned model on test inputs
        
        Args:
            test_inputs: List of test input strings
            
        Returns:
            Dictionary with evaluation results
        """
        if self.model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        results = []
        
        try:
            self.model.eval()
            with torch.inference_mode():
                for input_text in test_inputs:
                    # Tokenize input
                    inputs = self.tokenizer(input_text, return_tensors='pt', padding=True, truncation=True).to(DEVICE)
                    
                    # Generate output
                    outputs = self.model.generate(
                        **inputs,
                        max_length=100,
                        num_return_sequences=1,
                        temperature=0.7,
                        do_sample=True
                    )
                    
                    # Decode output
                    generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    results.append(generated_text)
        
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")
        
        return {
            "status": "evaluation completed",
            "results": results
        } 