import logging
import time
import torch
from typing import Dict, Any

from .base_model_service import BaseModelService, ModelServiceConfig
from fastapi import HTTPException

logger = logging.getLogger(__name__)

def parameters_to_fine_tune(model, mode: str):
    if mode == "all":
        return model.parameters()
    elif mode == "last":
        for param in model.parameters():
            param.requires_grad = False
        # Assuming the model has a 'transformer' attribute with blocks 'h'
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            for param in model.transformer.h[-1].parameters():
                param.requires_grad = True
        else:
            raise HTTPException(status_code=500, detail="Model does not support fine-tuning mode 'last'")
        return filter(lambda p: p.requires_grad, model.parameters())
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported fine-tuning mode: {mode}")


class FineTuneService(BaseModelService):
    async def load_model(self) -> None:
        # Use the same loading mechanism as the base service
        await super().load_model()

    async def fine_tune(self, dataset: str, mode: str, epochs: int = 1, learning_rate: float = 1e-4) -> Dict[str, Any]:
        if self.model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        # Select parameters to fine-tune based on the given mode
        params = list(parameters_to_fine_tune(self.model, mode))
        if not params:
            raise HTTPException(status_code=400, detail="No parameters selected for fine-tuning")
        
        self.model.train()
        optimizer = torch.optim.Adam(params, lr=learning_rate)
        loss_val = 0.0
        
        # Dummy fine-tuning loop: in a real scenario, load proper training data
        for epoch in range(epochs):
            optimizer.zero_grad()
            # Create dummy input and target tensors
            dummy_input = torch.randint(0, 100, (1, 10)).to(next(self.model.parameters()).device)
            dummy_target = torch.randint(0, 100, (1, 10)).to(next(self.model.parameters()).device)
            
            try:
                output = self.model(dummy_input)
            except Exception as e:
                logger.error(f"Error during forward pass: {e}")
                raise HTTPException(status_code=500, detail="Error during model forward pass")
            
            # For demonstration, use a dummy loss function
            loss = output.sum() * 1e-5
            loss.backward()
            optimizer.step()
            loss_val += loss.item()
            logger.info(f"Epoch {epoch+1}/{epochs} completed, loss: {loss.item():.6f}")
        
        self.model.eval()
        return {"status": "fine-tuning completed", "final_loss": loss_val, "epochs": epochs}

# End of FineTuneService 