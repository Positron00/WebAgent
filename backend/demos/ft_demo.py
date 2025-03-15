#!/usr/bin/env python
"""
Fine-Tuning Service Demo Script

This script demonstrates how to use the FineTuneService to fine-tune models
and track experiments with MLflow.

Usage:
    python -m backend.demos.ft_demo --model-id <model_id> --dataset <dataset> --mode <mode>

Example:
    python -m backend.demos.ft_demo --model-id gpt2-small --dataset squad --mode lora8
"""

import argparse
import asyncio
import logging
import os
import sys
import time
from typing import List, Dict, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.models.services.fine_tune_service import FineTuneService
from backend.models.services.base_model_service import ModelServiceConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger("ft_demo")

# Sample data for demonstration
SAMPLE_DATA = {
    "squad": {
        "train": [
            {
                "context": "The Apollo program, also known as Project Apollo, was the third United States human spaceflight program carried out by NASA, which succeeded in landing the first humans on the Moon from 1969 to 1972.",
                "question": "What was the Apollo program?",
                "answer": "The third United States human spaceflight program carried out by NASA"
            },
            {
                "context": "The Normans were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse raiders and pirates from Denmark, Iceland and Norway.",
                "question": "Who were the Normans?",
                "answer": "People who gave their name to Normandy in the 10th and 11th centuries"
            },
            {
                "context": "The Amazon rainforest, also known as Amazonia, is a moist broadleaf tropical rainforest in the Amazon biome that covers most of the Amazon basin of South America.",
                "question": "What is the Amazon rainforest?",
                "answer": "A moist broadleaf tropical rainforest in the Amazon biome"
            }
        ],
        "test": [
            "What was the purpose of the Apollo program?",
            "Where did the Normans come from?",
            "Where is the Amazon rainforest located?"
        ]
    },
    "summarization": {
        "train": [
            {
                "text": "The researchers found that the new drug reduced symptoms by 50% in patients with severe depression. The clinical trial involved 300 participants across 10 medical centers. Side effects were minimal and included mild headaches and nausea in some patients.",
                "summary": "New drug shows promising results for depression treatment with minimal side effects."
            },
            {
                "text": "The company announced a 20% increase in quarterly profits, exceeding analyst expectations. Revenue grew by 15% year-over-year, driven by strong performance in the Asian market. The board approved a dividend increase of 10 cents per share.",
                "summary": "Company reports better than expected profits with strong growth in Asia and increased dividends."
            },
            {
                "text": "The new electric vehicle can travel 400 miles on a single charge and recharge to 80% capacity in just 20 minutes. It features autonomous driving capabilities and an advanced infotainment system. Pricing starts at $45,000 before incentives.",
                "summary": "New electric vehicle offers impressive range, fast charging, and autonomous features starting at $45,000."
            }
        ],
        "test": [
            "Scientists have discovered a new species of deep-sea fish that can survive at extreme depths of up to 8,000 meters. The fish has evolved unique adaptations including specialized pressure-resistant cells and the ability to see in near-total darkness. Researchers believe studying these adaptations could lead to applications in medicine and materials science.",
            "The city council approved a new urban development plan that will transform the downtown area over the next decade. The plan includes pedestrian-only zones, expanded public transportation, and requirements for green spaces in all new construction. Critics argue that the plan may increase housing costs and displace current residents.",
            "The annual technology conference showcased several breakthrough innovations in artificial intelligence and robotics. Highlights included a robot that can perform complex surgical procedures with greater precision than human surgeons, and an AI system that can predict weather patterns with 95% accuracy up to two weeks in advance."
        ]
    }
}

async def prepare_data(dataset: str) -> Dict[str, List]:
    """
    Prepare training and testing data for the specified dataset.
    
    In a real-world scenario, this would load data from files or databases.
    For this demo, we use sample data defined above.
    
    Args:
        dataset: Name of the dataset to prepare
        
    Returns:
        Dictionary containing training and testing data
    """
    if dataset not in SAMPLE_DATA:
        logger.warning(f"Dataset {dataset} not found. Using summarization dataset.")
        dataset = "summarization"
    
    data = SAMPLE_DATA[dataset]
    
    if dataset == "squad":
        # Format QA data
        x_train = [f"{item['context']} Question: {item['question']}" for item in data["train"]]
        y_train = [item["answer"] for item in data["train"]]
    else:
        # Format summarization data
        x_train = [item["text"] for item in data["train"]]
        y_train = [item["summary"] for item in data["train"]]
    
    return {
        "x_train": x_train,
        "y_train": y_train,
        "test_inputs": data["test"]
    }

async def run_fine_tuning_demo(args):
    """
    Run the fine-tuning demo with the specified arguments.
    
    Args:
        args: Command-line arguments
    """
    logger.info(f"Starting fine-tuning demo with model: {args.model_id}, dataset: {args.dataset}, mode: {args.mode}")
    
    # Create service configuration
    config = ModelServiceConfig(
        model_id=args.model_id,
        model_path=args.model_path or f"models/{args.model_id}",
        device=args.device,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )
    
    # Initialize fine-tuning service
    service = FineTuneService(config)
    
    try:
        # Load the model
        logger.info(f"Loading model {args.model_id}...")
        await service.load_model()
        logger.info(f"Model loaded successfully")
        
        # Prepare data
        logger.info(f"Preparing {args.dataset} dataset...")
        data = await prepare_data(args.dataset)
        
        # Fine-tune the model
        logger.info(f"Starting fine-tuning with mode: {args.mode}")
        start_time = time.time()
        
        ft_results = await service.fine_tune(
            dataset=args.dataset,
            mode=args.mode,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            run_name=args.run_name
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"Fine-tuning completed in {elapsed_time:.2f} seconds")
        logger.info(f"Results: {ft_results}")
        
        # Evaluate the fine-tuned model
        logger.info("Evaluating fine-tuned model...")
        eval_results = await service.evaluate(data["test_inputs"], run_id=ft_results["mlflow_run_id"])
        logger.info(f"Evaluation metrics: {eval_results['metrics']}")
        
        # Save the fine-tuned model
        if args.save_model:
            save_name = args.save_name or f"{args.model_id}_{args.mode}_{args.dataset}"
            logger.info(f"Saving fine-tuned model as {save_name}...")
            save_results = await service.save_fine_tuned_model(
                model_name=save_name,
                description=f"Fine-tuned {args.model_id} on {args.dataset} using {args.mode} mode"
            )
            logger.info(f"Model saved: {save_results}")
        
        logger.info("Demo completed successfully!")
        logger.info(f"MLflow run ID: {ft_results['mlflow_run_id']}")
        logger.info(f"View results at: {os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5000')}")
        
    except Exception as e:
        logger.error(f"Error during demo: {str(e)}")
        raise
    finally:
        # Clean up resources
        if hasattr(service, "model") and service.model is not None:
            logger.info("Cleaning up resources...")
            service.model = None
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Fine-Tuning Service Demo")
    
    parser.add_argument("--model-id", type=str, default="gpt2-small", 
                        help="ID of the model to fine-tune")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to the model (defaults to models/{model_id})")
    parser.add_argument("--dataset", type=str, default="summarization",
                        choices=["squad", "summarization"],
                        help="Dataset to use for fine-tuning")
    parser.add_argument("--mode", type=str, default="lora8",
                        choices=["all", "last", "first", "middle", "lora4", "lora8", "lora16"],
                        help="Fine-tuning mode")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="Learning rate for optimization")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for training")
    parser.add_argument("--grad-accum", type=int, default=2,
                        help="Gradient accumulation steps")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cpu, cuda, mps)")
    parser.add_argument("--max-tokens", type=int, default=100,
                        help="Maximum number of tokens for generation")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for generation")
    parser.add_argument("--save-model", action="store_true",
                        help="Save the fine-tuned model")
    parser.add_argument("--save-name", type=str, default=None,
                        help="Name to save the fine-tuned model as")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Name for the MLflow run")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_fine_tuning_demo(args)) 