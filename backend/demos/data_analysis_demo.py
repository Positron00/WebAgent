#!/usr/bin/env python
"""
Data Analysis Agent Demo
=======================

This demo demonstrates the capabilities of the Data Analysis Agent,
which specializes in analyzing data and generating insights.

Usage:
    python -m backend.demos.data_analysis_demo

Requirements:
    - Configured environment variables (check .env.example)
    - Python packages: pandas, matplotlib, seaborn
"""

import os
import sys
import asyncio
import logging
from datetime import datetime
import tempfile
import json
import random
import csv
import pandas as pd
import numpy as np
from pprint import pprint
from pathlib import Path

# Setup path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("data_analysis_demo")

# Import agents and services
from backend.app.agents.data_analysis import get_data_analysis_agent
from backend.app.models.task import WorkflowState
from backend.app.core.config import settings
from backend.app.utils.metrics import timing_decorator
from backend.app.services.llm import confirm_llm_selection
from backend.demos.research_results_display import display_research_results

# Path to demo data
DEMO_DATA_DIR = os.path.join(current_dir, "demo_data")
os.makedirs(DEMO_DATA_DIR, exist_ok=True)

def create_sample_datasets():
    """Create sample datasets for testing."""
    created_files = {}
    
    # 1. User Activity Dataset
    user_activity_path = os.path.join(DEMO_DATA_DIR, "user_activity.csv")
    if not os.path.exists(user_activity_path):
        # Generate synthetic user activity data
        users = 200
        activities = ["search", "document_view", "chat", "export", "share", "login", "upload"]
        platforms = ["web", "mobile", "desktop", "api"]
        
        with open(user_activity_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["user_id", "date", "action", "platform", "duration_seconds", "success"])
            
            # Generate random data for the past 30 days
            for user_id in range(1, users + 1):
                # Each user has 5-20 activities
                num_activities = random.randint(5, 20)
                for _ in range(num_activities):
                    date = f"2025-03-{random.randint(1, 15):02d}"
                    action = random.choice(activities)
                    platform = random.choice(platforms)
                    duration = random.randint(5, 300)
                    success = random.random() > 0.1  # 10% failure rate
                    
                    writer.writerow([user_id, date, action, platform, duration, success])
        
        logger.info(f"Created sample dataset: {user_activity_path}")
    created_files["user_activity"] = user_activity_path
    
    # 2. Document Processing Performance
    doc_performance_path = os.path.join(DEMO_DATA_DIR, "document_processing.csv")
    if not os.path.exists(doc_performance_path):
        # Generate synthetic document processing data
        doc_types = ["pdf", "docx", "txt", "html", "md"]
        processing_types = ["extraction", "summary", "embedding", "translation"]
        
        with open(doc_performance_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["doc_id", "doc_type", "processing_type", "doc_size_kb", 
                             "processing_time_ms", "token_count", "error_occurred", "date"])
            
            # Generate 500 document processing records
            for doc_id in range(1, 501):
                doc_type = random.choice(doc_types)
                proc_type = random.choice(processing_types)
                
                # Size depends somewhat on document type
                if doc_type in ["pdf", "docx"]:
                    size = random.randint(100, 5000)  # larger files
                else:
                    size = random.randint(5, 500)  # smaller files
                
                # Processing time correlates with size
                time_ms = size * random.uniform(1.5, 3.0) + random.randint(-100, 100)
                time_ms = max(time_ms, 10)  # ensure positive time
                
                # Token count correlates with size
                token_count = int(size * random.uniform(2.5, 4.0) + random.randint(-50, 50))
                token_count = max(token_count, 1)  # ensure positive count
                
                # Error rate is higher for larger documents
                error_probability = 0.02 + (size / 10000)  # base 2% plus size factor
                error_occurred = random.random() < error_probability
                
                date = f"2025-03-{random.randint(1, 15):02d}"
                
                writer.writerow([doc_id, doc_type, proc_type, size, time_ms, token_count, error_occurred, date])
        
        logger.info(f"Created sample dataset: {doc_performance_path}")
    created_files["doc_performance"] = doc_performance_path
    
    # 3. Agent Performance Metrics
    agent_metrics_path = os.path.join(DEMO_DATA_DIR, "agent_metrics.csv")
    if not os.path.exists(agent_metrics_path):
        # Generate synthetic agent performance data
        agent_types = ["web_research", "document_extraction", "internal_research", 
                       "data_analysis", "coding_assistant", "team_manager"]
        models = ["meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", "meta-llama/Llama-3.1-8B", "claude-3-opus", "custom-model-v2"]
        
        with open(agent_metrics_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["agent_type", "model", "date", "total_runs", "avg_latency_ms", 
                             "success_rate", "token_usage", "user_rating"])
            
            # Generate data for each agent type for the past 15 days
            for day in range(1, 16):
                date = f"2025-03-{day:02d}"
                
                for agent in agent_types:
                    for model in models:
                        # Each agent has different performance characteristics
                        if agent == "web_research":
                            base_latency = 2500
                            base_success = 0.92
                            base_tokens = 4500
                        elif agent == "document_extraction":
                            base_latency = 1800
                            base_success = 0.89
                            base_tokens = 5200
                        elif agent == "coding_assistant":
                            base_latency = 3200
                            base_success = 0.85
                            base_tokens = 7500
                        else:
                            base_latency = 2200
                            base_success = 0.90
                            base_tokens = 4800
                        
                        # Models also affect performance
                        if model == "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free":
                            latency_factor = 1.0
                            success_factor = 1.0
                            token_factor = 1.0
                        elif model == "meta-llama/Llama-3.1-8B":
                            latency_factor = 0.8
                            success_factor = 0.97
                            token_factor = 0.85
                        elif model == "claude-3-opus":
                            latency_factor = 0.9
                            success_factor = 1.02
                            token_factor = 1.1
                        else:  # custom model
                            latency_factor = 0.75
                            success_factor = 0.95
                            token_factor = 0.8
                        
                        # Add some daily variance and overall improvement trend
                        day_factor = 1.0 - (day / 100)  # slight improvement over time
                        variance = random.uniform(0.9, 1.1)
                        
                        latency = base_latency * latency_factor * day_factor * variance
                        success = min(0.99, base_success * success_factor * (1 + (day / 300)))
                        tokens = base_tokens * token_factor * variance
                        
                        # Random number of runs
                        runs = random.randint(50, 200)
                        
                        # User rating (1-5 scale, correlates with success)
                        rating = min(5.0, 3.0 + (success * 2.5) + random.uniform(-0.5, 0.5))
                        rating = round(rating, 1)
                        
                        writer.writerow([agent, model, date, runs, round(latency), 
                                        round(success, 3), round(tokens), rating])
        
        logger.info(f"Created sample dataset: {agent_metrics_path}")
    created_files["agent_metrics"] = agent_metrics_path
    
    return created_files

@timing_decorator
async def run_data_analysis_demo():
    """Run the data analysis agent demo."""
    logger.info("Starting Data Analysis Agent Demo")
    
    # Initialize lists to store results for GUI display
    all_queries = []
    all_results = []
    all_errors = []
    
    # Ask user to confirm LLM provider and model selection
    print("\n" + "=" * 80)
    print("DATA ANALYSIS AGENT DEMO")
    print("=" * 80)
    print("This demo requires an LLM for analyzing data and generating insights.")
    print("The default LLM provider is TogetherAI with the Llama-3.3-70B model.")
    
    # Confirm LLM provider and model
    provider, model = confirm_llm_selection()
    
    # Create sample datasets
    datasets = create_sample_datasets()
    logger.info(f"Created {len(datasets)} sample datasets")
    
    # Get data analysis agent
    print("\nInitializing Data Analysis Agent...")
    analysis_agent = get_data_analysis_agent()
    
    # Define analysis tasks
    analysis_tasks = [
        {
            "title": "User Activity Analysis",
            "query": "Analyze the user activity data to identify trends in usage patterns. Which actions are most common? Is there a correlation between platform and action type? Visualize the data where appropriate.",
            "dataset": datasets["user_activity"]
        },
        {
            "title": "Document Processing Performance",
            "query": "Evaluate the document processing performance. Which document types take the longest to process? Is there a relationship between document size and processing time? What's the error rate by document type?",
            "dataset": datasets["doc_performance"]
        },
        {
            "title": "Agent Performance Comparison",
            "query": "Compare the performance of different agent types and models. Which combinations have the best success rates? Has performance improved over time? Is there a relationship between latency and user rating?",
            "dataset": datasets["agent_metrics"]
        }
    ]
    
    # Run through each analysis task
    for idx, task in enumerate(analysis_tasks):
        logger.info(f"\n\n=== ANALYSIS TASK {idx+1}: {task['title']} ===\n")
        
        # Create a workflow state for this task
        state = WorkflowState(
            query=task["query"],
            inputs={"data_path": task["dataset"]},
            context={
                "include_visualizations": True,
                "output_format": "markdown",
                "save_plots": True
            }
        )
        
        # Run data analysis agent
        logger.info(f"Running data analysis for: {task['title']}")
        print(f"\nRunning analysis with {provider} ({model}) as the LLM provider...")
        print(f"Analysis task: {task['title']}")
        try:
            start_time = datetime.now()
            result_state = await analysis_agent.run(state)
            duration = (datetime.now() - start_time).total_seconds()
            
            # Store results for display
            all_queries.append(f"{task['title']}: {task['query']}")
            all_results.append(result_state.output)
            all_errors.append(None)
            
            # Print results
            logger.info(f"Analysis Results (completed in {duration:.2f} seconds):")
            if result_state.output:
                # Format the analysis output
                if isinstance(result_state.output, dict):
                    logger.info("Analysis Summary:")
                    print(result_state.output.get("summary", "No summary provided"))
                    
                    # Check for insights
                    insights = result_state.output.get("insights", [])
                    if insights:
                        logger.info("\nKey Insights:")
                        for i, insight in enumerate(insights):
                            print(f"{i+1}. {insight}")
                    
                    # Check for visualizations
                    visualizations = result_state.output.get("visualizations", [])
                    if visualizations:
                        logger.info(f"\nGenerated {len(visualizations)} visualizations")
                        for viz in visualizations:
                            if isinstance(viz, dict) and "path" in viz:
                                logger.info(f"- {viz.get('title', 'Untitled')}: {viz['path']}")
                else:
                    pprint(result_state.output)
            else:
                logger.warning("No analysis results returned.")
                
        except Exception as e:
            logger.error(f"Error running data analysis agent: {e}")
            # Store error for display
            all_queries.append(f"{task['title']}: {task['query']}")
            all_results.append(None)
            all_errors.append(str(e))
    
    # Test analysis with specific parameters
    logger.info("\n\n=== TESTING ANALYSIS WITH SPECIFIC PARAMETERS ===\n")
    
    specific_task = {
        "query": "Perform a correlation analysis between all numerical variables in the document processing dataset. Identify which factors most strongly predict processing time.",
        "dataset": datasets["doc_performance"],
        "context": {
            "analysis_type": "correlation",
            "significance_threshold": 0.05,
            "include_visualizations": True,
            "output_format": "json"
        }
    }
    
    logger.info(f"Running analysis with specific parameters")
    try:
        # Create a workflow state with these parameters
        state = WorkflowState(
            query=specific_task["query"],
            inputs={"data_path": specific_task["dataset"]},
            context=specific_task["context"]
        )
        
        start_time = datetime.now()
        result_state = await analysis_agent.run(state)
        duration = (datetime.now() - start_time).total_seconds()
        
        # Store results for display
        all_queries.append(f"Correlation Analysis: {specific_task['query']}")
        all_results.append(result_state.output)
        all_errors.append(None)
        
        # Print results
        logger.info(f"Analysis Results (completed in {duration:.2f} seconds):")
        if result_state.output:
            # Save the full output to a file for examination
            output_dir = os.path.join(current_dir, "demo_outputs")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "correlation_analysis_output.json")
            
            with open(output_path, "w") as f:
                json.dump(result_state.output, f, indent=2)
            
            logger.info(f"Detailed analysis results saved to: {output_path}")
            
            # Show summary
            if isinstance(result_state.output, dict):
                correlation_data = result_state.output.get("correlation_data", {})
                if correlation_data:
                    logger.info("Top Correlations with Processing Time:")
                    for factor, corr in sorted(correlation_data.items(), key=lambda x: abs(x[1]) if isinstance(x[1], (int, float)) else 0, reverse=True):
                        if factor != "processing_time_ms" and isinstance(corr, (int, float)):
                            print(f"- {factor}: {corr:.3f}")
        else:
            logger.warning("No analysis results returned.")
            
    except Exception as e:
        logger.error(f"Error running data analysis agent: {e}")
        # Store error for display
        all_queries.append(f"Correlation Analysis: {specific_task['query']}")
        all_results.append(None)
        all_errors.append(str(e))
    
    logger.info("Data Analysis Agent Demo completed.")
    
    # Display all results in GUI
    logger.info("\nDisplaying results in GUI...")
    try:
        display_research_results(
            queries=all_queries,
            results=all_results,
            errors=all_errors,
            title="Data Analysis Results"
        )
    except Exception as e:
        logger.error(f"Error displaying results in GUI: {e}")

if __name__ == "__main__":
    # Run the demo
    asyncio.run(run_data_analysis_demo()) 