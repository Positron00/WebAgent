#!/usr/bin/env python
"""
Integrated Data Analysis and Coding Demo
========================================

This demo combines the Data Analysis Agent with the Coding Assistant and 
Coding Review Agents to demonstrate a workflow where:

1. Data Analysis Agent analyzes data and determines what code is needed
2. Coding Assistant Agent generates the necessary code
3. Coding Review Agent reviews, fixes, and optimizes the code
4. The improved code is executed to produce visualizations and insights

Usage:
    python -m backend.demos.integrated_data_analysis_demo

Requirements:
    - Configured environment variables (check .env.example)
    - Python packages: pandas, matplotlib, seaborn
"""

import os
import sys
import asyncio
import logging
import traceback
import tempfile
from datetime import datetime
import json
from pprint import pprint
import subprocess
import re
from pathlib import Path
import random
import pandas as pd
import numpy as np
import matplotlib
import argparse
matplotlib.use('Agg')  # Use non-interactive backend

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
logger = logging.getLogger("integrated_data_analysis_demo")

# Define a function to ensure proper model format based on provider
def get_proper_model_format(provider, model):
    """Ensure proper model format based on provider"""
    if provider == "openai":
        # Ensure OpenAI model has correct format
        if not model.startswith("gpt-"):
            model = f"gpt-{model}" if "-" in model else f"gpt-{model}"
    return model

# Import required modules
from backend.app.agents.data_analysis import get_data_analysis_agent, DataAnalysisAgent
from backend.app.agents.coding_assistant import get_coding_assistant_agent
from backend.app.agents.coding_review import get_coding_review_agent
from backend.app.models.task import WorkflowState
from backend.app.core.config import settings
from backend.app.utils.metrics import timing_decorator
from backend.app.services.llm import confirm_llm_selection
from backend.demos.research_results_display import display_research_results
from langgraph.graph import StateGraph, END

# Path to demo data and outputs
DEMO_DATA_DIR = os.path.join(current_dir, "demo_data")
DEMO_OUTPUT_DIR = os.path.join(current_dir, "demo_outputs")
os.makedirs(DEMO_DATA_DIR, exist_ok=True)
os.makedirs(DEMO_OUTPUT_DIR, exist_ok=True)

# Create a plots directory for visualizations
PLOTS_DIR = os.path.join(DEMO_OUTPUT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)
logger.info(f"Plots will be saved to: {PLOTS_DIR}")

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
            import csv
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
    
    # 2. Sales Data
    sales_data_path = os.path.join(DEMO_DATA_DIR, "sales_data.csv")
    if not os.path.exists(sales_data_path):
        # Generate synthetic sales data
        products = ["Product A", "Product B", "Product C", "Product D", "Product E"]
        regions = ["North", "South", "East", "West", "Central"]
        channels = ["Online", "Retail", "Direct", "Partner"]
        
        with open(sales_data_path, "w", newline="") as f:
            import csv
            writer = csv.writer(f)
            writer.writerow(["date", "product", "region", "channel", "units_sold", "revenue", "cost"])
            
            # Generate data for the past 12 months
            for month in range(1, 13):
                for _ in range(50):  # 50 entries per month
                    date = f"2024-{month:02d}-{random.randint(1, 28):02d}"
                    product = random.choice(products)
                    region = random.choice(regions)
                    channel = random.choice(channels)
                    
                    # Units sold depends on the product
                    base_units = {
                        "Product A": 100, 
                        "Product B": 50, 
                        "Product C": 200, 
                        "Product D": 75, 
                        "Product E": 150
                    }
                    units_sold = int(base_units[product] * random.uniform(0.5, 1.5))
                    
                    # Price and cost depend on the product
                    product_prices = {
                        "Product A": 29.99, 
                        "Product B": 99.99, 
                        "Product C": 14.99, 
                        "Product D": 49.99, 
                        "Product E": 39.99
                    }
                    product_costs = {
                        "Product A": 10.00, 
                        "Product B": 40.00, 
                        "Product C": 5.00, 
                        "Product D": 20.00, 
                        "Product E": 15.00
                    }
                    
                    # Add some regional and channel adjustment
                    region_factor = {
                        "North": 1.0, 
                        "South": 0.9, 
                        "East": 1.2, 
                        "West": 1.1, 
                        "Central": 0.95
                    }
                    channel_factor = {
                        "Online": 0.9, 
                        "Retail": 1.1, 
                        "Direct": 1.0, 
                        "Partner": 0.95
                    }
                    
                    price = product_prices[product] * region_factor[region] * channel_factor[channel]
                    cost = product_costs[product]
                    
                    revenue = units_sold * price
                    total_cost = units_sold * cost
                    
                    writer.writerow([date, product, region, channel, units_sold, round(revenue, 2), round(total_cost, 2)])
        
        logger.info(f"Created sample dataset: {sales_data_path}")
    created_files["sales_data"] = sales_data_path
    
    # 3. Customer Feedback Data
    feedback_data_path = os.path.join(DEMO_DATA_DIR, "customer_feedback.csv")
    if not os.path.exists(feedback_data_path):
        # Generate synthetic customer feedback data
        feedback_topics = ["UI/UX", "Performance", "Features", "Support", "Pricing"]
        sentiments = ["Positive", "Neutral", "Negative"]
        product_versions = ["v1.0", "v1.1", "v2.0", "v2.1", "v3.0"]
        
        with open(feedback_data_path, "w", newline="") as f:
            import csv
            writer = csv.writer(f)
            writer.writerow(["date", "customer_id", "topic", "sentiment", "product_version", "rating", "comment"])
            
            # Generate 500 feedback entries
            for i in range(500):
                # Date gradually shifting from older versions to newer ones
                month = random.randint(1, 12)
                day = random.randint(1, 28)
                version_weight = [0.4, 0.3, 0.2, 0.07, 0.03]  # More weighted toward older versions
                
                if month > 6:  # In later months, shift weight to newer versions
                    version_weight = [0.05, 0.15, 0.3, 0.3, 0.2]
                
                date = f"2024-{month:02d}-{day:02d}"
                customer_id = random.randint(1000, 9999)
                topic = random.choice(feedback_topics)
                product_version = random.choices(product_versions, weights=version_weight)[0]
                
                # Sentiment distribution varies by topic and version
                sentiment_weights = {
                    "UI/UX": {"v1.0": [0.3, 0.3, 0.4], "v2.0": [0.4, 0.3, 0.3], "v3.0": [0.6, 0.3, 0.1]},
                    "Performance": {"v1.0": [0.2, 0.3, 0.5], "v2.0": [0.5, 0.3, 0.2], "v3.0": [0.7, 0.2, 0.1]},
                    "Features": {"v1.0": [0.4, 0.4, 0.2], "v2.0": [0.5, 0.3, 0.2], "v3.0": [0.6, 0.3, 0.1]},
                    "Support": {"v1.0": [0.3, 0.4, 0.3], "v2.0": [0.4, 0.4, 0.2], "v3.0": [0.5, 0.3, 0.2]},
                    "Pricing": {"v1.0": [0.2, 0.3, 0.5], "v2.0": [0.3, 0.4, 0.3], "v3.0": [0.4, 0.4, 0.2]}
                }
                
                # Choose weights based on version family (v1.x, v2.x, v3.x)
                version_family = product_version[:2]
                weights = sentiment_weights[topic].get(version_family, [0.33, 0.34, 0.33])
                
                sentiment = random.choices(sentiments, weights=weights)[0]
                
                # Rating depends on sentiment
                rating_ranges = {
                    "Positive": (4, 5),
                    "Neutral": (3, 4),
                    "Negative": (1, 3)
                }
                rating_range = rating_ranges[sentiment]
                rating = round(random.uniform(rating_range[0], rating_range[1]), 1)
                
                # Generate simple placeholder comment
                comment = f"{sentiment} feedback on {topic} for {product_version}."
                
                writer.writerow([date, customer_id, topic, sentiment, product_version, rating, comment])
        
        logger.info(f"Created sample dataset: {feedback_data_path}")
    created_files["customer_feedback"] = feedback_data_path
    
    return created_files

# Custom data analysis agent for this demo to avoid template errors
async def custom_data_analysis_agent(state: WorkflowState) -> WorkflowState:
    """
    Custom data analysis agent that bypasses the ChatPromptTemplate variable error
    by directly constructing needed output.
    """
    try:
        logger.info(f"Data Analysis Agent processing query: {state.query}")
        
        # Check if we've processed this enough times and should stop
        da_count = state.context.get("data_analysis_count", 0)
        state.context["data_analysis_count"] = da_count + 1
        
        # After a couple of cycles, stop generating more code
        if da_count >= 2:
            logger.info(f"Data analysis has been called {da_count} times. Ending workflow to prevent infinite cycles.")
            data_analysis_report = {
                "status": "completed",
                "summary": f"Final analysis completed after {da_count} iterations",
                "insights": [
                    "Analysis complete with multiple code generation cycles",
                    "Summarizing results from previous iterations"
                ],
                "visualizations": [],
                "timestamp": datetime.now().isoformat()
            }
            
            # Add the report to the state
            state.update_with_agent_output("data_analysis", data_analysis_report)
            state.output = data_analysis_report
            
            # No more code generation needed
            state.context["needs_code_generation"] = False
            
            logger.info("Data Analysis Agent completed successfully")
            return state
            
        # In our demo context, we don't need to process via the template that's causing issues
        # Instead, we'll determine if we need code generation and set the state appropriately
        
        # Construct a basic output
        data_analysis_report = {
            "status": "completed",
            "summary": f"Analysis of data for query: {state.query}",
            "insights": [
                "Data analysis in demo mode - bypassing LLM chain",
                "Proceeding directly to code generation for this analysis task"
            ],
            "visualizations": [],
            "code_request": state.query,  # Pass the query directly as the code request
            "timestamp": datetime.now().isoformat()
        }
        
        # Add the report to the state
        state.update_with_agent_output("data_analysis", data_analysis_report)
        # Set the output to ensure it's accessible
        state.output = data_analysis_report
        
        # Always proceed to code generation in the demo
        state.context["needs_code_generation"] = True
        
        logger.info("Data Analysis Agent completed successfully")
        return state
        
    except Exception as e:
        logger.error(f"Error in Data Analysis Agent: {str(e)}", exc_info=True)
        state.mark_error(f"Data Analysis Agent error: {str(e)}")
        return state

def build_integrated_workflow():
    """
    Build a LangGraph workflow for data analysis, code generation, and review.
    
    Returns:
        StateGraph: The compiled workflow graph
    """
    logger.info("Building integrated data analysis and coding workflow graph")
    
    # Get the agent instances - use our custom data analysis agent
    data_analysis_agent = custom_data_analysis_agent
    coding_assistant = get_coding_assistant_agent()
    coding_review = get_coding_review_agent()
    
    # Create the graph
    workflow = StateGraph(WorkflowState)
    
    # Add nodes to the graph
    workflow.add_node("data_analysis", data_analysis_agent)
    workflow.add_node("coding_assistant", coding_assistant)
    workflow.add_node("coding_review", coding_review)
    
    # Define routing conditions
    def route_after_analysis(state: WorkflowState) -> str:
        """Route based on whether code generation is needed after analysis"""
        # Check if we're in a loop (cycling through data_analysis multiple times)
        da_count = state.context.get("data_analysis_count", 0)
        
        # If we've been through data_analysis more than twice, end the workflow
        if da_count >= 2:
            logger.info(f"Data analysis cycle count reached {da_count}. Ending workflow to prevent infinite loops.")
            return "end"
            
        # Check if the data analysis output explicitly sets needs_code_generation to False
        if state.context.get("needs_code_generation") is False:
            logger.info("Data analysis explicitly marked as complete. No code generation needed.")
            return "end"
            
        # Check if the data analysis output indicates code generation is needed
        if state.context.get("needs_code_generation", False):
            logger.info("Data analysis requires code generation. Routing to coding assistant.")
            # Extract the code generation request from the analysis output
            if state.output and isinstance(state.output, dict):
                # Use the code_request if available or fall back to code_description
                coding_query = state.output.get(
                    "code_request", 
                    state.output.get("code_description", "")
                )
                if coding_query:
                    state.query = coding_query
                    logger.info(f"Updated query for code generation: {coding_query}")
            return "coding_assistant"
        else:
            logger.info("Data analysis complete. No code generation needed.")
            return "end"
    
    # Define a function to determine when additional review iterations are needed
    def needs_further_review(state: WorkflowState) -> str:
        """
        Check if the code still needs further review after fixing.
        
        Rules:
        1. If it's already been through 2 review cycles, stop (prevent infinite loops)
        2. If the fixed code execution failed, go back to coding_review
        3. Otherwise, end the workflow
        """
        # Check the review count
        review_count = state.context.get("review_count", 0)
        state.context["review_count"] = review_count + 1
        
        # Check if we already hit the maximum review cycles (2)
        if review_count >= 2:
            logger.info(f"Max review count reached ({review_count}). Ending workflow.")
            return "end"
            
        # Check data analysis cycle count - if it's already high, don't go back to data_analysis
        da_count = state.context.get("data_analysis_count", 0)
        if da_count >= 2:
            logger.info(f"Data analysis cycle count already at {da_count}. Not returning to data analysis.")
            return "end"
            
        # Check if there's a coding_review report with fixed execution
        if "coding_review" in state.reports:
            review_report = state.reports["coding_review"]
            fixed_execution = review_report.get("fixed_execution", {})
            
            # Check if fixed code execution failed
            if fixed_execution and fixed_execution.get("success") is False:
                # Increment the review count (already done above)
                logger.info(f"Fixed code execution failed. Starting review cycle {review_count + 1}.")
                
                # Update the state to use the fixed code as the new input code
                if "fixed_code" in review_report:
                    review_report["original_code"] = review_report["fixed_code"]
                    review_report["fixed_code"] = ""
                    review_report["fixed_execution"] = {}
                
                return "coding_review"
        
        # After code review completes (and possibly fixes the code), go back to data analysis
        # to complete any final analysis with the generated code, but only once
        if state.context.get("return_to_analysis", False) and da_count < 2:
            logger.info("Code generation and review complete. Returning to data analysis for final processing.")
            
            # Update the context to indicate we now have code
            state.context["has_generated_code"] = True
            
            # Get the fixed code from the review report if available
            if "coding_review" in state.reports and "fixed_code" in state.reports["coding_review"]:
                state.context["generated_code"] = state.reports["coding_review"]["fixed_code"]
            elif "coding_assistant" in state.reports and "code" in state.reports["coding_assistant"]:
                state.context["generated_code"] = state.reports["coding_assistant"]["code"]
            
            return "data_analysis"
        
        # Default: end the workflow
        return "end"
    
    # Set up edges
    workflow.add_edge("__start__", "data_analysis")
    workflow.add_conditional_edges(
        "data_analysis",
        route_after_analysis,
        {
            "coding_assistant": "coding_assistant",
            "end": END
        }
    )
    workflow.add_edge("coding_assistant", "coding_review")
    workflow.add_conditional_edges(
        "coding_review",
        needs_further_review,
        {
            "coding_review": "coding_review",
            "data_analysis": "data_analysis",
            "end": END
        }
    )
    
    # Compile the workflow with default settings
    # Since we're using a custom data analysis agent that avoids recursion issues,
    # we don't need a custom recursion limit
    compiled_workflow = workflow.compile()
    logger.info("Compiled integrated workflow successfully")
    
    return compiled_workflow

@timing_decorator
async def run_integrated_data_analysis_demo(num_tasks=3):
    """Run the integrated data analysis and coding demo.
    
    Args:
        num_tasks (int): Number of tasks to run (default: 3)
    """
    logger.info(f"Starting Integrated Data Analysis and Coding Demo (running {num_tasks} tasks)")
    
    # Initialize lists to store results for display
    all_queries = []
    all_results = []
    all_errors = []
    
    print("\n" + "=" * 80)
    print("INTEGRATED DATA ANALYSIS AND CODING DEMO")
    print("=" * 80)
    print("This demo shows the complete workflow of data analysis, code generation,")
    print("and code review using a LangGraph-based architecture with OpenAI GPT-4o-mini")
    
    # Set model directly without interactive prompt
    provider = "openai" 
    model = "gpt-4o-mini"
    
    # Ensure proper model format
    model = get_proper_model_format(provider, model)
    
    # Set environment variables for the selected provider and model
    os.environ["CONFIRM_LLM_SELECTION"] = "true"
    os.environ["CONFIRMED_LLM_PROVIDER"] = provider
    os.environ["CONFIRMED_LLM_MODEL"] = model
    
    print(f"\nUsing {provider} ({model}) as the LLM provider")
    
    # Create sample datasets
    datasets = create_sample_datasets()
    logger.info(f"Created {len(datasets)} sample datasets")
    
    # Build the LangGraph workflow
    print("\nBuilding LangGraph workflow...")
    workflow = build_integrated_workflow()
    
    # Define analysis tasks
    analysis_tasks = [
        {
            "title": "Sales Performance Analysis with Visualization",
            "query": "Analyze the sales_data.csv to identify top-performing products, regions, and channels. Create visualizations of monthly revenue trends and product performance. Calculate profit margins and identify the most profitable combinations of product, region, and channel.",
            "datasets": {"sales_data": datasets["sales_data"]},
            "needs_code": True,
            "return_to_analysis": True
        },
        {
            "title": "Customer Feedback Analysis with Sentiment Trends",
            "query": "Analyze the customer_feedback.csv to identify sentiment trends across different product versions and topics. Create visualizations showing how sentiment and ratings have changed over time. Identify the most common feedback topics for negative sentiment ratings.",
            "datasets": {"customer_feedback": datasets["customer_feedback"]},
            "needs_code": True,
            "return_to_analysis": True
        },
        {
            "title": "User Activity Pattern Analysis",
            "query": "Analyze the user_activity.csv to identify usage patterns across different platforms and actions. Calculate success rates for different actions and identify any problematic areas. Create visualizations showing activity distribution by platform and time.",
            "datasets": {"user_activity": datasets["user_activity"]},
            "needs_code": True,
            "return_to_analysis": True
        }
    ]
    
    # Process each task
    for idx, task in enumerate(analysis_tasks[:num_tasks]):
        logger.info(f"\n\n=== TASK {idx+1}: {task['title']} ===\n")
        print(f"\n{'=' * 60}")
        print(f"TASK {idx+1}: {task['title']}")
        print(f"{'=' * 60}")
        print(f"Query: {task['query']}")
        
        # Store the query for results display
        all_queries.append(task["query"])
        
        try:
            # Create initial workflow state with datasets as inputs
            state = WorkflowState(
                query=task["query"],
                inputs=task["datasets"],
                context={
                    "needs_code_generation": task["needs_code"],
                    "return_to_analysis": task["return_to_analysis"],
                    "output_dir": PLOTS_DIR,
                    "include_visualizations": True
                }
            )
            
            # Execute the workflow
            print("\nExecuting LangGraph workflow...")
            start_time = datetime.now()
            final_state = await workflow.ainvoke(state)
            total_duration = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Workflow completed in {total_duration:.2f} seconds")
            print(f"Workflow completed in {total_duration:.2f} seconds")
            
            # For debugging: log the structure of the final state
            logger.debug(f"Final state structure: {type(final_state)}")
            
            # Process and display results
            # First, let's handle the fact that final_state could be different types
            print("\n--- WORKFLOW RESULTS ---")
            
            # Initialize output fields
            result = {}
            visualizations = []
            insights = []
            generated_code = None
            code_review_data = None
            
            # Handle potential outputs and reports
            outputs = None
            reports = None
            
            # Try different methods to access the final state data based on its type
            if hasattr(final_state, 'output') and final_state.output:
                outputs = final_state.output
            
            if hasattr(final_state, 'reports') and final_state.reports:
                reports = final_state.reports
                
            # If it's a dict-like object, try to access keys directly
            if isinstance(final_state, dict):
                if 'reports' in final_state:
                    reports = final_state['reports']
                if 'output' in final_state:
                    outputs = final_state['output']
                
                # Try other potential structures
                if 'values' in final_state:
                    values = final_state['values']
                    if isinstance(values, dict):
                        # Look for 'reports' or 'artifacts' in the values dict
                        if 'reports' in values:
                            reports = values['reports']
                        if 'artifacts' in values:
                            # Some LangGraph versions use artifacts instead of reports
                            reports = values['artifacts']
                        if 'output' in values:
                            outputs = values['output']
            
            # For LangGraph's return object - check its properties and methods
            for key in dir(final_state):
                if not key.startswith('_'):  # Skip private attributes
                    try:
                        # Try to access as attribute
                        value = getattr(final_state, key)
                        
                        # If we find reports or outputs, use them
                        if key == 'reports' or key == 'artifacts':
                            reports = value
                        elif key == 'output':
                            outputs = value
                        
                        # Look for coding assistant or coding review data directly
                        if key == 'coding_assistant':
                            coding_assistant_report = value
                        elif key == 'coding_review': 
                            code_review_data = value
                    except:
                        # Skip attributes we can't access
                        pass
            
            # Try to find coding assistant data if not already found
            if not generated_code:
                # Try different ways the coding data might be stored
                if isinstance(final_state, dict):
                    for key in final_state:
                        if key == 'coding_assistant' and isinstance(final_state[key], dict):
                            coding_assistant_report = final_state[key]
                            if 'code' in coding_assistant_report:
                                generated_code = coding_assistant_report['code']
                                break
                        elif isinstance(final_state[key], dict) and 'code' in final_state[key]:
                            generated_code = final_state[key]['code']
                            break
            
            # Extract coding assistant output
            if reports and 'coding_assistant' in reports:
                coding_assistant_report = reports['coding_assistant']
                
                print("\n=== CODING ASSISTANT OUTPUT ===")
                
                # Get code and explanation
                if 'code' in coding_assistant_report:
                    generated_code = coding_assistant_report['code']
                    code_explanation = coding_assistant_report.get('explanation', 'No explanation provided')
                    
                    print(f"\nCode Explanation: {code_explanation}")
                    print("\nGenerated Code:")
                    print("```python")
                    print(generated_code[:500] + ("..." if len(generated_code) > 500 else ""))
                    print("```")
            
            # Extract code review output
            if reports and 'coding_review' in reports:
                code_review_data = reports['coding_review']
                
                print("\n=== CODE REVIEW OUTPUT ===")
                
                if 'review' in code_review_data:
                    print(f"\nReview: {code_review_data['review']}")
                
                if 'fixed_code' in code_review_data:
                    print("\nFixed Code:")
                    print("```python")
                    print(code_review_data['fixed_code'][:500] + ("..." if len(code_review_data['fixed_code']) > 500 else ""))
                    print("```")
                
                if 'improvement_explanation' in code_review_data:
                    print(f"\nImprovements: {code_review_data['improvement_explanation']}")
                
                if 'fixed_execution' in code_review_data:
                    execution = code_review_data['fixed_execution']
                    success = execution.get('success', False)
                    print(f"\nFixed Code Execution: {'SUCCESS' if success else 'FAILURE'}")
                    if 'result' in execution:
                        print(f"Result: {execution['result']}")
            
            # Extract data analysis output if available
            if outputs and isinstance(outputs, dict):
                # Use available fields
                summary = outputs.get('summary', 'No summary available')
                insights = outputs.get('insights', [])
                visualizations = outputs.get('visualizations', [])
                
                print("\n=== DATA ANALYSIS OUTPUT ===")
                print(f"\nSummary: {summary}")
                
                if insights:
                    print("\nInsights:")
                    for i, insight in enumerate(insights):
                        print(f"{i+1}. {insight}")
                
                if visualizations:
                    print("\nVisualizations:")
                    for viz in visualizations:
                        if isinstance(viz, dict) and 'path' in viz:
                            title = viz.get('title', 'Unnamed visualization')
                            print(f"- {title}: {viz['path']}")
                
                result = {
                    'summary': summary,
                    'insights': insights
                }
            
            # Store results for display
            all_results.append({
                "analysis": result,
                "visualizations": visualizations,
                "insights": insights,
                "generated_code": generated_code,
                "code_review": code_review_data,
                "task_title": task["title"]
            })
            all_errors.append(None)
                
        except Exception as e:
            error_msg = f"Error running workflow: {str(e)}"
            logger.error(error_msg)
            print(f"Error: {error_msg}")
            traceback.print_exc()
            all_results.append(None)
            all_errors.append(error_msg)
    
    # Display summary of all results
    logger.info("\nDisplaying summary of all results...")
    try:
        # Convert all_results to a format compatible with result display
        display_results = []
        for result in all_results:
            if result is None:
                display_results.append(None)
                continue
                
            # Create a formatted result with all components
            formatted_result = ""
            
            # Add analysis section if available
            if "analysis" in result:
                analysis = result["analysis"]
                if isinstance(analysis, dict) and "summary" in analysis:
                    formatted_result += f"## Analysis Summary\n\n{analysis['summary']}\n\n"
                else:
                    formatted_result += f"## Analysis\n\n{str(analysis)}\n\n"
            
            # Add insights section if available
            if "insights" in result and result["insights"]:
                formatted_result += "## Key Insights\n\n"
                for i, insight in enumerate(result["insights"]):
                    formatted_result += f"{i+1}. {insight}\n"
                formatted_result += "\n"
            
            # Add visualizations section if available
            if "visualizations" in result and result["visualizations"]:
                formatted_result += "## Visualizations\n\n"
                for i, viz in enumerate(result["visualizations"]):
                    if isinstance(viz, dict) and "path" in viz:
                        title = viz.get("title", f"Visualization {i+1}")
                        path = viz["path"]
                        formatted_result += f"- {title}: {path}\n"
                formatted_result += "\n"
            
            # Add generated code section if available
            if "generated_code" in result and result["generated_code"]:
                formatted_result += "## Generated Code\n\n```python\n"
                formatted_result += result["generated_code"]
                formatted_result += "\n```\n\n"
            
            # Add code review section if available
            if "code_review" in result and result["code_review"]:
                review = result["code_review"]
                formatted_result += "## Code Review\n\n"
                
                if "review" in review:
                    formatted_result += f"### Review\n{review['review']}\n\n"
                
                if "improvement_explanation" in review:
                    formatted_result += f"### Improvements\n{review['improvement_explanation']}\n\n"
                
                if "fixed_execution" in review and "result" in review["fixed_execution"]:
                    formatted_result += f"### Execution Result\n{review['fixed_execution']['result']}\n\n"
            
            display_results.append(formatted_result)
        
        # Display results in GUI
        display_research_results(
            queries=all_queries,
            results=display_results,
            errors=all_errors,
            title="Integrated Data Analysis and Coding Results"
        )
    except Exception as e:
        logger.error(f"Error displaying results: {e}")
        traceback.print_exc()
    
    logger.info("Integrated Data Analysis and Coding Demo completed")
    print("\nDemo completed. Generated code files and plots saved to:", DEMO_OUTPUT_DIR)
    print(f"Visualizations are stored in: {PLOTS_DIR}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the integrated data analysis demo')
    parser.add_argument('--tasks', type=int, default=3, help='Number of tasks to run (1-3)')
    args = parser.parse_args()
    
    # Validate number of tasks
    num_tasks = max(1, min(3, args.tasks))  # Ensure between 1 and 3
    
    # Run the demo
    asyncio.run(run_integrated_data_analysis_demo(num_tasks)) 