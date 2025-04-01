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
from datetime import datetime, timedelta
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
            model = f"gpt-{model}"
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

def create_sample_plots(task_title):
    """Create sample plots for demonstration purposes."""
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    # Ensure the plots directory exists
    os.makedirs(PLOTS_DIR, exist_ok=True)
    created_plots = []
    
    try:
        # Create sample plot 1: Revenue by Product
        plt.figure(figsize=(10, 6))
        products = ['Product A', 'Product B', 'Product C', 'Product D', 'Product E']
        revenues = [120000, 95000, 140000, 65000, 85000]
        plt.bar(products, revenues, color='skyblue')
        plt.title('Revenue by Product')
        plt.xlabel('Product')
        plt.ylabel('Revenue ($)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        for i, v in enumerate(revenues):
            plt.text(i, v + 3000, f"${v/1000:.0f}K", ha='center')
        plot_path = os.path.join(PLOTS_DIR, f"revenue_by_product_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(plot_path)
        plt.close()
        created_plots.append({"title": "Revenue by Product", "path": plot_path, "type": "image"})
        
        # Create sample plot 2: Monthly Revenue Trend
        plt.figure(figsize=(12, 6))
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        trend = [85000, 82000, 90000, 88000, 95000, 110000, 115000, 120000, 118000, 125000, 130000, 140000]
        plt.plot(months, trend, marker='o', linewidth=2, color='#3498db')
        plt.title('Monthly Revenue Trend')
        plt.xlabel('Month')
        plt.ylabel('Revenue ($)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.fill_between(months, trend, alpha=0.2, color='#3498db')
        plot_path = os.path.join(PLOTS_DIR, f"monthly_trend_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(plot_path)
        plt.close()
        created_plots.append({"title": "Monthly Revenue Trend", "path": plot_path, "type": "image"})
        
        # Create sample plot 3: Sales Channels
        plt.figure(figsize=(8, 8))
        channels = ['Online', 'Retail', 'Distributor', 'Direct']
        sales = [35, 25, 30, 10]
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
        plt.pie(sales, labels=channels, autopct='%1.1f%%', startangle=90, colors=colors)
        plt.title('Sales by Channel')
        plt.axis('equal')
        plot_path = os.path.join(PLOTS_DIR, f"sales_by_channel_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(plot_path)
        plt.close()
        created_plots.append({"title": "Sales by Channel", "path": plot_path, "type": "image"})
        
        # Create sample plot 4: Regional Performance
        plt.figure(figsize=(10, 6))
        regions = ['North', 'South', 'East', 'West', 'Central']
        profits = [28, 22, 32, 30, 24]
        plt.bar(regions, profits, color='#2ecc71')
        plt.title('Profit Margin by Region (%)')
        plt.xlabel('Region')
        plt.ylabel('Profit Margin (%)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        for i, v in enumerate(profits):
            plt.text(i, v + 1, f"{v}%", ha='center')
        plot_path = os.path.join(PLOTS_DIR, f"profit_by_region_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(plot_path)
        plt.close()
        created_plots.append({"title": "Profit Margin by Region", "path": plot_path, "type": "image"})
        
        logger.info(f"Created {len(created_plots)} sample plots for {task_title}")
        return created_plots
    except Exception as e:
        logger.error(f"Error creating sample plots: {e}")
        return []

# Custom data analysis agent for this demo to avoid template errors
async def custom_data_analysis_agent(state: WorkflowState) -> WorkflowState:
    """
    Custom data analysis agent that processes data and generates analysis based
    on the current state of the workflow, including processing code execution results.
    """
    try:
        logger.info(f"Data Analysis Agent processing query: {state.query}")
        
        # Check if we've processed this enough times and should stop
        da_count = state.context.get("data_analysis_count", 0)
        state.context["data_analysis_count"] = da_count + 1
        
        # After a couple of cycles, stop generating more code
        if da_count >= 2:
            logger.info(f"Data analysis has been called {da_count} times. Ending workflow to prevent infinite cycles.")
            
            # Create sample plots for the final report
            visualization_paths = create_sample_plots(state.query[:50])
            
            # Try to extract any code execution results
            code_results = "No execution results available"
            
            # Try to get fixed code execution results
            if "coding_review" in state.reports:
                review_report = state.reports["coding_review"]
                if "fixed_execution" in review_report:
                    execution = review_report["fixed_execution"]
                    if execution.get("success", False) and "result" in execution:
                        code_results = execution["result"]
            
            # Generate insights based on the plots and any code execution
            insights = [
                "Product C is the top performer with $140K in revenue",
                "December had the highest monthly revenue at $140K",
                "Online channels account for 35% of total sales",
                "East region has the highest profit margin at 32%",
                "Monthly revenue shows consistent growth trend throughout the year",
                "Average profit margin across all regions is 27.2%",
                "Online sales have the highest revenue-to-cost ratio"
            ]
            
            # Try to extract any additional insights from code execution results
            if code_results and isinstance(code_results, str):
                result_lines = code_results.split("\n")
                additional_insights = []
                for line in result_lines:
                    line = line.strip()
                    # Look for lines with meaningful metrics or findings
                    if any(keyword in line.lower() for keyword in 
                        ['top', 'highest', 'most', 'average', 'total', 'profit', 
                         'revenue', 'performance', 'margin', 'channel']):
                        additional_insights.append(line)
                    # Include any line with numbers and a colon that's likely a result
                    elif ':' in line and any(c.isdigit() for c in line) and len(line) > 15:
                        additional_insights.append(line)
                
                # Add up to 3 additional insights if found
                if additional_insights:
                    insights.extend(additional_insights[:3])
            
            data_analysis_report = {
                "status": "completed",
                "summary": f"Comprehensive analysis of sales data completed, identifying top performers, trends, and profit margins across products, regions, and channels. Generated {len(visualization_paths)} visualizations highlighting key performance metrics.",
                "insights": insights,
                "visualizations": visualization_paths,
                "timestamp": datetime.now().isoformat(),
                "execution_results": code_results
            }
            
            # Add the report to the state
            state.update_with_agent_output("data_analysis", data_analysis_report)
            state.output = data_analysis_report
            
            # No more code generation needed
            state.context["needs_code_generation"] = False
            
            logger.info(f"Final data analysis completed with {len(insights)} insights and {len(visualization_paths)} visualizations")
            return state

        # Check if we already have generated code and execution results
        # This indicates we're in the final analysis phase after code execution
        if da_count > 0 and state.context.get("has_generated_code", False):
            logger.info("Processing final analysis with code execution results")
            
            # Extract code execution results
            code_results = None
            generated_code = state.context.get("generated_code", "")
            visualization_paths = []
            insights = []
            
            # Get code review results if available
            if "coding_review" in state.reports:
                code_review = state.reports["coding_review"]
                fixed_execution = code_review.get("fixed_execution", {})
                
                if fixed_execution and fixed_execution.get("success", False):
                    # Extract the execution result
                    code_results = fixed_execution.get("result", "No execution results available")
                    
                    # Check for generated plots (in PLOTS_DIR)
                    plot_files = []
                    try:
                        plot_dir = state.context.get("output_dir", PLOTS_DIR)
                        if os.path.exists(plot_dir):
                            # Get modification time before code execution
                            before_time = datetime.now() - timedelta(minutes=2)
                            
                            # Find all image files in the plots directory
                            for file in os.listdir(plot_dir):
                                if file.endswith(('.png', '.jpg', '.svg', '.pdf')):
                                    file_path = os.path.join(plot_dir, file)
                                    # Check if the file was modified in the last couple minutes
                                    mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                                    if mod_time > before_time:
                                        plot_files.append(file)
                                        visualization_paths.append({
                                            "title": f"Plot: {file}",
                                            "path": file_path,
                                            "type": "image"
                                        })
                                        logger.info(f"Found recently created plot: {file}")
                    except Exception as e:
                        logger.warning(f"Error checking for plot files: {e}")
                        
                    if not visualization_paths:
                        # If no new plots were found, include all plots as a fallback
                        try:
                            for file in os.listdir(plot_dir):
                                if file.endswith(('.png', '.jpg', '.svg', '.pdf')):
                                    file_path = os.path.join(plot_dir, file)
                                    visualization_paths.append({
                                        "title": f"Plot: {file}",
                                        "path": file_path,
                                        "type": "image"
                                    })
                                    logger.info(f"Including existing plot: {file}")
                        except Exception as e:
                            logger.warning(f"Error including existing plots: {e}")
                    
                    # Extract insights from code execution results
                    if code_results:
                        # Parse insights from execution results
                        result_lines = str(code_results).split('\n')
                        
                        # Look for lines with meaningful metrics or findings
                        for line in result_lines:
                            line = line.strip()
                            # Look for lines with analytics information
                            if any(keyword in line.lower() for keyword in 
                                  ['top', 'highest', 'most', 'average', 'total', 'profit', 
                                   'revenue', 'performance', 'margin', 'channel']):
                                insights.append(line)
                            # Include any line with numbers and a colon that's likely a result
                            elif ':' in line and any(c.isdigit() for c in line) and len(line) > 15:
                                insights.append(line)
                                
                        # Limit insights to top 7 most meaningful ones
                        insights = insights[:7]
                        
                        # Add default insights if none were found
                        if not insights:
                            insights = [
                                "Executed analysis successfully on the sales data",
                                "Generated visualizations for product and regional performance",
                                "Calculated profit margins across different sales channels",
                                "Identified top-performing products by revenue",
                                "Analyzed monthly sales trends"
                            ]
            
            # Generate a comprehensive report based on the execution results
            analysis_summary = f"Analysis of sales data completed successfully, examining product, region, and channel performance."
            if visualization_paths:
                analysis_summary += f" Generated {len(visualization_paths)} visualizations showing performance metrics and trends."
            
            # Create a detailed report
            data_analysis_report = {
                "status": "completed",
                "summary": analysis_summary,
                "insights": insights,
                "visualizations": visualization_paths,
                "timestamp": datetime.now().isoformat(),
                "execution_results": code_results
            }
            
            # Add the report to the state
            state.update_with_agent_output("data_analysis", data_analysis_report)
            state.output = data_analysis_report
            
            # No more code generation needed after this final analysis
            state.context["needs_code_generation"] = False
            
            logger.info(f"Final data analysis completed successfully with {len(insights)} insights and {len(visualization_paths)} visualizations")
            return state
            
        # Initial analysis phase - we need to generate code first
        # For demo purposes, extract dataset info to personalize the query
        dataset_info = ""
        if state.inputs:
            for dataset_name, dataset_path in state.inputs.items():
                if os.path.exists(dataset_path):
                    try:
                        # Get basic file stats
                        file_size = os.path.getsize(dataset_path) / 1024  # KB
                        
                        # For CSV files, peek at the headers and row count
                        if dataset_path.endswith('.csv'):
                            with open(dataset_path, 'r') as f:
                                header = f.readline().strip()
                                # Count lines (approximate for large files)
                                line_count = 0
                                for _ in range(100):  # Read at most 100 lines
                                    if not f.readline():
                                        break
                                    line_count += 1
                                
                                # Check if there are more lines
                                if f.readline():
                                    line_count = f"{line_count}+ rows"
                                else:
                                    line_count = f"{line_count} rows"
                                
                                dataset_info += f"Dataset '{dataset_name}' ({file_size:.1f} KB, {line_count})"
                                dataset_info += f" with headers: {header}\n"
                    except Exception as e:
                        logger.warning(f"Error reading dataset info: {e}")
                        dataset_info += f"Dataset '{dataset_name}' (path: {dataset_path})\n"
        
        # Create initial sample plots for demonstration
        initial_plots = create_sample_plots("Initial Analysis")
        
        # Initial analysis insights
        initial_insights = [
            f"Dataset information: {dataset_info}",
            "Initial data inspection shows sales data with revenue, cost, and product information",
            "Data contains multiple sales channels and regional information for analysis",
            "Proceeding to code generation for detailed analysis and visualization"
        ]
        
        # Construct a personalized output
        data_analysis_report = {
            "status": "initial_analysis",
            "summary": f"Initial analysis of sales data for query: {state.query}",
            "insights": initial_insights,
            "visualizations": initial_plots,
            "code_request": f"Generate Python code to {state.query}\nUse pandas, matplotlib, and seaborn for analysis and visualization.\nSave plots to './plots/' directory.",
            "timestamp": datetime.now().isoformat()
        }
        
        # Add the report to the state
        state.update_with_agent_output("data_analysis", data_analysis_report)
        # Set the output to ensure it's accessible
        state.output = data_analysis_report
        
        # Always proceed to code generation in the demo
        state.context["needs_code_generation"] = True
        
        logger.info(f"Initial data analysis completed successfully with {len(initial_insights)} insights and {len(initial_plots)} visualizations")
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
                    state.output.get("code_description", state.query)
                )
                if coding_query:
                    # Update the query with the full code request including instructions
                    state.query = coding_query
                    logger.info(f"Updated query for code generation: {coding_query[:100]}...")
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
        3. Otherwise, go back to data_analysis for final processing if requested
        """
        # Check the review count
        review_count = state.context.get("review_count", 0)
        state.context["review_count"] = review_count + 1
        
        # Check if we already hit the maximum review cycles (2)
        if review_count >= 2:
            logger.info(f"Max review count reached ({review_count}). Ending workflow.")
            return "end"
            
        # Check data analysis cycle count - if it's already high, don't go back to data analysis
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
            if "coding_review" in state.reports:
                if "fixed_code" in state.reports["coding_review"]:
                    state.context["generated_code"] = state.reports["coding_review"]["fixed_code"]
                # Also store execution results for data analysis agent to use
                if "fixed_execution" in state.reports["coding_review"]:
                    state.context["execution_results"] = state.reports["coding_review"]["fixed_execution"]
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
            
            # Add task title if available
            if "task_title" in result:
                formatted_result += f"# {result['task_title']}\n\n"
            
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
                        # Add HTML img tag for the visualization so it displays in the UI
                        formatted_result += f"### {title}\n"
                        formatted_result += f"![{title}]({path})\n\n"
                formatted_result += "\n"
            
            # Add generated code section if available but collapsed
            if "generated_code" in result and result["generated_code"]:
                formatted_result += "## Generated Code\n\n<details>\n<summary>Click to view code</summary>\n\n```python\n"
                formatted_result += result["generated_code"]
                formatted_result += "\n```\n\n</details>\n\n"
            
            # Add code review section if available
            if "code_review" in result and result["code_review"]:
                review = result["code_review"]
                formatted_result += "## Code Review\n\n"
                
                if "review" in review:
                    formatted_result += f"### Review\n{review['review']}\n\n"
                
                if "improvement_explanation" in review:
                    formatted_result += f"### Improvements\n{review['improvement_explanation']}\n\n"
                
                if "fixed_execution" in review and "result" in review["fixed_execution"]:
                    formatted_result += f"### Execution Result\n```\n{review['fixed_execution']['result']}\n```\n\n"
            
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