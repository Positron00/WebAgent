#!/usr/bin/env python
"""
Internal Research Agent Demo
===========================

This demo demonstrates the capabilities of the Internal Research Agent,
which specializes in retrieving information from internal vector databases.

Usage:
    python -m backend.demos.internal_research_demo

Requirements:
    - Configured environment variables (check .env.example)
    - Properly set up vector database with some documents
"""

import os
import sys
import asyncio
import logging
from datetime import datetime
import tempfile
from pathlib import Path
import json
from pprint import pprint

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
logger = logging.getLogger("internal_research_demo")

# Import agents and services
from backend.app.agents.internal_research import get_internal_research_agent
from backend.app.services.document_loader import get_document_loader_service
from backend.app.models.task import WorkflowState
from backend.app.core.config import settings
from backend.app.utils.metrics import timing_decorator

# Import the research results display
from backend.demos.research_results_display import display_research_results

# Path to demo documents
DEMO_DOCS_DIR = os.path.join(current_dir, "demo_docs")
os.makedirs(DEMO_DOCS_DIR, exist_ok=True)

# Sample documents to create
SAMPLE_DOCS = [
    {
        "filename": "ai_ethics_guidelines.txt",
        "content": """
AI Ethics Guidelines - WebAgent Corp - 2025
===========================================

1. Transparency
--------------
All AI systems should provide clear explanations of their capabilities, limitations, and decision-making processes.
Users must be informed when they are interacting with an AI system rather than a human.
Documentation must clearly state what data is being collected and how it's being used.

2. Fairness and Non-discrimination
---------------------------------
AI systems must be designed to treat all users fairly and avoid perpetuating biases.
Regular auditing processes should be implemented to detect and mitigate unfair bias.
Diversity in training data and development teams is essential to creating fair systems.

3. Privacy and Security
---------------------
All AI systems must respect user privacy and implement robust security measures.
Data minimization principles should be followed - only collect what's necessary.
Users should have control over their data, including the right to access, correct, and delete it.

4. Human Oversight
----------------
AI systems should support human autonomy and decision-making, not replace it.
Critical systems must have meaningful human oversight and intervention capabilities.
Regular human review of AI outputs should be conducted for high-risk applications.

5. Accountability
---------------
Clear lines of responsibility must be established for AI system outcomes.
Mechanisms for redress must be available when systems cause harm.
Impact assessments should be conducted before deploying AI in sensitive contexts.
        """
    },
    {
        "filename": "product_roadmap_2025.txt",
        "content": """
WebAgent Product Roadmap 2025
============================

Q1 2025
-------
- Launch WebAgent Enterprise 3.0 with enhanced multi-agent orchestration
- Implement graph database for document relationship tracking
- Improve vector database retrieval with hybrid sparse-dense embeddings
- Release batch document processing API for enterprise customers

Q2 2025
-------
- Introduce advanced fine-tuning capabilities for customer-specific agents
- Develop visual document understanding for diagrams and tables
- Enhance security features with role-based access control
- Launch mobile SDK for WebAgent capabilities

Q3 2025
-------
- Release real-time collaboration features for team-based research
- Implement multi-modal agent capabilities (text, image, audio)
- Develop domain-specific agents for legal, medical, and financial sectors
- Launch WebAgent Cloud with serverless deployment options

Q4 2025
-------
- Introduce agent memory and long-term knowledge retention
- Develop inter-organization knowledge sharing capabilities
- Launch WebAgent Marketplace for custom agent sharing
- Implement advanced analytics dashboard for agent performance

Key Metrics & Goals
------------------
- Increase enterprise customer base by 35%
- Reduce agent hallucination by 65% compared to 2024
- Improve retrieval accuracy by 28%
- Decrease token usage by 15% through optimization
        """
    },
    {
        "filename": "technical_architecture.txt",
        "content": """
WebAgent Technical Architecture Documentation
===========================================

Core Components
--------------
1. Agent Framework
   - Based on LangGraph with custom state management
   - Implements agent communication protocols
   - Handles error recovery and fallback mechanisms
   - Provides agent telemetry and observability

2. Vector Database Layer
   - Primary: ChromaDB for dense vector storage
   - Secondary: FAISS for high-performance similarity search
   - Hybrid retrieval combining BM25 and vector similarity
   - Document chunking with semantic boundaries

3. Graph Database
   - Neo4j for relationship modeling
   - Document-to-document relationships
   - Knowledge graph construction
   - Query-driven graph traversal

4. Model Framework
   - Model Registry for service discovery
   - Model Gateway API for unified access
   - Dedicated services for LLM, embedding, and specialized models
   - Support for model fallbacks and load balancing

5. Security Layer
   - JWT-based authentication
   - Role-based access control
   - Request validation and sanitization
   - Rate limiting and abuse prevention

Deployment Architecture
---------------------
- Containerized microservices with Kubernetes orchestration
- Multi-region deployment with data sovereignty controls
- Blue-green deployment strategy for zero-downtime updates
- Autoscaling based on request volume and model queue length

Integration Interfaces
--------------------
- REST API with OpenAPI documentation
- GraphQL API for complex data requests
- Webhook system for event notifications
- Streaming API for real-time agent interactions
        """
    }
]

def create_sample_documents():
    """Create sample documents for testing."""
    created_files = []
    
    for doc in SAMPLE_DOCS:
        doc_path = os.path.join(DEMO_DOCS_DIR, doc["filename"])
        
        if not os.path.exists(doc_path):
            with open(doc_path, "w") as f:
                f.write(doc["content"])
            logger.info(f"Created sample document: {doc_path}")
        
        created_files.append(doc_path)
        
    return created_files

@timing_decorator
async def run_internal_research_demo():
    """Run the internal research agent demo."""
    logger.info("Starting Internal Research Agent Demo")
    
    # Create sample documents
    sample_doc_paths = create_sample_documents()
    logger.info(f"Created {len(sample_doc_paths)} sample documents")
    
    # Get document loader service
    doc_loader = get_document_loader_service()
    
    # Get internal research agent
    research_agent = get_internal_research_agent()
    
    # Load documents into vector database
    for doc_path in sample_doc_paths:
        try:
            logger.info(f"Loading document: {doc_path}")
            artifact_id = await doc_loader.load_document(doc_path)
            logger.info(f"Document loaded. Artifact ID: {artifact_id}")
        except Exception as e:
            logger.error(f"Error loading document {doc_path}: {e}")
    
    # Define research queries to test
    research_queries = [
        "What are our company's AI ethics guidelines?",
        "What is the product roadmap for 2025?",
        "Explain our technical architecture for vector databases",
        "What security measures are in place in our system?"
    ]
    
    # Store results and errors for display
    all_results = []
    all_errors = []
    
    for idx, query in enumerate(research_queries):
        logger.info(f"\n\n=== QUERY {idx+1}: {query} ===\n")
        
        # Create a workflow state for this query
        state = WorkflowState(
            query=query,
            context={
                "research_priority": "accuracy", 
                "knowledge_source": "internal",
                "include_sources": True
            }
        )
        
        # Run internal research agent
        logger.info(f"Running internal research for: {query}")
        try:
            start_time = datetime.now()
            result_state = await research_agent.run(state)
            duration = (datetime.now() - start_time).total_seconds()
            
            # Print results
            logger.info(f"Internal Research Results (completed in {duration:.2f} seconds):")
            if result_state.output:
                # Format the research output
                if isinstance(result_state.output, dict):
                    logger.info("Summary:")
                    print(result_state.output.get("summary", "No summary provided"))
                    
                    logger.info("\nRetrieved Contexts:")
                    for idx, source in enumerate(result_state.output.get("contexts", [])):
                        print(f"{idx+1}. {source[:100]}...")
                else:
                    pprint(result_state.output)
                
                # Store the result
                all_results.append(result_state.output)
                all_errors.append(None)
            else:
                logger.warning("No research results returned.")
                all_results.append(None)
                all_errors.append(None)
                
        except Exception as e:
            logger.error(f"Error running internal research agent: {e}")
            all_results.append(None)
            all_errors.append(str(e))
    
    logger.info("Internal Research Agent Demo completed.")
    
    # Show results in GUI
    try:
        display_research_results(research_queries, results=all_results, errors=all_errors, title="Internal Research Results")
    except Exception as e:
        logger.error(f"Error displaying results in GUI: {e}")

if __name__ == "__main__":
    # Run the demo
    asyncio.run(run_internal_research_demo()) 