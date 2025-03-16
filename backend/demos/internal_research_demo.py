#!/usr/bin/env python
"""
Internal Research Agent Demo
===========================

This demo demonstrates the capabilities of the Internal Research Agent,
which specializes in retrieving information from internal vector databases.

Usage:
    python -m backend.demos.internal_research_demo [--debug]

Requirements:
    - Configured environment variables (check .env.example)
    - Properly set up vector database with some documents
"""

import os
import sys
import asyncio
import logging
import argparse
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
async def run_internal_research_demo(debug_mode=False):
    """Run the internal research agent demo."""
    logger.info("Starting Internal Research Agent Demo")
    
    # Show debug status
    if debug_mode:
        print("\n" + "=" * 80)
        print("RUNNING IN DEBUG MODE")
        print("Debug mode provides more verbose output and additional diagnostic information")
        print("=" * 80)
    
    # Ask for retrieval mode first
    print("\n" + "=" * 80)
    print("SELECT RETRIEVAL MODE")
    print("=" * 80)
    print("1. Hybrid (vector + LLM, default)")
    print("2. LLM only")
    print("3. Vector only (no LLM)")
    
    mode_choice = input("\nSelect retrieval mode (1-3, default=1): ").strip()
    
    retrieval_mode = "hybrid"
    use_llm = True
    
    if mode_choice == "2":
        retrieval_mode = "llm"
        use_llm = True
        print("\nSelected LLM-only retrieval mode")
    elif mode_choice == "3":
        retrieval_mode = "vector"
        use_llm = False
        print("\nSelected vector-only retrieval mode (no LLM)")
    else:
        print("\nSelected hybrid retrieval mode (vector + LLM)")
    
    # Only ask for LLM provider if using LLM
    llm_provider = "none"
    
    if use_llm:
        print("\n" + "=" * 80)
        print("SELECT LLM PROVIDER")
        print("=" * 80)
        print("1. TogetherAI (default)")
        print("2. Groq")
        print("3. OpenAI")
        print("4. Local Ollama (deepseek-r1:8b)")
        
        provider_choice = input("\nSelect LLM provider (1-4, default=1): ").strip()
        
        llm_provider = "together"
        if provider_choice == "2":
            llm_provider = "groq"
            print("\nSelected Groq as the LLM provider")
        elif provider_choice == "3":
            llm_provider = "openai"
            print("\nSelected OpenAI as the LLM provider")
        elif provider_choice == "4":
            llm_provider = "ollama"
            print("\nSelected local Ollama as the LLM provider")
        else:
            print("\nSelected TogetherAI as the LLM provider")
    
        # Ask for API confirmation if using an external LLM
        if llm_provider in ["together", "groq", "openai"]:
            print("\n" + "=" * 80)
            print(f"This demo will connect to the {llm_provider.upper()} API using your configured API key.")
            try:
                model_info = settings.LLM_PROVIDER if hasattr(settings, 'LLM_PROVIDER') else llm_provider
                model_name = settings.DEFAULT_MODEL if hasattr(settings, 'DEFAULT_MODEL') else {
                    'together': 'meta-llama/Llama-3.3-70B-Instruct-Turbo-Free',
                    'groq': 'llama3-70b-8192',
                    'openai': 'gpt-3.5-turbo',
                    'ollama': 'deepseek-r1:8b'
                }.get(llm_provider, 'unknown')
                print(f"The model that will be used is: {model_name} (Provider: {model_info})")
            except AttributeError:
                print(f"Using default {llm_provider.upper()} configuration")
            print("=" * 80)
            
            confirmation = input("\nDo you want to continue and connect to the LLM API? (Y/n): ")
            if confirmation.lower() not in ["y", "yes",""]:
                logger.info("User declined to connect to the LLM API. Demo aborted.")
                print("Demo aborted by user.")
                return
            
            logger.info(f"User confirmed to connect to the {llm_provider.upper()} API.")
            print(f"\nConnecting to {llm_provider.upper()} API...")
        elif llm_provider == "ollama":
            print("\n" + "=" * 80)
            print("This demo will connect to your local Ollama instance.")
            print("Make sure you have Ollama running with the deepseek-r1:8b model pulled.")
            print("=" * 80)
            
            confirmation = input("\nDo you want to continue and connect to Ollama? (Y/n): ")
            if confirmation.lower() not in ["y", "yes",""]:
                logger.info("User declined to connect to Ollama. Demo aborted.")
                print("Demo aborted by user.")
                return
            
            logger.info("User confirmed to connect to Ollama.")
            print("\nConnecting to local Ollama instance...")
        
        # Set environment variable to control which provider to use
        os.environ["LLM_PROVIDER"] = llm_provider
        os.environ["RETRIEVAL_MODE"] = retrieval_mode
    else:
        # Vector-only mode doesn't need API key
        print("\nRunning in vectorstore-only mode (no LLM API calls)")
        os.environ["RETRIEVAL_MODE"] = "vector"
    
    # Create sample documents
    sample_doc_paths = create_sample_documents()
    logger.info(f"Created {len(sample_doc_paths)} sample documents")
    
    # Get document loader service
    doc_loader = get_document_loader_service()
    
    # Get internal research agent
    if not debug_mode:
        research_agent = get_internal_research_agent()
    else:
        # Create the agent with debug configuration directly
        from backend.app.agents.internal_research import InternalResearchAgent
        research_agent = InternalResearchAgent(config={
            "debug": True,
            "verbose": True
        })
        print(f"Research agent created in debug mode with verbose output")

    # Add debug hooks to monitor retrieval process
    if debug_mode:
        # Get document retriever from the agent to monitor results
        retriever = getattr(research_agent, "_retriever", None)
        if retriever:
            # Hook into the retrieval process to monitor results
            original_get_relevant_documents = retriever.get_relevant_documents
            
            async def debug_get_relevant_documents(*args, **kwargs):
                documents = await original_get_relevant_documents(*args, **kwargs)
                
                # Print retrieved document details for debugging
                print("\n" + "=" * 40)
                print(f"RETRIEVAL RESULTS (found {len(documents)} documents):")
                print("=" * 40)
                
                for i, doc in enumerate(documents):
                    doc_content = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                    doc_meta = doc.metadata if hasattr(doc, "metadata") else {}
                    print(f"\nDOCUMENT {i+1}:")
                    print(f"Source: {doc_meta.get('source', 'Unknown')}")
                    print(f"Content preview: {doc_content}")
                    print("-" * 40)
                
                # Monitor reranker if available
                reranker = getattr(research_agent, "_reranker", None)
                if reranker and documents:
                    print("\nRERRANKER SCORES:")
                    print("-" * 40)
                    # Force reranking immediately for debugging
                    try:
                        query = args[0] if args else "unknown query"
                        top_docs = reranker.rerank(query, documents, top_n=min(5, len(documents)))
                        
                        # Check for NaN scores
                        has_nan_scores = check_for_nan_scores(top_docs)
                        
                        if has_nan_scores:
                            print("Detected NaN scores in reranker output. Using custom ranking instead.")
                            top_docs = custom_rank_documents(query, documents, top_n=min(5, len(documents)))
                        
                        # Print full document details with proper scores
                        for i, (doc, score) in enumerate(top_docs):
                            print(f"\n{i+1}. Score: {'N/A' if score != score else f'{score:.4f}'}")  # Check for NaN
                            
                            # Show document details
                            doc_content = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                            print(f"--- Document Content ---\n{doc_content}\n")
                            
                            # Show metadata if available
                            if hasattr(doc, "metadata") and doc.metadata:
                                print(f"--- Metadata ---")
                                for key, value in doc.metadata.items():
                                    print(f"{key}: {value}")
                    except Exception as e:
                        print(f"Error in reranker: {e}")
                        print("Using custom document ranking as fallback...")
                        query = args[0] if args else "unknown query"
                        top_docs = custom_rank_documents(query, documents, top_n=min(5, len(documents)))
                        
                        for i, (doc, score) in enumerate(top_docs):
                            print(f"\n{i+1}. Score: {score:.4f} (custom ranking)")
                            doc_content = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                            print(f"--- Document Content ---\n{doc_content}\n")
                
                print("=" * 40 + "\n")
                return documents
            
            # Replace with debugging version
            retriever.get_relevant_documents = debug_get_relevant_documents
        else:
            print("Warning: Could not get retriever from research agent for debugging")

    # Load documents into vector database
    for doc_path in sample_doc_paths:
        try:
            logger.info(f"Loading document: {doc_path}")
            artifact_id = await doc_loader.load_document(doc_path)
            logger.info(f"Document loaded. Artifact ID: {artifact_id}")
        except Exception as e:
            logger.error(f"Error loading document {doc_path}: {e}")
    
    # Display document information
    print("\n" + "=" * 80)
    print("LOADED DOCUMENTS:")
    print("=" * 80)
    for i, doc_path in enumerate(sample_doc_paths):
        doc_name = os.path.basename(doc_path)
        doc_size = os.path.getsize(doc_path)
        print(f"{i+1}. {doc_name} ({doc_size/1024:.1f} KB)")
    print("=" * 80 + "\n")
    
    # Add a debug inspection of the vector database content
    try:
        # Access the vector database from the agent if possible
        vectordb = getattr(research_agent, "_vectordb", None)
        if vectordb:
            collection_name = "default"  # This may need to be adjusted based on your setup
            if hasattr(vectordb, "get_collection"):
                collection = vectordb.get_collection(collection_name)
                if collection:
                    # Get count of documents in the collection
                    count = collection.count()
                    print(f"Vector database contains {count} chunks/documents")
                    
                    # If available, show a sample to verify embeddings were created
                    if count > 0 and hasattr(collection, "peek"):
                        print("\nSample from vector database:")
                        sample = collection.peek(5)  # Show up to 5 samples
                        for i, doc in enumerate(sample):
                            print(f"Document {i+1}: {doc[:150]}..." if len(doc) > 150 else doc)
                    print("=" * 80 + "\n")
    except Exception as e:
        print(f"Error inspecting vector database: {e}")
    
    # Debug: Check vectordb directly if in debug mode
    if debug_mode:
        print("\n" + "=" * 80)
        print("DEBUG: CHECKING VECTOR DATABASE DIRECTLY")
        print("=" * 80)
        
        try:
            # First test vectordb access
            from backend.app.services.vectordb import get_vectordb_service
            
            vectordb_service = get_vectordb_service()
            if vectordb_service:
                print("Successfully connected to vector database service")
                
                # Get document count
                doc_count = 0
                try:
                    # Try different ways to access the vectordb
                    # Method 1: try to access the internal _vectordb attribute
                    vectordb = getattr(vectordb_service, "_vectordb", None)
                    if not vectordb:
                        # Method 2: call the retrieve function to see if it works
                        print("Testing retrieval functionality directly...")
                        from backend.app.services.vectordb import retrieve_documents
                        test_docs = asyncio.run(retrieve_documents("ethics", k=2))
                        if test_docs:
                            print(f"Successfully retrieved {len(test_docs)} documents via retrieve_documents")
                            for i, doc in enumerate(test_docs):
                                print(f"\nRetrieved document {i+1}:")
                                content = doc.get("page_content", "")[:150]
                                print(f"{content}..." if len(content) > 150 else content)
                        else:
                            print("No documents found via direct retrieval")
                    else:
                        print(f"Successfully accessed vector database: {type(vectordb).__name__}")
                        # Try to get stats if available
                        if hasattr(vectordb, "get_collection"):
                            collection = vectordb.get_collection()
                            if collection:
                                count = collection.count()
                                print(f"Collection has {count} documents")
                        elif hasattr(vectordb, "_collection"):
                            count = getattr(vectordb._collection, "count", lambda: "unknown")()
                            print(f"Collection has {count} documents")
                except Exception as vdb_err:
                    print(f"Error accessing vector database: {vdb_err}")
            else:
                print("Failed to get vector database service")
        except Exception as e:
            print(f"Error inspecting vector database: {e}")
    
    # Define research queries
    research_queries = [
        "What are our guidelines for AI ethics?",
        "What is the technical architecture of WebAgent?",
        "What products are we planning to release in 2025?",
        # Add a simple test query that should clearly match documents we know exist
        "ethics AI" if debug_mode else None  
    ]

    # Filter out None values
    research_queries = [q for q in research_queries if q is not None]
    
    # Store results and errors for display
    all_results = []
    all_errors = []
    
    # Control whether to display results in GUI during processing
    display_results = True
    
    total_queries = len(research_queries)
    print(f"\nProcessing {total_queries} research queries using {retrieval_mode} retrieval mode...\n")
    
    for idx, query in enumerate(research_queries):
        logger.info(f"\n\n=== QUERY {idx+1}: {query} ===\n")
        
        # Show progress
        print(f"\n[{idx+1}/{total_queries}] Processing: {query}")
        
        # Create a workflow state for this query
        state = WorkflowState(
            query=query,
            context={
                "research_priority": "accuracy", 
                "knowledge_source": "internal",
                "include_sources": True,
                "retrieval_mode": retrieval_mode
            }
        )
        
        # Run internal research agent
        logger.info(f"Running internal research for: {query}")
        logger.info(f"Using retrieval mode: {retrieval_mode}")
        print(f"Running query with {retrieval_mode} retrieval mode...")
        
        # If we're in vector-only mode, skip the agent and just do direct retrieval
        if retrieval_mode == "vector":
            print("Using direct vector retrieval without LLM processing")
            try:
                start_time = datetime.now()
                documents = []
                
                # Try multiple approaches to get documents
                try:
                    # First try: Use the vectordb service directly
                    print("Attempting to retrieve documents via vectordb service...")
                    from backend.app.services.vectordb import retrieve_documents
                    documents = await retrieve_documents(query, k=10)
                    if documents and len(documents) > 0:
                        print(f"Retrieved {len(documents)} documents via vectordb service")
                except Exception as e1:
                    print(f"Error using vectordb service: {e1}")
                    
                    # Second try: Get from research agent's retriever
                    try:
                        print("Attempting to retrieve documents via research agent's retriever...")
                        if hasattr(research_agent, "_retriever") and research_agent._retriever:
                            documents = await research_agent._retriever.get_relevant_documents(query)
                            if documents and len(documents) > 0:
                                print(f"Retrieved {len(documents)} documents via agent's retriever")
                    except Exception as e2:
                        print(f"Error using agent's retriever: {e2}")
                        
                        # Third try: Direct vectordb access
                        try:
                            print("Attempting direct vectordb access...")
                            from backend.app.services.vectordb import get_vectordb_service
                            vectordb_service = get_vectordb_service()
                            if hasattr(vectordb_service, "_vectordb") and vectordb_service._vectordb:
                                documents = await asyncio.to_thread(
                                    vectordb_service._vectordb.similarity_search,
                                    query, k=10
                                )
                                if documents and len(documents) > 0:
                                    print(f"Retrieved {len(documents)} documents via direct vectordb access")
                        except Exception as e3:
                            print(f"Error using direct vectordb access: {e3}")
                
                # If we still don't have documents, try one last approach with a simple search
                if not documents or len(documents) == 0:
                    print("No documents found through standard methods. Trying manual search...")
                    
                    # Create a simple search function
                    def simple_search(query, sample_docs):
                        results = []
                        query_terms = query.lower().split()
                        for doc in sample_docs:
                            content = doc["content"].lower()
                            # Check if any query term is in the content
                            if any(term in content for term in query_terms):
                                from langchain.schema import Document
                                results.append(Document(
                                    page_content=doc["content"],
                                    metadata={"source": doc["filename"]}
                                ))
                        return results
                    
                    # Use our sample documents as a last resort
                    documents = simple_search(query, SAMPLE_DOCS)
                    if documents and len(documents) > 0:
                        print(f"Found {len(documents)} documents via manual search")
                
                if not documents or len(documents) == 0:
                    raise ValueError("No documents found through any retrieval method")
                
                # Convert documents to a consistent format if needed
                processed_docs = []
                for doc in documents:
                    if hasattr(doc, "page_content"):
                        # LangChain Document objects
                        processed_docs.append(doc)
                    elif isinstance(doc, dict) and "page_content" in doc:
                        # Dictionary with page_content
                        from langchain.schema import Document
                        processed_docs.append(Document(
                            page_content=doc["page_content"],
                            metadata=doc.get("metadata", {})
                        ))
                    elif isinstance(doc, str):
                        # Plain text string
                        from langchain.schema import Document
                        processed_docs.append(Document(
                            page_content=doc,
                            metadata={}
                        ))
                
                # Use the processed documents from now on
                documents = processed_docs
                
                # Create simple result with available documents
                summary = f"Found {len(documents)} documents related to '{query}'"
                contexts = [doc.page_content for doc in documents]
                key_points = []
                
                # Extract some key points from the document content
                for i, doc in enumerate(documents):
                    if hasattr(doc, "page_content"):
                        # Try to extract a few sentences as key points
                        content = doc.page_content
                        sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 20]
                        if sentences:
                            # Take up to 2 sentences from each document
                            for s in sentences[:2]:
                                key_points.append(s + ".")
                
                # Create output data
                result_data = {
                    "summary": summary,
                    "contexts": contexts,
                    "key_points": key_points[:5],  # Limit to 5 key points
                    "source": "vector_only_mode"
                }
                
                # Create result state
                result_state = WorkflowState(
                    query=query,
                    output=result_data
                )
                
                duration = (datetime.now() - start_time).total_seconds()
                print(f"Vector retrieval completed in {duration:.2f}s")
                
                # Store result
                all_results.append(result_state)
                all_errors.append(None)
                
                # Display the results using the result display module
                if display_results:
                    # Create a simple formatted output
                    formatted_results = {
                        "query": query,
                        "summary": result_data["summary"],
                        "key_points": result_data["key_points"],
                        "text_chunks": []
                    }
                    
                    # Add document chunks
                    for i, doc in enumerate(documents):
                        if hasattr(doc, "page_content") and doc.page_content:
                            chunk = {
                                "text": doc.page_content,
                                "source": getattr(doc, "metadata", {}).get("source", f"Document {i+1}"),
                                "relevance": 1.0  # No actual scoring in vector-only mode
                            }
                            formatted_results["text_chunks"].append(chunk)
                    
                    # Display results in a new window
                    print(f"Displaying results for query: {query}")
                    display_research_results(formatted_results)
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error in vector-only mode: {error_msg}")
                all_results.append(None)
                all_errors.append(error_msg)
                continue
        else:
            # Use the regular agent with LLM
            try:
                start_time = datetime.now()
                
                # Set environment variables based on the selected provider
                if llm_provider == "together":
                    os.environ["RESEARCH_LLM_PROVIDER"] = "together"
                elif llm_provider == "openai":
                    os.environ["RESEARCH_LLM_PROVIDER"] = "openai"
                elif llm_provider == "groq":
                    os.environ["RESEARCH_LLM_PROVIDER"] = "groq"
                elif llm_provider == "ollama":
                    os.environ["RESEARCH_LLM_PROVIDER"] = "ollama"
                    os.environ["OLLAMA_MODEL"] = "deepseek-r1:8b"
                
                # Fix reranker NaN issue by adding our own reranking if needed
                # First get the documents from the regular pipeline
                print("\nRunning research agent workflow...")
                
                try:
                    # First try with the regular agent
                    result_state = await research_agent.run(state)
                    
                    # Get the result from the state
                    result_data = result_state.output
                    if not result_data:
                        raise ValueError("Research agent returned no results")
                        
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Error in research agent: {error_msg}")
                    print(f"ERROR: {error_msg}")
                    
                    # If there's a NaN issue or the provider fails, try our backup approach
                    if "nan" in error_msg.lower() or "404" in error_msg or "api" in error_msg.lower():
                        print("\nDetected reranker issue or API failure. Attempting alternative approach...")
                        
                        # Extract documents ourselves and create a simpler result
                        try:
                            # Get documents from the agent's retriever
                            retriever = getattr(research_agent, "_retriever", None)
                            if not retriever:
                                raise ValueError("Could not access retriever")
                                
                            # Get documents directly
                            documents = await retriever.get_relevant_documents(query)
                            if not documents:
                                raise ValueError("No documents found")
                                
                            print(f"Retrieved {len(documents)} documents directly")
                            
                            # Apply our custom ranking
                            ranked_docs = custom_rank_documents(query, documents, top_n=5)
                            print(f"Applied custom ranking to documents")
                            
                            # Extract document contents from ranked results
                            contexts = []
                            key_points = []
                            
                            for doc, score in ranked_docs:
                                # Add document content
                                if hasattr(doc, "page_content"):
                                    contexts.append(doc.page_content)
                                    
                                    # Extract key points from content
                                    content = doc.page_content
                                    sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 20]
                                    if sentences:
                                        # Take up to 2 sentences from each document
                                        for s in sentences[:2]:
                                            key_points.append(s + ".")
                            
                            # Create a summary based on the found documents
                            summary = f"Found {len(ranked_docs)} relevant documents related to '{query}'"
                            if key_points:
                                summary += f". Key topics include: {', '.join(key_points[:2])}"
                            
                            # Create result data
                            result_data = {
                                "summary": summary,
                                "contexts": contexts,
                                "key_points": key_points[:5],  # Limit to 5 key points
                                "text_chunks": [
                                    {
                                        "text": doc.page_content if hasattr(doc, "page_content") else str(doc),
                                        "source": getattr(doc, "metadata", {}).get("source", f"Document {i+1}"),
                                        "relevance": score
                                    } for i, (doc, score) in enumerate(ranked_docs)
                                ],
                                "source": "direct_retrieval_fallback"
                            }
                            
                            # Create result state
                            result_state = WorkflowState(
                                query=query,
                                output=result_data
                            )
                            
                            print("Successfully created results using fallback approach")
                            
                        except Exception as fallback_error:
                            # If fallback also fails, re-raise the original error
                            logger.error(f"Fallback approach failed: {fallback_error}")
                            raise ValueError(f"Original error: {error_msg}. Fallback error: {fallback_error}")
                
                duration = (datetime.now() - start_time).total_seconds()
                print(f"Research completed in {duration:.2f}s")
                
                # Store results
                all_results.append(result_state)
                all_errors.append(None)
                
                # Display the results using the result display module
                if display_results and result_data:
                    print(f"Displaying results for query: {query}")
                    display_research_results(result_data)
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error in Internal Research Agent: {error_msg}")
                print(f"ERROR: {error_msg}")
                all_results.append(None)
                all_errors.append(error_msg)
        
        # Pause to avoid rate limits or to let the user read the results
        if idx < len(research_queries) - 1:
            delay_seconds = 1.5  # Slightly longer delay to be safe
            logger.info(f"Waiting {delay_seconds} seconds before next query to avoid rate limiting...")
            print(f"Waiting {delay_seconds} seconds to avoid API rate limits...")
            await asyncio.sleep(delay_seconds)
    
    logger.info("Internal Research Agent Demo completed.")
    
    # Show results in GUI
    try:
        display_research_results(research_queries, results=all_results, errors=all_errors, title="Internal Research Results")
    except Exception as e:
        logger.error(f"Error displaying results in GUI: {e}")

def custom_rank_documents(query, documents, top_n=5):
    """
    Simple custom ranker to handle cases where the neural reranker fails.
    Performs simple keyword matching and TF-IDF style ranking.
    
    Args:
        query: The search query
        documents: List of documents to rank
        top_n: Number of top documents to return
        
    Returns:
        List of (document, score) tuples
    """
    if not documents:
        return []
    
    # Process query - extract keywords
    query_terms = [term.lower() for term in query.split() if len(term) > 2]
    
    # Score each document
    scored_docs = []
    for doc in documents:
        score = 0
        
        # Get the document text
        if hasattr(doc, "page_content"):
            text = doc.page_content.lower()
        elif isinstance(doc, dict) and "page_content" in doc:
            text = doc["page_content"].lower()
        elif isinstance(doc, str):
            text = doc.lower()
        else:
            text = str(doc).lower()
        
        # Count keyword matches
        for term in query_terms:
            term_count = text.count(term)
            score += term_count
        
        # Boost score based on exact query match
        if query.lower() in text:
            score += 5
        
        # Normalize by document length (to avoid favoring long docs)
        text_len = max(len(text.split()), 1)  # Avoid division by zero
        score = score / (0.5 + 0.5 * (text_len / 100))  # Smoothed normalization
        
        scored_docs.append((doc, score))
    
    # Sort by score (descending) and take top_n
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return scored_docs[:top_n]


def check_for_nan_scores(reranked_documents):
    """
    Check if any reranked documents have NaN scores
    
    Args:
        reranked_documents: List of documents with scores
        
    Returns:
        True if any scores are NaN, False otherwise
    """
    for doc in reranked_documents:
        if isinstance(doc, tuple) and len(doc) >= 2:
            score = doc[1]
            if score != score:  # NaN check
                return True
        elif isinstance(doc, dict) and "score" in doc:
            score = doc["score"]
            if score != score:  # NaN check
                return True
    
    return False

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the internal research agent demo")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    # Run the demo
    asyncio.run(run_internal_research_demo(args.debug)) 