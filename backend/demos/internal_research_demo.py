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

Robustness Improvements:
-----------------------
1. Multiple retrieval modes:
   - Hybrid mode: Uses vector retrieval + LLM for comprehensive results
   - LLM-focused mode: Prioritizes LLM reasoning over vector retrieval
   - Vector-only mode: Works without an LLM for offline or API-limited scenarios

2. Error handling:
   - Graceful fallback to vector-only mode if LLM API fails
   - Detailed error messages for different API failure scenarios
   - Automatic retry with different LLM providers if one fails

3. Fallback components:
   - SimpleRetriever: Keyword-based document retrieval without vector DB
   - SimpleReranker: Custom document ranking without ML models
   - Custom document ranking based on term frequency and document structure

4. Provider flexibility:
   - Support for multiple LLM providers (TogetherAI, OpenAI, Groq, Ollama)
   - Configurable model selection for each provider
   - Automatic provider fallback in case of API issues
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
import re
from typing import List, Dict, Tuple, Any, Optional, Union
import time

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
from backend.app.services.document_ingestion_service import get_document_ingestion_service
from backend.app.models.task import WorkflowState
from backend.app.core.config import settings
from backend.app.utils.metrics import timing_decorator
from backend.app.services.llm import confirm_llm_selection

# Import the research results display
from backend.demos.research_results_display import display_research_results

# Path to demo documents
DEMO_DOCS_DIR = os.path.join(current_dir, "demo_docs")
os.makedirs(DEMO_DOCS_DIR, exist_ok=True)

class SimpleRetriever:
    """
    A simple retriever that searches through sample documents.
    Used as a fallback when the research agent's retriever is not available.
    """
    def __init__(self, sample_docs):
        self.sample_docs = sample_docs
        logger.info(f"Initialized SimpleRetriever with {len(sample_docs)} sample documents")
    
    async def get_relevant_documents(self, query, **kwargs):
        """
        Get relevant documents based on simple keyword matching.
        
        Args:
            query: The search query
            **kwargs: Additional arguments
            
        Returns:
            List of relevant documents
        """
        from langchain.schema import Document
        
        logger.info(f"SimpleRetriever searching for: {query}")
        
        # Break query into terms
        query_terms = [term.lower() for term in query.split() if len(term) > 2]
        
        # Search through sample documents
        results = []
        for doc_path in self.sample_docs:
            try:
                with open(doc_path, 'r') as f:
                    content = f.read()
                
                # Check if any query term is in the document
                matches = 0
                for term in query_terms:
                    if term in content.lower():
                        matches += content.lower().count(term)
                
                # If there are matches, add to results
                if matches > 0:
                    doc_name = os.path.basename(doc_path)
                    results.append(Document(
                        page_content=content,
                        metadata={
                            "source": doc_name,
                            "matches": matches,
                            "path": doc_path
                        }
                    ))
            except Exception as e:
                logger.error(f"Error reading document {doc_path}: {e}")
        
        # Sort by number of matches (descending)
        results.sort(key=lambda x: x.metadata.get("matches", 0), reverse=True)
        
        logger.info(f"SimpleRetriever found {len(results)} relevant documents")
        return results

class SimpleReranker:
    """
    A simple reranker that ranks documents based on relevance to a query.
    Used as a fallback when the research agent's reranker is not available.
    """
    def __init__(self):
        logger.info("Initialized SimpleReranker")
    
    async def rerank(self, query, documents, top_n=5):
        """
        Rerank documents based on relevance to the query.
        
        Args:
            query: The search query
            documents: List of documents to rerank
            top_n: Number of top documents to return
            
        Returns:
            List of reranked documents
        """
        logger.info(f"SimpleReranker reranking {len(documents)} documents")
        
        # Use the custom ranking function
        ranked_docs = custom_rank_documents(query, documents, top_n=top_n)
        
        # Convert to the expected format
        results = []
        for doc, score in ranked_docs:
            # Create a dictionary with the expected fields
            doc_dict = {
                "page_content": doc.page_content if hasattr(doc, "page_content") else str(doc),
                "metadata": getattr(doc, "metadata", {}),
                "score": score
            }
            results.append(doc_dict)
        
        logger.info(f"SimpleReranker returned {len(results)} reranked documents")
        return results

async def process_vector_only_query(query, retriever, reranker, top_n=5):
    """
    Process a query using only vector retrieval and reranking, without an LLM.
    
    Args:
        query: The search query
        retriever: The retriever to use
        reranker: The reranker to use
        top_n: Number of top documents to return
        
    Returns:
        Dictionary with query results
    """
    logger.info(f"Processing query in vector-only mode: {query}")
    start_time = time.time()
    
    try:
        # 1. Retrieve documents
        logger.info("Retrieving documents")
        documents = await retriever.retrieve(query, k=20)
        
        if not documents:
            logger.warning("No documents found")
            return {
                "status": "completed",
                "documents": [],
                "findings": "No relevant documents found for the query.",
                "query": query,
                "retrieval_time": time.time() - start_time
            }
        
        # 2. Rerank documents
        logger.info(f"Reranking {len(documents)} documents")
        reranked_docs = await reranker.rerank(query, documents, top_n=top_n)
        
        # 3. Format results
        result = {
            "status": "completed",
            "documents": reranked_docs,
            "findings": f"Found {len(reranked_docs)} relevant documents for the query.",
            "query": query,
            "retrieval_time": time.time() - start_time
        }
        
        logger.info(f"Vector-only processing completed in {time.time() - start_time:.2f}s")
        return result
        
    except Exception as e:
        logger.error(f"Error in vector-only processing: {e}")
        return {
            "status": "error",
            "error": str(e),
            "query": query,
            "retrieval_time": time.time() - start_time
        }

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
    print("SELECT RAG MODE")
    print("=" * 80)
    print("1. Hybrid (vector + semantic reranking + LLM, default)")
    print("2. LLM-focused (more weight on LLM reasoning)")
    print("3. Vector-only (no LLM required, fallback mode)")
    
    mode_choice = input("\nSelect RAG mode (1-3, default=1): ").strip()
    
    retrieval_mode = "hybrid"
    
    if mode_choice == "2":
        retrieval_mode = "llm"
        print("\nSelected LLM-focused retrieval mode")
    elif mode_choice == "3":
        retrieval_mode = "vector"
        print("\nSelected vector-only retrieval mode (no LLM required)")
    else:
        print("\nSelected hybrid retrieval mode (balanced vector + LLM)")
    
    # LLM provider selection - only if not in vector-only mode
    if retrieval_mode != "vector":
        print("\n" + "=" * 80)
        print("INTERNAL RESEARCH AGENT DEMO")
        print("=" * 80)
        print("This demo requires an LLM for processing research results.")
        print("The default LLM provider is TogetherAI with the Llama-3.3-70B model.")
        
        # Use standardized LLM selection function
        try:
            provider, model = confirm_llm_selection()
            logger.info(f"User selected LLM provider: {provider}")
            logger.info(f"User selected model: {model}")
            
            # Set environment variable for research agent
            os.environ["LLM_PROVIDER"] = provider
            os.environ["RESEARCH_LLM_PROVIDER"] = provider
            
            # Set provider-specific model environment variables
            if provider == "together":
                os.environ["TOGETHER_MODEL"] = model
            elif provider == "groq":
                os.environ["GROQ_MODEL"] = model
            elif provider == "openai":
                os.environ["OPENAI_MODEL"] = model
            elif provider == "anthropic":
                os.environ["ANTHROPIC_MODEL"] = model
            elif provider == "gemini":
                os.environ["GEMINI_MODEL"] = model
            elif provider == "ollama" or provider == "self":
                os.environ["OLLAMA_MODEL"] = model
                os.environ["SELF_HOSTED_MODEL"] = model
        except Exception as e:
            logger.error(f"Error during LLM selection: {str(e)}")
            print(f"Error selecting LLM: {e}")
            print("Falling back to vector-only mode.")
            retrieval_mode = "vector"
    else:
        # In vector-only mode, we don't need an LLM provider
        print("\nVector-only mode selected. No LLM API connection required.")
        provider = "none"
        model = "none"
        os.environ["LLM_PROVIDER"] = "none"
    
    # Set retrieval mode environment variable
    os.environ["RETRIEVAL_MODE"] = retrieval_mode
    
    # Create sample documents
    logger.info("Getting sample documents from document ingestion service")
    
    # Get document ingestion service
    doc_ingestion_service = get_document_ingestion_service()
    
    # Use the service to create and get documents
    sample_doc_paths, num_created = await doc_ingestion_service.create_sample_documents()
    logger.info(f"Created {num_created} sample documents")
    #print(f"\nCreated {num_created} sample documents in {doc_ingestion_service.sample_docs_dir}")
    
    # Get document loader service
    doc_loader = get_document_loader_service()
    
    # Create and initialize internal research agent
    research_agent = None
    retriever = None
    reranker = None
    
    try:
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
        
        # Try to access the retriever from the research agent
        try:
            retriever = getattr(research_agent, "retriever", None)
            if retriever:
                logger.info("Successfully accessed research agent's retriever")
            else:
                logger.warning("Research agent's retriever not available")
                retriever = SimpleRetriever(sample_doc_paths)
        except Exception as e:
            logger.error(f"Error accessing research agent's retriever: {e}")
            retriever = SimpleRetriever(sample_doc_paths)
        
        # Try to access the reranker from the research agent
        try:
            reranker = getattr(research_agent, "reranker", None)
            if reranker:
                logger.info("Successfully accessed research agent's reranker")
            else:
                logger.warning("Research agent's reranker not available")
                reranker = SimpleReranker()
        except Exception as e:
            logger.error(f"Error accessing research agent's reranker: {e}")
            reranker = SimpleReranker()
            
    except Exception as e:
        logger.error(f"Error creating research agent: {e}")
        print(f"\nERROR: Failed to initialize research agent: {e}")
        print("Falling back to simple retrieval mode without LLM")
        
        # Create simple retriever and reranker as fallbacks
        retriever = SimpleRetriever(sample_doc_paths)
        reranker = SimpleReranker()
        
        # Force vector-only mode
        retrieval_mode = "vector"
        os.environ["RETRIEVAL_MODE"] = "vector"
        print("\nForced to vector-only mode due to research agent initialization failure")

    # Ingest documents into vector database
    logger.info("Ingesting documents into vector database")
    artifact_ids = await doc_ingestion_service.ingest_documents(doc_loader, sample_doc_paths)
    
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
        #"What is the technical architecture of WebAgent?",
        #"What products are we planning to release in 2025?",
        #"How does our hybrid retrieval system work?",
        #"What are the capabilities of our research agent?",
        #"How does our document processing pipeline work?",
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
        
        # Run internal research agent or vector-only processing
        logger.info(f"Running internal research for: {query}")
        logger.info(f"Using retrieval mode: {retrieval_mode}")
        print(f"Running query with {retrieval_mode} retrieval mode...")
        
        if retrieval_mode == "vector" or research_agent is None:
            # Use vector-only mode
            print("\nRunning vector-only retrieval (no LLM)...")
            
            try:
                # Process the query using vector-only mode
                result = await process_vector_only_query(query, retriever, reranker)
                
                # Store the result
                all_results.append(result)
                all_errors.append(None)
                
                # Print debug info
                if debug_mode:
                    print("\nVector-only retrieval results:")
                    print(f"Found {len(result.get('documents', []))} relevant documents")
                    for i, doc in enumerate(result.get('documents', [])[:3]):
                        print(f"\nDocument {i+1} (Score: {doc.get('score', 0):.4f}):")
                        source = doc.get('metadata', {}).get('source', 'Unknown')
                        print(f"Source: {source}")
                        content = doc.get('page_content', '')[:150]
                        print(f"Content: {content}..." if len(content) > 150 else f"Content: {content}")
                
                # Display the results using the result display module
                if display_results:
                    print(f"Displaying results for query: {query}")
                    # Comment out the individual result display to avoid extra popups
                    # display_research_results(result)
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error in vector-only processing: {error_msg}")
                print(f"ERROR: {error_msg}")
                all_results.append(None)
                all_errors.append(error_msg)
                
                # Display the results
                if display_results:
                    print(f"Displaying vector-only results for query: {query}")
                    # Comment out the individual result display to avoid extra popups
                    # display_research_results(result)
        else:
            # Use the research agent
            print("\nRunning full RAG pipeline (retrieve -> rerank -> generate)...")
            
            try:
                # Run the research agent
                start_time = datetime.now()
                updated_state = await research_agent.run(state)
                
                # Try to extract result data
                agent_output = updated_state.reports.get("internal_research", {})
                result_data = agent_output.get("findings", None)
                documents = agent_output.get("documents", [])
                
                if not result_data:
                    # Check if there was an error message in the findings
                    findings = agent_output.get("findings", "")
                    if findings and "Error generating research findings:" in findings:
                        logger.error(f"Error in research agent: {findings}")
                        print(f"ERROR: {findings}")
                        all_results.append(None)
                        all_errors.append(findings)
                        
                        # Add a delay to avoid rate limits
                        if idx < len(research_queries) - 1:
                            delay_seconds = 1.5
                            logger.info(f"Waiting {delay_seconds} seconds before next query to avoid rate limiting...")
                            print(f"Waiting {delay_seconds} seconds to avoid API rate limits...")
                            await asyncio.sleep(delay_seconds)
                            
                        continue
                        
                    logger.error("Research agent returned no results")
                    print("ERROR: Research agent returned no results")
                    all_results.append(None)
                    all_errors.append("Research agent returned no results")
                else:
                    # Print debug info about the retrieval in debug mode
                    if debug_mode:
                        print("\nDetailed results:")
                        pprint(result_data, indent=2)
                    
                    duration = (datetime.now() - start_time).total_seconds()
                    print(f"Research completed in {duration:.2f}s")
                    
                    # Store results
                    all_results.append(updated_state)
                    all_errors.append(None)
                    
                    # Display the results using the result display module
                    if display_results and result_data:
                        print(f"Displaying results for query: {query}")
                        # Comment out the individual result display to avoid extra popups
                        # display_research_results(result_data)
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error in Internal Research Agent: {error_msg}")
                print(f"ERROR: {error_msg}")
                all_results.append(None)
                all_errors.append(error_msg)
                
                # If the research agent fails, try vector-only mode as a fallback
                print("\nFalling back to vector-only retrieval...")
                try:
                    result = await process_vector_only_query(query, retriever, reranker)
                    all_results[-1] = result  # Replace the None result with the vector-only result
                    all_errors[-1] = None  # Clear the error
                    
                    # Display the results
                    if display_results:
                        print(f"Displaying vector-only results for query: {query}")
                        # Comment out the individual result display to avoid extra popups
                        # display_research_results(result)
                except Exception as fallback_error:
                    logger.error(f"Error in vector-only fallback: {fallback_error}")
                    print(f"ERROR in fallback: {fallback_error}")
        
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

def check_for_nan_scores(ranked_docs): # not being used?
    """
    Check if any of the ranked documents have NaN scores.
    
    Args:
        ranked_docs: List of (document, score) tuples
        
    Returns:
        Boolean indicating if any NaN scores were found
    """
    if not ranked_docs:
        return False
        
    has_nan_scores = False
    for _, score in ranked_docs:
        if score != score:  # NaN check
            has_nan_scores = True
            break
            
    return has_nan_scores

def custom_rank_documents(query, documents, top_n=10):
    """
    Apply custom ranking to documents based on keyword matching and document structure.
    Used as a fallback when reranker has issues or returns NaN scores.
    
    Args:
        query: The search query
        documents: List of documents to rank
        top_n: Number of top documents to return
        
    Returns:
        List of (document, score) tuples, sorted by relevance
    """
    if not documents:
        return []
        
    # Break query into terms
    query_terms = [term.lower() for term in query.split() if len(term) > 2] # need to preprocess query to remove special characters and stop words
    
    # Score each document
    scored_docs = []
    
    for doc in documents:
        score = 0.0
        
        # Get document content
        if hasattr(doc, "page_content"):
            content = doc.page_content.lower()
        else:
            # Try to convert to string
            try:
                content = str(doc).lower()
            except:
                content = ""
                
        # Skip empty content
        if not content:
            continue
                
        # 1. Score based on term frequency
        for term in query_terms:
            term_count = content.count(term)
            if term_count > 0:
                # Diminishing returns for repeated terms
                score += min(1.0, 0.1 + (term_count / 10.0))
                
        # 2. Score based on exact phrase match
        if query.lower() in content:
            score += 2.0
            
        # 3. Score based on document structure (headings are more important)
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line looks like a heading
            if line.startswith('#') or line.startswith('=') or re.match(r'^[A-Z][a-z]+(\s+[A-Z][a-z]+){0,5}$', line):
                # Check if any query term is in the heading
                if any(term in line.lower() for term in query_terms):
                    score += 1.5
                    
        # 4. Document length penalty (prefer shorter docs slightly)
        length = len(content)
        length_factor = 1.0 if length < 1000 else (1000 / length) ** 0.5
        score *= length_factor
        
        # 5. Metadata bonus
        if hasattr(doc, "metadata"):
            # Check source filename for relevance
            source = doc.metadata.get("source", "")
            if isinstance(source, str):
                source_lower = source.lower()
                # If source filename contains query terms, boost score
                if any(term in source_lower for term in query_terms):
                    score += 1.0
        
        # Add document and its score to the list
        scored_docs.append((doc, score))
    
    # Sort by score (descending)
    sorted_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
    
    # Return top N documents
    return sorted_docs[:top_n]

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the internal research agent demo")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    # Run the demo
    asyncio.run(run_internal_research_demo(args.debug)) 