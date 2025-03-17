"""
Document Ingestion Service
=========================

Service for generating sample documents and ingesting them into the vector database.
This service can be used independently or called by other components when a vector database
needs to be initialized with documents.
"""

import os
import logging
import time
from pathlib import Path
import asyncio

from app.core.config import settings
from app.core.observability import (
    track_document_processing,
    record_document_size,
    record_data_quality,
    log_to_mlflow
)

# Configure logging
logger = logging.getLogger(__name__)

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
    },
    {
        "filename": "research_agent_capabilities.txt",
        "content": """
WebAgent Research Agent Capabilities
===================================

Overview
--------
The Research Agent is a specialized component of WebAgent designed to perform in-depth information retrieval and analysis 
from both internal knowledge bases and external sources. It enables faster, more accurate research with comprehensive citation 
and source verification.

Core Features
------------
1. Multi-modal Information Retrieval
   - Text-based document search and analysis
   - Image understanding and interpretation
   - Audio transcription and content extraction
   - Video content analysis and summarization

2. Advanced RAG Architecture
   - Hybrid retrieval combining dense and sparse methods
   - Multi-stage reranking for precision
   - Contextual compression for improved relevance
   - Adaptive chunking based on semantic boundaries

3. Source Verification and Validation
   - Automated fact-checking against trusted sources
   - Citation generation and validation
   - Information reliability scoring
   - Contradiction and inconsistency detection

4. Research Planning
   - Automated research plan generation
   - Sub-query decomposition for complex topics
   - Adaptive search refinement based on interim results
   - Structured output formatting based on user needs

Integration Capabilities
----------------------
- Direct integration with document management systems
- Citation export in multiple formats (APA, MLA, Chicago, etc.)
- Collaborative research workflows with multiple agents
- Research memory for preserving context across sessions

Performance Metrics
-----------------
- 82% reduction in research time compared to manual methods
- 76% accuracy increase in source retrieval relevance
- 94% user satisfaction rating based on output quality
- 68% reduction in hallucinations compared to standard RAG systems
        """
    },
    {
        "filename": "hybrid_retrieval_system.txt",
        "content": """
Hybrid Retrieval System Documentation
===================================

System Architecture
-----------------
The hybrid retrieval system combines multiple retrieval methods to maximize both recall and precision when searching through document repositories. By leveraging the strengths of different approaches, the system provides more relevant results than any single method.

Component Overview
----------------

1. Dense Vector Retrieval
   - Embedding Model: E5-large-v2 (768 dimensions)
   - Similarity: Cosine similarity with improved compression
   - Index: HNSW with 128 M neighbors and 16 ef_construction
   - Quantization: SQ8 for storage efficiency with minimal accuracy loss
   - Hardware acceleration: CUDA optimization for GPU throughput

2. Sparse Retrieval (Lexical)
   - Primary algorithm: BM25 with tuned parameters (k1=1.5, b=0.75)
   - Term weighting: Enhanced TF-IDF with contextual adjustments
   - Stopword handling: Domain-specific stopword lists
   - Stemming: Snowball stemmer with custom extensions
   - N-gram support: Unigrams, bigrams, and adaptive phrases

3. Reranking System
   - Model: Cross-encoder based on BERT architecture
   - Training: Fine-tuned on domain-specific relevance judgments
   - Score fusion: Reciprocal rank fusion with adaptive weights
   - Ensemble methods: Stacked ranking with linear combination
   - Diversity promotion: Maximum marginal relevance for result variety

Performance Characteristics
-------------------------
- Latency: <100ms for p95 query response (10M document index)
- Throughput: 200+ queries per second per node
- Recall@10: 92% compared to expert human search
- Precision@5: 87% relevance as judged by domain experts
- NDCG: 0.85 average across benchmark test sets

Integration Methods
----------------
- REST API with batch query support
- Streaming response option for incremental result display
- Asynchronous query capabilities with webhook callbacks
- Client libraries in Python, JavaScript, Java, and Go
- Observability hooks with detailed performance telemetry
        """
    },
    {
        "filename": "sample_financial_report.txt",
        "content": """
Financial Report - Q1 2025
==========================

Executive Summary
----------------
In Q1 2025, WebAgent Corp achieved significant growth with revenue reaching $12.5M, 
representing a 15% increase compared to Q1 2024. EBITDA improved by 22% to $3.2M, 
while net profit rose to $1.8M.

Financial Highlights
-------------------
- Revenue: $12.5M (+15% YoY)
- EBITDA: $3.2M (+22% YoY)
- Net Profit: $1.8M (+18% YoY)
- Operating Margin: 25.6% (up from 23.1%)
- Cash Reserves: $8.4M (as of March 31, 2025)

Department Breakdown
-------------------
1. Product Development: $4.2M investment, focused on AI agent capabilities
2. Sales & Marketing: $2.8M, with 65% digital marketing allocation
3. Operations: $1.9M, improved efficiency by 12%
4. Research: $2.1M, primarily for new vector database integration
5. Administration: $1.5M

Market Analysis
--------------
The AI agent market grew by approximately 28% in Q1 2025, with WebAgent gaining 2.3% 
market share. Key competitors include AgentWorks (18.2% share), AI Solutions (15.6%), 
and TechAgents (12.1%).

Outlook for Q2 2025
------------------
We project continued growth in Q2 with revenue estimates between $13.8M-$14.5M and 
EBITDA of $3.5M-$3.8M. Key initiatives include launching the new WebAgent Enterprise 
platform and expanding to three new international markets.
        """
    }
]

class DocumentIngestionService:
    """
    Service for generating sample documents and ingesting them into the vector database.
    """
    
    def __init__(self):
        """Initialize the document ingestion service."""
        self.sample_docs_dir = os.path.join(settings.DATA_DIR, "sample_docs")
        os.makedirs(self.sample_docs_dir, exist_ok=True)
        self.docs_directory = settings.SAMPLE_DOCS_DIR
        self.metrics = {
            "documents_processed": 0,
            "documents_failed": 0,
            "total_tokens": 0,
            "total_chunks": 0,
            "processing_time_seconds": 0
        }
    
    def get_sample_document_path(self, filename):
        """
        Get the path to a specific sample document, creating it if needed.
        
        Args:
            filename: The name of the sample document to get
            
        Returns:
            Path to the sample document
        """
        # Check if the file exists
        doc_path = os.path.join(self.sample_docs_dir, filename)
        
        # If the file doesn't exist, create it
        if not os.path.exists(doc_path):
            # Find the matching document in SAMPLE_DOCS
            for doc in SAMPLE_DOCS:
                if doc["filename"] == filename:
                    # Create the document
                    with open(doc_path, "w") as f:
                        f.write(doc["content"])
                    logger.info(f"Created sample document: {doc_path}")
                    break
            else:
                # If we get here, the document wasn't found in SAMPLE_DOCS
                logger.warning(f"Sample document {filename} not found in SAMPLE_DOCS")
        
        return doc_path
    
    @track_document_processing(doc_type="sample")
    async def create_sample_documents(self):
        """
        Create sample documents for ingestion.
        
        Returns:
            Tuple of (list of document paths, number of created files)
        """
        logger.info(f"Creating sample documents in {self.docs_directory}")
        os.makedirs(self.docs_directory, exist_ok=True)
        
        doc_paths = []
        
        for doc in SAMPLE_DOCS:
            doc_path = self.get_sample_document_path(doc["filename"])
            try:
                with open(doc_path, "w") as f:
                    f.write(doc["content"])
                
                doc_paths.append(doc_path)
                
                # Record document size for metrics
                file_size = os.path.getsize(doc_path)
                record_document_size(doc_type="sample", size_bytes=file_size)
                
                logger.debug(f"Created sample document: {doc_path}")
            except Exception as e:
                logger.error(f"Error creating sample document {doc_path}: {e}")
        
        logger.info(f"Created {len(doc_paths)} sample documents")
        return doc_paths, len(doc_paths)
    
    @track_document_processing(doc_type="ingestion")
    async def ingest_documents(self, doc_loader, doc_paths):
        """
        Ingest documents into the vector database using the document loader.
        
        Args:
            doc_loader: The document loader service
            doc_paths: List of document paths to ingest
            
        Returns:
            List of artifact IDs for the ingested documents
        """
        start_time = time.time()
        logger.info(f"Ingesting {len(doc_paths)} documents into vector database")
        
        successful = 0
        failed = 0
        total_tokens = 0
        total_chunks = 0
        
        # Get vector database service
        from app.services.vectordb import get_vectordb_service
        vector_db = get_vectordb_service()
        
        for i, doc_path in enumerate(doc_paths):
            try:
                doc_name = os.path.basename(doc_path)
                logger.info(f"[{i+1}/{len(doc_paths)}] Processing: {doc_name}")
                
                # Check if document already exists in the vector database
                # First, calculate a simple content hash to use as a check
                try:
                    with open(doc_path, 'r') as f:
                        content = f.read()
                    
                    # Extract metadata to help identify duplicates
                    doc_metadata = {
                        "source": doc_name,
                        "filename": doc_name,
                        "content_hash": hash(content[:200] + str(len(content)) + content[-200:]) if len(content) > 400 else hash(content)
                    }
                    
                    # Try to search for similar content in the vector database
                    search_query = content[:300]  # Use beginning of the document as search query
                    results = await vector_db.search(search_query, k=5)
                    
                    # Check if any results match our document by filename or content similarity
                    doc_exists = False
                    for doc, score in results:
                        result_filename = doc.metadata.get("source", "").split("/")[-1] if isinstance(doc.metadata.get("source", ""), str) else ""
                        
                        # Check if filenames match exactly
                        if result_filename == doc_name:
                            logger.info(f"Document '{doc_name}' already exists in vector database (filename match)")
                            doc_exists = True
                            break
                            
                        # Check if content is very similar (first 200 + last 200 chars)
                        if len(doc.page_content) > 400 and len(content) > 400:
                            if doc.page_content[:200] == content[:200] and doc.page_content[-200:] == content[-200:]:
                                logger.info(f"Document '{doc_name}' already exists in vector database (content match)")
                                doc_exists = True
                                break
                    
                    if doc_exists:
                        logger.info(f"Skipping document '{doc_name}' as it already exists in vector database")
                        failed += 1
                        continue
                    
                except Exception as check_err:
                    logger.warning(f"Error checking if document exists: {str(check_err)}. Proceeding with ingestion.")
                
                # Document doesn't exist, load it into vector database
                logger.info(f"Ingesting new document: {doc_name}")
                result = await doc_loader.ingest_document(doc_path)
                
                successful += 1
                total_tokens += result.get("tokens", 0)
                total_chunks += result.get("chunks", 0)
                
                # Basic data quality check - evaluate chunk quality
                if result.get("chunks", 0) > 0:
                    avg_chunk_size = result.get("tokens", 0) / result.get("chunks", 0)
                    # Score is 0-100, higher is better
                    # Assuming ideal chunk size is between 200-500 tokens
                    if 150 <= avg_chunk_size <= 600:
                        quality_score = 100
                    elif 100 <= avg_chunk_size < 150 or 600 < avg_chunk_size <= 800:
                        quality_score = 80
                    elif 50 <= avg_chunk_size < 100 or 800 < avg_chunk_size <= 1000:
                        quality_score = 60
                    else:
                        quality_score = 40
                    
                    record_data_quality(
                        dataset="document_chunks", 
                        metric_type="chunk_size", 
                        score=quality_score
                    )
                
                logger.info(f"Successfully ingested document: {doc_name} (ID: {result.get('artifact_id')})")
            except Exception as e:
                failed += 1
                logger.error(f"Error loading document {doc_path}: {e}")
                logger.error(f"Failed to ingest document: {doc_path}")
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Update metrics
        self.metrics["documents_processed"] += successful
        self.metrics["documents_failed"] += failed
        self.metrics["total_tokens"] += total_tokens
        self.metrics["total_chunks"] += total_chunks
        self.metrics["processing_time_seconds"] += processing_time
        
        # Log to MLflow if available
        log_to_mlflow(
            metrics={
                "documents_processed": successful,
                "documents_failed": failed,
                "total_tokens": total_tokens,
                "total_chunks": total_chunks,
                "processing_time_seconds": processing_time,
                "avg_tokens_per_document": total_tokens / max(1, successful),
                "avg_chunks_per_document": total_chunks / max(1, successful),
                "documents_per_second": successful / max(0.1, processing_time)
            },
            params={
                "document_count": len(doc_paths),
                "document_type": "sample"
            },
            experiment_name="document_ingestion"
        )
        
        logger.info("Document ingestion complete.")
        logger.info(f"Successfully ingested {successful} documents out of {len(doc_paths)} documents.")
        
        return successful
    
    async def generate_and_ingest_documents(self):
        """
        Generate sample documents and ingest them into the vector database.
        
        Returns:
            Tuple of (number of created documents, number of ingested documents)
        """
        # Create sample documents
        logger.info("Creating sample documents")
        sample_doc_paths, num_created = await self.create_sample_documents()
        logger.info(f"Created {num_created} sample documents")
        
        # Get document loader service
        from app.services.document_loader import get_document_loader_service
        logger.info("Initializing document loader service")
        doc_loader = get_document_loader_service()
        
        # Ingest documents into vector database
        logger.info("Ingesting documents into vector database")
        num_ingested = await self.ingest_documents(doc_loader, sample_doc_paths)
        
        # Display document information
        logger.info("LOADED DOCUMENTS:")
        for i, doc_path in enumerate(sample_doc_paths):
            doc_name = os.path.basename(doc_path)
            doc_size = os.path.getsize(doc_path)
            logger.info(f"{i+1}. {doc_name} ({doc_size/1024:.1f} KB)")
        
        return num_created, num_ingested
    
    async def run_document_ingestion(self):
        """
        Run the document generation and ingestion process.
        
        Returns:
            Dict with status and ingestion details
        """
        logger.info("Starting document generation and ingestion")
        
        start_time = time.time()
        
        try:
            # Generate and ingest documents
            num_created, num_ingested = await self.generate_and_ingest_documents()
            
            # Log completion
            elapsed_time = time.time() - start_time
            logger.info(f"Document generation and ingestion completed in {elapsed_time:.2f} seconds")
            logger.info(f"Created {num_created} documents, ingested {num_ingested} documents")
            
            return {
                "status": "success",
                "num_created": num_created,
                "num_ingested": num_ingested,
                "elapsed_time": elapsed_time
            }
            
        except Exception as e:
            logger.error(f"Error in document generation and ingestion: {e}")
            
            return {
                "status": "error",
                "error": str(e),
                "elapsed_time": time.time() - start_time
            }

# Singleton instance
_document_ingestion_service = None

def get_document_ingestion_service():
    """
    Get the document ingestion service instance.
    
    Returns:
        DocumentIngestionService instance
    """
    global _document_ingestion_service
    if _document_ingestion_service is None:
        _document_ingestion_service = DocumentIngestionService()
    return _document_ingestion_service 