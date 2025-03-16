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
        "filename": "rag_reranking_methods.txt",
        "content": """
RAG Reranking Methods Technical Specification
===========================================

Overview
--------
This document details the reranking methods used in our Retrieval-Augmented Generation (RAG) pipeline to 
improve the relevance of retrieved documents before they are provided to the language model.

Cross-Encoder Architecture
-------------------------
Our primary reranking is based on a cross-encoder architecture that directly compares query-document pairs.

1. Model Architecture
   - Base model: Distilled RoBERTa (135M parameters)
   - Input format: [CLS] query [SEP] document [SEP]
   - Attention mechanism: Full cross-attention between query and document
   - Output layer: Linear projection to scalar relevance score
   - Training objective: Pointwise MSE loss with teacher distillation

2. Training Methodology
   - Training data: 2.4M query-document pairs with human relevance judgments
   - Negative sampling: Hard negatives from BM25 top 100 results
   - Curriculum learning: Progressive introduction of harder examples
   - Optimization: AdamW with cosine learning rate schedule
   - Regularization: Dropout (0.1) and weight decay (0.01)

3. Score Calibration
   - Raw scores normalized using logistic function
   - Temperature scaling for confidence calibration
   - Score aggregation using exponentially weighted average
   - Threshold determination via precision-recall analysis

Integration with Vector Retrieval
-------------------------------
1. Two-stage Retrieval Process
   - Stage 1: High-recall vector retrieval (top-k=50)
   - Stage 2: High-precision reranking (top-k=5)
   
2. Hybrid Score Computation
   - Weighted combination: λ * vector_score + (1-λ) * reranker_score
   - Adaptive λ based on query characteristics
   - Score normalization using min-max scaling
   - Rank-based fusion for multi-modal queries

3. Performance Optimization
   - Batched inference for reranking (batch size=32)
   - Document caching for frequent queries
   - Incremental reranking for long documents
   - Asynchronous processing pipeline

Evaluation Metrics
----------------
- MRR: 0.87 (vs. 0.73 for vector-only retrieval)
- NDCG@10: 0.92 (vs. 0.78 for vector-only retrieval)
- MAP: 0.84 (vs. 0.69 for vector-only retrieval)
- Latency overhead: +45ms at p95 per query
        """
    },
    {
        "filename": "vector_embedding_techniques.txt",
        "content": """
Vector Embedding Techniques for Document Retrieval
================================================

Document Embedding Models
-----------------------
Our system uses multiple embedding models optimized for different aspects of semantic search:

1. E5-large Model
   - Architecture: Transformer encoder-only (Large)
   - Dimensions: 1024
   - Training corpus: 1.4B filtered web pages + academic papers
   - Contrastive learning with supervised fine-tuning
   - Specialized for long document representation
   - Support for up to 512 tokens with hierarchical pooling

2. SBERT Model
   - Architecture: Siamese BERT networks
   - Dimensions: 768
   - Training approach: Triplet loss with hard negative mining
   - Optimized for sentence-level similarity
   - Language support: 50+ languages with multilingual alignment
   - Special handling for technical terminology and domain concepts

3. Domain-Adaptive Embeddings
   - Base model: ContriverCL
   - Fine-tuned on domain-specific corpus (2.3M documents)
   - Hierarchical pooling strategy for document sectioning
   - Adversarial training for robustness to wording variations
   - Length-invariant representations with adaptive normalization

Embedding Enhancement Techniques
------------------------------
1. Query-Document Interaction
   - Cross-attention mechanisms for query-aware embeddings
   - Late interaction modules for efficient retrieval
   - Attention-based pooling for query-dependent weighting
   - Token importance weighting based on IDF statistics

2. Dimensionality Reduction
   - PCA with whitening for decorrelation
   - Product quantization for storage efficiency (32x compression)
   - Random projection with Johnson-Lindenstrauss guarantees
   - Sparse embedding representations with top-k activation

3. Contextual Embedding
   - Sliding window approach for long documents
   - Hierarchical embedding with section-level aggregation
   - Positional weighting based on document structure
   - Entity-centric embedding with higher weights for named entities

4. Multi-Modal Integration
   - Cross-modal alignment for text and image embeddings
   - Joint space mapping with contrastive objectives
   - Modal importance weighting based on query characteristics
   - Fallback mechanisms for single-modal queries

Retrieval Optimizations
---------------------
- Approximate nearest neighbor search with HNSW graph
- In-memory index with SSD extension for large collections
- Asynchronous index updates with versioning
- Query expansion using lightweight language models
- Cached embedding for frequent queries and documents
        """
    },
    {
        "filename": "document_processing_pipeline.txt",
        "content": """
Document Processing Pipeline Architecture
=======================================

Pipeline Overview
---------------
Our document processing pipeline transforms raw documents into structured, indexed, and retrievable knowledge units through a series of specialized stages.

Input Processing
--------------
1. Document Ingestion
   - Supported formats: PDF, DOCX, HTML, Markdown, TXT, CSV, JSON
   - OCR integration for scanned documents (Tesseract 5.0)
   - Image extraction and processing for embedded visuals
   - Table detection and structured data extraction
   - Header/footer identification and filtering

2. Content Normalization
   - Encoding standardization (UTF-8)
   - Character normalization and special character handling
   - Whitespace normalization and line break processing
   - Bullet point and numbering standardization
   - Language detection and script identification

Semantic Processing
-----------------
1. Document Segmentation
   - Structure-aware chunking based on section boundaries
   - Semantic chunking using sentence transformers
   - Sliding window approach with adaptive overlap
   - Hierarchical segmentation (document → section → paragraph)
   - Reference preservation across chunk boundaries

2. Metadata Extraction
   - Title and section heading identification
   - Author and publication information extraction
   - Date and version detection
   - Domain classification and topic modeling
   - Key entity recognition and relationship extraction

3. Content Enrichment
   - Named entity recognition and linking
   - Terminology standardization and synonym mapping
   - Citation parsing and validation
   - Fact extraction and verification
   - Knowledge graph integration

Indexing Pipeline
---------------
1. Embedding Generation
   - Multi-model embedding approach (E5 + domain-specific models)
   - Hierarchical embedding at multiple granularities
   - Sparse vector generation for lexical search
   - Embedding caching and reuse optimization
   - Batch processing for throughput maximization

2. Index Management
   - Versioned indices with atomic updates
   - Incremental indexing for document updates
   - Index partitioning based on document properties
   - Background reindexing with zero downtime
   - Index compression and optimization routines

3. Quality Assurance
   - Automated retrieval quality testing
   - Embedding drift detection
   - Retrieval performance monitoring
   - Index health checks and validation
   - Document processing error detection and recovery

System Integration
----------------
- Event-driven architecture with message queues
- Horizontal scaling for high-volume processing
- Stateless processing nodes for resilience
- Progress tracking and resumable processing
- Comprehensive logging and telemetry
        """
    },
    {
        "filename": "knowledge_graph_integration.txt",
        "content": """
Knowledge Graph Integration with RAG Systems
==========================================

Architecture Overview
-------------------
Our knowledge graph integration layer enhances RAG capabilities by incorporating structured entity relationships and semantic connections between documents.

Graph Data Model
--------------
1. Entity Representation
   - Entity types: Person, Organization, Product, Concept, Document
   - Property schema: Core properties + extensible attribute sets
   - Identifier system: URI-based with namespace conventions
   - Temporal aspects: Valid time ranges for relationships
   - Confidence scores for extracted information

2. Relationship Types
   - Hierarchical: IsA, PartOf, BelongsTo relationships
   - Associative: RelatesTo, References, Influences
   - Causal: CausedBy, Results, Impacts
   - Temporal: Precedes, Follows, DuringPeriod
   - Domain-specific: Technical dependencies, business impact

3. Document Integration
   - Document-entity links with relevance scores
   - Fragment-level entity mentions with context
   - Cross-document entity resolution
   - Citation and reference relationships
   - Provenance tracking for fact attribution

Integration Methods
----------------
1. Entity Extraction
   - NER pipeline with domain-specific models
   - Coreference resolution across documents
   - Entity linking to standard knowledge bases
   - Relationship extraction using dependency parsing
   - Attribute extraction with pattern-based approaches

2. Graph Construction
   - Incremental graph building from processed documents
   - Entity merging and deduplication strategies
   - Relationship inference and transitive closure
   - Confidence scoring for extracted relationships
   - Knowledge graph validation and consistency checking

3. Query Enhancement
   - Entity-based query expansion
   - Relationship-aware retrieval ranking
   - Path-based relevance scoring
   - Graph traversal for multi-hop question answering
   - Query interpretation using graph structure

Implementation Technology
----------------------
- Graph Database: Neo4j Enterprise with APOC extensions
- Entity Extraction: SpaCy with custom NER models
- Integration Framework: Custom ETL pipeline with transactional guarantees
- Query Interface: Cypher with REST API and Python bindings
- Scalability: Federated graph architecture with sharding

Performance Characteristics
-------------------------
- Entity extraction throughput: 500+ documents per minute
- Graph query latency: <50ms for 1-hop queries, <200ms for 3-hop queries
- Storage efficiency: Optimized property encoding, ~30% reduction vs. naive approach
- Update frequency: Near real-time for critical entities, batch updates for others
- Consistency: ACID-compliant transactions with optimistic locking
        """
    },
    {
        "filename": "retrieval_benchmark_results.txt",
        "content": """
WebAgent Retrieval System Benchmark Results
=========================================

Executive Summary
---------------
This document presents comprehensive benchmark results for WebAgent's retrieval systems, demonstrating significant performance improvements over baseline approaches and competitive commercial solutions.

Test Environment
--------------
- Hardware: 8-core CPU (Intel Xeon), 64GB RAM, NVIDIA A100 GPU
- Index size: 10 million documents (mixed domain corpus)
- Query set: 10,000 diverse queries from real user logs
- Evaluation metrics: MRR, NDCG@10, P@5, Recall@100, latency

Baseline System Performance
-------------------------
1. BM25 (Elasticsearch)
   - MRR: 0.62
   - NDCG@10: 0.71
   - P@5: 0.68
   - Recall@100: 0.83
   - Mean latency: 45ms

2. Dense Retrieval (SBERT)
   - MRR: 0.71
   - NDCG@10: 0.76
   - P@5: 0.73
   - Recall@100: 0.87
   - Mean latency: 78ms

WebAgent Hybrid System Performance
--------------------------------
1. Production Configuration
   - MRR: 0.86 (+38.7% vs. BM25, +21.1% vs. Dense)
   - NDCG@10: 0.89 (+25.4% vs. BM25, +17.1% vs. Dense)
   - P@5: 0.85 (+25.0% vs. BM25, +16.4% vs. Dense)
   - Recall@100: 0.94 (+13.3% vs. BM25, +8.0% vs. Dense)
   - Mean latency: 92ms

2. Optimized Research Configuration
   - MRR: 0.89 (+43.5% vs. BM25, +25.4% vs. Dense)
   - NDCG@10: 0.91 (+28.2% vs. BM25, +19.7% vs. Dense)
   - P@5: 0.88 (+29.4% vs. BM25, +20.5% vs. Dense)
   - Recall@100: 0.96 (+15.7% vs. BM25, +10.3% vs. Dense)
   - Mean latency: 157ms

Domain-Specific Performance
-------------------------
1. Scientific Documents
   - NDCG@10: 0.87 (vs. 0.73 baseline)
   - P@5: 0.83 (vs. 0.68 baseline)
   - Key improvement: Entity-aware embeddings

2. Technical Documentation
   - NDCG@10: 0.92 (vs. 0.78 baseline)
   - P@5: 0.89 (vs. 0.74 baseline)
   - Key improvement: Domain-specific vocabulary handling

3. Legal Documents
   - NDCG@10: 0.85 (vs. 0.69 baseline)
   - P@5: 0.81 (vs. 0.64 baseline)
   - Key improvement: Section-aware chunking

Scaling Characteristics
---------------------
- Linear scaling to 100M documents with <10% performance degradation
- Sub-linear latency increase with document count (optimized indexing)
- 2.5x throughput improvement with GPU acceleration
- 4x throughput on multi-node deployment (4 nodes)
- 99.9% availability in production environment

Competitive Analysis
------------------
- 32% better relevance than Commercial System A (NDCG@10)
- 18% better relevance than Commercial System B (NDCG@10)
- 45% lower latency than Commercial System C
- 27% better recall than Commercial System D
- More efficient token usage than all tested competitors
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
        
    return created_files, len(created_files)

@timing_decorator
async def ingest_documents(doc_loader, doc_paths):
    """
    Ingest documents into the vector database using the document loader.
    
    Args:
        doc_loader: The document loader service
        doc_paths: List of document paths to ingest
        
    Returns:
        List of artifact IDs for the ingested documents
    """
    artifact_ids = []
    
    print("\n" + "=" * 80)
    print("INGESTING DOCUMENTS INTO VECTOR DATABASE")
    print("=" * 80)
    
    for i, doc_path in enumerate(doc_paths):
        try:
            doc_name = os.path.basename(doc_path)
            print(f"[{i+1}/{len(doc_paths)}] Ingesting: {doc_name}")
            
            # Load the document into vector database
            artifact_id = await doc_loader.load_document(doc_path)
            
            artifact_ids.append(artifact_id)
            logger.info(f"Document loaded. Artifact ID: {artifact_id}")
            print(f"✓ Successfully ingested document: {doc_name} (ID: {artifact_id})")
        except Exception as e:
            logger.error(f"Error loading document {doc_path}: {e}")
            print(f"✗ Failed to ingest document: {doc_path}")
            print(f"  Error: {str(e)}")
    
    print("\nDocument ingestion complete.")
    print(f"Successfully ingested {len(artifact_ids)} out of {len(doc_paths)} documents.")
    print("=" * 80 + "\n")
    
    return artifact_ids

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
        documents = await retriever.get_relevant_documents(query)
        
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
        
        # Set environment variables to control which provider to use
        os.environ["LLM_PROVIDER"] = llm_provider
    else:
        # In vector-only mode, we don't need an LLM provider
        print("\nVector-only mode selected. No LLM API connection required.")
        llm_provider = "none"
        os.environ["LLM_PROVIDER"] = "none"
    
    # Set retrieval mode environment variable
    os.environ["RETRIEVAL_MODE"] = retrieval_mode
    
    # Set additional environment variables for specific providers
    if llm_provider == "together":
        # Make sure the environment has the right keys for Together AI
        os.environ["RESEARCH_LLM_PROVIDER"] = "together"
        # Use the free Llama 3.3 model which is reliable
        os.environ["TOGETHER_MODEL"] = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
        print(f"Using TogetherAI model: {os.environ.get('TOGETHER_MODEL', 'default')}")
    elif llm_provider == "openai":
        os.environ["RESEARCH_LLM_PROVIDER"] = "openai"
    elif llm_provider == "groq":
        os.environ["RESEARCH_LLM_PROVIDER"] = "groq"
    elif llm_provider == "ollama":
        os.environ["RESEARCH_LLM_PROVIDER"] = "ollama"
        os.environ["OLLAMA_MODEL"] = "deepseek-r1:8b"
    
    # Create sample documents
    sample_doc_paths, num_created = create_sample_documents()
    logger.info(f"Created {num_created} sample documents")
    print(f"\nCreated {num_created} sample documents in {DEMO_DOCS_DIR}")
    
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
    await ingest_documents(doc_loader, sample_doc_paths)
    
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
        "How does our hybrid retrieval system work?",
        "What are the capabilities of our research agent?",
        "How does our document processing pipeline work?",
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
                    display_research_results(result)
                    
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error in vector-only processing: {error_msg}")
                print(f"ERROR: {error_msg}")
                all_results.append(None)
                all_errors.append(error_msg)
        else:
            # Use the research agent
            print("\nRunning full RAG pipeline (retrieve -> rerank -> generate)...")
            
            try:
                # Run the research agent
                start_time = datetime.now()
                updated_state = await research_agent.run(state)
                
                # Try to extract result data
                agent_output = updated_state.get_agent_output("internal_research") or {}
                result_data = agent_output.get("report", None)
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
                        display_research_results(result_data)
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
                        display_research_results(result)
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

def check_for_nan_scores(ranked_docs):
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
    query_terms = [term.lower() for term in query.split() if len(term) > 2]
    
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