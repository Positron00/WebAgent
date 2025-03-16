"""
Internal Research Agent for the WebAgent backend.

This agent implements an advanced Retrieval-Augmented Generation (RAG) system that
searches internal knowledge bases and documents to find relevant information.

The RAG architecture consists of three primary components:
1. Retriever - Fetches candidate documents from vector database (using hybrid retrieval)
2. Reranker - Refines retrieved results using a cross-encoder to prioritize relevance
3. Generator - Produces final answers using the most relevant context documents

The implementation follows a modular approach for scalability and maintainability.
"""
from typing import Dict, List, Any, Optional, Tuple
import logging
import time
import os
from datetime import datetime
import asyncio
from functools import lru_cache

import torch
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

from app.models.task import WorkflowState
from app.services.llm import get_llm
from app.services.vectordb import retrieve_documents, get_vectordb_service
from app.agents.base_agent import BaseAgent
from app.core.config import settings
from app.utils.metrics import timing_decorator, log_memory_usage

logger = logging.getLogger(__name__)

# Prompt for the Internal Research Agent
INTERNAL_RESEARCH_PROMPT = """You are an Internal Research Agent specializing in retrieving and synthesizing information from internal knowledge bases.

USER QUERY: {query}
RESEARCH PLAN: {research_plan}

You have been provided with the following internal documents, ranked by relevance:
{documents}

{feedback_section}

Your task:
1. Analyze the documents and extract relevant information for the query
2. Organize the information in a clear, structured format
3. Cite your sources with document references
4. Only include information that is directly related to the query
5. Highlight key facts, statistics, and insights
6. Format your findings as markdown
7. If additional research was requested, focus on addressing the specific questions and missing information

IMPORTANT: Be thorough, accurate, and objective. Cite all sources. If the information is not in the provided documents, state that clearly.
"""

# Prompt for follow-up research
FOLLOWUP_RESEARCH_PROMPT = """
ADDITIONAL RESEARCH REQUESTED - ITERATION {iteration}

Previous research was rated {score}/10.

Missing information:
{missing_information}

Specific research questions to address:
{research_questions}

Please focus on finding this information in your analysis.
"""

# Experiment tracking configuration
TRACKING_ENABLED = os.environ.get("MLFLOW_TRACKING_URI") is not None
if TRACKING_ENABLED:
    import mlflow
    mlflow.set_experiment("rag_agent_performance")

class RAGRetriever:
    """
    Advanced retrieval component that combines multiple retrieval strategies.
    
    Features:
    - Hybrid retrieval (dense + sparse)
    - Configurable retrieval parameters
    - Performance metrics tracking
    """
    
    def __init__(self, config: Dict = None):
        """Initialize the RAG Retriever with optional configuration."""
        self.config = config or {}
        self.initialize_retrievers()
        
    @timing_decorator
    def initialize_retrievers(self):
        """Set up dense and sparse retrievers for hybrid search."""
        # Get basic vector DB service (using OpenAI embeddings)
        self.vectordb_service = get_vectordb_service()
        
        # Initialize a more advanced dense retriever with a better model
        try:
            # Use the same vectordb path as the rest of the application
            # Don't create a separate 'advanced' subdirectory to avoid path inconsistencies
            vectordb_path = settings.VECTOR_DB_PATH
            os.makedirs(vectordb_path, exist_ok=True)
            logger.info(f"Using vector database path: {vectordb_path}")
            
            # First, check if the vector database already exists and get its dimension
            expected_dim = None
            try:
                # Initialize a temporary Chroma client without embeddings to check dimensions
                from langchain_chroma import Chroma
                from chromadb import Client as ChromaClient
                from chromadb.config import Settings as ChromaSettings
                
                # Check if an existing collection exists and its dimensionality
                client = ChromaClient(Settings=ChromaSettings(
                    persist_directory=vectordb_path,
                    anonymized_telemetry=False
                ))
                
                expected_dim = 1536  # Default OpenAI dimension
                collection_exists = False
                
                if len(client.list_collections()) > 0:
                    collection_exists = True
                    # Get the first collection to check metadata
                    collection = client.get_collection(client.list_collections()[0].name)
                    if collection:
                        # Try to get a document to determine dimensionality
                        try:
                            results = collection.peek(1)
                            if results and 'embeddings' in results and len(results['embeddings']) > 0:
                                expected_dim = len(results['embeddings'][0])
                                logger.info(f"Detected vector database dimension: {expected_dim}")
                            else:
                                logger.warning("Couldn't find embeddings in collection peek results")
                        except Exception as dim_err:
                            logger.warning(f"Could not detect embedding dimension: {str(dim_err)}")
                else:
                    logger.info("No existing collections found. Creating new vector database.")
                    
                # Store whether the database exists for later checks
                self.vectordb_exists = collection_exists
                # Store the dimension for reference
                self.embedding_dim = expected_dim
                
            except Exception as check_err:
                logger.warning(f"Error checking vector database dimensions: {str(check_err)}")
                self.vectordb_exists = False
            
            # Choose the appropriate embedding model based on the database dimension
            if expected_dim == 1536 or (expected_dim is None and not self.vectordb_exists):
                # For OpenAI compatible embeddings (1536 dim)
                if expected_dim is None:
                    logger.info("Creating new database with OpenAI-compatible embeddings (1536 dim)")
                else:
                    logger.info("Using OpenAI-compatible embeddings (1536 dim) to match database")
                
                # Set the dimension explicitly
                self.embedding_dim = 1536
                
                from langchain_openai import OpenAIEmbeddings
                
                # Try to use the OpenAI embeddings
                try:
                    self.advanced_embeddings = OpenAIEmbeddings(
                        model="text-embedding-ada-002"  # 1536 dimensions
                    )
                    logger.info("Successfully initialized OpenAI embeddings")
                except Exception as oai_err:
                    logger.warning(f"Could not initialize OpenAI embeddings: {str(oai_err)}")
                    # Fallback to a HuggingFace model with dimension transformation
                    logger.info("Falling back to HuggingFace embeddings with dimension adaptation")
                    
                    # Try models with close to 1536 dimensions
                    hf_models_to_try = [
                        "BAAI/bge-large-en-v1.5",  # 1024 dimensions
                        "intfloat/e5-large-v2",  # 1024 dimensions
                        "sentence-transformers/all-mpnet-base-v2"  # 768 dimensions
                    ]
                    
                    success = False
                    for model_name in hf_models_to_try:
                        try:
                            from langchain_community.embeddings import HuggingFaceEmbeddingsWithDimension
                            
                            # Custom class to enforce dimension
                            class Dim1536Embeddings(HuggingFaceEmbeddings):
                                def embed_documents(self, texts):
                                    # Get base embeddings
                                    embeddings = super().embed_documents(texts)
                                    # Apply dimension transformation to match 1536
                                    import numpy as np
                                    
                                    # Simple approach: resize with padding or truncation
                                    resized_embeddings = []
                                    for emb in embeddings:
                                        emb_array = np.array(emb)
                                        if len(emb) < 1536:
                                            # Pad with zeros
                                            padding = np.zeros(1536 - len(emb))
                                            resized = np.concatenate([emb_array, padding])
                                        else:
                                            # Truncate
                                            resized = emb_array[:1536]
                                        # Normalize
                                        resized = resized / np.linalg.norm(resized)
                                        resized_embeddings.append(resized.tolist())
                                    return resized_embeddings
                                
                                def embed_query(self, text):
                                    # Single query embedding resize
                                    emb = super().embed_query(text)
                                    import numpy as np
                                    emb_array = np.array(emb)
                                    if len(emb) < 1536:
                                        # Pad with zeros
                                        padding = np.zeros(1536 - len(emb))
                                        resized = np.concatenate([emb_array, padding])
                                    else:
                                        # Truncate
                                        resized = emb_array[:1536]
                                    # Normalize
                                    resized = resized / np.linalg.norm(resized)
                                    return resized.tolist()
                            
                            # Initialize with dimension adaptation
                            self.advanced_embeddings = Dim1536Embeddings(
                                model_name=model_name,
                                model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
                                encode_kwargs={"normalize_embeddings": True}
                            )
                            logger.info(f"Successfully initialized HuggingFace embeddings with model {model_name} and dimension adaptation to 1536")
                            success = True
                            break
                        except Exception as hf_err:
                            logger.warning(f"Failed to initialize {model_name}: {str(hf_err)}")
                    
                    if not success:
                        logger.error("Failed to initialize any compatible embedding model")
                        raise ValueError("No compatible embedding models available")
                        
            elif expected_dim is not None:
                # Use HuggingFace model that matches the detected dimension
                logger.info(f"Matching existing database with dimension {expected_dim}")
                
                # Select model based on dimension
                if expected_dim == 768:
                    model_name = "sentence-transformers/all-mpnet-base-v2"  # 768 dimensions
                elif expected_dim == 1024:
                    model_name = "BAAI/bge-large-en-v1.5"  # 1024 dimensions 
                else:
                    # Use the configured model as fallback
                    model_name = self.config.get("embedding_model", "sentence-transformers/all-mpnet-base-v2")
                
                logger.info(f"Using HuggingFace embeddings: {model_name}")
                self.advanced_embeddings = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
                    encode_kwargs={"normalize_embeddings": True}
                )
            else:
                # Use default model 
                model_name = self.config.get("embedding_model", "sentence-transformers/all-mpnet-base-v2")
                logger.info(f"Using default HuggingFace embeddings: {model_name}")
                self.advanced_embeddings = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
                    encode_kwargs={"normalize_embeddings": True}
                )
            
            # Initialize the vector store with the embeddings
            logger.info("Initializing vector database with embeddings")
            self.advanced_vectordb = Chroma(
                persist_directory=vectordb_path,
                embedding_function=self.advanced_embeddings
            )
            
            # Create retrievers
            self.dense_retriever = self.advanced_vectordb.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.config.get("dense_k", 20)}
            )
            
            # Create BM25 sparse retriever for keyword-based search
            try:
                # Create a proper Document object with page_content
                from langchain.schema import Document
                
                # Check if the collection exists and has documents directly
                try:
                    # First check the collection count
                    collection_info = self.advanced_vectordb._collection.count()
                    logger.info(f"Vector database has {collection_info} documents")
                    
                    # Set initial document count based on collection info
                    self.vector_db_document_count = collection_info
                    
                    # If the collection exists but we need doc content for BM25
                    if collection_info > 0:
                        documents = self.advanced_vectordb.get()
                        if documents and documents.get('documents'):
                            # Convert to Document objects
                            doc_content = documents.get('documents', [])
                            doc_metadata = documents.get('metadatas', [{} for _ in doc_content])
                            documents = [Document(page_content=content, metadata=meta) 
                                        for content, meta in zip(doc_content, doc_metadata)]
                        else:
                            # If we couldn't get documents but collection count > 0, create dummy document
                            logger.warning("Collection count > 0 but couldn't retrieve documents. Using dummy document.")
                            documents = [Document(page_content="Document content placeholder", metadata={})]
                    else:
                        logger.info("Vector database is empty. Using placeholder document.")
                        documents = [Document(page_content="Empty repository placeholder", metadata={})]
                        self.vector_db_document_count = 0
                except Exception as doc_err:
                    logger.warning(f"Error accessing vector database: {str(doc_err)}. Using placeholder document.")
                    documents = [Document(page_content="Error accessing repository", metadata={})]
                    # Don't assume 0 documents just because of an error
                    self.vector_db_document_count = None  # None means "unknown" rather than definitely empty
                
                # Create the sparse retriever with available documents
                self.sparse_retriever = BM25Retriever.from_documents(
                    documents,
                    preprocess_func=lambda text: text.lower().split(),
                    k=self.config.get("sparse_k", 20)
                )
            except Exception as e:
                logger.error(f"Error creating BM25Retriever: {str(e)}")
                # Create a mock retriever that returns empty results
                try:
                    # Try multiple import paths for backward/forward compatibility
                    try:
                        from langchain.schema.retriever import BaseRetriever
                    except ImportError:
                        try:
                            from langchain.retrievers.base import BaseRetriever
                        except ImportError:
                            from langchain._retriever_base import BaseRetriever
                    
                    class EmptyRetriever(BaseRetriever):
                        def _get_relevant_documents(self, query, **kwargs):
                            return []
                    self.sparse_retriever = EmptyRetriever()
                except Exception as inner_e:
                    # Last resort fallback - just create a minimal fake retriever
                    logger.error(f"Error creating retriever: {str(inner_e)}. Using minimal mock retriever.")
                    from langchain.schema import Document
                    
                    class MinimalRetriever:
                        def get_relevant_documents(self, query, **kwargs):
                            return [Document(page_content="Empty result", metadata={})]
                        
                    self.sparse_retriever = MinimalRetriever()
            
            # Create an ensemble retriever that combines both approaches
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[self.dense_retriever, self.sparse_retriever],
                weights=[0.7, 0.3]  # Favor dense retrieval but include sparse results
            )
            
            logger.info(f"Advanced RAG Retriever initialized with {model_name}")
            self.initialized = True
            
        except Exception as e:
            logger.error(f"Error initializing advanced retrievers: {str(e)}")
            # Fall back to basic retriever
            self.initialized = False
            self.vector_db_document_count = None  # Unknown instead of definitely empty
    
    async def ensure_vector_db_has_documents(self):
        """
        Check if the vector database has documents and populate it if necessary.
        
        Returns:
            Boolean indicating if the vector database now has documents
        """
        # Check if vector DB has documents
        if self.vector_db_document_count is not None and self.vector_db_document_count > 0:
            logger.info(f"Vector database already has {self.vector_db_document_count} documents")
            return True
            
        # If document count is None (unknown), try to verify directly
        if self.vector_db_document_count is None:
            try:
                # Direct check of vectordb service
                try:
                    # Try to perform a test retrieval to check if documents exist
                    from app.services.vectordb import retrieve_documents
                    test_docs = await retrieve_documents("test", k=1)
                    
                    if test_docs and len(test_docs) > 0:
                        logger.info(f"Vector database verification successful: found {len(test_docs)} documents")
                        self.vector_db_document_count = len(test_docs)
                        return True
                except Exception as verify_err:
                    logger.warning(f"Vector database verification failed: {str(verify_err)}")
                
                # Try to check the collection count directly
                if hasattr(self.advanced_vectordb, "_collection") and hasattr(self.advanced_vectordb._collection, "count"):
                    count = self.advanced_vectordb._collection.count()
                    if count > 0:
                        logger.info(f"Direct collection count check: {count} documents found")
                        self.vector_db_document_count = count
                        return True
            except Exception as e:
                logger.warning(f"Failed to verify vector database contents: {str(e)}")
        
        # Vector DB is empty or verification failed, need to generate and ingest documents
        logger.info("Vector database appears empty. Generating and ingesting documents.")
        
        # Continue with the existing document ingestion process
        try:
            # Import the document ingestion service
            from app.services.document_ingestion_service import get_document_ingestion_service
            
            # Get the document ingestion service
            document_ingestion_service = get_document_ingestion_service()
            
            # Run the document ingestion process
            logger.info("Running document ingestion process")
            result = await document_ingestion_service.run_document_ingestion()
            
            if result["status"] == "success":
                logger.info(f"Document ingestion completed successfully. Created {result['num_created']} documents, ingested {result['num_ingested']} documents")
                
                # Try using the shared vectordb service directly
                try:
                    # Import retrieve_documents function which is the actual API
                    from app.services.vectordb import retrieve_documents
                    
                    # Try a test retrieval
                    test_docs = await retrieve_documents("ethics", k=5)
                    
                    if test_docs and len(test_docs) > 0:
                        logger.info(f"Successfully retrieved {len(test_docs)} documents using retrieve_documents")
                        
                        # Update the document count
                        self.vector_db_document_count = len(test_docs)
                        
                        # If using the vectordb service works, just use that for retrieval
                        self.initialized = True
                        return True
                except Exception as retrieve_err:
                    logger.error(f"Error retrieving documents after ingestion: {str(retrieve_err)}")
                
                # Try directly accessing the Chroma database in the default location
                try:
                    # The default vectordb path might be different from what we expect
                    # Try various potential paths
                    potential_db_paths = [
                        settings.VECTOR_DB_PATH,
                        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "vectordb"),
                        "./data/vectordb"
                    ]
                    
                    for db_path in potential_db_paths:
                        if os.path.exists(db_path):
                            logger.info(f"Found vector database at {db_path}")
                            
                            # Try to initialize with this path
                            try:
                                # Reinitialize with this path
                                from langchain_chroma import Chroma
                                
                                self.advanced_vectordb = Chroma(
                                    persist_directory=db_path,
                                    embedding_function=self.advanced_embeddings
                                )
                                
                                # Check if it has documents
                                collection_info = self.advanced_vectordb._collection.count()
                                
                                if collection_info > 0:
                                    logger.info(f"Vector database at {db_path} has {collection_info} documents")
                                    
                                    # Update the retrievers
                                    self.dense_retriever = self.advanced_vectordb.as_retriever(
                                        search_type="similarity",
                                        search_kwargs={"k": self.config.get("dense_k", 20)}
                                    )
                                    
                                    # Create a simple sparse retriever
                                    from langchain.schema import Document
                                    try:
                                        # Try to create BM25Retriever
                                        self.sparse_retriever = BM25Retriever.from_documents(
                                            [Document(page_content="Dummy content for BM25", metadata={})],
                                            preprocess_func=lambda text: text.lower().split(),
                                            k=self.config.get("sparse_k", 20)
                                        )
                                        
                                        # Create ensemble retriever
                                        self.ensemble_retriever = EnsembleRetriever(
                                            retrievers=[self.dense_retriever, self.sparse_retriever],
                                            weights=[0.9, 0.1]  # Heavily favor dense retrieval
                                        )
                                    except Exception as sparse_err:
                                        logger.error(f"Error creating sparse retriever: {str(sparse_err)}")
                                        # Just use the dense retriever directly
                                        self.ensemble_retriever = self.dense_retriever
                                    
                                    # Update state
                                    self.vector_db_document_count = collection_info
                                    self.initialized = True
                                    return True
                            except Exception as db_init_err:
                                logger.error(f"Error initializing vector database at {db_path}: {str(db_init_err)}")
                except Exception as path_err:
                    logger.error(f"Error trying alternative database paths: {str(path_err)}")
                
                # If we get here, we couldn't access the vector database after ingestion
                logger.error("Could not access vector database after document ingestion")
                return False
            else:
                logger.error(f"Document ingestion failed: {result.get('error', 'Unknown error')}")
                return False
                
        except Exception as e:
            logger.error(f"Error ensuring vector database has documents: {str(e)}")
            return False
    
    @timing_decorator
    async def retrieve(self, query: str, k: int = 20) -> List[Dict]:
        """
        Retrieve documents using the ensemble retriever if available,
        otherwise fall back to the basic retriever.
        
        Args:
            query: The search query
            k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents
        """
        start_time = time.time()
        
        try:
            # First, ensure the vector database has documents
            if not await self.ensure_vector_db_has_documents():
                logger.warning("Vector database has no documents even after ingestion attempt")
                return []
            
            if self.initialized:
                # Use the ensemble retriever for better results
                docs = await asyncio.to_thread(
                    self.ensemble_retriever.get_relevant_documents,
                    query
                )
                
                # Convert to the expected format
                results = []
                for doc in docs:
                    # Handle different document types appropriately
                    if hasattr(doc, 'page_content'):
                        # LangChain Document objects have page_content attribute
                        results.append({
                            "page_content": doc.page_content,
                            "metadata": doc.metadata,
                            "score": 1.0  # Placeholder score for ensemble retriever
                        })
                    elif isinstance(doc, dict):
                        # If it's already a dictionary, ensure it has page_content
                        if "page_content" not in doc and "content" in doc:
                            doc["page_content"] = doc["content"]
                        results.append(doc)
                    elif isinstance(doc, str):
                        # If it's just a string, wrap it as a document
                        results.append({
                            "page_content": doc,
                            "metadata": {},
                            "score": 1.0
                        })
            else:
                # Fall back to basic retriever
                results = await retrieve_documents(query, k=k)
            
            retrieval_time = time.time() - start_time
            
            if TRACKING_ENABLED:
                mlflow.log_metric("retrieval_time", retrieval_time)
                mlflow.log_metric("num_retrieved_docs", len(results))
            
            logger.info(f"Retrieved {len(results)} documents in {retrieval_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"Error in document retrieval: {str(e)}")
            if TRACKING_ENABLED:
                mlflow.log_metric("retrieval_error", 1)
            return []


class RAGReranker:
    """
    Document reranker that uses a cross-encoder model to refine retrieval results.
    
    Features:
    - Cross-encoder reranking for improved relevance
    - Configurable reranking parameters
    - Performance monitoring
    """
    
    def __init__(self, config: Dict = None):
        """Initialize the reranker with optional configuration."""
        self.config = config or {}
        self.initialize_reranker()
    
    @timing_decorator
    def initialize_reranker(self):
        """Initialize the cross-encoder reranker model."""
        try:
            # Use a better cross-encoder model for reranking
            # ms-marco-MiniLM-L12-v2 has better performance than L6-v2 (NDCG@10: 74.31 vs 74.30)
            model_name = self.config.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L12-v2")
            
            # Direct initialization for more control
            self.reranker = CrossEncoder(
                model_name,
                device="cuda" if torch.cuda.is_available() else "cpu",
                max_length=512
            )
            
            # Also create a LangChain compatible reranker for integration with retrievers
            self.lc_reranker = HuggingFaceCrossEncoder(model_name=model_name)
            self.lc_compressor = CrossEncoderReranker(
                model=self.lc_reranker,
                top_n=self.config.get("rerank_top_n", 5)
            )
            
            logger.info(f"Reranker initialized with {model_name}")
            self.initialized = True
            
        except Exception as e:
            logger.error(f"Error initializing reranker: {str(e)}")
            self.initialized = False
    
    @timing_decorator
    async def rerank(self, query: str, documents: List[Dict], top_n: int = 5) -> List[Dict]:
        """
        Rerank documents based on relevance to the query.
        
        Args:
            query: The search query
            documents: List of documents to rerank
            top_n: Number of top documents to return after reranking
            
        Returns:
            List of reranked documents
        """
        start_time = time.time()
        
        if not self.initialized or not documents:
            return documents[:top_n] if documents else []
        
        try:
            # Prepare query-document pairs
            doc_texts = [doc.get("page_content") for doc in documents]
            pairs = [[query, text] for text in doc_texts]
            
            # Get relevance scores using cross-encoder
            scores = await asyncio.to_thread(self.reranker.predict, pairs)
            
            # Attach scores to documents
            for i, doc in enumerate(documents):
                doc["score"] = float(scores[i])
            
            # Sort documents by score (descending)
            reranked_docs = sorted(documents, key=lambda d: d.get("score", 0), reverse=True)
            
            # Keep only the top_n documents
            reranked_docs = reranked_docs[:top_n]
            
            # Log performance metrics
            rerank_time = time.time() - start_time
            if TRACKING_ENABLED:
                mlflow.log_metric("rerank_time", rerank_time)
                mlflow.log_metric("top_score", reranked_docs[0].get("score", 0) if reranked_docs else 0)
            
            logger.info(f"Reranked documents in {rerank_time:.2f}s, top score: {reranked_docs[0].get('score', 0):.4f}" if reranked_docs else "No documents to rerank")
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Error in document reranking: {str(e)}")
            if TRACKING_ENABLED:
                mlflow.log_metric("rerank_error", 1)
            return documents[:top_n]


class InternalResearchAgent(BaseAgent):
    """
    Internal Research Agent that implements an advanced RAG architecture to search
    knowledge bases and internal documents.
    
    Features:
    - Multi-stage RAG pipeline: retrieval → rerank → generation
    - Experiment tracking integration
    - Performance monitoring
    - Configurable components
    """
    
    def __init__(self, config: Dict = None):
        """Initialize the Internal Research Agent with optional configuration."""
        super().__init__(name="internal_research")
        
        # Configuration
        self.config = config or {}
        
        # Initialize RAG components
        self.retriever = RAGRetriever(config)
        self.reranker = RAGReranker(config)
        
        # Initialize LLM and chain
        self.prompt = ChatPromptTemplate.from_template(INTERNAL_RESEARCH_PROMPT)
        
        try:
            # Use the default model from settings instead of hardcoded "gpt-4-turbo"
            # which is not supported by TogetherAI
            from app.core.config import settings
            default_model = settings.DEFAULT_MODEL
            # Only use the model specified in config if it's not "gpt-4-turbo"
            model_name = self.config.get("llm_model")
            if not model_name or model_name.startswith("gpt-"):
                model_name = default_model
                logger.info(f"Using default model from settings: {model_name}")
            
            self.llm = get_llm(model_name)
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
            logger.warning("Using fallback OpenAI LLM due to initialization error")
            self.llm = get_llm("gpt-3.5-turbo")  # Fallback to more reliable model
            
        self.parser = StrOutputParser()
        self.chain = self.prompt | self.llm | self.parser
        self.chain = self.trace_chain(self.chain)
        
        logger.info("Advanced RAG-based Internal Research Agent initialized")
    
    def _format_internal_documents(self, documents: List[Dict[str, Any]]) -> str:
        """Format internal documents for the prompt, including relevance scores."""
        if not documents:
            return "No relevant internal documents found."
            
        formatted_docs = ""
        for i, doc in enumerate(documents, 1):
            score = doc.get("score", 0)
            formatted_docs += f"DOCUMENT {i} (Relevance: {score:.4f}):\n"
            formatted_docs += f"Title: {doc.get('metadata', {}).get('title', 'No title')}\n"
            formatted_docs += f"Source: {doc.get('metadata', {}).get('source', 'Internal knowledge base')}\n"
            formatted_docs += f"Content: {doc.get('page_content', 'No content')}\n\n"
            
        return formatted_docs
    
    def _format_feedback(self, state: WorkflowState) -> str:
        """Format feedback from Senior Research Agent if available."""
        # Check if there's feedback in the context
        feedback = state.context.get("research_feedback")
        iteration = state.context.get("research_iteration", 1)
        
        if not feedback or iteration <= 1:
            return ""
            
        # Format the feedback section
        score = feedback.get("score", 0)
        missing_info = feedback.get("missing_information", [])
        questions = feedback.get("research_questions", [])
        
        # Format missing information as bullet points
        missing_info_str = "\n".join([f"- {item}" for item in missing_info])
        
        # Format research questions as bullet points
        questions_str = "\n".join([f"- {q}" for q in questions])
        
        return FOLLOWUP_RESEARCH_PROMPT.format(
            iteration=iteration,
            score=score,
            missing_information=missing_info_str,
            research_questions=questions_str
        )
    
    async def _generate_findings_with_retry(self, prompt_inputs: dict, max_retries: int = 3) -> str:
        """
        Generate research findings with retry logic for rate limiting.
        
        Args:
            prompt_inputs: The inputs for the LLM prompt
            max_retries: Maximum number of retry attempts
            
        Returns:
            The generated findings or an error message
        """
        attempts = 0
        base_delay = 2  # Start with 2 seconds delay
        
        while attempts < max_retries:
            try:
                # Generate the research findings using the LLM
                return await self.chain.ainvoke(prompt_inputs)
            except Exception as e:
                error_msg = str(e)
                attempts += 1
                
                # Check for rate limiting errors
                if "429" in error_msg or "Too Many Requests" in error_msg:
                    delay = base_delay * (2 ** attempts)  # Exponential backoff
                    logger.warning(f"Rate limit hit. Retrying in {delay}s (attempt {attempts}/{max_retries})")
                    await asyncio.sleep(delay)
                    continue
                
                # For other errors, return an error message
                logger.error(f"Error generating research findings: {error_msg}")
                return f"Error generating research findings: {error_msg}"
                
        # If we've exhausted retries
        return "Error generating research findings: Rate limit exceeded after multiple retries."
    
    @timing_decorator
    @log_memory_usage
    async def run(self, state: WorkflowState) -> WorkflowState:
        """
        Execute the RAG pipeline to search, rerank, and analyze internal documents.
        
        Args:
            state: The current workflow state
            
        Returns:
            Updated workflow state with internal research results
        """
        logger.info(f"Internal Research Agent running for query: {state.query}")
        state.current_step = "internal_research"
        
        if TRACKING_ENABLED:
            mlflow.start_run(run_name="rag_agent_run")
            mlflow.log_param("query", state.query)
        
        start_time = time.time()
        
        try:
            # Get the research plan
            research_plan = state.context.get("research_plan", {})
            
            # Get the current research iteration
            iteration = state.context.get("research_iteration", 1)
            is_followup = iteration > 1
            
            # If internal knowledge is not required, skip this step
            if not research_plan.get("requires_internal_knowledge", True):
                logger.info("Internal knowledge not required by research plan. Skipping.")
                state.update_with_agent_output("internal_research", {
                    "status": "skipped",
                    "reason": "Internal knowledge not required by research plan",
                    "timestamp": datetime.now().isoformat()
                })
                
                if TRACKING_ENABLED:
                    mlflow.log_param("status", "skipped")
                    mlflow.end_run()
                
                return state
            
            # Modify search query if this is a follow-up research request
            search_query = state.query
            if is_followup:
                feedback = state.context.get("research_feedback", {})
                questions = feedback.get("research_questions", [])
                
                # If there are specific questions, use them to enhance the query
                if questions:
                    search_query = f"{state.query} {questions[0]}"
                    logger.info(f"Using enhanced query for search: {search_query}")
            
            # 1. Retrieve documents using the RAG Retriever
            logger.info(f"Retrieving internal documents (iteration {iteration})")
            retrieved_documents = await self.retriever.retrieve(
                query=search_query,
                k=self.config.get("retrieve_k", 20)
            )
            
            # If follow-up research and there are specific questions, get additional documents
            if is_followup:
                feedback = state.context.get("research_feedback", {})
                questions = feedback.get("research_questions", [])[1:]  # Get remaining questions
                
                for i, question in enumerate(questions[:2]):  # Limit to 2 additional questions
                    logger.info(f"Performing additional document retrieval for: {question}")
                    additional_docs = await self.retriever.retrieve(
                        query=question,
                        k=self.config.get("retrieve_k", 10)  # Fewer documents for follow-up questions
                    )
                    # Add unique documents that weren't already retrieved
                    existing_ids = {doc.get("id") for doc in retrieved_documents if "id" in doc}
                    for doc in additional_docs:
                        if doc.get("id") not in existing_ids:
                            retrieved_documents.append(doc)
                            existing_ids.add(doc.get("id"))
            
            if not retrieved_documents:
                logger.warning("No internal documents found.")
                state.update_with_agent_output("internal_research", {
                    "status": "completed",
                    "documents": [],
                    "findings": "No relevant information found in internal knowledge base.",
                    "timestamp": datetime.now().isoformat()
                })
                
                if TRACKING_ENABLED:
                    mlflow.log_param("status", "no_documents")
                    mlflow.end_run()
                
                return state
            
            # 1.5 Deduplicate retrieved documents
            logger.info(f"Deduplicating {len(retrieved_documents)} retrieved documents")
            retrieved_documents = self._deduplicate_documents(retrieved_documents)
            logger.info(f"After deduplication: {len(retrieved_documents)} documents")
            
            # Get rerank_top_n configuration
            rerank_top_n = self.config.get("rerank_top_n", 5)
            
            # 2. Rerank documents if we have more than rerank_top_n documents
            if len(retrieved_documents) <= rerank_top_n:
                logger.info(f"Skipping reranking as we have fewer than {rerank_top_n} documents")
                reranked_documents = retrieved_documents
            else:
                logger.info("Reranking retrieved documents")
                reranked_documents = await self.reranker.rerank(
                    query=search_query,
                    documents=retrieved_documents,
                    top_n=rerank_top_n
                )
            
            # Format internal documents for the prompt
            formatted_docs = self._format_internal_documents(reranked_documents)
            
            # Format feedback section if this is a follow-up research
            feedback_section = self._format_feedback(state)
            
            # Validate reranked documents before generating findings
            if not reranked_documents:
                logger.warning("No documents available after reranking. Cannot generate findings.")
                state.update_with_agent_output("internal_research", {
                    "status": "completed",
                    "documents": [],
                    "findings": "No relevant documents were found to answer this query.",
                    "timestamp": datetime.now().isoformat()
                })
                return state
                
            # Log information about reranked documents for debugging
            logger.info(f"Reranked documents summary (top {len(reranked_documents)}):")
            for i, doc in enumerate(reranked_documents):
                doc_id = doc.get("id", "unknown")
                score = doc.get("score", 0)
                source = doc.get("metadata", {}).get("source", "unknown")
                content_preview = doc.get("page_content", "")[:100] + "..." if len(doc.get("page_content", "")) > 100 else doc.get("page_content", "")
                logger.info(f"  [{i+1}] ID: {doc_id}, Score: {score:.4f}, Source: {source}")
                logger.info(f"      Preview: {content_preview}")
            
            # 3. Generate answer using the LLM with reranked documents as context
            logger.info("Generating research findings")
            prompt_inputs = {                
                "research_plan": research_plan.get("analysis", "No research plan provided."),
                "documents": formatted_docs,
                "feedback_section": feedback_section,
                "query": state.query
            }
            
            # Generate findings with retry logic
            analysis = await self._generate_findings_with_retry(prompt_inputs)
            
            # Check if we got a valid response
            if not analysis or len(analysis.strip()) == 0:
                logger.warning("Empty research findings from LLM.")
                analysis = "The research agent was unable to generate a response. This may be due to rate limiting or API issues."
            
            # Create the internal research report
            internal_research_report = {
                "status": "completed",
                "documents": reranked_documents,
                "findings": analysis,
                "iteration": iteration,
                "timestamp": datetime.now().isoformat(),
                "metrics": {
                    "total_time": time.time() - start_time,
                    "num_retrieved": len(retrieved_documents),
                    "num_reranked": len(reranked_documents),
                    "top_document_score": reranked_documents[0].get("score", 0) if reranked_documents else 0
                }
            }
            
            # Update the state with the internal research report
            state.update_with_agent_output("internal_research", internal_research_report)
            
            # Add the findings to the context for other agents to use
            state.context["internal_research_findings"] = analysis
            
            if TRACKING_ENABLED:
                mlflow.log_param("status", "completed")
                mlflow.log_param("iteration", iteration)
                mlflow.log_metric("total_time", internal_research_report["metrics"]["total_time"])
                mlflow.log_metric("num_retrieved", internal_research_report["metrics"]["num_retrieved"])
                mlflow.log_metric("num_reranked", internal_research_report["metrics"]["num_reranked"])
                mlflow.end_run()
            
            logger.info(f"Internal Research completed with {len(reranked_documents)} relevant documents in {internal_research_report['metrics']['total_time']:.2f}s (iteration {iteration}).")
            return state
            
        except Exception as e:
            logger.error(f"Error in Internal Research Agent: {str(e)}")
            state.mark_error(f"Internal Research Agent failed: {str(e)}")
            
            if TRACKING_ENABLED:
                mlflow.log_param("status", "error")
                mlflow.log_param("error", str(e))
                mlflow.end_run()
                
            return state

    def _deduplicate_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicate documents based on content similarity.
        
        Args:
            documents: List of documents to deduplicate
            
        Returns:
            List of deduplicated documents
        """
        if not documents:
            return []
            
        # Use a set to track unique document content hashes
        unique_docs = []
        content_hashes = set()
        
        for doc in documents:
            # Get document content
            content = doc.get("page_content", "")
            
            # Create a simple hash of the content (first 100 chars + length + last 100 chars)
            # This is a simple but effective way to identify near-duplicate content
            content_preview = f"{content[:100]}_{len(content)}_{content[-100:]}"
            content_hash = hash(content_preview)
            
            # If this content hash is new, add the document
            if content_hash not in content_hashes:
                content_hashes.add(content_hash)
                unique_docs.append(doc)
        
        return unique_docs


def get_internal_research_agent() -> InternalResearchAgent:
    """
    Get an instance of the advanced RAG-based Internal Research Agent.
    
    Returns:
        An InternalResearchAgent instance with RAG capabilities
    """
    return InternalResearchAgent() 