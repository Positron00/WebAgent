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

from app.core import task_tracking

import torch
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers.document_compressors import CrossEncoderReranker
from sentence_transformers import CrossEncoder

from app.models.task import WorkflowState
from app.models.research_plan import research_query
from app.security.hop_guard import acheck_hop, kept_text, task_items
from app.security.types import ContextItem
from app.services.vectordb import retrieve_documents, get_vectordb_service, get_embeddings
from app.security import egress as security_egress
from app.security.tool_guard import guard_tool
from app.security.guarded_llm import register_template
from app.agents.base_agent import BaseAgent
from app.core.config import settings
from app.core.log_context import describe_text
from app.core.loadEnvYAML import get_agents_config
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

# Static prompt text is never scored by the safety net (SEC-032).
register_template("internal_research", INTERNAL_RESEARCH_PROMPT)
register_template("internal_research", FOLLOWUP_RESEARCH_PROMPT)

# Experiment tracking configuration
TRACKING_ENABLED = os.environ.get("MLFLOW_TRACKING_URI") is not None


def _prepare_experiment() -> None:
    import mlflow

    # Always use settings.MLFLOW_ARTIFACT_LOCATION for artifacts. The first call on an empty store also
    # creates MLflow's default experiment, which several workers can race on: retry once (the image's
    # startup checks prepare the store before the workers start, so this is a second line).
    try:
        experiment = mlflow.get_experiment_by_name("rag_agent_performance")
    except Exception:  # noqa: BLE001 - "Yaml file ... exists" from a concurrent first write
        experiment = mlflow.get_experiment_by_name("rag_agent_performance")
    if experiment is None:
        # Several uvicorn workers import this module at once on a fresh data volume; if another
        # worker created the experiment first, that is fine (it used to kill this worker).
        try:
            mlflow.create_experiment(
                "rag_agent_performance",
                artifact_location=settings.MLFLOW_ARTIFACT_LOCATION
            )
        except mlflow.exceptions.MlflowException:
            if mlflow.get_experiment_by_name("rag_agent_performance") is None:
                raise
    else:
        mlflow.set_experiment("rag_agent_performance")


if TRACKING_ENABLED:
    # Experiment tracking is optional: a tracking store that can't be used turns it off for this
    # worker instead of stopping the server.
    try:
        _prepare_experiment()
    except Exception as exc:  # noqa: BLE001
        logger.warning("MLflow tracking disabled for internal research: %s", exc)
        TRACKING_ENABLED = False


def _remember_private_text(state: WorkflowState, documents) -> None:
    """Internal documents are private: a web search copying them is the egress signal (checkpoint2-intent SEC-106)."""
    for doc in documents or []:
        text = doc.get("page_content") if isinstance(doc, dict) else getattr(doc, "page_content", None)
        if isinstance(text, str):
            security_egress.remember(state, text)

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
            if not settings.VECTOR_DB_PATH or settings.VECTOR_DB_PATH.strip() == '':
                vectordb_path = os.path.join(settings.DATA_DIR, "vectordb")
                logger.warning(f"VECTOR_DB_PATH is empty, using default path: {vectordb_path}")
            else:
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
                client = ChromaClient(settings=ChromaSettings(
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
            
            self.advanced_embeddings = get_embeddings()
            if expected_dim is not None:
                logger.info(
                    "Using configured embeddings (provider=%s); existing collection dimension=%s",
                    settings.EMBEDDINGS_PROVIDER,
                    expected_dim,
                )
            else:
                logger.info(
                    "Using configured embeddings (provider=%s) for a new vector database",
                    settings.EMBEDDINGS_PROVIDER,
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
            
            # Ensure model_name is defined in all code paths
            if not locals().get('model_name'):
                model_name = "default-embedding-model"
                
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
                    # Only use the path from settings
                    potential_db_paths = [
                        settings.VECTOR_DB_PATH
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
                    lambda q: self.ensemble_retriever.invoke(q),
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
                # Fall back to basic retriever - which may return [(doc, score), ...] tuples
                result_tuples = await retrieve_documents(query, k=k)
                
                # Convert tuple results to consistent dictionary format
                results = []
                for item in result_tuples:
                    if isinstance(item, tuple) and len(item) == 2:
                        # Handle (doc, score) tuple format
                        doc, score = item
                        if hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):
                            # Convert Document object to dictionary
                            results.append({
                                "page_content": doc.page_content,
                                "metadata": doc.metadata,
                                "score": float(score)
                            })
                        else:
                            # Handle other tuple formats
                            logger.warning(f"Unexpected document format in tuple: {type(doc)}")
                            results.append({
                                "page_content": str(doc),
                                "metadata": {},
                                "score": float(score)
                            })
                    elif isinstance(item, dict):
                        # Already in dictionary format
                        results.append(item)
                    else:
                        # Unknown format, attempt conversion
                        logger.warning(f"Unexpected result format: {type(item)}")
                        try:
                            if hasattr(item, 'page_content'):
                                results.append({
                                    "page_content": item.page_content,
                                    "metadata": getattr(item, 'metadata', {}),
                                    "score": 1.0
                                })
                            else:
                                results.append({
                                    "page_content": str(item),
                                    "metadata": {},
                                    "score": 1.0
                                })
                        except Exception as conv_err:
                            logger.error(f"Error converting result: {str(conv_err)}")
            
            retrieval_time = time.time() - start_time
            
            if TRACKING_ENABLED:
                task_tracking.log_metric("retrieval_time", retrieval_time)
                task_tracking.log_metric("num_retrieved_docs", len(results))
            
            logger.info(f"Retrieved {len(results)} documents in {retrieval_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"Error in document retrieval: {str(e)}")
            if TRACKING_ENABLED:
                task_tracking.log_metric("retrieval_error", 1)
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
                task_tracking.log_metric("rerank_time", rerank_time)
                task_tracking.log_metric("top_score", reranked_docs[0].get("score", 0) if reranked_docs else 0)
            
            logger.info(f"Reranked documents in {rerank_time:.2f}s, top score: {reranked_docs[0].get('score', 0):.4f}" if reranked_docs else "No documents to rerank")
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Error in document reranking: {str(e)}")
            if TRACKING_ENABLED:
                task_tracking.log_metric("rerank_error", 1)
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
        self.max_documents = get_agents_config().internal_research.max_documents
        
        # Initialize RAG components
        self.retriever = RAGRetriever(config)
        self.reranker = RAGReranker(config)
        
        # Initialize LLM and prompt template
        self.prompt = ChatPromptTemplate.from_template(INTERNAL_RESEARCH_PROMPT)
        
        try:
            # Use the default model from settings instead of hardcoded "gpt-4-turbo"
            # which is not supported by TogetherAI
            from app.services.llm import get_agent_llm

            self.llm = get_agent_llm("internal_research")
            logger.info("Successfully initialized internal research LLM from agents config")

        except Exception as e:
            logger.error(f"Error initializing primary LLM: {str(e)}")
            if os.getenv("WEBAGENT_ALLOW_FAKE_LLM", "false").lower() == "true":
                from langchain_community.llms.fake import FakeListLLM

                self.llm = FakeListLLM(
                    responses=[
                        "ERROR: Unable to initialize any LLM. Internal research results cannot be generated."
                    ]
                )
                logger.warning("Using mock LLM as emergency fallback")
            else:
                raise
        
        # We don't create a chain anymore since we're now invoking the LLM directly in the
        # _generate_findings_with_retry method to avoid tags conflicts
        
        logger.info("Advanced RAG-based Internal Research Agent initialized")
    
    def _format_internal_documents(self, documents: List[Dict[str, Any]]) -> str:
        """Format internal documents for the prompt, including relevance scores."""
        if not documents:
            return "No relevant internal documents found."
        
        # First, combine chunks from the same source documents
        combined_docs = self._combine_document_chunks(documents)
            
        formatted_docs = ""
        for i, doc in enumerate(combined_docs, 1):
            score = doc.get("score", 0)
            source = doc.get("metadata", {}).get("source", "Internal knowledge base")
            section = doc.get("metadata", {}).get("document_section", "")
            chunk_info = ""
            if "chunk_id" in doc.get("metadata", {}) and "total_chunks" in doc.get("metadata", {}):
                chunk_id = doc.get("metadata", {}).get("chunk_id")
                total_chunks = doc.get("metadata", {}).get("total_chunks")
                chunk_info = f" (Section {chunk_id+1}/{total_chunks})"
            
            formatted_docs += f"DOCUMENT {i} (Relevance: {score:.4f}):\n"
            formatted_docs += f"Title: {doc.get('metadata', {}).get('title', 'No title')}\n"
            formatted_docs += f"Source: {source}{chunk_info}\n"
            
            if section:
                formatted_docs += f"Section: {section.capitalize()} of document\n"
                
            formatted_docs += f"Content: {doc.get('page_content', 'No content')}\n\n"
            
        return formatted_docs
        
    def _combine_document_chunks(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Combine chunks from the same source document to avoid duplication in citations.
        
        Args:
            documents: List of document chunks
            
        Returns:
            List of combined documents (one per unique source)
        """
        if not documents:
            return []
            
        # Group documents by source
        source_groups = {}
        for doc in documents:
            source = doc.get("metadata", {}).get("source", "unknown")
            if source not in source_groups:
                source_groups[source] = []
            source_groups[source].append(doc)
        
        # For each source, either select the best chunk or combine chunks
        combined_docs = []
        for source, docs in source_groups.items():
            if len(docs) == 1:
                # If only one chunk, just use it
                combined_docs.append(docs[0])
            else:
                # Sort chunks by chunk_id if available
                docs.sort(key=lambda d: d.get("metadata", {}).get("chunk_id", 0))
                
                # If there are multiple chunks, we have options:
                # 1. Take the highest scoring chunk
                best_chunk = max(docs, key=lambda d: d.get("score", 0))
                
                # 2. Or combine them (but this can make very long docs)
                # Only combine if we have less than 3 chunks to avoid making it too long
                if len(docs) <= 3:
                    # Combine content with section markers
                    combined_content = ""
                    max_score = 0
                    for i, doc in enumerate(docs):
                        score = doc.get("score", 0)
                        max_score = max(max_score, score)
                        section_num = doc.get("metadata", {}).get("chunk_id", i)
                        total_sections = doc.get("metadata", {}).get("total_chunks", len(docs))
                        section_name = doc.get("metadata", {}).get("document_section", f"Section {section_num+1}/{total_sections}")
                        
                        combined_content += f"[{section_name.upper()}]\n"
                        combined_content += doc.get("page_content", "") + "\n\n"
                    
                    # Create a new combined document
                    combined_doc = docs[0].copy()  # Use the first doc as a base
                    combined_doc["page_content"] = combined_content
                    combined_doc["score"] = max_score  # Use the highest score
                    combined_doc["metadata"] = combined_doc.get("metadata", {}).copy()
                    combined_doc["metadata"]["is_combined"] = True
                    combined_doc["metadata"]["num_chunks_combined"] = len(docs)
                    
                    combined_docs.append(combined_doc)
                else:
                    # If too many chunks, just use the best one
                    combined_docs.append(best_chunk)
        
        # Resort the combined documents by score
        combined_docs.sort(key=lambda d: d.get("score", 0), reverse=True)
        
        logger.info(f"Combined {len(documents)} chunks into {len(combined_docs)} unique source documents")
        return combined_docs
    
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
    
    async def _generate_findings_with_retry(self, prompt_inputs: dict, state: WorkflowState, max_retries: int = 3) -> str:
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
                # Checkpoint 1 first: the prompt is built from the retrieved text it kept (SEC-031).
                context_key = "context" if "context" in prompt_inputs else "documents"
                context_blob = str(prompt_inputs.get(context_key) or "")
                hop = await acheck_hop(
                    "internal_research",
                    task_items(state)
                    + ([ContextItem("retrieved_doc", context_blob, ref="retrieval")] if context_blob else []),
                    state,
                )
                if context_blob:
                    prompt_inputs = {**prompt_inputs, context_key: kept_text(hop, "retrieval", context_blob)}
                formatted_prompt = self.prompt.format_messages(**prompt_inputs)
                
                # Invoke the LLM directly, avoiding the chain to prevent tags conflicts
                from langchain_core.output_parsers import StrOutputParser
                
                logger.info("Invoking LLM directly without chain to avoid tags conflict")
                raw_llm_output = await self.llm.ainvoke(formatted_prompt)
                
                # Extract the content from the LLM output
                if hasattr(raw_llm_output, "content"):
                    # Handle ChatMessage or similar objects
                    return raw_llm_output.content
                elif isinstance(raw_llm_output, str):
                    # Handle direct string output
                    return raw_llm_output
                elif isinstance(raw_llm_output, dict) and "content" in raw_llm_output:
                    # Handle dictionary with content field
                    return raw_llm_output["content"]
                else:
                    # Try to convert to string
                    return str(raw_llm_output)
                
            except Exception as e:
                error_msg = str(e)
                attempts += 1
                logger.error(f"LLM invocation attempt {attempts} failed: {error_msg}")
                
                # Check for rate limiting errors
                if "429" in error_msg or "Too Many Requests" in error_msg:
                    delay = base_delay * (2 ** attempts)  # Exponential backoff
                    logger.warning(f"Rate limit hit. Retrying in {delay}s (attempt {attempts}/{max_retries})")
                    await asyncio.sleep(delay)
                    continue
                
                # Try alternative approach if there's a tags error
                if "tags" in error_msg and attempts < max_retries:
                    try:
                        logger.warning("Trying alternative LLM invocation method")
                        # Create a brand new LLM instance without any config
                        from app.services.llm import get_agent_llm
                        temporary_llm = get_agent_llm("internal_research")
                        
                        # Format the prompt and invoke directly
                        raw_output = await temporary_llm.ainvoke(formatted_prompt)
                        
                        # Extract content from output
                        if hasattr(raw_output, "content"):
                            return raw_output.content
                        elif isinstance(raw_output, str):
                            return raw_output
                        elif isinstance(raw_output, dict) and "content" in raw_output:
                            return raw_output["content"]
                        else:
                            return str(raw_output)
                    except Exception as alt_error:
                        logger.error(f"Alternative LLM invocation failed: {str(alt_error)}")
                        # Continue to next attempt or return error
                
                # For other errors, continue to the next attempt
                if attempts >= max_retries:
                    # Return error message on final attempt
                    logger.error(f"All attempts failed. Last error: {error_msg}")
                    return f"Error generating research findings: {error_msg}"
                
        # If we've exhausted retries
        return "Error generating research findings: Maximum retry attempts exceeded."
    
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
        logger.info(f"Internal Research Agent running for query ({describe_text(state.query)})")
        if settings.WEBAGENT_ENV == "dev":
            logger.debug("Query: %s", state.query)
        state.current_step = "internal_research"
        
        if TRACKING_ENABLED:
            task_tracking.start_run("rag_agent_run", {"query": state.query})
        
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
                    task_tracking.log_param("status", "skipped")
                    task_tracking.end_run()
                
                return state
            
            # Modify search query if this is a follow-up research request
            search_query = research_query(state)
            if is_followup:
                feedback = state.context.get("research_feedback", {})
                questions = feedback.get("research_questions", [])
                
                # If there are specific questions, use them to enhance the query
                if questions:
                    search_query = f"{research_query(state)} {questions[0]}"
                    logger.info(f"Using enhanced query for search ({describe_text(search_query)})")
            
            # 1. Retrieve documents using the RAG Retriever
            logger.info(f"Retrieving internal documents (iteration {iteration})")
            if await guard_tool(
                "tool::internal_retrieve",
                {"query": search_query},
                state,
                principal="internal_research",
            ):
                retrieved_documents = await self.retriever.retrieve(
                    query=search_query,
                    k=min(self.config.get("retrieve_k", 20), self.max_documents)
                )
                _remember_private_text(state, retrieved_documents)
            else:
                retrieved_documents = []
            
            # If follow-up research and there are specific questions, get additional documents
            if is_followup:
                feedback = state.context.get("research_feedback", {})
                questions = feedback.get("research_questions", [])[1:]  # Get remaining questions
                
                for i, question in enumerate(questions[:2]):  # Limit to 2 additional questions
                    logger.info(f"Performing additional document retrieval ({describe_text(question)})")
                    if not await guard_tool(
                        "tool::internal_retrieve",
                        {"query": question},
                        state,
                        principal="internal_research",
                    ):
                        continue
                    additional_docs = await self.retriever.retrieve(
                        query=question,
                        k=min(self.config.get("retrieve_k", 10), self.max_documents)  # Fewer documents for follow-up questions
                    )
                    _remember_private_text(state, additional_docs)
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
                    task_tracking.log_param("status", "no_documents")
                    task_tracking.end_run()
                
                return state
            
            # 1.5 Deduplicate retrieved documents
            logger.info(f"Deduplicating {len(retrieved_documents)} retrieved documents")
            
            # Log source distribution before deduplication
            source_counts_before = {}
            for doc in retrieved_documents:
                source = doc.get("metadata", {}).get("source", "Unknown")
                source_counts_before[source] = source_counts_before.get(source, 0) + 1
            
            # Log the most common sources before deduplication
            logger.info("Source distribution before deduplication:")
            for source, count in sorted(source_counts_before.items(), key=lambda x: x[1], reverse=True)[:5]:
                logger.info(f"  - {source}: {count} documents")
            
            # Perform deduplication
            retrieved_documents = self._deduplicate_documents(retrieved_documents)
            
            # Log source distribution after deduplication
            source_counts_after = {}
            for doc in retrieved_documents:
                source = doc.get("metadata", {}).get("source", "Unknown")
                source_counts_after[source] = source_counts_after.get(source, 0) + 1
            
            # Log the most common sources after deduplication
            logger.info("Source distribution after deduplication:")
            for source, count in sorted(source_counts_after.items(), key=lambda x: x[1], reverse=True)[:5]:
                logger.info(f"  - {source}: {count} documents")
                
            logger.info(f"After deduplication: {len(retrieved_documents)} documents ({len(source_counts_after)} unique sources)")
            
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
            analysis = await self._generate_findings_with_retry(prompt_inputs, state)
            
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
                    "top_document_score": reranked_documents[0].get("score", 0) if reranked_documents else 0,
                    "deduplication": {
                        "sources_before": len(source_counts_before),
                        "sources_after": len(source_counts_after),
                        "documents_before_dedup": sum(source_counts_before.values()),
                        "documents_after_dedup": sum(source_counts_after.values()),
                        "source_counts": source_counts_after
                    }
                }
            }
            
            # Update the state with the internal research report
            state.update_with_agent_output("internal_research", internal_research_report)
            
            # Add the findings to the context for other agents to use
            state.context["internal_research_findings"] = analysis
            
            if TRACKING_ENABLED:
                task_tracking.log_param("status", "completed")
                task_tracking.log_param("iteration", iteration)
                task_tracking.log_metric("total_time", internal_research_report["metrics"]["total_time"])
                task_tracking.log_metric("num_retrieved", internal_research_report["metrics"]["num_retrieved"])
                task_tracking.log_metric("num_reranked", internal_research_report["metrics"]["num_reranked"])
                task_tracking.end_run()
            
            logger.info(f"Internal Research completed with {len(reranked_documents)} relevant documents in {internal_research_report['metrics']['total_time']:.2f}s (iteration {iteration}).")
            return state
            
        except Exception as e:
            logger.error(f"Error in Internal Research Agent: {str(e)}")
            state.mark_error(f"Internal Research Agent failed: {str(e)}")
            
            if TRACKING_ENABLED:
                task_tracking.log_param("status", "error")
                task_tracking.log_param("error", str(e))
                task_tracking.end_run()
                
            return state

    def _deduplicate_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicate documents based on content similarity and source.
        
        Args:
            documents: List of documents to deduplicate
            
        Returns:
            List of deduplicated documents
        """
        if not documents:
            return []
            
        # Use sets to track unique document content hashes and sources
        unique_docs = []
        content_hashes = set()
        source_hashes = set()
        
        # Track detailed info for logging
        dedup_reasons = {
            "exact_content_match": 0,
            "source_and_content_similarity": 0,
            "kept": 0
        }
        
        for doc in documents:
            # Get document content
            content = doc.get("page_content", "")
            
            # Get document source 
            metadata = doc.get("metadata", {})
            source = metadata.get("source", "")
            
            # Create content hash
            content_preview = f"{content[:100]}_{len(content)}_{content[-100:]}"
            content_hash = hash(content_preview)
            
            # Create source hash
            source_hash = hash(source) if source else None
            
            # Skip this document if we've seen identical content
            if content_hash in content_hashes:
                dedup_reasons["exact_content_match"] += 1
                continue
                
            # If we have a source and it's already been seen, check content similarity
            if source_hash and source_hash in source_hashes:
                # Compare with already accepted documents with the same source
                for accepted_doc in unique_docs:
                    accepted_source = accepted_doc.get("metadata", {}).get("source", "")
                    if accepted_source == source:
                        # Compare content similarity
                        accepted_content = accepted_doc.get("page_content", "")
                        similarity = self._calculate_content_similarity(content, accepted_content)
                        
                        # If similarity is above threshold, skip this document
                        if similarity > 0.7:  # Threshold for similarity
                            dedup_reasons["source_and_content_similarity"] += 1
                            # Add the similarity score to the document for debugging
                            doc["dedup_similarity"] = similarity
                            break
                else:
                    # No similar document found, add this one
                    content_hashes.add(content_hash)
                    if source_hash:
                        source_hashes.add(source_hash)
                    unique_docs.append(doc)
                    dedup_reasons["kept"] += 1
            else:
                # New content and source, add it
                content_hashes.add(content_hash)
                if source_hash:
                    source_hashes.add(source_hash)
                unique_docs.append(doc)
                dedup_reasons["kept"] += 1
        
        # Log deduplication results
        logger.info(f"Deduplicated {len(documents)} documents to {len(unique_docs)} unique documents")
        logger.info(f"Deduplication reasons: {dedup_reasons}")
        
        return unique_docs

    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """
        Calculate similarity between two content strings.
        
        Args:
            content1: First content string
            content2: Second content string
            
        Returns:
            Similarity score between 0 and 1
        """
        # Simple word overlap similarity
        if not content1 or not content2:
            return 0.0
            
        # Convert to lowercase and split into words
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union == 0:
            return 0.0
            
        return intersection / union


_internal_research_function = None


def get_internal_research_agent():
    """
    Get the shared RAG-based Internal Research Agent (created once, reused on every research pass).

    Returns:
        A callable function that takes a WorkflowState and calls the agent's run method.
        This makes it compatible with LangGraph which expects nodes to be callable.
    """
    global _internal_research_function
    if _internal_research_function is None:
        agent = InternalResearchAgent()

        async def internal_research_function(state: WorkflowState) -> WorkflowState:
            """Callable wrapper for the internal research agent"""
            return await agent.run(state)

        _internal_research_function = internal_research_function
    return _internal_research_function