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
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.embeddings import HuggingFaceEmbeddings
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

Your task:
1. Analyze the documents and extract relevant information for the query
2. Organize the information in a clear, structured format
3. Cite your sources with document references
4. Only include information that is directly related to the query
5. Highlight key facts, statistics, and insights
6. Format your findings as markdown

IMPORTANT: Be thorough, accurate, and objective. Cite all sources. If the information is not in the provided documents, state that clearly.
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
        
        # Initialize a more advanced dense retriever with Contriever or another modern embedding model
        try:
            # Using Contriever for better semantic understanding
            model_name = self.config.get("embedding_model", "facebook/contriever")
            self.advanced_embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
            
            # Create a persistent directory for the advanced vector store
            advanced_persist_dir = os.path.join(settings.VECTOR_DB_PATH, "advanced")
            os.makedirs(advanced_persist_dir, exist_ok=True)
            
            # Initialize the advanced vector store
            self.advanced_vectordb = Chroma(
                persist_directory=advanced_persist_dir,
                embedding_function=self.advanced_embeddings
            )
            
            # Create retrievers
            self.dense_retriever = self.advanced_vectordb.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.config.get("dense_k", 20)}
            )
            
            # Create BM25 sparse retriever for keyword-based search
            self.sparse_retriever = BM25Retriever.from_documents(
                self.advanced_vectordb.get(),
                preprocess_func=lambda text: text.lower().split(),
                k=self.config.get("sparse_k", 20)
            )
            
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
            if self.initialized:
                # Use the ensemble retriever for better results
                docs = await asyncio.to_thread(
                    self.ensemble_retriever.get_relevant_documents,
                    query
                )
                
                # Convert to the expected format
                results = []
                for doc in docs:
                    results.append({
                        "page_content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": 1.0  # Placeholder score for ensemble retriever
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
            # Use a cross-encoder model for reranking
            model_name = self.config.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
            
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
        self.llm = get_llm(self.config.get("llm_model", "gpt-4-turbo"))
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
            
            # 1. Retrieve documents using the RAG Retriever
            logger.info("Retrieving internal documents")
            retrieved_documents = await self.retriever.retrieve(
                query=state.query,
                k=self.config.get("retrieve_k", 20)
            )
            
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
            
            # 2. Rerank documents to prioritize the most relevant ones
            logger.info("Reranking retrieved documents")
            reranked_documents = await self.reranker.rerank(
                query=state.query,
                documents=retrieved_documents,
                top_n=self.config.get("rerank_top_n", 5)
            )
            
            # Format internal documents for the prompt
            formatted_docs = self._format_internal_documents(reranked_documents)
            
            # 3. Generate answer using the LLM with reranked documents as context
            logger.info("Generating research findings")
            analysis = await self.chain.ainvoke({
                "query": state.query,
                "research_plan": research_plan.get("analysis", "No research plan provided."),
                "documents": formatted_docs
            })
            
            # Create the internal research report
            internal_research_report = {
                "status": "completed",
                "documents": reranked_documents,
                "findings": analysis,
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
                mlflow.log_metric("total_time", internal_research_report["metrics"]["total_time"])
                mlflow.log_metric("num_retrieved", internal_research_report["metrics"]["num_retrieved"])
                mlflow.log_metric("num_reranked", internal_research_report["metrics"]["num_reranked"])
                mlflow.end_run()
            
            logger.info(f"Internal Research completed with {len(reranked_documents)} relevant documents in {internal_research_report['metrics']['total_time']:.2f}s.")
            return state
            
        except Exception as e:
            logger.error(f"Error in Internal Research Agent: {str(e)}")
            state.mark_error(f"Internal Research Agent failed: {str(e)}")
            
            if TRACKING_ENABLED:
                mlflow.log_param("status", "error")
                mlflow.log_param("error", str(e))
                mlflow.end_run()
                
            return state


def get_internal_research_agent() -> InternalResearchAgent:
    """
    Get an instance of the advanced RAG-based Internal Research Agent.
    
    Returns:
        An InternalResearchAgent instance with RAG capabilities
    """
    return InternalResearchAgent() 