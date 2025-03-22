"""
Vector database service for the WebAgent backend.
Provides document storage and retrieval.
"""
from typing import Dict, List, Optional, Any
import logging
import os
from pathlib import Path
from functools import lru_cache
import uuid
import time

from langchain.schema.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from app.core.config import settings
from app.services.graphdb import get_graph_db_service
from app.utils.metrics import timing_decorator
from app.core.observability import (
    track_vectordb_query,
    record_data_quality,
    VECTORDB_QUERY_COUNT,
    VECTORDB_QUERY_LATENCY,
    log_to_mlflow
)

logger = logging.getLogger(__name__)

class VectorDBService:
    """
    Vector database service for storing and retrieving documents.
    """
    _instance = None
    
    def __new__(cls):
        """Singleton pattern to ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super(VectorDBService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the vector database."""
        self.initialized = False
        
        try:
            # Handle empty VECTOR_DB_PATH
            if not settings.VECTOR_DB_PATH or settings.VECTOR_DB_PATH.strip() == '':
                default_path = os.path.join(settings.DATA_DIR, "vectordb")
                logger.warning(f"VECTOR_DB_PATH is empty, using default path: {default_path}")
                self.vector_db_path = default_path
            else:
                self.vector_db_path = settings.VECTOR_DB_PATH
            
            # Ensure the directory exists
            os.makedirs(self.vector_db_path, exist_ok=True)
            
            # Create embeddings
            self.embeddings = OpenAIEmbeddings(
                api_key=settings.OPENAI_API_KEY
            )
            
            # Create vector database
            self.vectordb = Chroma(
                persist_directory=self.vector_db_path,
                embedding_function=self.embeddings
            )
            
            # Get graph database service
            self.graph_db = get_graph_db_service()
            
            self.initialized = True
            logger.info(f"Vector database initialized at {self.vector_db_path}")
        except Exception as e:
            logger.error(f"Error initializing vector database: {str(e)}")
            self.error = str(e)
    
    @timing_decorator
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the vector database and graph database.
        
        Args:
            documents: List of documents to add
            
        Returns:
            List of document IDs
        """
        if not self.initialized:
            logger.error("Vector database not initialized")
            return []
        
        if not documents:
            logger.warning("No documents provided to add_documents")
            return []
        
        try:
            # Add to vector database (not async, so don't use await)
            doc_ids = self.vectordb.add_documents(documents)
            
            # Add to graph database and track document relationships
            for i, doc in enumerate(documents):
                doc_id = doc_ids[i] if doc_ids else f"doc_{i}_{uuid.uuid4()}"
                
                # Add document node to graph
                if self.graph_db.initialized:
                    # Extract metadata for graph
                    metadata = doc.metadata.copy() if hasattr(doc, 'metadata') else {}
                    metadata['content_preview'] = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                    
                    # Add document node
                    self.graph_db.add_document(doc_id, metadata)
                    
                    # Link related documents (if they're from the same source)
                    if 'source' in metadata:
                        # Find other documents with same source
                        for j, other_doc in enumerate(documents):
                            if i != j and 'source' in other_doc.metadata and other_doc.metadata['source'] == metadata['source']:
                                other_id = doc_ids[j] if doc_ids else f"doc_{j}_{uuid.uuid4()}"
                                self.graph_db.add_relationship(
                                    doc_id, 
                                    other_id, 
                                    "SAME_SOURCE", 
                                    {"source": metadata['source']}
                                )
            
            logger.info(f"Added {len(documents)} documents to vector database and graph database")
            return doc_ids
        except Exception as e:
            logger.error(f"Error adding documents to database: {str(e)}")
            return []
    
    @track_vectordb_query()
    async def search(self, query: str, top_k: int = 5, namespace: str = None):
        """
        Search for documents most similar to the query.
        
        Args:
            query: The query string
            top_k: Number of results to return
            namespace: Optional namespace to search in
            
        Returns:
            List of document chunks and their similarity scores
        """
        start_time = time.time()
        try:
            logger.debug(f"Searching vector database for: '{query}'")
            
            # Get embeddings for the query
            embedding = await self.get_embedding(query)
            
            if not embedding:
                logger.error("Failed to generate embedding for query")
                return []
                
            # Get the vector database client
            if not self.client:
                await self.connect()
                
            # Search the vector database
            results = await self.client.search(
                collection_name=self.collection_name,
                query_vector=embedding,
                top_k=top_k,
                namespace=namespace or self.namespace
            )
            
            processing_time = time.time() - start_time
            
            # Log to MLflow
            log_to_mlflow(
                metrics={
                    "search_latency": processing_time,
                    "search_result_count": len(results) if results else 0
                },
                params={
                    "query_length": len(query),
                    "top_k": top_k,
                    "namespace": namespace or self.namespace
                },
                experiment_name="vectordb_operations"
            )
            
            # Record the latency in our metrics
            VECTORDB_QUERY_LATENCY.observe(processing_time)
            
            logger.debug(f"Vector search completed in {processing_time:.4f}s with {len(results)} results")
            
            if not results:
                logger.warning(f"No results found for query: '{query}'")
                VECTORDB_QUERY_COUNT.labels(status="empty").inc()
                return []
                
            # Format results for return
            formatted_results = []
            for result in results:
                metadata = result.get("metadata", {})
                formatted_results.append({
                    "score": float(result.get("score", 0)),
                    "text": result.get("payload", {}).get("text", ""),
                    "metadata": {
                        "document_id": metadata.get("document_id", ""),
                        "source": metadata.get("source", ""),
                        "chunk_id": metadata.get("chunk_id", ""),
                        "page_number": metadata.get("page_number", 0)
                    }
                })
                
            # Track successful query
            VECTORDB_QUERY_COUNT.labels(status="success").inc()
            
            # Basic quality check
            if formatted_results:
                top_score = formatted_results[0]["score"]
                quality_score = min(max(top_score * 100, 0), 100)  
                record_data_quality("vector_search", "relevance", quality_score)
                
            return formatted_results
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error searching vector database: {str(e)}")
            
            # Track failed query
            VECTORDB_QUERY_COUNT.labels(status="error").inc()
            
            # Log to MLflow
            log_to_mlflow(
                metrics={
                    "search_latency": processing_time,
                    "search_error": 1
                },
                params={
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                },
                experiment_name="vectordb_operations"
            )
            
            return []
    
    @timing_decorator
    def get_document(self, document_id: str) -> Optional[Document]:
        """
        Retrieve a document by its ID.
        
        Args:
            document_id: The ID of the document to retrieve
            
        Returns:
            The document if found, None otherwise
        """
        try:
            # Get the document from the collection
            results = self.vectordb._collection.get(ids=[document_id])
            
            if not results or not results['documents'] or len(results['documents']) == 0:
                logger.warning(f"Document with ID {document_id} not found in vector database")
                return None
                
            # Create a Document object from the retrieved data
            metadata = results['metadatas'][0] if results['metadatas'] else {}
            content = results['documents'][0]
            
            document = Document(
                page_content=content,
                metadata=metadata
            )
            
            return document
        except Exception as e:
            logger.error(f"Error retrieving document from vector database: {str(e)}")
            return None
            
    @timing_decorator
    def get_document_by_artifact_id(self, artifact_id: str) -> Optional[Document]:
        """
        Retrieve document chunks by artifact ID and combine them.
        
        Args:
            artifact_id: The artifact ID of the document
            
        Returns:
            Combined document if found, None otherwise
        """
        try:
            # Query for documents with matching artifact_id in metadata
            results = self.vectordb._collection.get(
                where={"artifact_id": artifact_id}
            )
            
            if not results or not results['documents'] or len(results['documents']) == 0:
                logger.warning(f"No documents found with artifact_id {artifact_id}")
                return None
                
            # Combine all chunks into a single document
            combined_content = "\n".join(results['documents'])
            
            # Use metadata from the first chunk
            metadata = results['metadatas'][0] if results['metadatas'] else {}
            
            document = Document(
                page_content=combined_content,
                metadata=metadata
            )
            
            return document
        except Exception as e:
            logger.error(f"Error retrieving document by artifact_id: {str(e)}")
            return None

# Create instance
_db_service = None

def get_vectordb_service():
    """Get the vector database service instance."""
    global _db_service
    if _db_service is None:
        _db_service = VectorDBService()
    return _db_service

async def retrieve_documents(query: str, k: int = 5):
    """
    Retrieve documents related to the query.
    
    Args:
        query: The search query
        k: Number of results to return
        
    Returns:
        List of documents
    """
    db = get_vectordb_service()
    return await db.search(query, k)

async def get_vectordb_status():
    """
    Check the status of the vector database.
    
    Returns:
        Dict with status information
    """
    db = get_vectordb_service()
    
    if not db.initialized:
        return {
            "status": "error",
            "error": getattr(db, "error", "Not initialized"),
            "path": settings.VECTOR_DB_PATH
        }
    
    try:
        # Count documents
        doc_count = len(db.vectordb._collection.get()["documents"])
        
        return {
            "status": "ok",
            "document_count": doc_count,
            "path": settings.VECTOR_DB_PATH
        }
    except Exception as e:
        logger.error(f"Vector DB health check failed: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "path": settings.VECTOR_DB_PATH
        } 