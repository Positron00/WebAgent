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

from langchain.schema.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from app.core.config import settings
from app.services.graphdb import get_graph_db_service

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
            # Ensure the directory exists
            os.makedirs(settings.VECTOR_DB_PATH, exist_ok=True)
            
            # Create embeddings
            self.embeddings = OpenAIEmbeddings(
                api_key=settings.OPENAI_API_KEY
            )
            
            # Create vector database
            self.vectordb = Chroma(
                persist_directory=settings.VECTOR_DB_PATH,
                embedding_function=self.embeddings
            )
            
            # Get graph database service
            self.graph_db = get_graph_db_service()
            
            self.initialized = True
            logger.info(f"Vector database initialized at {settings.VECTOR_DB_PATH}")
        except Exception as e:
            logger.error(f"Error initializing vector database: {str(e)}")
            self.error = str(e)
    
    @timing_decorator
    async def add_documents(self, documents: List[Document]) -> List[str]:
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
            # Add to vector database
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
    
    async def search(self, query: str, k: int = 5):
        """
        Search for documents related to the query.
        
        Args:
            query: The search query
            k: Number of results to return
            
        Returns:
            List of documents
        """
        if not self.initialized:
            logger.error("Vector database not initialized")
            return []
        
        try:
            results = await self.vectordb.asimilarity_search_with_score(
                query=query,
                k=k
            )
            return results
        except Exception as e:
            logger.error(f"Error searching vector database: {str(e)}")
            return []

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