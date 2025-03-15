"""
Vector database service for the WebAgent backend.
Provides document storage and retrieval.
"""
from typing import Dict, List, Optional, Any
import logging
import os
from pathlib import Path
from functools import lru_cache

from langchain.schema.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from app.core.config import settings

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
            
            self.initialized = True
            logger.info(f"Vector database initialized at {settings.VECTOR_DB_PATH}")
        except Exception as e:
            logger.error(f"Error initializing vector database: {str(e)}")
            self.error = str(e)
    
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