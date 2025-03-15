"""
Document API endpoints for WebAgent backend.

These endpoints provide document operations as a data-as-a-service interface.
"""
from typing import Dict, List, Optional, Any
import logging
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import json
import os
from pathlib import Path
import uuid
import tempfile
import shutil
import asyncio

from app.services.document_loader import get_document_loader_service
from app.services.vectordb import get_vectordb_service
from app.services.graphdb import get_graph_db_service
from app.core.config import settings
from app.utils.metrics import timing_decorator

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Models

class DocumentMetadata(BaseModel):
    """Document metadata model."""
    category: Optional[str] = Field(None, description="Document category")
    tags: Optional[List[str]] = Field(None, description="Document tags")
    source: Optional[str] = Field(None, description="Document source")
    custom_metadata: Optional[Dict[str, Any]] = Field(None, description="Custom metadata")

class DocumentSearchQuery(BaseModel):
    """Document search query model."""
    query: str = Field(..., description="Search query text")
    limit: int = Field(10, description="Maximum number of results")
    include_content: bool = Field(False, description="Whether to include document content")

class DocumentResponse(BaseModel):
    """Document response model."""
    id: str = Field(..., description="Document ID")
    filename: str = Field(..., description="Original filename")
    metadata: Dict[str, Any] = Field(..., description="Document metadata")
    upload_time: str = Field(..., description="Upload timestamp")
    content_preview: Optional[str] = Field(None, description="Content preview")

class DocumentSearchResponse(BaseModel):
    """Document search response model."""
    id: str = Field(..., description="Document ID")
    score: float = Field(..., description="Relevance score")
    metadata: Dict[str, Any] = Field(..., description="Document metadata")
    content: Optional[str] = Field(None, description="Document content")

class BatchImportResponse(BaseModel):
    """Batch import response model."""
    total: int = Field(..., description="Total number of files processed")
    successful: int = Field(..., description="Number of successfully processed files")
    failed: int = Field(..., description="Number of failed files")
    document_ids: List[str] = Field(..., description="List of created document IDs")
    errors: Dict[str, str] = Field(..., description="Map of filename to error message")

# Endpoints

@router.post("/upload", response_model=DocumentResponse)
@timing_decorator
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    metadata_json: Optional[str] = Form(None)
) -> DocumentResponse:
    """
    Upload a single document for processing.
    
    Args:
        file: The file to upload
        metadata_json: Optional JSON string with document metadata
        
    Returns:
        Document information
    """
    document_loader = get_document_loader_service()
    
    # Parse metadata
    metadata = {}
    if metadata_json:
        try:
            metadata = json.loads(metadata_json)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid metadata JSON: {str(e)}")
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
        # Write uploaded file data to temp file
        shutil.copyfileobj(file.file, temp_file)
        temp_file_path = temp_file.name
    
    try:
        # Add original filename to metadata
        metadata["original_filename"] = file.filename
        
        # Process the document
        artifact_id = await document_loader.load_document(temp_file_path, metadata)
        
        # Get artifact metadata
        artifact_dir = Path(settings.ARTIFACTS_PATH) / artifact_id
        with open(artifact_dir / "metadata.json", "r") as f:
            artifact_metadata = json.load(f)
        
        return DocumentResponse(
            id=artifact_id,
            filename=file.filename,
            metadata=metadata,
            upload_time=artifact_metadata["upload_time"],
            content_preview=metadata.get("content_preview", "")
        )
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
    finally:
        # Clean up temp file
        os.unlink(temp_file_path)

@router.post("/search", response_model=List[DocumentSearchResponse])
@timing_decorator
async def search_documents(
    query: DocumentSearchQuery
) -> List[DocumentSearchResponse]:
    """
    Search for documents by content.
    
    Args:
        query: Search parameters
        
    Returns:
        List of matching documents
    """
    vector_db = get_vectordb_service()
    
    try:
        # Search for documents
        results = await vector_db.search(query.query, query.limit)
        
        # Format results
        responses = []
        for doc, score in results:
            # Extract metadata
            metadata = doc.metadata.copy() if hasattr(doc, 'metadata') else {}
            
            # Create response
            response = DocumentSearchResponse(
                id=metadata.get("artifact_id", "unknown"),
                score=score,
                metadata=metadata,
                content=doc.page_content if query.include_content else None
            )
            responses.append(response)
            
        return responses
    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error searching documents: {str(e)}")

@router.get("/document/{document_id}", response_model=DocumentResponse)
@timing_decorator
async def get_document(
    document_id: str,
    include_content: bool = Query(False, description="Whether to include document content")
) -> DocumentResponse:
    """
    Get document information by ID.
    
    Args:
        document_id: Document ID
        include_content: Whether to include document content
        
    Returns:
        Document information
    """
    # Check if document exists
    artifact_dir = Path(settings.ARTIFACTS_PATH) / document_id
    if not artifact_dir.exists():
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        # Get artifact metadata
        with open(artifact_dir / "metadata.json", "r") as f:
            artifact_metadata = json.load(f)
        
        # Find the original file
        files = list(artifact_dir.glob("*"))
        original_file = None
        for file in files:
            if file.name != "metadata.json":
                original_file = file
                break
        
        if not original_file:
            raise HTTPException(status_code=404, detail="Document file not found")
        
        # Create base response
        response = DocumentResponse(
            id=document_id,
            filename=artifact_metadata["original_filename"],
            metadata=artifact_metadata,
            upload_time=artifact_metadata["upload_time"],
            content_preview=artifact_metadata.get("content_preview", "")
        )
        
        # Add content if requested
        if include_content:
            # Handle by content type
            content_type = artifact_metadata.get("content_type", "")
            if content_type in ["txt", "md", "html", "csv", "json"]:
                with open(original_file, "r") as f:
                    response.metadata["content"] = f.read()
        
        return response
    except Exception as e:
        logger.error(f"Error retrieving document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving document: {str(e)}")

@router.get("/related/{document_id}", response_model=List[DocumentResponse])
@timing_decorator
async def get_related_documents(
    document_id: str,
    relationship_type: Optional[str] = Query(None, description="Filter by relationship type"),
    depth: int = Query(1, description="Relationship depth"),
    limit: int = Query(10, description="Maximum number of results")
) -> List[DocumentResponse]:
    """
    Get documents related to the specified document.
    
    Args:
        document_id: Document ID
        relationship_type: Optional relationship type filter
        depth: Relationship depth to traverse
        limit: Maximum number of results
        
    Returns:
        List of related documents
    """
    graph_db = get_graph_db_service()
    
    try:
        # Get related documents from graph database
        related = graph_db.find_related_documents(document_id, relationship_type, depth)
        
        # Limit results
        related = related[:limit]
        
        # Format responses
        responses = []
        for doc in related:
            doc_id = doc.get("id")
            
            # Skip if no valid ID
            if not doc_id:
                continue
                
            # Get artifact metadata if available
            artifact_dir = Path(settings.ARTIFACTS_PATH) / doc_id
            if artifact_dir.exists():
                try:
                    with open(artifact_dir / "metadata.json", "r") as f:
                        artifact_metadata = json.load(f)
                        
                    # Create response
                    response = DocumentResponse(
                        id=doc_id,
                        filename=artifact_metadata.get("original_filename", "unknown"),
                        metadata=artifact_metadata,
                        upload_time=artifact_metadata.get("upload_time", ""),
                        content_preview=doc.get("content_preview", "")
                    )
                    responses.append(response)
                except Exception as e:
                    logger.warning(f"Error getting metadata for related document {doc_id}: {str(e)}")
            
        return responses
    except Exception as e:
        logger.error(f"Error getting related documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting related documents: {str(e)}") 