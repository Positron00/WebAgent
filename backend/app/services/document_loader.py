"""
Document Loader Service
======================

Provides batch document loading functionality for the WebAgent backend.
"""
from typing import Dict, List, Optional, Union, Any
import logging
import os
import json
import asyncio
from pathlib import Path
from datetime import datetime
import uuid
import shutil
import tempfile
import time

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader, 
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader,
    CSVLoader,
    JSONLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

from app.core.config import settings
from app.services.vectordb import get_vectordb_service
from app.utils.metrics import timing_decorator

logger = logging.getLogger(__name__)

class DocumentLoaderService:
    """Service for batch loading documents into the system."""

    # Map file extensions to appropriate loaders
    LOADER_MAP = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".md": UnstructuredMarkdownLoader,
        ".docx": Docx2txtLoader,
        ".doc": Docx2txtLoader,
        ".html": UnstructuredHTMLLoader,
        ".htm": UnstructuredHTMLLoader,
        ".csv": CSVLoader,
        ".json": lambda path: JSONLoader(file_path=path, jq_schema='.', text_content=False)
    }

    def __init__(self):
        self.vector_db = get_vectordb_service()
        self.artifacts_path = Path(settings.ARTIFACTS_PATH)
        self.chunk_size = settings.DOCUMENT_CHUNK_SIZE
        self.chunk_overlap = settings.DOCUMENT_CHUNK_OVERLAP
        
        # Ensure artifacts directory exists
        self.artifacts_path.mkdir(parents=True, exist_ok=True)
        self.loader_map = self.LOADER_MAP

    @classmethod
    def supported_extensions(cls) -> frozenset:
        """Lower-case file suffixes (with the dot) that a loader exists for."""
        return frozenset(cls.LOADER_MAP)
    
    def _get_loader_for_file(self, file_path: Union[str, Path]) -> Optional[Any]:
        """Get the appropriate document loader for a file based on its extension."""
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        return self.loader_map.get(extension)
    
    def _store_artifact(self, file_path: Union[str, Path]) -> str:
        """
        Store a document as an artifact with a unique ID.
        
        Args:
            file_path: Path to the document
            
        Returns:
            str: The unique ID of the stored artifact
        """
        file_path = Path(file_path)
        artifact_id = str(uuid.uuid4())
        
        # Create a directory for this artifact
        artifact_dir = self.artifacts_path / artifact_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy the file
        dest_path = artifact_dir / file_path.name
        shutil.copy2(file_path, dest_path)
        
        # Create metadata file
        metadata = {
            "original_filename": file_path.name,
            "upload_time": datetime.now().isoformat(),
            "file_size": os.path.getsize(file_path),
            "content_type": file_path.suffix.lstrip("."),
        }
        
        with open(artifact_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
        return artifact_id
    
    @timing_decorator
    async def load_document(self, file_path: Union[str, Path], metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Load a single document into the system.
        
        Args:
            file_path: Path to the document
            metadata: Optional metadata to associate with the document
            
        Returns:
            str: The unique ID of the loaded document
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        # Get appropriate loader
        loader_class = self._get_loader_for_file(file_path)
        if loader_class is None:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        
        logger.info(f"Loading document: {file_path}")
        # Drift samples (OBS-002): numbers and the extension only, never the name or the text.
        started = time.perf_counter()
        size = file_path.stat().st_size
        doc_type = file_path.suffix.lower().lstrip(".") or "none"
        loaded = False

        try:
            # Load document
            loader = loader_class(str(file_path))
            documents = loader.load()
            
            # Add any provided metadata
            if metadata:
                for doc in documents:
                    doc.metadata.update(metadata)
            
            # Store as artifact
            artifact_id = self._store_artifact(file_path)
            
            # Add artifact ID to metadata
            for doc in documents:
                doc.metadata["artifact_id"] = artifact_id
                doc.metadata["source"] = str(file_path)
            
            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            chunks = text_splitter.split_documents(documents)
            
            # Add chunk identifiers to metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata["chunk_id"] = i
                chunk.metadata["total_chunks"] = len(chunks)
                # Also add a document section identifier to help with citations
                start_percent = int((i / len(chunks)) * 100)
                end_percent = int(((i + 1) / len(chunks)) * 100)
                if start_percent == 0 and end_percent <= 33:
                    chunk.metadata["document_section"] = "beginning"
                elif start_percent >= 33 and end_percent <= 66:
                    chunk.metadata["document_section"] = "middle"
                else:
                    chunk.metadata["document_section"] = "end"
            
            # Add to vector database (not async, so don't use await)
            self.vector_db.add_documents(chunks)
            
            logger.info(f"Document loaded successfully: {file_path}, ID: {artifact_id}")
            loaded = True
            return artifact_id

        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            raise
        finally:
            _record_document_samples(time.perf_counter() - started, size, doc_type, loaded)
    
    @timing_decorator
    async def batch_load_documents(self, directory_path: Union[str, Path], glob_pattern: str = "*.*", metadata: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """
        Load multiple documents from a directory into the system.
        
        Args:
            directory_path: Path to the directory containing documents
            glob_pattern: Pattern to match files (default: "*.*")
                          Can be comma-separated patterns like "*.pdf,*.txt"
            metadata: Optional metadata to associate with all documents
            
        Returns:
            Dict[str, str]: Mapping of filenames to their artifact IDs
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists() or not directory_path.is_dir():
            raise NotADirectoryError(f"Directory not found: {directory_path}")
        
        # Find all matching files - handle comma-separated patterns
        matching_files = []
        
        # Split by comma if present
        patterns = glob_pattern.split(',')
        for pattern in patterns:
            pattern = pattern.strip()
            matching_files.extend(list(directory_path.glob(pattern)))
        
        # Remove duplicates and sort
        matching_files = sorted(set(matching_files))
        
        if not matching_files:
            logger.warning(f"No files matching pattern '{glob_pattern}' found in {directory_path}")
            return {}
        
        logger.info(f"Found {len(matching_files)} files to process in {directory_path}")
        
        # Process all files concurrently
        artifact_ids = {}
        tasks = []
        
        for file_path in matching_files:
            if file_path.is_file():
                # Add file-specific metadata
                file_metadata = metadata.copy() if metadata else {}
                file_metadata.update({
                    "batch_import": True,
                    "import_time": datetime.now().isoformat(),
                    "directory": str(directory_path)
                })
                
                # Create task
                task = self.load_document(file_path, file_metadata)
                tasks.append((file_path.name, task))
        
        # Await all tasks
        for filename, task in tasks:
            try:
                artifact_id = await task
                artifact_ids[filename] = artifact_id
            except Exception as e:
                logger.error(f"Error loading {filename}: {str(e)}")
                artifact_ids[filename] = f"ERROR: {str(e)}"
        
        return artifact_ids

# Singleton instance
_document_loader_service = None

def get_document_loader_service():
    """Get the document loader service instance."""
    global _document_loader_service
    if _document_loader_service is None:
        _document_loader_service = DocumentLoaderService()
    return _document_loader_service


def _record_document_samples(seconds: float, size: int, doc_type: str, loaded: bool) -> None:
    from app.core import drift_store

    drift_store.add("doc_processing_time", round(seconds, 4))
    drift_store.add("doc_size", size)
    drift_store.add("doc_type", doc_type)
    drift_store.add("doc_outcome", "ok" if loaded else "error")
    try:  # the dashboard's "Document processing time (p95)" panel
        from app.core.observability import DOCUMENT_PROCESSING_LATENCY

        DOCUMENT_PROCESSING_LATENCY.labels(doc_type=doc_type).observe(seconds)
    except Exception as exc:  # noqa: BLE001 - metrics must never break an upload
        logger.debug("Document latency not recorded: %s", exc)
