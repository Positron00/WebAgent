"""
Test suite for the document loader service.
"""
import os
import pytest
import tempfile
import asyncio
from pathlib import Path
import shutil
from unittest.mock import patch, MagicMock

from backend.app.services.document_loader import DocumentLoaderService, get_document_loader_service
from backend.app.services.vectordb import VectorDBService
from backend.app.services.graphdb import GraphDBService
from langchain.schema.document import Document

class TestDocumentLoaderService:
    """Tests for the document loader service."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_files(self, temp_dir):
        """Create sample files for testing."""
        # Create a text file
        txt_path = os.path.join(temp_dir, "sample.txt")
        with open(txt_path, "w") as f:
            f.write("This is a sample text file for testing document loading.")
        
        # Create a markdown file
        md_path = os.path.join(temp_dir, "sample.md")
        with open(md_path, "w") as f:
            f.write("# Sample Markdown\n\nThis is a sample markdown file for testing.")
        
        return {
            "txt": txt_path,
            "md": md_path,
            "dir": temp_dir
        }
    
    @pytest.fixture
    def mock_vector_db(self):
        """Mock the vector database service."""
        with patch("backend.app.services.document_loader.get_vectordb_service") as mock:
            vector_db = MagicMock()
            vector_db.initialized = True
            vector_db.add_documents.return_value = ["doc1", "doc2"]
            mock.return_value = vector_db
            yield vector_db
    
    @pytest.fixture
    def mock_document_loader(self, mock_vector_db):
        """Create a document loader service with mocked dependencies."""
        with patch("backend.app.services.document_loader.settings") as mock_settings:
            # Configure settings
            mock_settings.ARTIFACTS_PATH = tempfile.mkdtemp()
            mock_settings.DOCUMENT_CHUNK_SIZE = 500
            mock_settings.DOCUMENT_CHUNK_OVERLAP = 50
            
            # Create service
            loader = DocumentLoaderService()
            
            yield loader
            
            # Clean up
            shutil.rmtree(mock_settings.ARTIFACTS_PATH)
    
    @pytest.mark.asyncio
    async def test_load_single_document(self, mock_document_loader, sample_files):
        """Test loading a single document."""
        # Mock the text splitter
        with patch("backend.app.services.document_loader.RecursiveCharacterTextSplitter") as mock_splitter:
            # Configure the mock
            mock_splitter_instance = MagicMock()
            mock_splitter_instance.split_documents.return_value = [
                Document(page_content="Chunk 1", metadata={}),
                Document(page_content="Chunk 2", metadata={})
            ]
            mock_splitter.return_value = mock_splitter_instance
            
            # Load a text file
            artifact_id = await mock_document_loader.load_document(sample_files["txt"])
            
            # Verify the artifact was created
            assert artifact_id is not None
            assert len(artifact_id) > 0
            
            # Verify the vector DB was called
            mock_document_loader.vector_db.add_documents.assert_called_once()
            
            # Verify artifact was stored
            artifact_dir = Path(mock_document_loader.artifacts_path) / artifact_id
            assert artifact_dir.exists()
            assert (artifact_dir / "metadata.json").exists()
    
    @pytest.mark.asyncio
    async def test_batch_load_documents(self, mock_document_loader, sample_files):
        """Test batch loading documents."""
        # Mock the load_document method to track calls
        original_load_document = mock_document_loader.load_document
        artifact_ids = {
            "sample.txt": "id1",
            "sample.md": "id2"
        }
        
        async def mock_load_document(file_path, metadata=None):
            return artifact_ids.get(os.path.basename(file_path), "unknown")
        
        with patch.object(mock_document_loader, "load_document", side_effect=mock_load_document):
            # Before patching the glob method, verify the files exist
            temp_dir = Path(sample_files["dir"])
            txt_file = temp_dir / "sample.txt"
            md_file = temp_dir / "sample.md"
            assert txt_file.exists(), f"Test file {txt_file} does not exist"
            assert md_file.exists(), f"Test file {md_file} does not exist"
            
            # Mock the glob method to return our sample files
            with patch("pathlib.Path.glob") as mock_glob:
                mock_glob.return_value = [
                    Path(sample_files["txt"]), 
                    Path(sample_files["md"])
                ]
                
                # Batch load documents
                results = await mock_document_loader.batch_load_documents(
                    sample_files["dir"],
                    "*.txt,*.md",  # Changed pattern format
                    {"batch": "test"}
                )
                
                # Verify results
                assert len(results) == 2
                assert "sample.txt" in results
                assert "sample.md" in results
                assert results["sample.txt"] == "id1"
                assert results["sample.md"] == "id2"
    
    @pytest.mark.asyncio
    async def test_get_document_loader_service(self):
        """Test the singleton factory function."""
        # Mock the document loader class
        with patch("backend.app.services.document_loader.DocumentLoaderService") as mock_cls:
            # First call should create a new instance
            service1 = get_document_loader_service()
            mock_cls.assert_called_once()
            
            # Second call should return the same instance
            mock_cls.reset_mock()
            service2 = get_document_loader_service()
            mock_cls.assert_not_called()
            
            # Verify same instance
            assert service1 == service2

if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 