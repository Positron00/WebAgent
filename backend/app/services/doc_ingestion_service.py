"""Document Ingestion Service

This module handles document ingestion and processing.
"""

import time
import threading
from typing import Dict, List, Any, Optional

from app.core.logger import logger
from app.core.observability import (
    record_processing_time,
    record_document_count,
    record_document_size,
    record_error_count,
    log_to_mlflow
)
from app.core.drift_detection import (
    create_numerical_drift_detector,
    create_categorical_drift_detector,
    create_performance_drift_detector
)

# Initialize drift detectors
processing_time_detector = create_numerical_drift_detector("doc_processing_time", window_size=500, threshold=0.05)
doc_size_detector = create_numerical_drift_detector("doc_size", window_size=500, threshold=0.1)
doc_type_detector = create_categorical_drift_detector("doc_type", window_size=500, threshold=0.05)
doc_quality_detector = create_performance_drift_detector("doc_quality", window_size=200, threshold=0.15)

# Lock for thread safety
_lock = threading.RLock()

class DocIngestionService:
    """Service for document ingestion and processing"""
    
    def __init__(self):
        """Initialize the document ingestion service."""
        logger.info("Initialized document ingestion service")
        
        # Track if we're in baseline collection mode
        self.collecting_baseline = True
        self.baseline_count = 0
        self.max_baseline_samples = 50
    
    def process_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a document.
        
        Args:
            document: Document to process
            
        Returns:
            Dictionary with processing results
        """
        start_time = time.time()
        
        try:
            # Extract document metadata
            doc_id = document.get("id", "unknown")
            doc_type = document.get("type", "unknown")
            doc_content = document.get("content", "")
            doc_size = len(doc_content)
            
            # Here would be the actual document processing
            # For now we'll simulate it
            time.sleep(0.1)  # Simulated processing time
            
            # Simulate processing
            tokens = len(doc_content.split())
            entities = ["entity1", "entity2"]  # Simulated entities
            
            # Calculate quality metrics
            quality_score = min(0.95, len(doc_content) / 10000)  # Simulated quality score
            ocr_confidence = 0.9 if doc_type == "pdf" else 1.0  # Simulated OCR confidence
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Log metrics
            record_processing_time("document_processing", processing_time)
            record_document_count(doc_type, 1)
            record_document_size(doc_type, doc_size)
            
            # Track drift
            with _lock:
                if self.collecting_baseline and self.baseline_count < self.max_baseline_samples:
                    processing_time_detector.add_reference_sample(processing_time)
                    doc_size_detector.add_reference_sample(doc_size)
                    doc_type_detector.add_reference_sample(doc_type)
                    self.baseline_count += 1
                    
                    if self.baseline_count >= self.max_baseline_samples:
                        self.collecting_baseline = False
                        logger.info(f"Completed baseline collection for document drift detection with {self.baseline_count} samples")
                else:
                    # Add current samples and check for drift
                    time_score, time_drift = processing_time_detector.add_current_sample(processing_time)
                    size_score, size_drift = doc_size_detector.add_current_sample(doc_size)
                    type_score, type_drift = doc_type_detector.add_current_sample(doc_type)
                    
                    # Add quality metrics
                    quality_score, quality_drift = doc_quality_detector.add_performance_metrics(
                        accuracy=quality_score,
                        confidence=ocr_confidence
                    )
                    
                    # Log any detected drift
                    if time_drift or size_drift or type_drift or quality_drift:
                        logger.warning(f"Drift detected in document processing - Time: {time_drift}, Size: {size_drift}, Type: {type_drift}, Quality: {quality_drift}")
                        
                        # Check if pipeline adjustments are recommended
                        if doc_quality_detector.should_retrain():
                            logger.warning("Document pipeline adjustments recommended due to quality drift")
            
            # Log document processing to MLflow
            log_to_mlflow(
                metrics={
                    "processing_time": processing_time,
                    "doc_size": doc_size,
                    "token_count": tokens,
                    "quality_score": quality_score,
                    "ocr_confidence": ocr_confidence
                },
                params={
                    "doc_id": doc_id,
                    "doc_type": doc_type,
                    "entity_count": len(entities)
                },
                experiment_name="document_processing"
            )
            
            return {
                "doc_id": doc_id,
                "processing_time": processing_time,
                "tokens": tokens,
                "entities": entities,
                "quality_score": quality_score,
                "status": "success"
            }
            
        except Exception as e:
            # Record error
            processing_time = time.time() - start_time
            record_error_count("document_processing", 1)
            record_processing_time("document_processing", processing_time)
            
            logger.error(f"Error in document processing: {str(e)}")
            
            return {
                "error": str(e),
                "processing_time": processing_time,
                "status": "error"
            }
    
    def batch_process_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process a batch of documents.
        
        Args:
            documents: List of documents to process
            
        Returns:
            Dictionary with batch processing results
        """
        start_time = time.time()
        results = []
        success_count = 0
        error_count = 0
        
        for doc in documents:
            result = self.process_document(doc)
            results.append(result)
            
            if result.get("status") == "success":
                success_count += 1
            else:
                error_count += 1
        
        # Calculate total processing time
        total_processing_time = time.time() - start_time
        
        # Log batch metrics
        log_to_mlflow(
            metrics={
                "batch_size": len(documents),
                "batch_processing_time": total_processing_time,
                "success_rate": success_count / max(1, len(documents)),
                "error_rate": error_count / max(1, len(documents))
            },
            experiment_name="document_processing_batch"
        )
        
        return {
            "batch_size": len(documents),
            "success_count": success_count,
            "error_count": error_count,
            "total_processing_time": total_processing_time,
            "results": results
        }
    
    def get_drift_status(self) -> Dict[str, Any]:
        """
        Get the current drift status for the document processing service.
        
        Returns:
            Dictionary with drift statistics
        """
        time_stats = processing_time_detector.get_stats() if processing_time_detector else {}
        size_stats = doc_size_detector.get_stats() if doc_size_detector else {}
        type_stats = doc_type_detector.get_stats() if doc_type_detector else {}
        quality_stats = doc_quality_detector.get_stats() if doc_quality_detector else {}
        
        return {
            "collecting_baseline": self.collecting_baseline,
            "baseline_count": self.baseline_count,
            "processing_time_drift": time_stats,
            "doc_size_drift": size_stats,
            "doc_type_drift": type_stats,
            "doc_quality_drift": quality_stats,
            "pipeline_adjustments_recommended": doc_quality_detector.should_retrain() if doc_quality_detector else False
        }
    
    def reset_drift_baselines(self):
        """Reset all drift detection baselines."""
        with _lock:
            self.collecting_baseline = True
            self.baseline_count = 0
            
            # Create new detectors
            global processing_time_detector, doc_size_detector, doc_type_detector, doc_quality_detector
            processing_time_detector = create_numerical_drift_detector("doc_processing_time", window_size=500, threshold=0.05)
            doc_size_detector = create_numerical_drift_detector("doc_size", window_size=500, threshold=0.1)
            doc_type_detector = create_categorical_drift_detector("doc_type", window_size=500, threshold=0.05)
            doc_quality_detector = create_performance_drift_detector("doc_quality", window_size=200, threshold=0.15)
            
            logger.info("Reset drift baselines for document processing service")

# Global instance
doc_ingestion_service = DocIngestionService() 