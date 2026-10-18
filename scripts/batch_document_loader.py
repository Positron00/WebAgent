#!/usr/bin/env python
"""
Batch Document Loader Script
============================

This script provides a command-line interface for batch loading documents into the WebAgent system.
It loads a directory or single files into the vector store, the graph store and the local
artifacts folder (WEBAGENT_ARTIFACTS_PATH). There is no remote artifact registry.

Usage:
    python batch_document_loader.py --dir /path/to/documents --pattern "*.pdf"
    python batch_document_loader.py --file /path/to/document.pdf --metadata '{"category": "reports"}'
"""

import argparse
import asyncio
import json
import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import time

# Add the parent directory to the path so we can import the backend modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("batch_loader")

# Import backend services
from backend.app.services.document_loader import get_document_loader_service
from backend.app.core.config import settings

async def load_documents(args):
    """Load documents based on command-line arguments."""
    document_loader = get_document_loader_service()
    
    # Parse metadata if provided
    metadata = {}
    if args.metadata:
        try:
            metadata = json.loads(args.metadata)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing metadata JSON: {str(e)}")
            return
    
    # Add any additional metadata from arguments
    if args.category:
        metadata["category"] = args.category
    if args.tags:
        metadata["tags"] = args.tags.split(",")
    if args.source:
        metadata["source"] = args.source
        
    # Add timestamp
    metadata["import_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Process based on mode (file or directory)
    start_time = time.time()
    if args.file:
        # Single file mode
        logger.info(f"Loading single file: {args.file}")
        try:
            artifact_id = await document_loader.load_document(args.file, metadata)
            logger.info(f"File loaded successfully. Artifact ID: {artifact_id}")

        except Exception as e:
            logger.error(f"Error loading file: {str(e)}")
    else:
        # Directory mode
        if not args.dir:
            logger.error("Either --file or --dir must be specified")
            return
            
        logger.info(f"Loading documents from directory: {args.dir}")
        pattern = args.pattern or "*.*"
        try:
            results = await document_loader.batch_load_documents(args.dir, pattern, metadata)
            
            # Count successes and failures
            successes = sum(1 for v in results.values() if not str(v).startswith("ERROR"))
            failures = len(results) - successes
            
            logger.info(f"Batch processing complete. Loaded {successes} documents, {failures} failures.")
            logger.info(f"Total processing time: {time.time() - start_time:.2f} seconds")
            
            # Print details of failures if any
            if failures > 0:
                logger.info("Failed documents:")
                for filename, error in results.items():
                    if str(error).startswith("ERROR"):
                        logger.info(f"  - {filename}: {error}")

        except Exception as e:
            logger.error(f"Error processing directory: {str(e)}")
            
    logger.info("Document loading process completed")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Batch document loader for WebAgent")
    
    # Define mode group (file or directory)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--file", help="Path to a single file to load")
    mode_group.add_argument("--dir", help="Path to a directory of files to load")
    
    # Directory mode options
    parser.add_argument("--pattern", help="Glob pattern for matching files (e.g., '*.pdf')")
    
    # Metadata options
    parser.add_argument("--metadata", help="JSON string with metadata to attach to all documents")
    parser.add_argument("--category", help="Category for all documents")
    parser.add_argument("--tags", help="Comma-separated tags for all documents")
    parser.add_argument("--source", help="Source attribution for all documents")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the async function
    asyncio.run(load_documents(args))

if __name__ == "__main__":
    main() 