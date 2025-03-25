#!/usr/bin/env python
"""
Debug script to check code extraction from LLM responses
"""

import os
import sys
import re
import json
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("debug_code_extraction")

# Sample LLM responses for testing
SAMPLES = [
    """
Here's a function to parse a CSV file:

```python
import csv

def parse_users_over_18(csv_file_path):
    '''
    Parse a CSV file containing user data and return users over 18.
    
    Args:
        csv_file_path (str): Path to the CSV file
        
    Returns:
        list: List of dictionaries containing user data for users over 18
    '''
    users_over_18 = []
    
    try:
        with open(csv_file_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                try:
                    age = int(row.get('age', 0))
                    if age > 18:
                        users_over_18.append(row)
                except (ValueError, TypeError):
                    # Skip rows with invalid age values
                    continue
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        
    return users_over_18
```

This function opens the CSV file, reads it using the DictReader, and returns a list of users whose age is greater than 18.
    """,
    
    """
{
  "explanation": "This code connects to ChromaDB and embeds documents using OpenAI",
  "code": "import chromadb\\nfrom openai import OpenAI\\n\\ndef embed_document(document, collection_name):\\n    # Connect to ChromaDB\\n    client = chromadb.Client()\\n    # Create collection\\n    collection = client.create_collection(collection_name)\\n    # Embed document\\n    openai_client = OpenAI()\\n    response = openai_client.embeddings.create(input=document, model='text-embedding-ada-002')\\n    embedding = response.data[0].embedding\\n    # Store in vector database\\n    collection.add(documents=[document], embeddings=[embedding], ids=['doc1'])\\n    return True"
}
    """,
    
    """
I'll create a FastAPI endpoint for file uploads with validation:

```python
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import os
import magic
import uuid
from typing import Optional

app = FastAPI()

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    '''
    Upload a PDF file with size and type validation.
    
    Args:
        file: The file to upload
        
    Returns:
        JSONResponse: Metadata about the uploaded file
    '''
    # Check file size
    file_size = 0
    contents = await file.read(MAX_FILE_SIZE + 1)
    file_size = len(contents)
    
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large")
    
    # Reset file position
    await file.seek(0)
    
    # Check file type
    mime = magic.Magic(mime=True)
    file_mime = mime.from_buffer(contents)
    
    if file_mime != "application/pdf":
        raise HTTPException(status_code=415, detail="Only PDF files are allowed")
    
    # Generate unique filename
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    
    # Save file (in production, you'd want to save to cloud storage)
    with open(f"uploads/{unique_filename}", "wb") as f:
        f.write(contents)
    
    # Return metadata
    return JSONResponse({
        "filename": file.filename,
        "stored_filename": unique_filename,
        "content_type": file_mime,
        "size": file_size
    })
```

This FastAPI endpoint validates PDF files for size and type before storing them.
    """
]

def extract_code_from_llm_response(response):
    """Extract code from an LLM response using different methods."""
    logger.debug(f"Processing response of length: {len(response)}")
    
    # Method 1: Try parsing as JSON
    try:
        if response.strip().startswith("{") and response.strip().endswith("}"):
            code_data = json.loads(response)
            if "code" in code_data:
                logger.debug("Successfully extracted code from JSON")
                return code_data["code"]
    except json.JSONDecodeError as e:
        logger.debug(f"JSON parse error: {e}")
    
    # Method 2: Look for JSON in the response
    try:
        import re
        json_match = re.search(r'(\{.*\})', response, re.DOTALL)
        if json_match:
            json_text = json_match.group(1)
            code_data = json.loads(json_text)
            if "code" in code_data:
                logger.debug("Successfully extracted code from JSON in text")
                return code_data["code"]
    except (json.JSONDecodeError, AttributeError) as e:
        logger.debug(f"JSON in text extraction error: {e}")
    
    # Method 3: Extract code blocks
    code_blocks = re.findall(r'```(?:python)?\s*([\s\S]*?)```', response, re.DOTALL)
    if code_blocks:
        longest_code = max(code_blocks, key=len).strip()
        logger.debug(f"Extracted code block of length: {len(longest_code)}")
        return longest_code
    
    # Method 4: Use the entire response as code
    logger.debug("No code extraction methods worked, using full response")
    return response.strip()

def save_code_to_file(code, filename):
    """Save extracted code to a file."""
    # Use the absolute path from the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "demo_outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w") as f:
        f.write(code)
    
    logger.info(f"Code saved to: {filepath}")
    return filepath

def main():
    """Test code extraction on sample responses."""
    logger.info("Testing code extraction from LLM responses")
    
    for i, sample in enumerate(SAMPLES):
        logger.info(f"\n\n=== Testing Sample {i+1} ===\n")
        
        # Extract code
        extracted_code = extract_code_from_llm_response(sample)
        
        # Print extracted code
        print("\n--- Extracted Code ---")
        print(extracted_code)
        print("--- End of Code ---\n")
        
        # Save to file
        filename = f"sample_{i+1}_extracted.py"
        filepath = save_code_to_file(extracted_code, filename)

if __name__ == "__main__":
    main() 