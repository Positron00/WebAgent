# Web API Endpoint
# Write a FastAPI endpoint that accepts file uploads, validates them for size and type (PDF only), and returns a JSON response with the file's metadata.

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

app = FastAPI()

# Constants for validation
MAX_FILE_SIZE_MB = 5
ALLOWED_FILE_TYPE = "application/pdf"

# Sample data for visualization
sample_data = {
    'file_name': ['sample1.pdf', 'sample2.pdf', 'sample3.pdf'],
    'file_size_MB': [1.2, 3.4, 2.1]
}

def create_sample_visualizations():
    """Create visualizations from sample data."""
    df = pd.DataFrame(sample_data)
    
    # Bar chart showing the count of files
    plt.figure(figsize=(10, 5))
    df['file_name'].value_counts().plot(kind='bar', color='skyblue')
    plt.title('Count of Uploaded PDF Files')
    plt.xlabel('File Name')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('uploads_count.png')
    plt.close()

    # Pie chart showing file size distribution
    plt.figure(figsize=(8, 8))
    plt.pie(df['file_size_MB'], labels=df['file_name'], autopct='%1.1f%%', startangle=140)
    plt.title('File Size Distribution of Uploaded PDFs')
    plt.axis('equal')  # Equal aspect ratio ensures pie chart is circular.
    plt.savefig('file_size_distribution.png')
    plt.close()

@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    """Endpoint to upload a PDF file and return its metadata."""
    try:
        # Validate the file type
        if file.content_type != ALLOWED_FILE_TYPE:
            raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are allowed.")
        
        # Validate the file size
        file_size_mb = len(await file.read()) / (1024 * 1024)  # Convert bytes to MB
        if file_size_mb > MAX_FILE_SIZE_MB:
            raise HTTPException(status_code=400, detail=f"File size exceeds the limit of {MAX_FILE_SIZE_MB} MB.")
        
        # Prepare response metadata
        metadata = {
            "filename": file.filename,
            "content_type": file.content_type,
            "size_MB": round(file_size_mb, 2)
        }
        
        return JSONResponse(content=metadata)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Ensure the file is closed
        await file.close()

# Generate visualizations from sample data
try:
    create_sample_visualizations()
except Exception as e:
    print(f"An error occurred while generating visualizations: {e}")