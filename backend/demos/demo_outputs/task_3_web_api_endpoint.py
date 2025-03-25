# Web API Endpoint
# Write a FastAPI endpoint that accepts file uploads, validates them for size and type (PDF only), and returns a JSON response with the file's metadata.

# Import necessary libraries
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import os

# Initialize FastAPI app
app = FastAPI()

# Set the maximum file size (in bytes) for uploads
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """
    Endpoint to upload a PDF file. Validates file type and size.
    
    Args:
        file (UploadFile): The uploaded file.

    Returns:
        JSONResponse: Metadata of the uploaded file or error message.
    """
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    # Validate file size
    file_size = await file.read(0)  # Read size of the file
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 10 MB.")

    # Reset file pointer to the beginning
    await file.seek(0)

    # Collect file metadata
    file_metadata = {
        "filename": file.filename,
        "content_type": file.content_type,
        "size": file_size,
        "upload_success": True
    }

    # Return metadata as JSON response
    return JSONResponse(content=file_metadata)

# To run the FastAPI app, use the command:
# uvicorn filename:app --reload