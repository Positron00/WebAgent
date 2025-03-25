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