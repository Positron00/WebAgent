# Web API Endpoint
# Write a FastAPI endpoint that accepts file uploads, validates them for size and type (PDF only), and returns a JSON response with the file's metadata.

# Import necessary libraries
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import matplotlib.pyplot as plt
import numpy as np
import os

# Initialize FastAPI app
app = FastAPI()

# Allow CORS for all origins (for testing purposes, restrict as needed in production)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Define constants
MAX_FILE_SIZE_MB = 5  # Maximum file size in MB
ALLOWED_EXTENSIONS = {'pdf'}

# Initialize upload statistics
upload_stats = {
    'successful': 0,
    'failed': 0
}

def allowed_file(filename: str) -> bool:
    """Check if the uploaded file is a PDF."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    """Endpoint to upload a file and return its metadata."""
    
    # Validate file type
    if not allowed_file(file.filename):
        upload_stats['failed'] += 1
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF allowed.")

    # Validate file size
    file_size = len(await file.read())
    if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
        upload_stats['failed'] += 1
        raise HTTPException(status_code=400, detail=f"File size exceeds {MAX_FILE_SIZE_MB} MB limit.")

    # Reset file pointer to the beginning after reading
    await file.seek(0)

    # Simulate saving the file (In production, save to a directory)
    # with open(os.path.join("uploads", file.filename), "wb") as f:
    #     f.write(await file.read())

    # Update upload statistics on success
    upload_stats['successful'] += 1

    # Return file metadata as JSON response
    return JSONResponse(content={
        "filename": file.filename,
        "content_type": file.content_type,
        "size": file_size,
        "status": "success"
    })

@app.get("/visualize/")
def visualize_uploads():
    """Visualize the upload statistics."""
    
    # Prepare data for visualization
    labels = ['Successful Uploads', 'Failed Uploads']
    sizes = [upload_stats['successful'], upload_stats['failed']]
    
    # Create a bar chart
    fig, ax = plt.subplots()
    ax.bar(labels, sizes, color=['green', 'red'])
    ax.set_title('File Upload Statistics')
    ax.set_ylabel('Number of Files')
    ax.set_ylim(0, max(sizes) + 1)  # Set y-axis limit slightly higher than max value

    # Annotate bars with their counts
    for i, count in enumerate(sizes):
        ax.text(i, count + 0.1, str(count), ha='center')

    # Save the plot as an image file
    plt.savefig("upload_statistics.png")
    plt.close(fig)  # Close the figure to free up memory

    return {"message": "Visualization created", "file": "upload_statistics.png"}

# To run the FastAPI application, use: `uvicorn filename:app --reload`