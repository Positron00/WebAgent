from flask import Flask, request, jsonify
import os
import sys
from pathlib import Path

# Add the current directory to the path
current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
if str(current_dir.parent.parent) not in sys.path:
    sys.path.insert(0, str(current_dir.parent.parent))

# Import application modules
from chat import handle_chat

app = Flask(__name__)

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Simple health check endpoint.
    """
    return jsonify({
        "status": "ok",
        "service": "WebAgent API",
        "version": "2.9.5",
    })

# Ping endpoint
@app.route('/api/ping', methods=['GET'])
def ping():
    """
    Simple ping endpoint to check if the API is responsive.
    """
    return jsonify({
        "status": "ok",
        "message": "pong",
    })

# Chat endpoint
@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Handle chat requests to the LLM API.
    """
    return handle_chat()

# Task endpoints
@app.route('/api/task', methods=['POST'])
def handle_task():
    """
    Handle agentic task requests.
    This is a serverless function for Vercel deployment.
    """
    try:
        data = request.json
        query = data.get('query', '')
        image = data.get('image')
        
        # Log the request
        print(f"Received task request: {query[:100]}...")
        
        # For now, return a simple response
        # In production, you would integrate with your LangGraph system
        return jsonify({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": f"I received your task request: {query[:50]}... This is a placeholder response as the multi-agent system is in serverless mode."
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        })
    except Exception as e:
        print(f"Error processing task request: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500

@app.route('/api/task/<task_id>', methods=['GET'])
def get_task_status(task_id):
    """
    Get the status of an asynchronous task.
    This is a serverless function for Vercel deployment.
    """
    try:
        # Log the request
        print(f"Checking status for task: {task_id}")
        
        # For now, return a simple completed response
        # In production, you would check the actual task status
        return jsonify({
            "status": "completed",
            "result": {
                "content": f"Task {task_id} has been processed. This is a placeholder response for serverless mode."
            }
        })
    except Exception as e:
        print(f"Error checking task status: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500
        
# For local development
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True) 