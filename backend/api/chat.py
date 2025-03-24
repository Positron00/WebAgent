from flask import request, jsonify
import os
import json
import requests
from datetime import datetime

def handle_chat():
    """
    Handle chat API requests for Vercel serverless functions.
    This is a simple handler that forwards requests to Together AI.
    """
    try:
        # Get API key from environment variables
        api_key = os.environ.get('TOGETHER_API_KEY')
        if not api_key:
            return jsonify({
                "error": "API key not configured",
                "message": "TOGETHER_API_KEY environment variable is not set"
            }), 500
        
        # Parse request data
        data = request.json
        messages = data.get('messages', [])
        model = data.get('model', 'meta-llama/Llama-3.3-70b-instruct-turbo-free')
        max_tokens = data.get('max_tokens', 1000)
        temperature = data.get('temperature', 0.7)
        top_p = data.get('top_p', 0.9)
        
        # Log request details (without sensitive content)
        print(f"Chat request received: model={model}, message_count={len(messages)}")
        
        # Forward request to Together AI
        together_endpoint = "https://api.together.xyz/v1/chat/completions"
        
        # Prepare Together AI request
        together_request = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p
        }
        
        # Make API call to Together AI
        response = requests.post(
            together_endpoint,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json=together_request,
            timeout=60
        )
        
        # Check response status
        if not response.ok:
            error_data = response.json() if response.text else {"error": response.reason}
            return jsonify(error_data), response.status_code
        
        # Return Together AI response
        return jsonify(response.json())
        
    except Exception as e:
        print(f"Error in chat handler: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500 