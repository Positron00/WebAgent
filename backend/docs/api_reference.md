# WebAgent API Reference

This document provides a comprehensive reference for the WebAgent Backend API (v2.4.0).

## Base URL

All API paths are relative to the base URL:

```
https://api.webagent.example.com/api/v1
```

## Authentication

API authentication is handled using JWT (JSON Web Token) bearer tokens. Include the token in your request's `Authorization` header:

```
Authorization: Bearer <your_access_token>
```

To obtain a token, use the authentication endpoint with valid credentials.

## Common Response Codes

| Code | Description |
|------|-------------|
| 200  | Success |
| 201  | Created (resource created successfully) |
| 400  | Bad Request (invalid input) |
| 401  | Unauthorized (missing or invalid token) |
| 403  | Forbidden (insufficient permissions) |
| 404  | Not Found (resource not found) |
| 429  | Too Many Requests (rate limit exceeded) |
| 500  | Internal Server Error |

## Rate Limiting

The API implements rate limiting to protect against abuse. Limits are as follows:

- 100 requests per minute per client IP address
- 10 concurrent requests per client IP address

When rate limited, you'll receive a `429 Too Many Requests` response with a `Retry-After` header indicating the number of seconds to wait before retrying.

## Chat API

### Start a Chat Workflow

Initiates a new multi-agent chat workflow.

**Endpoint:** `POST /chat`

**Request Body:**

```json
{
  "message": "What are the benefits of LLM observability?",
  "context": {
    "additional_information": "Focus on LangSmith specifically",
    "previous_messages": [
      {
        "role": "user",
        "content": "Tell me about LLMs"
      },
      {
        "role": "assistant",
        "content": "LLMs are large language models..."
      }
    ]
  },
  "session_id": "optional-session-id"
}
```

**Response:**

```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "message": "Your request is being processed. Check the status using the task ID."
}
```

### Get Chat Status

Retrieves the status and results of a chat task.

**Endpoint:** `GET /chat/{task_id}`

**Parameters:**
- `task_id` (path) - The unique ID of the task

**Response:**

```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "message": "Task complete",
  "result": {
    "title": "Benefits of LLM Observability with LangSmith",
    "content": "LangSmith provides several benefits for LLM observability...",
    "visualizations": [
      {
        "type": "chart",
        "description": "Performance comparison",
        "image_data": "base64_encoded_image_data",
        "code": "python code used to generate the visualization"
      }
    ],
    "sources": [
      {
        "title": "LangSmith Documentation",
        "content": "Excerpt from LangSmith docs...",
        "url": "https://docs.langchain.com/langsmith/",
        "relevance": 0.95,
        "source_type": "web"
      }
    ]
  }
}
```

## Frontend Compatibility API

These endpoints provide compatibility with the frontend by mimicking other LLM service APIs.

### Chat Completions

Handles chat completions in a format compatible with the frontend.

**Endpoint:** `POST /v1/chat/completions`

**Request Body:**

```json
{
  "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
  "messages": [
    {
      "role": "user",
      "content": "What is the capital of France?"
    }
  ],
  "max_tokens": 4096,
  "temperature": 0.7,
  "top_p": 1.0,
  "frequency_penalty": 0.0,
  "presence_penalty": 0.0
}
```

**Response:**

```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "Your request is being processed (Task ID: 550e8400-e29b-41d4-a716-446655440000). Please check back in a moment."
      },
      "finish_reason": "processing"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 0,
    "total_tokens": 10
  }
}
```

### Get Chat Completion Status

Gets the status of a chat completion task.

**Endpoint:** `GET /v1/chat/status/{task_id}`

**Parameters:**
- `task_id` (path) - The unique ID of the task

**Response:**

```json
{
  "status": "completed",
  "response": {
    "choices": [
      {
        "message": {
          "role": "assistant",
          "content": "The capital of France is Paris. It is known as the 'City of Light' and is famous for landmarks such as the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral."
        },
        "finish_reason": "stop"
      }
    ],
    "usage": {
      "prompt_tokens": 10,
      "completion_tokens": 37,
      "total_tokens": 47
    }
  }
}
```

### Backend Status

Gets the status of the backend services.

**Endpoint:** `GET /v1/status`

**Response:**

```json
{
  "status": "ok",
  "version": "2.4.0",
  "llm": {
    "status": "ok",
    "provider": "together",
    "default_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    "available_models": [
      "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
      "meta-llama/Llama-3.3-8B-Instruct"
    ]
  },
  "services": {
    "workflow": "ok",
    "database": "ok"
  }
}
```

## Task Management API

### List Tasks

Lists all active and completed tasks.

**Endpoint:** `GET /tasks`

**Response:**

```json
{
  "tasks": [
    {
      "task_id": "550e8400-e29b-41d4-a716-446655440000",
      "status": "completed",
      "created_at": "2025-03-09T10:30:00Z",
      "completed_at": "2025-03-09T10:31:05Z",
      "message": "Task complete",
      "query": "What is the capital of France?",
      "current_step": null
    },
    {
      "task_id": "550e8400-e29b-41d4-a716-446655440001",
      "status": "processing",
      "created_at": "2025-03-09T10:35:00Z",
      "completed_at": null,
      "message": "Task in progress",
      "query": "Explain quantum computing",
      "current_step": "web_research"
    }
  ]
}
```

### Get Task Details

Gets details about a specific task.

**Endpoint:** `GET /tasks/{task_id}`

**Parameters:**
- `task_id` (path) - The unique ID of the task

**Response:**

```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "created_at": "2025-03-09T10:30:00Z",
  "completed_at": "2025-03-09T10:31:05Z",
  "message": "Task complete",
  "query": "What is the capital of France?",
  "current_step": null
}
```

### Delete Task

Deletes a task and its results.

**Endpoint:** `DELETE /tasks/{task_id}`

**Parameters:**
- `task_id` (path) - The unique ID of the task

**Response:**

```json
{
  "status": "success",
  "message": "Task 550e8400-e29b-41d4-a716-446655440000 deleted"
}
```

## Health API

### Basic Health Check

Performs a basic health check.

**Endpoint:** `GET /health`

**Response:**

```json
{
  "status": "ok",
  "version": "2.4.0",
  "environment": "production"
}
```

### Detailed Health Check

Performs a detailed health check with information about connected services.

**Endpoint:** `GET /health/detailed`

**Response:**

```json
{
  "api": {
    "status": "ok",
    "version": "2.4.0"
  },
  "llm_service": {
    "status": "ok",
    "provider": "together",
    "latency_ms": 245
  },
  "vector_db": {
    "status": "ok",
    "count": 1250,
    "size_mb": 156
  }
}
```

### Metrics

Prometheus metrics endpoint.

**Endpoint:** `GET /metrics`

**Response:**
Plain text Prometheus metrics format

## Errors

Errors are returned as JSON objects with the following structure:

```json
{
  "error": "Error Type",
  "message": "Detailed error message",
  "error_id": "unique-error-id-for-tracking"
}
```

## Support

For API support, please contact:
- Email: api-support@webagent.example.com
- API Status Page: https://status.webagent.example.com 