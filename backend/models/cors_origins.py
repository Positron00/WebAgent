"""Shared CORS origin list for model gateway and model services."""
import os


def get_model_service_cors_origins() -> list[str]:
    raw = os.getenv("WEBAGENT_CORS_ORIGINS", "http://localhost:3000")
    return [origin.strip() for origin in raw.split(",") if origin.strip()]
