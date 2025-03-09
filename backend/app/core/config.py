"""
Configuration settings for the WebAgent backend microservice.
Loads environment variables and provides settings throughout the application.
"""
import os
from typing import List, Optional

from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from app.core.loadEnvYAML import get_config, get_api_config, get_cors_config, get_llm_config, get_database_config

# Load environment variables from .env file first (lowest priority)
load_dotenv(".env")

# Then load from YAML configuration (higher priority)
try:
    app_config = get_config()
    api_config = get_api_config()
    cors_config = get_cors_config()
    llm_config = get_llm_config()
    db_config = get_database_config()
except Exception as e:
    # Fall back to .env and default values if YAML config fails
    print(f"Warning: Failed to load YAML configuration: {e}")
    print("Falling back to .env file and default values")
    app_config = None


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # API settings
    API_V1_STR: str = os.getenv("WEBAGENT_API_PREFIX", "/api/v1")
    PROJECT_NAME: str = "WebAgent Backend"
    PROJECT_DESCRIPTION: str = "Multi-Agent Research and Analysis Platform"
    VERSION: str = "2.0.0"
    
    # Server settings
    HOST: str = os.getenv("WEBAGENT_API_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("WEBAGENT_API_PORT", "8000"))
    DEBUG_MODE: bool = os.getenv("WEBAGENT_API_DEBUG_MODE", "False").lower() == "true"
    
    # CORS
    CORS_ORIGINS: List[str] = os.getenv("WEBAGENT_CORS_ORIGINS", "http://localhost:3000").split(",")
    
    # OpenAI
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    DEFAULT_MODEL: str = os.getenv("WEBAGENT_LLM_DEFAULT_MODEL", "gpt-4-turbo")
    
    # Tavily for web search
    TAVILY_API_KEY: Optional[str] = os.getenv("TAVILY_API_KEY")
    
    # Vector database settings
    VECTOR_DB_PATH: str = os.getenv("WEBAGENT_DATABASE_VECTOR_DB_PATH", "./data/vectordb")
    
    # Redis
    REDIS_HOST: str = os.getenv("WEBAGENT_DATABASE_REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("WEBAGENT_DATABASE_REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("WEBAGENT_DATABASE_REDIS_DB", "0"))
    REDIS_PASSWORD: Optional[str] = os.getenv("WEBAGENT_DATABASE_REDIS_PASSWORD")
    
    # Security
    SECRET_KEY: str = os.getenv("WEBAGENT_SECURITY_SECRET_KEY", "insecure_key_for_dev_only_change_in_production")
    ALGORITHM: str = os.getenv("WEBAGENT_SECURITY_ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("WEBAGENT_SECURITY_TOKEN_EXPIRE_MINUTES", "30"))

    class Config:
        """Pydantic config class"""
        case_sensitive = True
        env_file = ".env"


# Override the settings from YAML if available
if app_config:
    class YAMLConfigSettings(Settings):
        """Settings loaded from YAML configuration"""
        
        # API settings
        API_V1_STR: str = api_config.prefix
        HOST: str = api_config.host
        PORT: int = api_config.port
        DEBUG_MODE: bool = api_config.debug_mode
        
        # CORS
        CORS_ORIGINS: List[str] = cors_config.origins
        
        # LLM
        DEFAULT_MODEL: str = llm_config.default_model
        
        # Database
        VECTOR_DB_PATH: str = db_config.vector_db.path
        REDIS_HOST: str = db_config.redis.host
        REDIS_PORT: int = db_config.redis.port
        REDIS_DB: int = db_config.redis.db
        REDIS_PASSWORD: Optional[str] = db_config.redis.password
    
    # Create settings instance with YAML values
    settings = YAMLConfigSettings()
else:
    # Create settings instance from .env and default values
    settings = Settings() 