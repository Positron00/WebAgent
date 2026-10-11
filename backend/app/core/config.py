"""
Configuration settings for the WebAgent backend microservice.
Loads environment variables and provides settings throughout the application.

Configuration Priority:
1. .env.local file (highest priority) - For API keys and sensitive information
   - Should contain: TOGETHER_API_KEY, OPENAI_API_KEY, LANGSMITH_API_KEY, etc.
   - Never commit this file to version control
   - Exception: WEBAGENT_ENV set in the shell or container wins over .env.local,
     so `WEBAGENT_ENV=prod` always selects prod.

2. YAML configuration (middle priority) - For environment-specific settings
   - Located in config/{env}.yaml where env is dev, uat, or prod
   - Selected via WEBAGENT_ENV environment variable

3. .env file (lowest priority) - For default fallback values
   - General settings that can be safely committed to version control
   - Used when no other source provides the value

4. Default values in code - Ultimate fallback
"""
import os
import sys
from typing import Dict, List, Optional, Tuple, Any, ClassVar
from pathlib import Path
import logging
import platform
from datetime import datetime

from dotenv import load_dotenv
from pydantic import Field, ConfigDict
from pydantic_settings import BaseSettings
from app import __version__
from app.core.loadEnvYAML import get_config, get_api_config, get_cors_config, get_llm_config, get_database_config
from app.core.config_def import TriageConfig

# The backend folder holds app/, config/ and data/. In the Docker image it is /app,
# so data paths are built from BACKEND_DIR rather than from the repo root.
BACKEND_DIR = str(Path(__file__).resolve().parents[2])
# Repo root (where .env.local lives in a local checkout)
BASE_DIR = str(Path(BACKEND_DIR).parent)

# Priority order for configuration:
# 1. .env.local file (highest priority for API keys)
# 2. YAML configuration (high priority for non-API key settings)
# 3. .env file (lowest priority, fallback)

# Shell/container WEBAGENT_* values must win over .env / .env.local (same as WEBAGENT_ENV).
_preserved_webagent_env = {
    key: value for key, value in os.environ.items() if key.startswith("WEBAGENT_")
}
_selected_env = _preserved_webagent_env.get("WEBAGENT_ENV")

# Load environment variables from .env file first (lowest priority)
load_dotenv(".env")

# Load API keys from .env.local file (highest priority)
env_local_path = os.path.join(BASE_DIR, ".env.local")
if os.path.exists(env_local_path):
    load_dotenv(env_local_path, override=True)

for key, value in _preserved_webagent_env.items():
    os.environ[key] = value

# Then load from YAML configuration (middle priority)
try:
    app_config = get_config()
    api_config = get_api_config()
    cors_config = get_cors_config()
    llm_config = get_llm_config()
    db_config = get_database_config()
except Exception as e:
    # Fall back to .env and default values if YAML config fails
    logging.getLogger(__name__).warning(
        "Failed to load YAML configuration (%s); falling back to .env and defaults", e
    )
    app_config = None

# YAML export writes WEBAGENT_* into os.environ; shell/container values still win.
for key, value in _preserved_webagent_env.items():
    os.environ[key] = value


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # API settings
    API_V1_STR: str = os.getenv("WEBAGENT_API_PREFIX", "/api/v1")
    PROJECT_NAME: str = "WebAgent Backend"
    PROJECT_DESCRIPTION: str = "Multi-Agent Research and Analysis Platform"
    VERSION: str = __version__
    
    # Environment
    WEBAGENT_ENV: str = os.getenv("WEBAGENT_ENV", "dev")
    
    # Server settings
    HOST: str = os.getenv("WEBAGENT_API_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("WEBAGENT_API_PORT", "8000"))
    DEBUG_MODE: bool = os.getenv("WEBAGENT_API_DEBUG_MODE", "False").lower() == "true"
    # On for local demos; backend/main.py turns it off for the API server.
    ENABLE_CODE_EXECUTION: bool = (
        os.getenv("WEBAGENT_ENABLE_CODE_EXECUTION", "true").lower() == "true"
    )
    
    # CORS
    CORS_ORIGINS: List[str] = os.getenv("WEBAGENT_CORS_ORIGINS", "http://localhost:3000").split(",")
    
    # Default LLM services
    LLM_DEFAULT_PROVIDER: str = os.getenv("WEBAGENT_LLM_DEFAULT_PROVIDER", "together")
    LLM_DEFAULT_MODEL: str = os.getenv("WEBAGENT_LLM_DEFAULT_MODEL", "meta-models/Muse-Glimmer-30B")
    DEFAULT_MODEL: str = os.getenv(
        "WEBAGENT_LLM_DEFAULT_MODEL", "meta-models/Muse-Glimmer-30B"
    )
    
    # OpenAI
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: Optional[str] = os.getenv("OPENAI_MODEL", "gpt-4o-mini-2024-07-18")
    
    # Anthropic
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    ANTHROPIC_MODEL: Optional[str] = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")
    
    # Gemini
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL: Optional[str] = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-002")
    
    # Groq
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
    GROQ_MODEL: Optional[str] = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    # LangSmith for tracing and observability
    LANGSMITH_API_KEY: Optional[str] = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_PROJECT_NAME: str = os.getenv("LANGSMITH_PROJECT_NAME", "webagent-research")
    LANGSMITH_TRACING_ENABLED: bool = os.getenv("LANGSMITH_TRACING_ENABLED", "True").lower() == "true"
    
    # Alias for LANGSMITH_TRACING_ENABLED for backwards compatibility
    @property
    def LANGCHAIN_TRACING_V2(self) -> bool:
        """Alias for LANGSMITH_TRACING_ENABLED for consistent usage across codebase."""
        return self.LANGSMITH_TRACING_ENABLED
    
    # Tavily for web search
    TAVILY_API_KEY: Optional[str] = os.getenv("TAVILY_API_KEY")
    
    # Together AI for LLM services
    TOGETHER_API_KEY: Optional[str] = os.getenv("TOGETHER_API_KEY")
    TOGETHER_MODEL: Optional[str] = os.getenv("TOGETHER_MODEL", "meta-models/Muse-Glimmer-30B")
    
    # Data directory settings
    DATA_DIR: str = Field(
        default=os.getenv("WEBAGENT_DATA_DIR", os.path.join(BACKEND_DIR, "data")),
        description="Base directory for all data storage",
    )
    
    # Sample documents directory
    SAMPLE_DOCS_DIR: str = Field(
        default=os.getenv("WEBAGENT_SAMPLE_DOCS_DIR", os.path.join(BACKEND_DIR, "demos", "demo_docs")),
        description="Directory for sample documents used in demonstrations",
    )
    
    # Vector database settings
    VECTOR_DB_PATH: str = Field(
        default=os.getenv("WEBAGENT_VECTOR_DB_PATH", os.path.join(BACKEND_DIR, "data", "vectordb")),
        description="Path to the vector database",
    )
    
    # Document processing settings
    DOCUMENT_CHUNK_SIZE: int = Field(
        default=1000,
        description="Size of document chunks for storage"
    )
    DOCUMENT_CHUNK_OVERLAP: int = Field(
        default=200,
        description="Overlap between document chunks"
    )
    
    # Artifacts storage settings
    ARTIFACTS_PATH: str = Field(
        default=os.getenv("WEBAGENT_ARTIFACTS_PATH", os.path.join(BACKEND_DIR, "data", "artifacts")),
        description="Path to store document artifacts"
    )
    
    # MLflow artifacts storage settings
    MLRUNS_PATH: str = Field(
        default=os.path.join(BACKEND_DIR, "mlruns"),
        description="Path to store MLflow runs and artifacts"
    )
    
    # MLflow artifact location (file URI format)
    MLFLOW_ARTIFACT_LOCATION: str = Field(
        default=f"file://{os.path.join(BACKEND_DIR, 'mlruns')}",
        description="File URI path for MLflow artifacts"
    )
    
    ARTIFACTS_REGISTRY_URL: Optional[str] = Field(
        default=None,
        description="URL to remote artifact registry"
    )
    
    # Database settings
    DATABASE_GRAPH_DB_URL: Optional[str] = Field(
        default=None,
        description="URL to the graph database (Neo4j)"
    )
    
    # Redis
    REDIS_HOST: str = os.getenv("WEBAGENT_DATABASE_REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("WEBAGENT_DATABASE_REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("WEBAGENT_DATABASE_REDIS_DB", "0"))
    REDIS_PASSWORD: Optional[str] = os.getenv("WEBAGENT_DATABASE_REDIS_PASSWORD")
    
    # Security
    SECRET_KEY: str = os.getenv("WEBAGENT_SECURITY_SECRET_KEY", "insecure_key_for_dev_only_change_in_production")
    ALGORITHM: str = os.getenv("WEBAGENT_SECURITY_ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("WEBAGENT_SECURITY_TOKEN_EXPIRE_MINUTES", "30"))
    
    # Rate limiting settings
    RATE_LIMIT_ENABLED: bool = os.getenv("WEBAGENT_SECURITY_RATE_LIMIT_ENABLED", "True").lower() == "true"
    RATE_LIMIT_MAX_REQUESTS: int = int(os.getenv("WEBAGENT_SECURITY_RATE_LIMIT_MAX_REQUESTS", "60"))  # requests per window
    RATE_LIMIT_WINDOW_SIZE: int = int(os.getenv("WEBAGENT_SECURITY_RATE_LIMIT_WINDOW_SIZE", "60"))  # window size in seconds
    RATE_LIMIT_BURST_LIMIT: int = int(os.getenv("WEBAGENT_SECURITY_RATE_LIMIT_BURST_LIMIT", "120"))  # burst limit
    
    # Request size limiting
    REQUEST_SIZE_LIMIT: int = int(os.getenv("WEBAGENT_SECURITY_REQUEST_SIZE_LIMIT", "10485760"))  # 10MB default
    
    # Authentication settings
    JWT_ENABLED: bool = os.getenv("WEBAGENT_SECURITY_JWT_ENABLED", "False").lower() == "true"
    API_KEY_ENABLED: bool = os.getenv("WEBAGENT_SECURITY_API_KEY_ENABLED", "False").lower() == "true"
    
    # MLflow tracking
    MLFLOW_TRACKING_URI: Optional[str] = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:8080")
    MLFLOW_DEFAULT_ARTIFACT_ROOT: str = os.getenv(
        "MLFLOW_DEFAULT_ARTIFACT_ROOT", 
        f"file://{os.path.join(BACKEND_DIR, 'mlruns')}"
    )

    # Grafana settings
    GRAFANA_URL: Optional[str] = os.getenv("GRAFANA_URL", "http://127.0.0.1:3000")
    GRAFANA_API_KEY: Optional[str] = os.getenv("GRAFANA_API_KEY")
    GRAFANA_SERVICE_ACCOUNT_TOKEN: Optional[str] = os.getenv("GRAFANA_SERVICE_ACCOUNT_TOKEN")

    model_config = ConfigDict(
        case_sensitive=True,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
    )


def _triage_bool(env_name: str, default: bool) -> bool:
    """`true`/`false` (any case); anything else logs a warning and keeps the YAML value."""
    raw = os.getenv(env_name)
    if raw in (None, ""):
        return default
    value = raw.strip().lower()
    if value in ("true", "false"):
        return value == "true"
    logging.getLogger(__name__).warning("Invalid %s=%s; using YAML", env_name, raw)
    return default


def _build_triage_settings() -> TriageConfig:
    base = getattr(app_config, "triage", None) if app_config else None
    if base is None:
        base = TriageConfig()
    log = logging.getLogger(__name__)
    threshold = base.threshold
    raw_thr = os.getenv("WEBAGENT_TRIAGE_THRESHOLD")
    if raw_thr not in (None, ""):
        try:
            value = float(raw_thr)
            if not 0.0 <= value <= 1.0:
                raise ValueError("outside 0-1")
            threshold = value
        except ValueError:
            log.warning("Invalid WEBAGENT_TRIAGE_THRESHOLD=%s; ignoring", raw_thr)
    yaml_mode = (base.mode or "").strip().lower()
    if yaml_mode not in ("shadow", "act"):
        log.warning("Invalid triage.mode=%s in YAML; using act", base.mode)
        yaml_mode = "act"
    mode = (os.getenv("WEBAGENT_TRIAGE_MODE") or yaml_mode).strip().lower()
    if mode not in ("shadow", "act"):
        log.warning("Invalid WEBAGENT_TRIAGE_MODE=%s; using YAML", mode)
        mode = yaml_mode
    return TriageConfig(
        enabled=_triage_bool("WEBAGENT_TRIAGE_ENABLED", base.enabled),
        mode=mode,
        threshold=threshold,
        uncertain_band=base.uncertain_band,
        llm_tiebreak=_triage_bool("WEBAGENT_TRIAGE_LLM_TIEBREAK", base.llm_tiebreak),
        tiebreak_upper=base.tiebreak_upper,
        collect=_triage_bool("WEBAGENT_TRIAGE_COLLECT", base.collect),
        collect_retention_days=base.collect_retention_days,
        model_dir=base.model_dir,
        encoder_name=base.encoder_name,
        short_query_words=base.short_query_words,
    )


def _attach_triage(settings_obj):
    settings_obj.TRIAGE = _build_triage_settings()
    return settings_obj


def _security_setting(yaml_name: str, env_name: str, default):
    """Value of a security setting: the environment variable if set, else YAML, else the default."""
    raw = os.getenv(env_name)
    if raw not in (None, ""):
        return raw.lower() == "true" if isinstance(default, bool) else int(raw)
    security = getattr(app_config, "security", None)
    return getattr(security, yaml_name, default) if security is not None else default


# Override the settings from YAML if available
if app_config:
    class YAMLConfigSettings(Settings):
        """Settings loaded from YAML configuration"""

        model_config = ConfigDict(case_sensitive=True, extra="allow")
        
        # API settings
        API_V1_STR: str = api_config.prefix
        HOST: str = api_config.host
        PORT: int = api_config.port
        DEBUG_MODE: bool = api_config.debug_mode
        
        # Environment
        WEBAGENT_ENV: str = os.getenv("WEBAGENT_ENV", "dev")
        
        # CORS
        CORS_ORIGINS: List[str] = cors_config.origins
        
        # LLM
        DEFAULT_MODEL: str = llm_config.default_model
        
        # Database
        VECTOR_DB_PATH: str = db_config.vector_db.path if hasattr(db_config.vector_db, 'path') and db_config.vector_db.path else os.path.join(BACKEND_DIR, "data", "vectordb")
        REDIS_HOST: str = db_config.redis.host
        REDIS_PORT: int = db_config.redis.port
        REDIS_DB: int = db_config.redis.db
        REDIS_PASSWORD: Optional[str] = db_config.redis.password
        
        # Security settings: a WEBAGENT_SECURITY_* environment variable wins over the YAML value.
        # (Their names differ from the field names, so pydantic doesn't apply them on its own.)
        RATE_LIMIT_ENABLED: bool = _security_setting("rate_limit_enabled", "WEBAGENT_SECURITY_RATE_LIMIT_ENABLED", True)
        RATE_LIMIT_MAX_REQUESTS: int = _security_setting("rate_limit_max_requests", "WEBAGENT_SECURITY_RATE_LIMIT_MAX_REQUESTS", 60)
        RATE_LIMIT_WINDOW_SIZE: int = _security_setting("rate_limit_window_size", "WEBAGENT_SECURITY_RATE_LIMIT_WINDOW_SIZE", 60)
        RATE_LIMIT_BURST_LIMIT: int = _security_setting("rate_limit_burst_limit", "WEBAGENT_SECURITY_RATE_LIMIT_BURST_LIMIT", 120)
        REQUEST_SIZE_LIMIT: int = _security_setting("request_size_limit", "WEBAGENT_SECURITY_REQUEST_SIZE_LIMIT", 10485760)
        JWT_ENABLED: bool = _security_setting("jwt_enabled", "WEBAGENT_SECURITY_JWT_ENABLED", False)
        API_KEY_ENABLED: bool = _security_setting("api_key_enabled", "WEBAGENT_SECURITY_API_KEY_ENABLED", False)
        
        # Anthropic API
        ANTHROPIC_API_KEY: Optional[str] = getattr(app_config.security, 'anthropic_api_key', None) if hasattr(app_config, 'security') else None
        
        # MLflow tracking
        MLFLOW_TRACKING_URI: Optional[str] = getattr(app_config.security, 'mlflow_tracking_uri', None) if hasattr(app_config, 'security') else None
    
    # Create settings instance with YAML values
    settings = _attach_triage(YAMLConfigSettings())
else:
    # Create settings instance from .env and default values
    settings = _attach_triage(Settings())