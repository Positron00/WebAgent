"""
Configuration definitions using Pydantic models.
These models define the structure and types of the configuration loaded from YAML files.
"""
from enum import Enum
from typing import Dict, List, Optional, Union, Literal
from pydantic import BaseModel, Field, HttpUrl, field_validator


class LogLevel(str, Enum):
    """Log level enum for type safety."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ApiConfig(BaseModel):
    """API server configuration."""
    version: str = Field(default="v1")
    prefix: str = Field(default="/api/v1")
    debug_mode: bool = Field(default=False)
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)


class CorsConfig(BaseModel):
    """CORS configuration."""
    origins: List[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    allow_credentials: bool = Field(default=True)
    allow_methods: List[str] = Field(default_factory=lambda: ["*"])
    allow_headers: List[str] = Field(default_factory=lambda: ["*"])


class VectorDBConfig(BaseModel):
    """Vector database configuration."""
    path: str = Field(default="./data/vectordb")
    embedding_dimension: int = Field(default=1536)
    distance_metric: str = Field(default="cosine")


class RedisConfig(BaseModel):
    """Redis configuration."""
    host: str = Field(default="localhost")
    port: int = Field(default=6379)
    db: int = Field(default=0)
    password: Optional[str] = Field(default=None)
    key_prefix: str = Field(default="webagent:")


class DatabaseConfig(BaseModel):
    """Database configuration."""
    vector_db: VectorDBConfig = Field(default_factory=VectorDBConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)


class LLMConfig(BaseModel):
    """LLM provider configuration."""
    provider: str = Field(default="openai")
    default_model: str = Field(default="gpt-4-turbo")
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    timeout_seconds: int = Field(default=60, gt=0)
    max_tokens: int = Field(default=4096, gt=0)
    retry_attempts: int = Field(default=3, ge=0)
    batch_size: int = Field(default=10, gt=0)


class WebSearchConfig(BaseModel):
    """Web search configuration."""
    provider: str = Field(default="tavily")
    search_depth: Literal["basic", "comprehensive"] = Field(default="basic")
    max_results: int = Field(default=10, gt=0)
    include_domains: List[str] = Field(default_factory=list)
    exclude_domains: List[str] = Field(default_factory=list)
    timeout_seconds: int = Field(default=30, gt=0)


class TasksConfig(BaseModel):
    """Task management configuration."""
    max_concurrent: int = Field(default=10, gt=0)
    result_ttl_minutes: int = Field(default=60, gt=0)
    history_ttl_days: int = Field(default=7, gt=0)
    max_retries: int = Field(default=3, ge=0)


class AgentConfig(BaseModel):
    """Base agent configuration."""
    model: str = Field(default="gpt-4-turbo")
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)


class WebResearchAgentConfig(AgentConfig):
    """Web research agent configuration."""
    max_searches_per_task: int = Field(default=5, gt=0)


class InternalResearchAgentConfig(AgentConfig):
    """Internal research agent configuration."""
    max_documents: int = Field(default=20, gt=0)


class SeniorResearchAgentConfig(AgentConfig):
    """Senior research agent configuration."""
    max_follow_up_questions: int = Field(default=3, ge=0)


class CodingAgentConfig(AgentConfig):
    """Coding agent configuration."""
    allowed_modules: List[str] = Field(
        default_factory=lambda: ["matplotlib", "seaborn", "numpy", "pandas"]
    )
    timeout_seconds: int = Field(default=30, gt=0)


class AgentsConfig(BaseModel):
    """Agent configurations."""
    supervisor: AgentConfig = Field(default_factory=AgentConfig)
    web_research: WebResearchAgentConfig = Field(default_factory=WebResearchAgentConfig)
    internal_research: InternalResearchAgentConfig = Field(default_factory=InternalResearchAgentConfig)
    senior_research: SeniorResearchAgentConfig = Field(default_factory=SeniorResearchAgentConfig)
    data_analysis: AgentConfig = Field(default_factory=AgentConfig)
    coding: CodingAgentConfig = Field(default_factory=CodingAgentConfig)
    team_manager: AgentConfig = Field(default_factory=AgentConfig)


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: LogLevel = Field(default=LogLevel.INFO)
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    log_to_file: bool = Field(default=True)
    log_file: str = Field(default="./logs/webagent.log")
    rotate_logs: bool = Field(default=True)
    max_file_size_mb: int = Field(default=10, gt=0)
    backup_count: int = Field(default=5, ge=0)


class SecurityConfig(BaseModel):
    """Security configuration."""
    token_expire_minutes: int = Field(default=30, gt=0)
    algorithm: str = Field(default="HS256")
    password_min_length: int = Field(default=8, ge=8)


class LangSmithConfig(BaseModel):
    """LangSmith configuration for tracing and observability."""
    project_name: str = Field(default="webagent-research")
    tracing_enabled: bool = Field(default=True)
    log_level: LogLevel = Field(default=LogLevel.INFO)


class AppConfig(BaseModel):
    """Main application configuration."""
    api: ApiConfig = Field(default_factory=ApiConfig)
    cors: CorsConfig = Field(default_factory=CorsConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    web_search: WebSearchConfig = Field(default_factory=WebSearchConfig)
    tasks: TasksConfig = Field(default_factory=TasksConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    langsmith: LangSmithConfig = Field(default_factory=LangSmithConfig)
    
    @field_validator('cors')
    @classmethod
    def validate_origins(cls, cors):
        """Validate that origins are properly formatted."""
        for origin in cors.origins:
            if not (origin == "*" or origin.startswith(("http://", "https://"))):
                raise ValueError(f"Invalid origin: {origin}. Must start with http:// or https://")
        return cors 