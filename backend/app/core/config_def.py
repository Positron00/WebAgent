"""
Configuration definitions using Pydantic models.
These models define the structure and types of the configuration loaded from YAML files.
"""
from enum import Enum
from typing import Dict, List, Optional, Union, Literal
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


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
    path: str = Field(default="")  # Will be set from settings.VECTOR_DB_PATH
    embedding_dimension: int = Field(default=1536)
    distance_metric: str = Field(default="cosine")
    document_chunk_size: int = Field(default=1000, description="Size of document chunks for storage")
    document_chunk_overlap: int = Field(default=200, description="Overlap between document chunks")


class RedisConfig(BaseModel):
    """Redis configuration."""
    host: str = Field(default="localhost")
    port: int = Field(default=6379)
    db: int = Field(default=0)
    password: Optional[str] = Field(default=None)
    key_prefix: str = Field(default="webagent:")


class ArtifactsConfig(BaseModel):
    """Artifacts storage configuration."""
    path: str = Field(default="")  # Will be set from settings.ARTIFACTS_PATH
    registry_url: str = Field(default=None, description="URL to remote artifact registry")
    enable_versioning: bool = Field(default=True)


class DatabaseConfig(BaseModel):
    """Database configuration."""
    vector_db: VectorDBConfig = Field(default_factory=VectorDBConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    artifacts: ArtifactsConfig = Field(default_factory=ArtifactsConfig)
    graph_db_url: Optional[str] = Field(default=None)


class LLMConfig(BaseModel):
    """LLM provider configuration."""
    provider: str = Field(default="openai")
    default_model: str = Field(default="meta-models/Muse-Glimmer-30B")
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    timeout_seconds: int = Field(default=60, gt=0)
    max_tokens: int = Field(default=4096, gt=0)
    retry_attempts: int = Field(default=3, ge=0)
    batch_size: int = Field(default=10, gt=0)
    # Self-hosted LLM specific settings
    self_hosted_url: Optional[str] = Field(default="http://localhost:8080")
    model_path: Optional[str] = Field(default=None, description="Path to the model for self-hosted LLM")


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
    model: str = Field(default="meta-models/Muse-Glimmer-30B")
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)
    # Output limit for this agent's model calls; lower than llm.max_tokens wins. None = use llm.max_tokens.
    max_tokens: Optional[int] = Field(default=None, gt=0)


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
    document_extraction: AgentConfig = Field(default_factory=AgentConfig)


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: LogLevel = Field(default=LogLevel.INFO)
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    log_to_file: bool = Field(default=True)
    log_file: str = Field(default="")  # Will be set from settings
    rotate_logs: bool = Field(default=True)
    max_file_size_mb: int = Field(default=10, gt=0)
    backup_count: int = Field(default=5, ge=0)


class SecurityConfig(BaseModel):
    """Security configuration."""
    token_expire_minutes: int = Field(default=30, gt=0)
    algorithm: str = Field(default="HS256")
    password_min_length: int = Field(default=8, ge=8)
    api_key_enabled: bool = Field(default=False)
    jwt_enabled: bool = Field(default=False)
    rate_limit_enabled: bool = Field(default=True)
    rate_limit_max_requests: int = Field(default=60, gt=0)
    rate_limit_window_size: int = Field(default=60, gt=0)
    rate_limit_burst_limit: int = Field(default=120, gt=0)
    request_size_limit: int = Field(default=10485760, gt=0)


class LangSmithConfig(BaseModel):
    """LangSmith configuration for tracing and observability."""
    project_name: str = Field(default="webagent-research")
    tracing_enabled: bool = Field(default=True)
    log_level: LogLevel = Field(default=LogLevel.INFO)


class SecurityGuardConfig(BaseModel):
    """Agentic security guard (checkpoints 1 and 2)."""

    enabled: bool = Field(default=True)
    input_mode: Literal["shadow", "act"] = Field(default="shadow")
    tool_mode: Literal["shadow", "act"] = Field(default="shadow")
    judge: bool = Field(default=False)
    # Checkpoint 1 judges for borderline user turns (SEC-035a). "moderation" is OpenAI's moderation
    # endpoint (needs OPENAI_API_KEY; built for this content, so it doesn't refuse); "llm" is the
    # trusted chat model, which may refuse to judge clearly harmful text (a refusal counts as flagged).
    harm_judge: Literal["moderation", "llm", "off"] = Field(default="moderation")
    # "prompt_guard" is Meta's Llama Prompt Guard 2, run locally from models/security/prompt_guard
    # (gated download, Llama 4 Community License); nothing leaves the server and it can't refuse.
    jailbreak_judge: Literal["prompt_guard", "llm", "off"] = Field(default="llm")
    prompt_guard_threshold: float = Field(default=0.5, gt=0.0, lt=1.0)
    collect: bool = Field(default=False)
    collect_retention_days: int = Field(default=30, gt=0)
    model_dir: str = Field(default="models/security")
    policy_dir: str = Field(default="config/security")


class TriageConfig(BaseModel):
    """Agentic triage (Off / Auto / Always routing)."""

    model_config = ConfigDict(protected_namespaces=())
    enabled: bool = Field(default=True)
    mode: str = Field(default="act")
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    uncertain_band: float = Field(default=0.15, ge=0.0, le=0.5)
    llm_tiebreak: bool = Field(default=False)
    tiebreak_upper: float = Field(default=0.65, ge=0.0, le=1.0)
    collect: bool = Field(default=False)
    collect_retention_days: int = Field(default=30, gt=0)
    model_dir: str = Field(default="models/triage")
    encoder_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    short_query_words: int = Field(default=6, gt=0)


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
    triage: TriageConfig = Field(default_factory=TriageConfig)
    security_guard: SecurityGuardConfig = Field(default_factory=SecurityGuardConfig)
    
    @field_validator('cors')
    @classmethod
    def validate_origins(cls, cors):
        """Validate that origins are properly formatted."""
        for origin in cors.origins:
            if not (origin == "*" or origin.startswith(("http://", "https://"))):
                raise ValueError(f"Invalid origin: {origin}. Must start with http:// or https://")
        return cors 