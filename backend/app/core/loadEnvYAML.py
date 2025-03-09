"""
YAML configuration loader for the WebAgent backend.
Loads environment-specific configuration from YAML files and updates environment variables.
"""
import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from app.core.config_def import AppConfig

logger = logging.getLogger(__name__)

# Constants
ENV_VAR_PREFIX = "WEBAGENT_"
DEFAULT_ENV = "dev"
CONFIG_DIR = Path(__file__).parents[2] / "config"


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """
    Flatten a nested dictionary into a single level dictionary with keys joined by separator.
    
    Args:
        d: The dictionary to flatten
        parent_key: The parent key for nested dictionaries
        sep: Separator character for joined keys
        
    Returns:
        Flattened dictionary with concatenated keys
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
            
    return dict(items)


def load_yaml_config(env: str = None) -> Dict[str, Any]:
    """
    Load configuration from the appropriate YAML file based on environment.
    
    Args:
        env: Environment name (dev, uat, prod). If None, will use WEBAGENT_ENV or default to dev.
        
    Returns:
        Loaded configuration as a dictionary
    """
    # Determine which environment to use
    env = env or os.environ.get("WEBAGENT_ENV", DEFAULT_ENV)
    
    # Validate environment
    if env not in ["dev", "uat", "prod"]:
        logger.warning(f"Invalid environment: {env}. Using default: {DEFAULT_ENV}")
        env = DEFAULT_ENV
    
    # Construct file path
    config_file = CONFIG_DIR / f"{env}.yaml"
    
    # Check if file exists
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    # Load YAML configuration
    logger.info(f"Loading configuration from {config_file}")
    with open(config_file, 'r') as f:
        try:
            config = yaml.safe_load(f)
            logger.info(f"Successfully loaded configuration for environment: {env}")
            return config
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise


def export_config_to_env(config: Dict[str, Any]) -> None:
    """
    Export configuration to environment variables with WEBAGENT_ prefix.
    
    Args:
        config: Configuration dictionary
    """
    # Flatten the nested dictionary
    flat_config = flatten_dict(config, parent_key=ENV_VAR_PREFIX.rstrip('_'))
    
    # Export to environment variables
    for key, value in flat_config.items():
        # Skip if it's a complex structure, we'll handle those specially
        if isinstance(value, (dict, list)):
            continue
        
        # Convert to string
        env_value = str(value) if value is not None else ""
        
        # Set environment variable
        os.environ[key.upper()] = env_value
        
    logger.debug(f"Exported {len(flat_config)} configuration values to environment variables")


def validate_config(config: Dict[str, Any]) -> Optional[AppConfig]:
    """
    Validate configuration with Pydantic model.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Validated AppConfig object or None if validation fails
    """
    try:
        app_config = AppConfig(**config)
        logger.info("Configuration validation successful")
        return app_config
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        raise


def load_config(env: str = None) -> AppConfig:
    """
    Load, validate, and export configuration.
    
    Args:
        env: Environment name (dev, uat, prod)
        
    Returns:
        Validated AppConfig object
    """
    # Load configuration
    config = load_yaml_config(env)
    
    # Validate configuration
    app_config = validate_config(config)
    
    # Export to environment variables
    export_config_to_env(config)
    
    return app_config


# Create global config instance
_config_instance = None


def get_config() -> AppConfig:
    """
    Get the global configuration instance.
    
    Returns:
        AppConfig instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = load_config()
    return _config_instance


# Allow direct access to specific sections
def get_api_config():
    """Get API configuration section."""
    return get_config().api


def get_cors_config():
    """Get CORS configuration section."""
    return get_config().cors


def get_database_config():
    """Get database configuration section."""
    return get_config().database


def get_llm_config():
    """
    Get LLM configuration settings.
    """
    config = get_config()
    return config.llm


def get_langsmith_config():
    """
    Get LangSmith configuration settings.
    """
    config = get_config()
    return config.langsmith


def get_web_search_config():
    """Get web search configuration section."""
    return get_config().web_search


def get_tasks_config():
    """Get tasks configuration section."""
    return get_config().tasks


def get_agents_config():
    """Get agents configuration section."""
    return get_config().agents


def get_logging_config():
    """Get logging configuration section."""
    return get_config().logging


def get_security_config():
    """Get security configuration section."""
    return get_config().security 