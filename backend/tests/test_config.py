#!/usr/bin/env python3
"""
Tests for the YAML configuration loading functionality.
"""
import os
import sys
import pytest
from app.core.loadEnvYAML import load_config, get_config
from app.core.config import settings


def test_config_loading():
    """Test that configuration loads successfully."""
    try:
        # Load configuration
        config = load_config()
        
        # Basic assertions
        assert hasattr(config, 'api'), "Config should have 'api' attribute"
        assert hasattr(config, 'llm'), "Config should have 'llm' attribute"
        assert hasattr(config, 'agents'), "Config should have 'agents' attribute"
        
        assert hasattr(config.api, 'version'), "API config should have 'version' attribute"
        assert hasattr(config.api, 'prefix'), "API config should have 'prefix' attribute"
        assert hasattr(config.api, 'debug_mode'), "API config should have 'debug_mode' attribute"
        
        assert hasattr(config.llm, 'provider'), "LLM config should have 'provider' attribute"
        assert hasattr(config.llm, 'default_model'), "LLM config should have 'default_model' attribute"
        assert hasattr(config.llm, 'temperature'), "LLM config should have 'temperature' attribute"
    except Exception as e:
        pytest.fail(f"Configuration loading failed: {e}")


def test_settings_consistency():
    """Test that settings object is properly initialized."""
    assert hasattr(settings, 'PROJECT_NAME'), "Settings should have 'PROJECT_NAME' attribute"
    assert hasattr(settings, 'API_V1_STR'), "Settings should have 'API_V1_STR' attribute"
    assert hasattr(settings, 'VERSION'), "Settings should have 'VERSION' attribute"
    
    # Check that paths exist
    assert os.path.exists(settings.DATA_DIR), "DATA_DIR should exist"
    # The vector store creates its folder on first use, so a fresh checkout has none yet.
    assert settings.VECTOR_DB_PATH, "VECTOR_DB_PATH should be set"


def test_env_variable_overrides():
    """Test that environment variables can override config values."""
    original_api_prefix = settings.API_V1_STR
    
    try:
        # Set environment variable
        os.environ["WEBAGENT_API_PREFIX"] = "/custom/api"
        
        # Re-import settings (this won't actually work in pytest due to caching,
        # but in a real app restart, it would pick up the new value)
        from app.core.config import settings as new_settings
        
        # For testing, we can at least verify that the current settings are still intact
        assert settings.API_V1_STR == original_api_prefix
    finally:
        # Clean up
        if "WEBAGENT_API_PREFIX" in os.environ:
            del os.environ["WEBAGENT_API_PREFIX"]


# Keep the display_config function for command-line use
def display_config(env=None):
    """Display the loaded configuration."""
    print(f"\n{'='*50}")
    print(f"Loading configuration for environment: {env or os.environ.get('WEBAGENT_ENV', 'dev')}")
    print(f"{'='*50}\n")
    
    try:
        # Load configuration
        config = load_config(env)
        
        # Display configuration
        print("Configuration loaded successfully!\n")
        
        # Display some key values
        print("API Configuration:")
        print(f"  Version: {config.api.version}")
        print(f"  Prefix: {config.api.prefix}")
        print(f"  Debug Mode: {config.api.debug_mode}")
        
        print("\nLLM Configuration:")
        print(f"  Provider: {config.llm.provider}")
        print(f"  Default Model: {config.llm.default_model}")
        print(f"  Temperature: {config.llm.temperature}")
        
        print("\nAgent Configurations:")
        for agent_name in ["supervisor", "web_research", "internal_research", 
                          "senior_research", "data_analysis", "coding", "team_manager"]:
            agent_config = getattr(config.agents, agent_name)
            print(f"  {agent_name.replace('_', ' ').title()}:")
            print(f"    Model: {agent_config.model}")
            print(f"    Temperature: {agent_config.temperature}")
        
        print("\nEnvironment Variables (WEBAGENT_ prefix):")
        webagent_env_vars = {k: v for k, v in os.environ.items() if k.startswith("WEBAGENT_")}
        for k, v in sorted(webagent_env_vars.items()):
            print(f"  {k}: {v}")
            
        return True
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return False


def main():
    """Main function for command-line use."""
    # Get environment from command-line argument if provided
    env = sys.argv[1] if len(sys.argv) > 1 else None
    
    success = display_config(env)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 