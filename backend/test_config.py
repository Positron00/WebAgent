#!/usr/bin/env python3
"""
Script to test the YAML configuration loading.
"""
import os
import sys
import pprint
from app.core.loadEnvYAML import load_config, get_config


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
    """Main function."""
    # Get environment from command-line argument if provided
    env = sys.argv[1] if len(sys.argv) > 1 else None
    
    success = display_config(env)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 