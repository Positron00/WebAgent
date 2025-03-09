#!/usr/bin/env python3
"""
Script to verify that all required dependencies are installed and configured properly.
"""
import importlib
import os
import sys
from dotenv import load_dotenv

def check_module(module_name, package_name=None):
    """Check if a module can be imported."""
    package_name = package_name or module_name
    try:
        importlib.import_module(module_name)
        print(f"✅ {module_name} is installed")
        return True
    except ImportError:
        print(f"❌ {module_name} is NOT installed. Install with: pip install {package_name}")
        return False

def check_env_var(name):
    """Check if an environment variable is set."""
    value = os.getenv(name)
    if value:
        masked_value = value[:4] + "..." + value[-4:] if len(value) > 10 else "***"
        print(f"✅ {name} is set to: {masked_value}")
        return True
    else:
        print(f"❌ {name} is NOT set")
        return False

def main():
    """Main function."""
    print("Checking WebAgent backend installation...\n")
    
    # Load environment variables
    load_dotenv()
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major >= 3 and python_version.minor >= 10:
        print(f"✅ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        print(f"❌ Python version: {python_version.major}.{python_version.minor}.{python_version.micro} (3.10+ recommended)")
    
    # Check core packages
    modules = [
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("pydantic", "pydantic"),
        ("langchain", "langchain"),
        ("langgraph", "langgraph"),
        ("langchain_openai", "langchain-openai"),
        ("langchain_community", "langchain-community"),
        ("dotenv", "python-dotenv"),
    ]
    
    all_modules_installed = all(check_module(m, p) for m, p in modules)
    
    print("\nChecking environment variables:")
    all_env_vars_set = check_env_var("OPENAI_API_KEY")
    # Optional env vars
    check_env_var("TAVILY_API_KEY")
    
    # Final status
    print("\nSummary:")
    if all_modules_installed and all_env_vars_set:
        print("✅ All required packages and environment variables are installed and configured.")
        print("You're good to go! Start the server with: uvicorn main:app --reload")
    else:
        print("❌ Some requirements are missing. Please fix the issues above.")

if __name__ == "__main__":
    main() 