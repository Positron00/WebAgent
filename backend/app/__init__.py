"""
WebAgent App Package
==================

This package contains the core components of the WebAgent platform.
"""

# Import the path setup to ensure imports work correctly
import os
import sys
import logging
import logging.config
from pathlib import Path

# Setup colored logging
try:
    import colorama
    
    # Initialize colorama for cross-platform colored terminal output
    colorama.init()
    
    # Define color codes
    class ColorCodes:
        GREY = "\033[38;5;240m"
        GREEN = "\033[32m"
        YELLOW = "\033[33m"
        ORANGE = "\033[38;5;208m" 
        RED = "\033[31m"
        BOLD_RED = "\033[31;1m"
        BLUE = "\033[34m"
        RESET = "\033[0m"
    
    # Custom formatter with colors
    class ColoredFormatter(logging.Formatter):
        FORMATS = {
            logging.DEBUG: ColorCodes.GREY + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + ColorCodes.RESET,
            logging.INFO: "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            logging.WARNING: ColorCodes.ORANGE + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + ColorCodes.RESET,
            logging.ERROR: ColorCodes.RED + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + ColorCodes.RESET,
            logging.CRITICAL: ColorCodes.BOLD_RED + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + ColorCodes.RESET,
        }
    
        def format(self, record):
            log_fmt = self.FORMATS.get(record.levelno)
            formatter = logging.Formatter(log_fmt)
            return formatter.format(record)
    
    # Configure root logger with colored output
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredFormatter())
    
    # Get the root logger and add our handler
    root_logger = logging.getLogger()
    
    # Remove any existing handlers and add our colored handler
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    root_logger.addHandler(console_handler)
    
    # Set default level to INFO but let individual loggers override
    root_logger.setLevel(logging.INFO)
    
except ImportError:
    # Fall back to standard logging if colorama isn't available
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

# Add the backend directory to the path if needed
backend_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if backend_dir.name == 'backend' and str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# Version is defined once, in the repo-root package.json.
def _read_version() -> str:
    import json

    try:
        package_json = Path(__file__).resolve().parents[2] / "package.json"
        return json.loads(package_json.read_text())["version"]
    except (OSError, KeyError, ValueError):
        return "0.0.0"


__version__ = _read_version()
