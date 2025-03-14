"""
Logging configuration for the WebAgent backend.

This module sets up a structured logger with different levels,
log rotation, and formatting based on the environment.
"""
import os
import sys
import json
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Dict, Any, Optional, Union

from app.core.config import settings

# Create logs directory if it doesn't exist
logs_dir = Path("./logs")
logs_dir.mkdir(exist_ok=True)

# Get environment or default to 'dev'
env = getattr(settings, "WEBAGENT_ENV", os.getenv("WEBAGENT_ENV", "dev"))

# Determine log file path
log_file = logs_dir / f"webagent-{env}.log"

# Configure logging levels based on environment
if settings.DEBUG_MODE:
    log_level = logging.DEBUG
else:
    log_level = logging.INFO

# Create custom formatter for structured logging
class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as a JSON string."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "module": record.module,
            "path": record.pathname,
            "line": record.lineno,
        }
        
        # Add exception info if available
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info),
            }
        
        # Add extra fields if available
        if hasattr(record, "extra"):
            log_data.update(record.extra)
        
        return json.dumps(log_data)

# Create custom logger class with extra fields
class StructuredLogger(logging.Logger):
    """Logger that supports structured logging with extra fields."""
    
    def _log_with_extra(self, level: int, msg: str, extra: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """Log with extra fields."""
        if extra is None:
            extra = {}
        
        # Add extra fields
        kwargs["extra"] = {"extra": extra}
        
        # Call parent method
        super().log(level, msg, **kwargs)
    
    def debug(self, msg: str, extra: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """Log a debug message with extra fields."""
        self._log_with_extra(logging.DEBUG, msg, extra, **kwargs)
    
    def info(self, msg: str, extra: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """Log an info message with extra fields."""
        self._log_with_extra(logging.INFO, msg, extra, **kwargs)
    
    def warning(self, msg: str, extra: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """Log a warning message with extra fields."""
        self._log_with_extra(logging.WARNING, msg, extra, **kwargs)
    
    def error(self, msg: str, extra: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """Log an error message with extra fields."""
        self._log_with_extra(logging.ERROR, msg, extra, **kwargs)
    
    def critical(self, msg: str, extra: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """Log a critical message with extra fields."""
        self._log_with_extra(logging.CRITICAL, msg, extra, **kwargs)

# Register custom logger class
logging.setLoggerClass(StructuredLogger)

# Create logger
logger = logging.getLogger("webagent")
logger.setLevel(log_level)

# Create console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(log_level)

# Create file handler with rotation
if env == "prod":
    # In production, rotate logs daily and keep 30 days of logs
    file_handler = TimedRotatingFileHandler(
        filename=log_file,
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8",
    )
else:
    # In development and UAT, rotate logs when they reach 10 MB
    file_handler = RotatingFileHandler(
        filename=log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )

file_handler.setLevel(log_level)

# Set formatters
if env == "dev":
    # Use readable format for development
    console_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_format = console_format
else:
    # Use JSON formatter for UAT and production
    console_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_format = JsonFormatter()

console_handler.setFormatter(console_format)
file_handler.setFormatter(file_format)

# Add handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Add request_id context
class RequestIdFilter(logging.Filter):
    """Filter that adds request_id to log records."""
    
    def __init__(self, name: str = "", default_request_id: str = "no_request_id"):
        super().__init__(name)
        self.default_request_id = default_request_id
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add request_id to the log record."""
        if not hasattr(record, "request_id"):
            record.request_id = self.default_request_id
        return True

# Add filter to add request_id
request_id_filter = RequestIdFilter()
logger.addFilter(request_id_filter)

def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    return logging.getLogger(f"webagent.{name}")

def setup_logger(name: str, log_level: int = None) -> logging.Logger:
    """
    Set up a logger with the given name and log level.
    
    Args:
        name: Name of the logger
        log_level: Log level (defaults to the level set in settings)
        
    Returns:
        Configured logger instance
    """
    # Get the logger
    logger_instance = get_logger(name)
    
    # Set log level (use the one from settings if not specified)
    if log_level is None:
        log_level = logging.DEBUG if settings.DEBUG_MODE else logging.INFO
    
    logger_instance.setLevel(log_level)
    
    # Ensure the logger has handlers (might already have them from root logger)
    if not logger_instance.handlers:
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        
        # Set formatter
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger_instance.addHandler(console_handler)
    
    return logger_instance

# Log startup information
logger.info(
    f"Logger initialized - Environment: {env}, "
    f"Level: {logging.getLevelName(log_level)}, "
    f"File: {log_file}"
) 