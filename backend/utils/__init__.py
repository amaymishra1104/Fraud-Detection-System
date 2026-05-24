"""
Backend utility functions and helpers.
"""

import logging
import json
from typing import Any, Dict

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """
    Configure logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        )
        logging.getLogger().addHandler(file_handler)


def format_response(data: Any, message: str = "Success", status: str = "success") -> Dict:
    """
    Format API response.
    
    Args:
        data: Response data
        message: Response message
        status: Response status
    
    Returns:
        Formatted response dictionary
    """
    return {
        "status": status,
        "message": message,
        "data": data,
        "timestamp": str(__import__('datetime').datetime.utcnow())
    }


def error_response(message: str, error_code: str = "ERROR", status_code: int = 400) -> Dict:
    """
    Format error response.
    
    Args:
        message: Error message
        error_code: Error code identifier
        status_code: HTTP status code
    
    Returns:
        Formatted error response dictionary
    """
    return {
        "status": "error",
        "message": message,
        "error_code": error_code,
        "status_code": status_code,
        "timestamp": str(__import__('datetime').datetime.utcnow())
    }
