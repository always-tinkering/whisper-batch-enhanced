"""
Whisper Batch Enhanced - Batch audio/video transcription with improved CUDA error handling.

This tool enhances the OpenAI Whisper transcription model with batch processing
capabilities and improved CUDA error handling for Windows environments.
"""

import logging
import sys
import os
from pathlib import Path
import datetime
import traceback

# Version information
__version__ = "1.0.0"
__author__ = "Whisper Batch Team"
__email__ = "info@example.com"

# Default log format
DEFAULT_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

# Default locations for logs
DEFAULT_LOG_DIR = Path.home() / ".whisper_batch" / "logs"
DEFAULT_ERROR_LOG = DEFAULT_LOG_DIR / "error.log"

def setup_logging(log_level=logging.INFO, log_file=None):
    """
    Set up logging to both console and file.
    
    Args:
        log_level: The logging level to use.
        log_file: Optional file path for logging. If None, no file logging.
    
    Returns:
        The configured logger.
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(DEFAULT_LOG_FORMAT)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if requested
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

def handle_exception(exc_type, exc_value, exc_traceback):
    """
    Global exception handler to log unhandled exceptions.
    
    Args:
        exc_type: Exception type.
        exc_value: Exception value.
        exc_traceback: Exception traceback.
    """
    # Skip KeyboardInterrupt exceptions
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    # Create log directory if it doesn't exist
    if not DEFAULT_LOG_DIR.exists():
        DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Log the exception
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    error_log = DEFAULT_ERROR_LOG
    
    with open(error_log, 'a') as f:
        f.write(f"\n\n--- Unhandled Exception: {timestamp} ---\n")
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=f)
    
    # Also output to console
    print(f"An unhandled exception occurred. Details have been logged to: {error_log}", file=sys.stderr)
    traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stderr)

# Set the global exception handler
sys.excepthook = handle_exception