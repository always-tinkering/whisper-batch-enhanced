"""
Whisper Batch - Automated Video Transcription Tool

A tool for batch processing video files and transcribing them using Whisper.
"""

__version__ = '0.1.0'
__author__ = 'Your Name'

# Configure shared logging format
import logging
import os
from pathlib import Path
import sys
import traceback
from datetime import datetime

# Standard logging format to be used across all modules
DEFAULT_LOG_FORMAT = '%(asctime)s - %(levelname)s - [%(module)s] %(message)s'

# Setup logging with file output
def setup_logging(log_to_file=True, log_level=logging.INFO):
    """
    Configure logging with console and optionally file output.
    
    Args:
        log_to_file: Whether to log to a file in addition to console.
        log_level: The logging level to use.
        
    Returns:
        The path to the log file if created, None otherwise.
    """
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(DEFAULT_LOG_FORMAT)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    log_file_path = None
    if log_to_file:
        # Create logs directory if it doesn't exist
        logs_dir = Path.home() / 'whisper_batch_logs'
        logs_dir.mkdir(exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file_path = logs_dir / f'whisper_batch_{timestamp}.log'
        
        # Create file handler
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(DEFAULT_LOG_FORMAT)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        logging.info(f"Log file created at: {log_file_path}")
    
    return log_file_path

# Global exception handler
def handle_exception(exc_type, exc_value, exc_traceback):
    """
    Global exception handler to log unhandled exceptions.
    """
    if issubclass(exc_type, KeyboardInterrupt):
        # Don't log keyboard interrupt
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
        
    # Log the exception
    logger = logging.getLogger(__name__)
    logger.critical("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))
    
    # Write to error file for easier access
    error_file = Path.home() / 'whisper_batch_error.log'
    try:
        with open(error_file, 'a', encoding='utf-8') as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"\n\n--- UNHANDLED EXCEPTION [{timestamp}] ---\n")
            traceback.print_exception(exc_type, exc_value, exc_traceback, file=f)
            f.write("\n--- END EXCEPTION ---\n")
    except Exception as e:
        logger.error(f"Failed to write to error file: {e}")
    
    # Show error to the user
    sys.__excepthook__(exc_type, exc_value, exc_traceback) 