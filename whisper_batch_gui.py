#!/usr/bin/env python
"""
Whisper Batch GUI Launcher

This script launches the Whisper Batch GUI application.
"""

import sys
import os
import logging
from pathlib import Path
import traceback

# Add the project root to the Python path
script_path = Path(__file__).resolve()
project_root = script_path.parent  
sys.path.insert(0, str(project_root))

try:
    # Import the enhanced logging setup and exception handler
    from src.whisper_batch import setup_logging, handle_exception
    from src.gui.tkinter_app import main
    
    # Set up global exception handling
    sys.excepthook = handle_exception
    
    # Configure logging with file output
    log_file = setup_logging(log_to_file=True, log_level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    logger.info(f"Whisper Batch GUI starting, version 0.1.0")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Running from: {project_root}")
    
    if __name__ == "__main__":
        try:
            # Launch the GUI
            logger.info("Launching GUI application")
            main()
        except Exception as e:
            logger.critical(f"Failed to start GUI: {e}", exc_info=True)
            # Re-raise to let the global exception handler deal with it
            raise
except Exception as e:
    # Fallback error handling (in case the import fails)
    error_file = Path.home() / 'whisper_batch_startup_error.log'
    try:
        with open(error_file, 'a', encoding='utf-8') as f:
            f.write("\n\n--- STARTUP ERROR ---\n")
            traceback.print_exc(file=f)
            f.write(f"\nError: {str(e)}\n")
            f.write("--- END ERROR ---\n")
        print(f"Error during startup. See log file: {error_file}")
    except:
        print(f"Critical error during startup: {str(e)}")
        traceback.print_exc()
    sys.exit(1) 