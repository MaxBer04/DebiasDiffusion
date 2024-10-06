"""
Logger Module for DebiasDiffusion

This module provides a custom logging functionality for the DebiasDiffusion project.
It creates a CustomLogger class that sets up both file and console logging.

Usage:
    from src.utils.logger import get_logger

    logger = get_logger(log_dir="/path/to/logs", experiment_name="my_experiment")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")

Note:
    The logger creates log files in the specified directory with the experiment name.
"""

import logging
import os
from datetime import datetime
from typing import Optional

class CustomLogger:
    def __init__(self, log_dir: str, experiment_name: str):
        """
        Initialize the CustomLogger.

        Args:
            log_dir (str): Directory to save log files.
            experiment_name (str): Name of the experiment for log file naming.
        """
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(logging.INFO)
        
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        log_file = os.path.join(log_dir, f"{experiment_name}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def info(self, message: str) -> None:
        """Log an info message."""
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """Log a warning message."""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log an error message."""
        self.logger.error(message)

def get_logger(log_dir: str, experiment_name: str) -> CustomLogger:
    """
    Create and return a CustomLogger instance.

    Args:
        log_dir (str): Directory to save log files.
        experiment_name (str): Name of the experiment for log file naming.

    Returns:
        CustomLogger: An instance of the CustomLogger class.
    """
    return CustomLogger(log_dir, experiment_name)