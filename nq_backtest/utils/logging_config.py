"""Logging configuration for the NQ backtest system."""

import logging
import logging.handlers
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Colored console formatter for better readability."""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        """Format the log record with colors."""
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset_color = self.COLORS['RESET']
        
        # Create colored level name
        record.levelname = f"{log_color}{record.levelname}{reset_color}"
        
        return super().format(record)


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: str = "logs",
    enable_console: bool = True,
    enable_file: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """Setup comprehensive logging for the NQ backtest system.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Specific log file name (if None, auto-generated)
        log_dir: Directory for log files
        enable_console: Enable console logging
        enable_file: Enable file logging
        max_file_size: Maximum size of each log file
        backup_count: Number of backup log files to keep
        
    Returns:
        Configured root logger
    """
    # Convert log level string to constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)-20s | %(funcName)-15s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = ColoredFormatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)-15s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # File handler with rotation
    if enable_file:
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"nq_backtest_{timestamp}.log"
        
        file_path = log_path / log_file
        
        # Use rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            filename=file_path,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(file_handler)
    
    # Configure specific loggers
    _configure_module_loggers(numeric_level)
    
    # Log startup message
    root_logger.info(f"Logging initialized - Level: {log_level}, File: {log_file if enable_file else 'Disabled'}")
    
    return root_logger


def _configure_module_loggers(log_level: int):
    """Configure logging levels for specific modules."""
    
    # Set specific log levels for external libraries
    logging.getLogger('ib_insync').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    
    # NQ backtest module loggers
    nq_loggers = [
        'nq_backtest.data',
        'nq_backtest.strategy',
        'nq_backtest.backtest',
        'nq_backtest.utils',
        'nq_backtest.reports'
    ]
    
    for logger_name in nq_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


class LoggingContext:
    """Context manager for temporary logging configuration."""
    
    def __init__(self, logger_name: str, level: str):
        """Initialize logging context.
        
        Args:
            logger_name: Name of the logger to modify
            level: Temporary log level
        """
        self.logger = logging.getLogger(logger_name)
        self.original_level = self.logger.level
        self.new_level = getattr(logging, level.upper(), logging.INFO)
    
    def __enter__(self):
        """Enter the context and set new log level."""
        self.logger.setLevel(self.new_level)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context and restore original log level."""
        self.logger.setLevel(self.original_level)


def log_function_call(func):
    """Decorator to log function calls with parameters and execution time."""
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        
        # Log function entry
        logger.debug(f"Entering {func.__name__} with args={args}, kwargs={kwargs}")
        
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"Exiting {func.__name__} - Execution time: {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Exception in {func.__name__} after {execution_time:.3f}s: {e}")
            raise
    
    return wrapper


def log_performance(operation_name: str):
    """Decorator to log performance metrics for operations."""
    import functools
    import time
    import psutil
    import os
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            
            # Get initial metrics
            process = psutil.Process(os.getpid())
            start_memory = process.memory_info().rss / 1024 / 1024  # MB
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                # Calculate metrics
                end_time = time.time()
                end_memory = process.memory_info().rss / 1024 / 1024  # MB
                
                execution_time = end_time - start_time
                memory_delta = end_memory - start_memory
                
                logger.info(
                    f"Performance - {operation_name}: "
                    f"Time={execution_time:.3f}s, "
                    f"Memory={end_memory:.1f}MB ({memory_delta:+.1f}MB)"
                )
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"Performance - {operation_name} FAILED after {execution_time:.3f}s: {e}")
                raise
        
        return wrapper
    return decorator