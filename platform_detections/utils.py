"""
Platform Detection Framework - Utilities Module

Part of QuantJourney, a project for high-performance investment research and
quantitative finance, build on AI infrastructure for Investors and Quantitative
Researchers available at https://quantjourney.substack.com/

This module provides standardized utilities for error handling, 
logging, JSON serialization, and other common operations used 
across the Platform Detection Framework.

This code is licensed under the MIT License.
Copyright (c) 2023 Jakub Polec

"""

import json
import logging
import warnings
import datetime
import platform
import sys
import os
from typing import Dict, Any, List, Optional, Union, Callable

# Import constants
from .constants import (
    ERROR_LEVEL_DEBUG, ERROR_LEVEL_INFO, ERROR_LEVEL_WARNING,
    ERROR_LEVEL_ERROR, ERROR_LEVEL_CRITICAL,
    JSON_METADATA_KEY, JSON_TIMESTAMP_KEY, JSON_VERSION_KEY, 
    FRAMEWORK_VERSION
)

# Set up logging
logger = logging.getLogger("platform_detection")


class PlatformError(Exception):
    """Base exception for all platform detection errors"""
    def __init__(self, message: str, category: str = "general", level: int = ERROR_LEVEL_ERROR):
        self.message = message
        self.category = category
        self.level = level
        super().__init__(message)


class ErrorHandler:
    """
    Provides centralized error handling and logging for the framework
    
    This class offers consistent methods for handling errors, logging,
    and raising appropriate exceptions across the framework.
    """
    
    def __init__(self, log_level: int = logging.WARNING, enable_warnings: bool = True):
        """
        Initialize the error handler
        
        Args:
            log_level: The logging level to use
            enable_warnings: Whether to enable Python warnings
        """
        # Use default WARNING level if None is provided
        self.log_level = logging.WARNING if log_level is None else log_level
        self.enable_warnings = enable_warnings
        
        # Configure logging
        self._configure_logging()
        
        # Configure warnings
        if not enable_warnings:
            warnings.filterwarnings("ignore")
    
    def _configure_logging(self):
        """Configure the logging system"""
        # Check if handlers already exist
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        logger.setLevel(self.log_level)
    
    def debug(self, message: str, category: str = "general"):
        """Log a debug message"""
        logger.debug(f"[{category}] {message}")
    
    def info(self, message: str, category: str = "general"):
        """Log an info message"""
        logger.info(f"[{category}] {message}")
    
    def warning(self, message: str, category: str = "general"):
        """Log a warning message"""
        logger.warning(f"[{category}] {message}")
        if self.enable_warnings:
            warnings.warn(f"[{category}] {message}")
    
    def error(self, message: str, category: str = "general", 
              raise_exception: bool = False):
        """
        Log an error message and optionally raise an exception
        
        Args:
            message: The error message
            category: The error category
            raise_exception: Whether to raise a PlatformError
        """
        logger.error(f"[{category}] {message}")
        if raise_exception:
            raise PlatformError(message, category, ERROR_LEVEL_ERROR)
    
    def critical(self, message: str, category: str = "general",
                raise_exception: bool = True):
        """
        Log a critical error message and optionally raise an exception
        
        Args:
            message: The error message
            category: The error category
            raise_exception: Whether to raise a PlatformError
        """
        logger.critical(f"[{category}] {message}")
        if raise_exception:
            raise PlatformError(message, category, ERROR_LEVEL_CRITICAL)
    
    def handle_exception(self, e: Exception, message: str = None,
                         category: str = "general", level: int = ERROR_LEVEL_ERROR):
        """
        Handle an exception with appropriate logging
        
        Args:
            e: The exception to handle
            message: Optional message to include
            category: The error category
            level: The error level
        """
        error_msg = message if message else str(e)
        full_msg = f"{error_msg} - {str(e)}" if message else str(e)
        
        if level == ERROR_LEVEL_DEBUG:
            logger.debug(f"[{category}] {full_msg}")
        elif level == ERROR_LEVEL_INFO:
            logger.info(f"[{category}] {full_msg}")
        elif level == ERROR_LEVEL_WARNING:
            logger.warning(f"[{category}] {full_msg}")
            if self.enable_warnings:
                warnings.warn(f"[{category}] {full_msg}")
        elif level == ERROR_LEVEL_ERROR:
            logger.error(f"[{category}] {full_msg}")
        elif level == ERROR_LEVEL_CRITICAL:
            logger.critical(f"[{category}] {full_msg}")


class JSONSerializer:
    """
    Provides standardized JSON serialization and deserialization for the framework
    
    This class handles conversion of platform detection results to and from JSON,
    ensuring consistent handling of non-serializable types and metadata.
    """
    
    @staticmethod
    def to_json(data: Dict[str, Any], prettify: bool = True,
               include_metadata: bool = True) -> str:
        """
        Convert a dictionary to a JSON string
        
        Args:
            data: The dictionary to convert
            prettify: Whether to format the JSON for readability
            include_metadata: Whether to include framework metadata
            
        Returns:
            JSON string representation
        """
        # Make a copy to avoid modifying the original
        serializable_data = JSONSerializer._make_serializable(data)
        
        # Add metadata if requested
        if include_metadata:
            metadata = {
                JSON_TIMESTAMP_KEY: datetime.datetime.now().isoformat(),
                JSON_VERSION_KEY: FRAMEWORK_VERSION,
                "platform": platform.platform(),
                "python_version": platform.python_version(),
            }
            serializable_data[JSON_METADATA_KEY] = metadata
        
        # Convert to JSON
        indent = 2 if prettify else None
        return json.dumps(serializable_data, indent=indent)
    
    @staticmethod
    def to_file(data: Dict[str, Any], file_path: str, 
               prettify: bool = True, include_metadata: bool = True) -> None:
        """
        Write a dictionary to a JSON file
        
        Args:
            data: The dictionary to convert
            file_path: Path to the output file
            prettify: Whether to format the JSON for readability
            include_metadata: Whether to include framework metadata
        """
        # Make serializable and add metadata
        serializable_data = JSONSerializer._make_serializable(data)
        
        # Add metadata if requested
        if include_metadata:
            metadata = {
                JSON_TIMESTAMP_KEY: datetime.datetime.now().isoformat(),
                JSON_VERSION_KEY: FRAMEWORK_VERSION,
                "platform": platform.platform(),
                "python_version": platform.python_version(),
            }
            serializable_data[JSON_METADATA_KEY] = metadata
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # Write to file
        indent = 2 if prettify else None
        with open(file_path, 'w') as f:
            json.dump(serializable_data, f, indent=indent)
    
    @staticmethod
    def from_json(json_string: str) -> Dict[str, Any]:
        """
        Convert a JSON string to a dictionary
        
        Args:
            json_string: The JSON string to convert
            
        Returns:
            Dictionary representation
        """
        return json.loads(json_string)
    
    @staticmethod
    def from_file(file_path: str) -> Dict[str, Any]:
        """
        Read a dictionary from a JSON file
        
        Args:
            file_path: Path to the input file
            
        Returns:
            Dictionary representation
        """
        with open(file_path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def _make_serializable(obj: Any) -> Any:
        """
        Make an object JSON-serializable
        
        Args:
            obj: The object to convert
            
        Returns:
            JSON-serializable representation
        """
        if isinstance(obj, dict):
            return {k: JSONSerializer._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple, set)):
            return [JSONSerializer._make_serializable(item) for item in obj]
        elif hasattr(obj, "tolist"):  # For numpy arrays
            return JSONSerializer._make_serializable(obj.tolist())
        elif hasattr(obj, "item"):  # For numpy scalars
            return obj.item()
        elif isinstance(obj, bytes):
            return obj.decode('utf-8', errors='replace')
        elif isinstance(obj, datetime.datetime):
            return obj.isoformat()
        else:
            # Check if basic type or at least string-convertible
            try:
                json.dumps(obj)
                return obj
            except (TypeError, OverflowError):
                return str(obj)


class BaseDetector:
    """
    Base class for all platform detectors
    
    Provides common functionality for detection, error handling,
    and result standardization across all detector implementations.
    """
    
    def __init__(self, error_handler: Optional[ErrorHandler] = None):
        """
        Initialize the base detector
        
        Args:
            error_handler: Optional error handler to use
        """
        self.capabilities = {}
        self.error_handler = error_handler or ErrorHandler()
        
    def detect(self) -> Dict[str, Any]:
        """
        Perform platform detection
        
        Returns:
            Dictionary with detected capabilities
        """
        # Base implementation - should be overridden
        return self.capabilities
    
    def get_summary(self, capabilities: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get a summary of detected capabilities
        
        Args:
            capabilities: Optional pre-detected capabilities
            
        Returns:
            Dictionary with capability summary
        """
        # Base implementation - should be overridden
        if capabilities is None:
            capabilities = self.capabilities
        return capabilities
    
    def to_json(self, prettify: bool = True, include_metadata: bool = True) -> str:
        """
        Convert capabilities to a JSON string
        
        Args:
            prettify: Whether to format the JSON for readability
            include_metadata: Whether to include framework metadata
            
        Returns:
            JSON string representation
        """
        return JSONSerializer.to_json(
            self.capabilities, prettify, include_metadata
        )
    
    def to_file(self, file_path: str, prettify: bool = True, 
               include_metadata: bool = True) -> None:
        """
        Write capabilities to a JSON file
        
        Args:
            file_path: Path to the output file
            prettify: Whether to format the JSON for readability
            include_metadata: Whether to include framework metadata
        """
        JSONSerializer.to_file(
            self.capabilities, file_path, prettify, include_metadata
        )
    
    def _log_debug(self, message: str, category: str = "detection"):
        """Log a debug message"""
        self.error_handler.debug(message, category)
    
    def _log_info(self, message: str, category: str = "detection"):
        """Log an info message"""
        self.error_handler.info(message, category)
    
    def _log_warning(self, message: str, category: str = "detection"):
        """Log a warning message"""
        self.error_handler.warning(message, category)
    
    def _log_error(self, message: str, category: str = "detection",
                  raise_exception: bool = False):
        """Log an error message"""
        self.error_handler.error(message, category, raise_exception)
    
    def _run_safe(self, func: Callable, default_value: Any = None,
                 error_message: Optional[str] = None,
                 error_category: str = "detection") -> Any:
        """
        Run a function safely, handling any exceptions
        
        Args:
            func: The function to run
            default_value: Value to return if an exception occurs
            error_message: Optional error message to log
            error_category: Category for error logging
            
        Returns:
            The function result or default_value if an exception occurs
        """
        try:
            return func()
        except Exception as e:
            if error_message:
                self._log_warning(f"{error_message}: {str(e)}", error_category)
            else:
                self._log_warning(str(e), error_category)
            return default_value