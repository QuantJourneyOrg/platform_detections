"""
Platform Detection Framework - Base Detector Module

This module provides the BaseDetector class that all specific detectors
should inherit from to ensure consistent interface and behavior.

Part of QuantJourney, a project for high-performance investment research and
quantitative finance, build on AI infrastructure for Investors and Quantitative
Researchers available at https://quantjourney.substack.com/

This code is licensed under the MIT License.
Copyright (c) 2023 Jakub Polec

"""

from typing import Dict, Any, List, Optional, Callable
import abc

from ..utils import ErrorHandler, JSONSerializer


class BaseDetector(metaclass=abc.ABCMeta):
    """
    Abstract base class for all platform detectors
    
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
        
    @abc.abstractmethod
    def detect(self) -> Dict[str, Any]:
        """
        Perform platform detection
        
        Returns:
            Dictionary with detected capabilities
        """
        raise NotImplementedError("Subclasses must implement detect()")
    
    @abc.abstractmethod
    def get_summary(self, capabilities: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get a summary of detected capabilities
        
        Args:
            capabilities: Optional pre-detected capabilities
            
        Returns:
            Dictionary with capability summary
        """
        if capabilities is None:
            capabilities = self.capabilities.copy()
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
    
    def _normalize_flags(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize capability flags to use standard format
        
        Args:
            info: Dictionary with capability information
            
        Returns:
            Dictionary with normalized capability flags
        """
        normalized = {}
        
        for key, value in info.items():
            # Normalize capability flags to has_* format
            if key.endswith("_available") and isinstance(value, bool):
                new_key = key.replace("_available", "")
                new_key = f"has_{new_key}"
                normalized[new_key] = value
            # Normalize is_* to has_* format
            elif key.startswith("is_") and isinstance(value, bool):
                new_key = key.replace("is_", "has_")
                normalized[new_key] = value
            else:
                normalized[key] = value
                
        return normalized