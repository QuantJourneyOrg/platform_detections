"""
Platform Detection Framework
A comprehensive framework for detecting hardware, software, and OS-specific
capabilities to optimize computational workloads across different platforms.

Part of QuantJourney, a project for high-performance investment research and
quantitative finance, build on AI infrastructure for Investors and Quantitative
Researchers.

This module provides a unified interface for detecting and utilizing various
compute backends, including CPU, GPU, and specialized hardware. It also includes
utilities for error handling and data serialization.

This code is licensed under the MIT License.
Copyright (c) 2023 Jakub Polec

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

__version__ = "1.0.0"
__author__ = "Jakub Polec"

from .orchestrator import PlatformOrchestrator, ComputeBackend
from .utils import ErrorHandler, JSONSerializer

# Convenience function to get an orchestrator instance
def get_detector(force_detect=True, log_level=None, enable_warnings=True):
    """
    Get a PlatformOrchestrator instance with the specified settings
    
    Args:
        force_detect: Whether to run detection immediately
        log_level: Logging level to use
        enable_warnings: Whether to enable Python warnings
        
    Returns:
        PlatformOrchestrator instance
    """
    return PlatformOrchestrator(
        force_detect=force_detect,
        log_level=log_level,
        enable_warnings=enable_warnings
    )

# Convenience decorator for optimization
def optimize(operation_type=None, data_size_estimator=None):
    """
    Decorator that automatically selects the optimal backend for a function
    
    Args:
        operation_type: Type of operation ('matrix', 'stat', 'ml', etc.)
        data_size_estimator: Function to estimate data size from arguments
        
    Returns:
        Decorator function
    """
    from .backends import use_backend
    
    # Get or create a detector instance
    detector = get_detector()
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Determine the operation type
            op_type = operation_type or "matrix"
            
            # Estimate data size if provided
            data_size = 0
            if data_size_estimator:
                data_size = data_size_estimator(*args, **kwargs)
                
            # Get the recommended backend
            backend = detector.get_backend_for_operation(op_type, data_size)
            
            # Execute with the appropriate backend
            with use_backend(backend):
                return func(*args, **kwargs)
                
        return wrapper
    return decorator