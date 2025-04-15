"""
Platform Detection Framework - Orchestrator Module

Part of QuantJourney, a project for high-performance investment research and
quantitative finance, build on AI infrastructure for Investors and Quantitative
Researchers available at https://quantjourney.substack.com/

This module coordinates the detection of hardware and software capabilities
across different platforms to optimize computational workloads.

This code is licensed under the MIT License.
Copyright (c) 2023 Jakub Polec

"""

import platform
import sys
import os
import importlib
import datetime
from enum import Enum
from typing import Dict, Any, List, Optional, Union, Set

# Import standardized utilities and constants
from .utils import ErrorHandler, JSONSerializer, PlatformError
from .constants import (
    FLAG_NUMPY, FLAG_CUDA, FLAG_METAL, FLAG_OPENCL, 
    FLAG_APPLE_SILICON, FLAG_ACCELERATE, FLAG_POLARS,
    FLAG_NUMBA, FLAG_CYTHON, FLAG_BOTTLENECK, FLAG_DASK,
    DATA_SIZE_SMALL, DATA_SIZE_MEDIUM, DATA_SIZE_LARGE,
    OPERATION_TYPE_MATRIX, OPERATION_TYPE_STAT, OPERATION_TYPE_ML,
    OPERATION_TYPE_DATA, OPERATION_TYPE_FINANCE
)

# Import platform-specific detector modules
from platform_detections.detectors.os import (
    DarwinDetector, LinuxDetector, WindowsDetector
)

from platform_detections.detectors import HardwareDetector, SoftwareDetector

class ComputeBackend(Enum):
    """Available computation backends"""
    NUMPY = "numpy"          # Basic NumPy (fallback)
    NUMBA = "numba"          # Numba JIT compilation
    CUDA = "cuda"            # NVIDIA GPU
    OPENCL = "opencl"        # Generic GPU
    METAL = "metal"          # Apple Metal GPU
    ACCELERATE = "accelerate"  # Apple Accelerate framework
    CYTHON = "cython"        # Cython optimized
    POLARS = "polars"        # Polars DataFrames
    MKL = "mkl"              # Intel MKL optimized
    DASK = "dask"            # Dask distributed
    BOTTLENECK = "bottleneck"  # Bottleneck optimized functions
    TENSORFLOW = "tensorflow"  # TensorFlow
    PYTORCH = "pytorch"      # PyTorch
    JAX = "jax"              # JAX
    DARWIN = "darwin"        # Enhanced macOS/Darwin backend
    LINUX = "linux"          # Linux-specific optimizations
    WINDOWS = "windows"      # Windows-specific optimizations


class PlatformOrchestrator:
    """
    Orchestrates platform detection across different modules
    
    This class coordinates the detection of hardware, software, and OS-specific
    capabilities, combining them into a unified view of available resources.
    """
    
    def __init__(self, force_detect: bool = True, log_level: int = None, 
                enable_warnings: bool = True):
        """
        Initialize the orchestrator
        
        Args:
            force_detect: Whether to run detection immediately
            log_level: Logging level to use
            enable_warnings: Whether to enable Python warnings
        """
        # Set up error handling
        self.error_handler = ErrorHandler(
            log_level=log_level, 
            enable_warnings=enable_warnings
        )
        
        # Initialize capabilities
        self.capabilities = {}
        
        # Initialize detectors
        self.hw_detector = HardwareDetector(error_handler=self.error_handler)
        self.sw_detector = SoftwareDetector(error_handler=self.error_handler)
        
        # OS-specific detector
        os_name = platform.system().lower()
        self.os_name = os_name
        
        if os_name == "darwin":
            self.os_detector = DarwinDetector(error_handler=self.error_handler)
        elif os_name == "linux":
            self.os_detector = LinuxDetector(error_handler=self.error_handler)
        elif os_name == "windows":
            self.os_detector = WindowsDetector(error_handler=self.error_handler)
        else:
            self.os_detector = None
            self.error_handler.warning(
                f"Unknown OS: {os_name}, OS-specific detection will be skipped", 
                "orchestrator"
            )
            
        # Run detection if requested
        if force_detect:
            self.detect()
    
    def detect(self) -> Dict[str, Any]:
        """
        Perform comprehensive detection of platform capabilities
        
        Returns:
            Dictionary with detected capabilities
        """
        self.error_handler.info("Starting platform detection", "orchestrator")
        
        # Basic info to start with
        self.capabilities = {
            "os": self.os_name,
            "os_version": platform.version(),
            "os_release": platform.release(),
            "python_version": platform.python_version(),
            "python_bits": 64 if sys.maxsize > 2**32 else 32,
            "python_implementation": platform.python_implementation(),
            "python_compiler": platform.python_compiler(),
            "detection_timestamp": datetime.datetime.now().isoformat(),
        }
        
        # Run hardware detection
        self.error_handler.info("Detecting hardware capabilities", "orchestrator")
        try:
            hw_info = self.hw_detector.detect()
            self.capabilities.update(hw_info)
        except Exception as e:
            self.error_handler.error(
                f"Error during hardware detection: {str(e)}", 
                "orchestrator"
            )
        
        # Run software detection
        self.error_handler.info("Detecting software capabilities", "orchestrator")
        try:
            sw_info = self.sw_detector.detect()
            self.capabilities.update(sw_info)
        except Exception as e:
            self.error_handler.error(
                f"Error during software detection: {str(e)}", 
                "orchestrator"
            )
        
        # Run OS-specific detection
        if self.os_detector:
            self.error_handler.info(
                f"Detecting {self.os_name}-specific capabilities", 
                "orchestrator"
            )
            try:
                os_info = self.os_detector.detect()
                self.capabilities.update(os_info)
            except Exception as e:
                self.error_handler.error(
                    f"Error during OS-specific detection: {str(e)}", 
                    "orchestrator"
                )
                
        self.error_handler.info(
            f"Platform detection complete, found {len(self.capabilities)} capabilities", 
            "orchestrator"
        )
        return self.capabilities
    
    def get_optimal_backend(self) -> ComputeBackend:
        """
        Determines the optimal backend based on platform capabilities

        Returns:
            ComputeBackend: Recommended backend enum for this platform
        """
        info = self.capabilities
        
        # If we haven't detected capabilities yet, do it now
        if not info:
            self.detect()
            info = self.capabilities

        # macOS-specific selection with Metal and Accelerate
        if info.get("os") == "darwin":
            if info.get(FLAG_APPLE_SILICON, False) and info.get(FLAG_METAL, False):
                return ComputeBackend.DARWIN  # Use enhanced Darwin backend for Apple Silicon with Metal
            elif info.get(FLAG_METAL, False):
                return ComputeBackend.METAL
            elif info.get(FLAG_ACCELERATE, False):
                return ComputeBackend.ACCELERATE

        # CUDA is preferred if available
        if info.get("has_gpu", False) and info.get("gpu_vendor") == "nvidia" and info.get(FLAG_CUDA, False):
            return ComputeBackend.CUDA

        # OpenCL is a good choice for AMD/Intel GPUs
        if info.get("has_gpu", False) and info.get("gpu_vendor") in ["amd", "intel"] and info.get(FLAG_OPENCL, False):
            return ComputeBackend.OPENCL

        # OS-specific optimized backends if available
        if info.get("os") == "linux":
            return ComputeBackend.LINUX
        elif info.get("os") == "windows":
            return ComputeBackend.WINDOWS
            
        # MKL if NumPy is using it
        if info.get("numpy_using_mkl", False):
            return ComputeBackend.MKL
            
        # Polars if available (often faster than pandas)
        if info.get(FLAG_POLARS, False):
            return ComputeBackend.POLARS
            
        # Numba is a good CPU-based option
        if info.get(FLAG_NUMBA, False):
            return ComputeBackend.NUMBA
            
        # Cython if available
        if info.get(FLAG_CYTHON, False):
            return ComputeBackend.CYTHON

        # Fallback to basic NumPy
        return ComputeBackend.NUMPY
    
    def get_backend_for_operation(self, operation_type: str, data_size: int = 0) -> ComputeBackend:
        """
        Get the optimal backend for a specific type of operation and data size
        
        Args:
            operation_type: Type of operation ('matrix', 'stat', 'ml', etc.)
            data_size: Size of data to process
            
        Returns:
            Optimal backend for the specified operation
        """
        info = self.capabilities
        default_backend = self.get_optimal_backend()
        
        # For very small data, avoid GPU overhead
        if data_size < DATA_SIZE_SMALL:
            if default_backend in [ComputeBackend.CUDA, ComputeBackend.OPENCL, ComputeBackend.METAL]:
                if info.get(FLAG_NUMBA, False):
                    return ComputeBackend.NUMBA
                elif info.get("numpy_using_mkl", False):
                    return ComputeBackend.MKL
                else:
                    return ComputeBackend.NUMPY
        
        # For matrix operations
        if operation_type.lower() == OPERATION_TYPE_MATRIX:
            # Large matrix operations benefit from GPU
            if data_size > DATA_SIZE_MEDIUM:
                if info.get(FLAG_CUDA, False):
                    return ComputeBackend.CUDA
                elif info.get(FLAG_METAL, False) and info.get(FLAG_APPLE_SILICON, False):
                    return ComputeBackend.DARWIN
                elif info.get(FLAG_METAL, False):
                    return ComputeBackend.METAL
                elif info.get(FLAG_OPENCL, False):
                    return ComputeBackend.OPENCL
            
            # Medium matrix operations
            if data_size > DATA_SIZE_SMALL:
                if info.get("os") == "darwin" and info.get(FLAG_ACCELERATE, False):
                    return ComputeBackend.ACCELERATE
                elif info.get("numpy_using_mkl", False):
                    return ComputeBackend.MKL
        
        # For statistical operations
        if operation_type.lower() == OPERATION_TYPE_STAT:
            if info.get(FLAG_BOTTLENECK, False):
                return ComputeBackend.BOTTLENECK
            elif info.get(FLAG_NUMBA, False):
                return ComputeBackend.NUMBA
        
        # For ML operations
        if operation_type.lower() == OPERATION_TYPE_ML:
            if info.get("has_tensorflow", False) and info.get("tensorflow_gpu_available", False):
                return ComputeBackend.TENSORFLOW
            elif info.get("has_pytorch", False) and info.get("pytorch_gpu_available", False):
                return ComputeBackend.PYTORCH
            elif info.get("has_jax", False):
                return ComputeBackend.JAX
        
        # For data processing
        if operation_type.lower() == OPERATION_TYPE_DATA:
            if data_size > DATA_SIZE_MEDIUM and info.get(FLAG_DASK, False):
                return ComputeBackend.DASK
            elif info.get(FLAG_POLARS, False):
                return ComputeBackend.POLARS
        
        # Finance-specific operations (often benefit from low-latency options)
        if operation_type.lower() == OPERATION_TYPE_FINANCE:
            if info.get(FLAG_NUMBA, False):
                return ComputeBackend.NUMBA
            elif info.get("numpy_using_mkl", False):
                return ComputeBackend.MKL
            # For large financial datasets
            elif data_size > DATA_SIZE_MEDIUM and info.get(FLAG_POLARS, False):
                return ComputeBackend.POLARS
                
        # Default to the general optimal backend
        return default_backend
    
    def get_hardware_summary(self) -> Dict[str, Any]:
        """Get a summary of hardware capabilities"""
        return self.hw_detector.get_summary(self.capabilities)
        
    def get_software_summary(self) -> Dict[str, Any]:
        """Get a summary of software capabilities"""
        return self.sw_detector.get_summary(self.capabilities)
        
    def get_os_summary(self) -> Dict[str, Any]:
        """Get a summary of OS-specific capabilities"""
        if self.os_detector:
            return self.os_detector.get_summary(self.capabilities)
        return {}
    
    def get_full_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of all capabilities
        
        Returns:
            Dictionary with all capability summaries
        """
        return {
            "hardware": self.get_hardware_summary(),
            "software": self.get_software_summary(),
            "os": self.get_os_summary(),
            "optimal_backend": self.get_optimal_backend().value,
            "detection_time": self.capabilities.get("detection_timestamp", "Unknown")
        }
        
    def json_dump(self, file_path: Optional[str] = None, prettify: bool = True,
                 include_metadata: bool = True) -> Optional[str]:
        """
        Dump capabilities to JSON file or string
        
        Args:
            file_path: Optional path to save JSON file
            prettify: Whether to format the JSON for readability
            include_metadata: Whether to include framework metadata
            
        Returns:
            JSON string if file_path is None, else None
        """
        # Ensure capabilities are detected
        if not self.capabilities:
            self.detect()
            
        # Use the standardized JSONSerializer
        if file_path:
            JSONSerializer.to_file(
                self.capabilities, file_path, prettify, include_metadata
            )
            self.error_handler.info(
                f"Saved capabilities to {file_path}", "orchestrator"
            )
            return None
        else:
            return JSONSerializer.to_json(
                self.capabilities, prettify, include_metadata
            )
    
    def __str__(self) -> str:
        """String representation with key platform information"""
        info = self.capabilities
        
        if not info:
            return "Platform capabilities not yet detected. Call detect() first."
            
        os_info = f"{info.get('os', 'unknown')} {info.get('os_version', '')}"
        cpu_info = f"{info.get('cpu_brand', info.get('processor', 'Unknown'))} ({info.get('cpu_count', 0)} cores)"
        
        gpu_info = "None detected"
        if info.get("has_gpu", False):
            gpu_vendor = info.get("gpu_vendor", "unknown")
            gpu_name = info.get("gpu_name", "unknown")
            gpu_info = f"{gpu_vendor} {gpu_name}"
            
        optimal = self.get_optimal_backend().value
        
        return (f"Platform: {os_info}\n"
                f"CPU: {cpu_info}\n"
                f"GPU: {gpu_info}\n"
                f"Optimal Backend: {optimal}")


# Simple usage example
if __name__ == "__main__":
    import argparse
    import logging
    
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Platform Detection Framework")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--file", type=str, help="Save output to specified file")
    parser.add_argument("--summary", action="store_true", help="Show summary instead of full details")
    parser.add_argument("--quiet", action="store_true", help="Suppress warnings and info messages")
    parser.add_argument("--verbose", action="store_true", help="Show verbose debug output")
    
    args = parser.parse_args()
    
    # Set log level based on arguments
    if args.verbose:
        log_level = logging.DEBUG
    elif args.quiet:
        log_level = logging.ERROR
    else:
        log_level = logging.INFO
    
    # Initialize detector with appropriate settings
    detector = PlatformOrchestrator(log_level=log_level, enable_warnings=not args.quiet)
    
    if args.json:
        # Output as JSON
        if args.file:
            detector.json_dump(args.file)
            print(f"Platform capabilities saved to {args.file}")
        else:
            # Print JSON to stdout
            print(detector.json_dump())
    elif args.summary:
        # Show summary
        summary = detector.get_full_summary()
        
        # Pretty print the summary
        print("=== Platform Detection Summary ===")
        
        # Hardware summary
        hw = summary["hardware"]
        print(f"\nHardware:")
        print(f"  CPU: {hw['cpu']['brand']} ({hw['cpu']['cores']} cores)")
        print(f"  Architecture: {hw['cpu']['architecture']}")
        print(f"  SIMD Support: {', '.join(hw['cpu']['simd_support'])}")
        
        if hw['gpu']['available']:
            print(f"  GPU: {hw['gpu']['vendor']} {hw['gpu']['name']}")
        else:
            print("  GPU: None detected")
        
        # Software summary
        sw = summary["software"]
        print(f"\nSoftware:")
        print(f"  Python: {sw['python']['version']} ({sw['python']['implementation']})")
        
        # Print key packages
        print("  Key packages:")
        for pkg, info in sw["packages"].items():
            if info.get("available", False):
                version = info.get("version", "Unknown version")
                print(f"    - {pkg}: {version}")
        
        # OS summary
        os_summary = summary["os"]
        os_name = detector.os_name.capitalize()
        print(f"\n{os_name}-specific:")
        
        if os_name == "Darwin":
            macos = os_summary["macos"]
            print(f"  macOS: {macos['name']} {macos['version']}")
            if macos["is_apple_silicon"]:
                print(f"  Chip: {macos['apple_chip']}")
                print(f"  Cores: {macos['cores']['performance']} performance, {macos['cores']['efficiency']} efficiency")
            print(f"  Frameworks: {'Metal, ' if os_summary['frameworks']['metal'] else ''}{'Accelerate' if os_summary['frameworks']['accelerate'] else ''}")
            
        elif os_name == "Linux":
            linux = os_summary["distribution"]
            print(f"  Distribution: {linux['name']} {linux['version']}")
            print(f"  Desktop: {os_summary['desktop']['environment']}")
            
            # Print available BLAS implementations
            blas_impls = []
            for blas, available in os_summary["optimizations"].items():
                if available and blas in ["mkl", "openblas", "atlas"]:
                    blas_impls.append(blas.upper())
            if blas_impls:
                print(f"  BLAS: {', '.join(blas_impls)}")
            
        elif os_name == "Windows":
            windows = os_summary["windows"]
            print(f"  Windows: {windows['name']} {windows['edition']}")
            print(f"  Build: {windows['build']}")
            
            if os_summary["graphics"]["directx"]["available"]:
                print(f"  DirectX: {os_summary['graphics']['directx']['version']}")
            
            if os_summary["wsl"]["available"]:
                print(f"  WSL: Version {os_summary['wsl']['version']}")
                if os_summary["wsl"]["distros"]:
                    print(f"  WSL Distros: {', '.join(os_summary['wsl']['distros'])}")
                    
        # Optimal backend
        print(f"\nOptimal computation backend: {summary['optimal_backend']}")
    else:
        # Show default output
        capabilities = detector.capabilities
        print(f"Detected {len(capabilities)} platform capabilities")
        print(f"Optimal backend: {detector.get_optimal_backend().value}")
        print(str(detector))
        
        # Save to file if requested
        if args.file:
            detector.json_dump(args.file)
            print(f"Platform capabilities saved to {args.file}")