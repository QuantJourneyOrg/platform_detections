"""
Platform Detection Framework - Darwin (macOS) Detection Module

This module provides specialized detection for macOS (Darwin) systems, 
including Apple Silicon capabilities, frameworks, and optimizations.
"""

import platform
import subprocess
import os
import sys
import warnings
from typing import Dict, Any, List, Optional

from platform_detections.detectors.base import BaseDetector
from platform_detections.utils import ErrorHandler
from platform_detections.constants import (
    FLAG_APPLE_SILICON, FLAG_ACCELERATE, FLAG_METAL, FLAG_COREML,
    FLAG_NEURAL_ENGINE, FLAG_OPENCL, FLAG_OPENCL
)


class DarwinDetector(BaseDetector):
    """
    Detects macOS-specific hardware and software capabilities, with special 
    focus on Apple Silicon, Metal, and Accelerate framework optimizations.
    """
    
    def __init__(self, error_handler: Optional[ErrorHandler] = None):
        """Initialize the macOS detector"""
        super().__init__(error_handler)
        self.capabilities = {}
        
    def detect(self) -> Dict[str, Any]:
        """
        Perform macOS-specific detection
        
        Returns:
            Dictionary with macOS capabilities
        """
        self._log_info("Starting macOS/Darwin detection", "darwin")
        
        # Basic macOS information
        info = {
            "macos_version": platform.mac_ver()[0],
            "macos_name": self._get_macos_name(platform.mac_ver()[0]),
            "has_apple_silicon": False,
            "has_accelerate": False,
            "has_metal": False,
            "metal_support": False,
            "performance_cores": 0,
            "efficiency_cores": 0,
            "simd_support": [],
        }
        
        # Check if running on Apple Silicon
        self._log_info("Detecting Apple Silicon capabilities", "darwin")
        info.update(self._detect_apple_silicon())
        
        # Check for Apple frameworks
        self._log_info("Detecting Apple frameworks", "darwin")
        info.update(self._detect_apple_frameworks())
        
        # Check for specialized frameworks and libraries
        self._log_info("Detecting specialized libraries", "darwin")
        info.update(self._detect_specialized_libraries())
        
        # Detect compilers
        self._log_info("Detecting compiler information", "darwin")
        info.update(self._detect_compilers())
        
        # Detect memory information
        self._log_info("Detecting memory information", "darwin")
        info.update(self._detect_memory_info())
        
        # Store capabilities
        self.capabilities = info
        
        # Normalize capability flags
        normalized_info = self._normalize_flags(info)
        
        self._log_info("macOS/Darwin detection complete", "darwin")
        return normalized_info
    
    def _detect_apple_silicon(self) -> Dict[str, Any]:
        """
        Detect if running on Apple Silicon and get related info
        
        Returns:
            Dictionary with Apple Silicon information
        """
        info = {
            "has_apple_silicon": False,
            "apple_chip": "unknown",
            "performance_cores": 0,
            "efficiency_cores": 0,
        }
        
        # Check if running on Apple Silicon - more robust method
        try:
            # Check CPU brand string
            output = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"]
            ).decode("utf-8").strip()
            
            info["has_apple_silicon"] = "Apple" in output
            
            if not info["has_apple_silicon"]:
                # Check architecture as fallback
                arch = platform.machine()
                if arch == "arm64":
                    info["has_apple_silicon"] = True
                    
            # If Apple Silicon, identify the chip
            if info["has_apple_silicon"]:
                chip_info = output.lower() if "Apple" in output else "apple silicon"
                info["apple_chip"] = self._identify_apple_chip(chip_info)
                
                # Get core configuration based on identified chip
                perf_cores, eff_cores = self._get_core_configuration(info["apple_chip"], chip_info)
                info["performance_cores"] = perf_cores
                info["efficiency_cores"] = eff_cores
                
                # Apple Silicon has Neon SIMD instructions
                info["simd_support"] = ["neon"]
                info["has_neon"] = True
                
                # Check for Apple Neural Engine (ANE)
                info["has_neural_engine"] = True  # All Apple Silicon Macs have ANE
                
            else:
                # For Intel Macs, detect SIMD instructions
                info["simd_support"] = self._detect_intel_simd()
                
        except Exception as e:
            # Fallback to simple architecture check
            info["has_apple_silicon"] = platform.machine() == "arm64"
            if info["has_apple_silicon"]:
                info["simd_support"] = ["neon"]
                info["has_neon"] = True
            self._log_warning(f"Error detecting Apple Silicon: {str(e)}", "darwin")
            
        return info
    
    def _detect_apple_frameworks(self) -> Dict[str, Any]:
        """
        Detect availability of Apple frameworks
        
        Returns:
            Dictionary with framework information
        """
        info = {
            "has_accelerate": False,
            "has_metal": False,
            "metal_support": False,
            "has_coreml": False,
            "has_vision": False,
            "has_core_image": False,
            "has_core_video": False,
            "has_core_audio": False,
            "has_core_bluetooth": False,
            "has_core_location": False,
            "has_core_media": False,
            "has_core_motion": False,
            "has_av_foundation": False,
        }
        
        # Check for Accelerate framework
        try:
            import ctypes
            try:
                accelerate = ctypes.cdll.LoadLibrary(
                    "/System/Library/Frameworks/Accelerate.framework/Accelerate"
                )
                info["has_accelerate"] = True
            except OSError:
                info["has_accelerate"] = False
        except ImportError:
            info["has_accelerate"] = False
            
        # Enhanced Metal detection that works with or without PyObjC
        # First try PyObjC for advanced Metal detection
        metal_detected = False
        try:
            import objc
            import Foundation
            import Metal
            
            device = Metal.MTLCreateSystemDefaultDevice()
            if device:
                info["has_metal"] = True
                info["metal_support"] = True
                info["metal_device_name"] = device.name()
                metal_detected = True
                
                # Get more Metal device details
                info["metal_device"] = {
                    "name": device.name(),
                    "registry_id": device.registryID(),
                    "low_power": device.isLowPower(),
                    "headless": device.isHeadless(),
                    "removable": device.isRemovable(),
                    "max_threads_per_threadgroup": (
                        device.maxThreadsPerThreadgroup().width,
                        device.maxThreadsPerThreadgroup().height,
                        device.maxThreadsPerThreadgroup().depth
                    ),
                }
        except ImportError:
            # If PyObjC fails, continue to the alternative method
            self._log_debug("PyObjC not available for Metal detection", "darwin")
            
        # If Metal wasn't detected with PyObjC, try system_profiler
        if not metal_detected:
            try:
                result = subprocess.run(
                    ["system_profiler", "SPDisplaysDataType"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode == 0 and "Metal" in result.stdout:
                    info["has_metal"] = True
                    info["metal_support"] = True
                    
                    # Try to extract GPU model
                    for line in result.stdout.split("\n"):
                        if "Chipset Model:" in line:
                            info["metal_device"] = line.split("Chipset Model:")[1].strip()
                            break
            except (FileNotFoundError, subprocess.SubprocessError):
                # If system_profiler fails, just check for the framework
                info["has_metal"] = os.path.exists(
                    "/System/Library/Frameworks/Metal.framework"
                )
                info["metal_support"] = info["has_metal"]
        
        # Check for other Apple frameworks
        frameworks_to_check = [
            ("has_coreml", "CoreML"),
            ("has_vision", "Vision"), 
            ("has_accelerate", "Accelerate"),
            ("has_core_image", "CoreImage"),
            ("has_core_video", "CoreVideo"),
            ("has_core_audio", "CoreAudio"),
            ("has_core_bluetooth", "CoreBluetooth"),
            ("has_core_location", "CoreLocation"),
            ("has_core_media", "CoreMedia"),
            ("has_core_motion", "CoreMotion"),
            ("has_av_foundation", "AVFoundation"),
        ]
        
        for key, framework in frameworks_to_check:
            framework_path = f"/System/Library/Frameworks/{framework}.framework"
            info[key] = os.path.exists(framework_path)
            
        return info
    
    def _detect_specialized_libraries(self) -> Dict[str, Any]:
        """
        Detect macOS-specific libraries and optimizations
        
        Returns:
            Dictionary with specialized library information
        """
        info = {
            "has_veclib": False,
            "has_opencl": False,
            "has_opengl": False,
            "has_grand_central_dispatch": True,  # GCD is always available on modern macOS
        }
        
        # Check for VecLib (part of Accelerate)
        try:
            import ctypes
            try:
                veclib = ctypes.cdll.LoadLibrary(
                    "/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/vecLib"
                )
                info["has_veclib"] = True
            except OSError:
                info["has_veclib"] = False
        except ImportError:
            info["has_veclib"] = False
            
        # Check for OpenCL
        info["has_opencl"] = os.path.exists(
            "/System/Library/Frameworks/OpenCL.framework"
        )
        
        # Check for OpenGL
        info["has_opengl"] = os.path.exists(
            "/System/Library/Frameworks/OpenGL.framework"
        )
        
        # Check for PyObjC
        try:
            import objc
            info["has_pyobjc"] = True
            info["pyobjc_version"] = objc.__version__
        except ImportError:
            info["has_pyobjc"] = False
            
        # Check for Numba
        try:
            import numba
            info["has_numba"] = True
            info["numba_version"] = numba.__version__
            
            # Check Numba threading layer
            info["numba_threading_layer"] = numba.config.THREADING_LAYER
            
        except ImportError:
            info["has_numba"] = False
            
        # Check for Bottleneck
        try:
            import bottleneck
            info["has_bottleneck"] = True
            info["bottleneck_version"] = bottleneck.__version__
        except ImportError:
            info["has_bottleneck"] = False
            
        return info
    
    def _detect_compilers(self) -> Dict[str, Any]:
        """
        Detect macOS compiler information
        
        Returns:
            Dictionary with compiler information
        """
        info = {
            "has_c_compiler": False,
            "c_compiler_type": None,
            "c_compiler_version": None,
        }
        
        # Check for Xcode command line tools / clang
        try:
            # Try to find the C compiler
            cc_path = subprocess.check_output(["which", "cc"]).decode("utf-8").strip()
            if cc_path:
                info["has_c_compiler"] = True
                
                # Get C compiler version
                result = subprocess.run(
                    [cc_path, "--version"], 
                    capture_output=True, 
                    text=True, 
                    check=False
                )
                if result.returncode == 0:
                    output = result.stdout
                    
                    if "Apple clang" in output:
                        info["c_compiler_type"] = "Apple Clang"
                    elif "clang" in output:
                        info["c_compiler_type"] = "Clang"
                    elif "gcc" in output.lower():
                        info["c_compiler_type"] = "GCC"
                        
                    # Extract version
                    first_line = output.split("\n")[0]
                    info["c_compiler_version"] = first_line
                    
                    # Try to extract just the version number
                    import re
                    version_match = re.search(r'version\s+(\d+\.\d+\.\d+)', first_line)
                    if version_match:
                        info["c_compiler_version_number"] = version_match.group(1)
        except (FileNotFoundError, subprocess.SubprocessError):
            pass
            
        # Check for Swift
        try:
            result = subprocess.run(
                ["swift", "--version"], 
                capture_output=True, 
                text=True, 
                check=False
            )
            if result.returncode == 0:
                info["has_swift"] = True
                info["swift_version"] = result.stdout.split("\n")[0]
                
                # Extract version number
                import re
                version_match = re.search(r'version\s+(\d+\.\d+(\.\d+)?)', result.stdout)
                if version_match:
                    info["swift_version_number"] = version_match.group(1)
            else:
                info["has_swift"] = False
        except (FileNotFoundError, subprocess.SubprocessError):
            info["has_swift"] = False
            
        # Check for Cython
        try:
            import Cython
            info["has_cython"] = True
            info["cython_version"] = Cython.__version__
            
            try:
                from Cython.Compiler.Version import version as cython_compiler_version
                info["cython_compiler_version"] = cython_compiler_version
            except ImportError:
                pass
                
        except ImportError:
            info["has_cython"] = False
            
        return info
        
    def _detect_memory_info(self) -> Dict[str, Any]:
        """
        Detect memory information on macOS
        
        Returns:
            Dictionary with memory information
        """
        info = {
            "memory_total": 0,
            "memory_available": 0,
            "memory_used": 0,
            "memory_percent": 0,
        }
        
        # Try to use psutil for cross-platform memory info
        try:
            import psutil
            mem = psutil.virtual_memory()
            info["memory_total"] = mem.total
            info["memory_available"] = mem.available
            info["memory_used"] = mem.used
            info["memory_percent"] = mem.percent
            
            # Get swap memory info
            swap = psutil.swap_memory()
            info["swap_total"] = swap.total
            info["swap_used"] = swap.used
            info["swap_free"] = swap.free
            info["swap_percent"] = swap.percent
            
        except ImportError:
            # macOS-specific fallback for memory detection
            try:
                # Get total physical memory
                mem_total = subprocess.check_output(
                    ["sysctl", "-n", "hw.memsize"]
                ).decode("utf-8").strip()
                info["memory_total"] = int(mem_total)
                
                # Try to get VM stats for more memory info
                vm_stat = subprocess.check_output(
                    ["vm_stat"]
                ).decode("utf-8").strip()
                
                # Parse vm_stat output
                page_size = 4096  # Default page size on macOS
                free_pages = 0
                for line in vm_stat.split("\n"):
                    if "page size of" in line:
                        page_size_str = line.split("page size of")[1].strip().split()[0]
                        try:
                            page_size = int(page_size_str)
                        except ValueError:
                            pass
                    elif "Pages free:" in line:
                        free_str = line.split(":")[1].strip().rstrip(".")
                        try:
                            free_pages = int(free_str)
                        except ValueError:
                            pass
                
                # Calculate available memory
                info["memory_available"] = free_pages * page_size
                
                # If we have both total and available, calculate used and percent
                if info["memory_total"] > 0 and info["memory_available"] > 0:
                    info["memory_used"] = info["memory_total"] - info["memory_available"]
                    info["memory_percent"] = (info["memory_used"] / info["memory_total"]) * 100
            except (subprocess.SubprocessError, ValueError):
                pass
                
        return info
    
    def _get_macos_name(self, version: str) -> str:
        """
        Convert macOS version number to marketing name
        
        Args:
            version: macOS version string (e.g., "10.15.7")
            
        Returns:
            Marketing name of the macOS version
        """
        macos_names = {
            "10.9": "Mavericks",
            "10.10": "Yosemite",
            "10.11": "El Capitan",
            "10.12": "Sierra",
            "10.13": "High Sierra",
            "10.14": "Mojave",
            "10.15": "Catalina",
            "11": "Big Sur",
            "12": "Monterey",
            "13": "Ventura",
            "14": "Sonoma",
            "15": "Sequoia",  # Future version (speculative)
        }
        
        # Extract major version
        major_version = version.split(".")[0]
        
        # For versions 10.x, use first two components
        if major_version == "10":
            major_minor = ".".join(version.split(".")[:2])
            return macos_names.get(major_minor, "Unknown macOS")
        
        # For newer versions (11+), just use major version
        return macos_names.get(major_version, "Unknown macOS")
    
    def _identify_apple_chip(self, chip_info: str) -> str:
        """
        Identify Apple Silicon chip from CPU information
        
        Args:
            chip_info: CPU information string
            
        Returns:
            Identified Apple chip name
        """
        chip_info = chip_info.lower()
        
        # M3 series
        if "m3" in chip_info:
            if "ultra" in chip_info:
                return "M3 Ultra"
            elif "max" in chip_info:
                return "M3 Max"
            elif "pro" in chip_info:
                return "M3 Pro"
            else:
                return "M3"
                
        # M2 series
        elif "m2" in chip_info:
            if "ultra" in chip_info:
                return "M2 Ultra"
            elif "max" in chip_info:
                return "M2 Max"
            elif "pro" in chip_info:
                return "M2 Pro"
            else:
                return "M2"
                
        # M1 series
        elif "m1" in chip_info:
            if "ultra" in chip_info:
                return "M1 Ultra"
            elif "max" in chip_info:
                return "M1 Max"
            elif "pro" in chip_info:
                return "M1 Pro"
            else:
                return "M1"
        
        # Future chips or unknown
        return "Apple Silicon (Unknown)"
    
    def _get_core_configuration(self, chip_name: str, chip_info: str) -> tuple:
        """
        Get performance and efficiency core counts for Apple Silicon chips
        
        Args:
            chip_name: Identified Apple chip name
            chip_info: Original chip information string
            
        Returns:
            Tuple of (performance_cores, efficiency_cores)
        """
        chip_info = chip_info.lower()
        
        # M1 series configurations
        if chip_name == "M1":
            return 4, 4
        elif chip_name == "M1 Pro":
            return 8, 2  # Can be 6P+2E or 8P+2E
        elif chip_name == "M1 Max":
            return 8, 2
        elif chip_name == "M1 Ultra":
            return 16, 4  # Essentially 2x M1 Max
            
        # M2 series configurations
        elif chip_name == "M2":
            return 4, 4
        elif chip_name == "M2 Pro":
            return 10, 4  # Can be 6P+4E or 8P+4E or 10P+4E
        elif chip_name == "M2 Max":
            return 12, 4  # Can be 10P+4E or 12P+4E
        elif chip_name == "M2 Ultra":
            return 24, 8  # Essentially 2x M2 Max
            
        # M3 series configurations
        elif chip_name == "M3":
            return 4, 4
        elif chip_name == "M3 Pro":
            return 6, 4  # Can be 6P+4E or 12P+4E
        elif chip_name == "M3 Max":
            return 14, 6  # Can be 14P+6E or 16P+4E
        elif chip_name == "M3 Ultra":
            return 32, 8  # Essentially 2x M3 Max (or potential config)
        
        # Unknown or future chips - provide conservative defaults
        return 4, 4

    def _detect_intel_simd(self) -> List[str]:
        """
        Detect SIMD instruction sets available on Intel Macs
        
        Returns:
            List of detected SIMD instruction sets
        """
        simd_features = []
        
        try:
            # Use sysctl to get CPU features
            output = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.features"]
            ).decode("utf-8").strip().lower()
            
            # Check for common SIMD instruction sets
            if "sse" in output:
                simd_features.append("sse")
            if "sse2" in output:
                simd_features.append("sse2")
            if "sse3" in output:
                simd_features.append("sse3")
            if "ssse3" in output:
                simd_features.append("ssse3")
            if "sse4.1" in output or "sse4_1" in output:
                simd_features.append("sse4_1")
            if "sse4.2" in output or "sse4_2" in output:
                simd_features.append("sse4_2")
            if "avx" in output and "avx2" not in output:
                simd_features.append("avx")
            if "avx2" in output:
                simd_features.append("avx2")
            if "avx512" in output:
                simd_features.append("avx512")
            
            # Also check for advanced instruction sets
            leaf7_output = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.leaf7_features"]
            ).decode("utf-8").strip().lower()
            
            # Check specific AVX-512 variants
            if "avx512f" in leaf7_output:
                simd_features.append("avx512f")
            if "avx512vl" in leaf7_output:
                simd_features.append("avx512vl")
            if "avx512bw" in leaf7_output:
                simd_features.append("avx512bw")
            if "avx512dq" in leaf7_output:
                simd_features.append("avx512dq")
            
        except (subprocess.SubprocessError, FileNotFoundError):
            # Fallback method - check processor model
            try:
                model = subprocess.check_output(
                    ["sysctl", "-n", "machdep.cpu.brand_string"]
                ).decode("utf-8").strip()
                
                # Simple heuristic based on processor generation
                if "i9" in model or "i7" in model or "i5" in model:
                    # Extract generation if possible
                    import re
                    gen_match = re.search(r'i[7539]-(\d)(\d+)', model)
                    if gen_match:
                        gen = int(gen_match.group(1))
                        if gen >= 10:  # 10th gen or newer
                            simd_features = ["sse", "sse2", "sse3", "ssse3", "sse4_1", "sse4_2", "avx", "avx2"]
                        elif gen >= 6:  # 6th-9th gen
                            simd_features = ["sse", "sse2", "sse3", "ssse3", "sse4_1", "sse4_2", "avx", "avx2"]
                        elif gen >= 3:  # 3rd-5th gen
                            simd_features = ["sse", "sse2", "sse3", "ssse3", "sse4_1", "sse4_2", "avx"]
                        else:  # Older
                            simd_features = ["sse", "sse2", "sse3", "ssse3"]
                    else:
                        # Default to common instruction sets for modern Intel Macs
                        simd_features = ["sse", "sse2", "sse3", "ssse3", "sse4_1", "sse4_2"]
                elif "Xeon" in model:
                    # Most Xeon processors in Macs support at least these
                    simd_features = ["sse", "sse2", "sse3", "ssse3", "sse4_1", "sse4_2", "avx"]
            except (subprocess.SubprocessError, FileNotFoundError):
                # Very basic fallback - assume at least SSE2 is available on all Intel Macs
                simd_features = ["sse", "sse2"]
        
        return simd_features

    def get_summary(self, capabilities=None) -> Dict[str, Any]:
        """
        Get a summary of macOS-specific capabilities
        
        Args:
            capabilities: Optional pre-detected capabilities
            
        Returns:
            Dictionary with macOS capability summary
        """
        if capabilities is None:
            capabilities = self.detect() if not self.capabilities else self.capabilities
        
        summary = {
            "macos": {
                "name": capabilities.get("macos_name", "Unknown"),
                "version": capabilities.get("macos_version", "Unknown"),
                "is_apple_silicon": capabilities.get("has_apple_silicon", False),
                "apple_chip": capabilities.get("apple_chip", "Unknown"),
            },
            "cores": {
                "performance": capabilities.get("performance_cores", 0),
                "efficiency": capabilities.get("efficiency_cores", 0),
            },
            "frameworks": {
                "metal": capabilities.get("has_metal", False),
                "accelerate": capabilities.get("has_accelerate", False),
                "coreml": capabilities.get("has_coreml", False),
                "opencl": capabilities.get("has_opencl", False),
                "opengl": capabilities.get("has_opengl", False),
            },
            "optimizations": {
                "neural_engine": capabilities.get("has_neural_engine", False),
                "grand_central_dispatch": capabilities.get("has_grand_central_dispatch", True),
                "veclib": capabilities.get("has_veclib", False),
            },
            "development": {
                "c_compiler": {
                    "available": capabilities.get("has_c_compiler", False),
                    "type": capabilities.get("c_compiler_type", "Unknown"),
                    "version": capabilities.get("c_compiler_version_number", "Unknown"),
                },
                "swift": {
                    "available": capabilities.get("has_swift", False),
                    "version": capabilities.get("swift_version_number", "Unknown"),
                },
                "pyobjc": {
                    "available": capabilities.get("has_pyobjc", False),
                    "version": capabilities.get("pyobjc_version", "Unknown"),
                },
            },
            "simd_support": capabilities.get("simd_support", []),
        }
        
        return summary