"""
Platform Detection Framework - Hardware Detection Module

This module provides hardware detection functions for identifying
CPU, GPU, memory, and other hardware capabilities.

Part of QuantJourney, a project for high-performance investment research and
quantitative finance, build on AI infrastructure for Investors and Quantitative
Researchers available at https://quantjourney.substack.com/

This code is licensed under the MIT License.
Copyright (c) 2023 Jakub Polec

"""

import platform
import multiprocessing
import subprocess
import os
import sys
import warnings
from typing import Dict, Any, List, Optional

from .base import BaseDetector
from ..constants import (
    FLAG_GPU, FLAG_CPU_CORES, FLAG_CPU_BRAND, FLAG_MEMORY_TOTAL,
    FLAG_MEMORY_AVAILABLE, FLAG_AVX, FLAG_AVX2, FLAG_NEON
)


class HardwareDetector(BaseDetector):
    """
    Detects hardware capabilities such as CPU features, GPU availability,
    memory capacity, and other hardware-specific information.
    """
    
    def __init__(self, error_handler: Optional[ErrorHandler] = None):
        """Initialize the hardware detector"""
        super().__init__(error_handler)
        self.cpu_info = {}
        self.gpu_info = {}
        self.memory_info = {}
        
    def detect(self) -> Dict[str, Any]:
        """
        Perform comprehensive hardware detection
        
        Returns:
            Dictionary with hardware capabilities
        """
        self._log_info("Starting hardware detection", "hardware")
        
        # Initialize with basic info
        info = {
            "cpu_count": multiprocessing.cpu_count(),
            "arch": platform.machine(),
            "processor": platform.processor(),
            "has_avx": False,
            "has_avx2": False,
            "has_avx512f": False,
            "has_gpu": False,
            "gpu_vendor": None,
            "gpu_memory": None,
        }
        
        # Detect CPU features
        self._log_info("Detecting CPU features", "hardware")
        cpu_features = self._detect_cpu_features()
        info.update(cpu_features)
        self.cpu_info = cpu_features
        
        # Detect GPU capabilities
        self._log_info("Detecting GPU capabilities", "hardware")
        gpu_features = self._detect_gpu_capabilities()
        info.update(gpu_features)
        self.gpu_info = gpu_features
        
        # Detect memory info
        self._log_info("Detecting memory information", "hardware")
        memory_info = self._detect_memory_info()
        info.update(memory_info)
        self.memory_info = memory_info
        
        # Normalize capability flags
        normalized_info = self._normalize_flags(info)
        
        self._log_info("Hardware detection complete", "hardware")
        return normalized_info
    
    def _detect_cpu_features(self) -> Dict[str, Any]:
        """Detect CPU features and capabilities"""
        info = {
            "cpu_brand": "Unknown CPU",
            "has_sse": False,
            "has_sse2": False,
            "has_sse3": False,
            "has_ssse3": False,
            "has_sse4_1": False,
            "has_sse4_2": False,
            "has_avx": False,
            "has_avx2": False,
            "has_avx512f": False,
            "has_neon": False,
            "simd_support": [],
            "cpu_flags": [],
        }
        
        # Try to use py-cpuinfo
        try:
            import cpuinfo
            cpu_data = cpuinfo.get_cpu_info()
            
            # SIMD instruction sets
            if "flags" in cpu_data:
                info["has_sse"] = "sse" in cpu_data["flags"]
                info["has_sse2"] = "sse2" in cpu_data["flags"]
                info["has_sse3"] = "sse3" in cpu_data["flags"]
                info["has_ssse3"] = "ssse3" in cpu_data["flags"]
                info["has_sse4_1"] = "sse4_1" in cpu_data["flags"]
                info["has_sse4_2"] = "sse4_2" in cpu_data["flags"]
                info["has_avx"] = "avx" in cpu_data["flags"]
                info["has_avx2"] = "avx2" in cpu_data["flags"]
                info["has_avx512f"] = "avx512f" in cpu_data["flags"]
                info["has_neon"] = "neon" in cpu_data["flags"]
                
                # Create SIMD support list
                for feature in ["sse", "sse2", "sse3", "ssse3", "sse4_1", "sse4_2", 
                              "avx", "avx2", "avx512f", "neon"]:
                    if feature in cpu_data["flags"]:
                        info["simd_support"].append(feature)
                
                # Store all flags
                info["cpu_flags"] = list(cpu_data["flags"])
            
            # CPU details
            info["cpu_brand"] = cpu_data.get("brand_raw", "Unknown CPU")
            info["cpu_hz"] = cpu_data.get("hz_actual", (0, ""))
            info["cpu_arch"] = cpu_data.get("arch", "unknown")
            info["cpu_bits"] = cpu_data.get("bits", 0)
            info["cpu_vendor"] = cpu_data.get("vendor_id_raw", "unknown")
            info["cpu_family"] = cpu_data.get("family", 0)
            info["cpu_model"] = cpu_data.get("model", 0)
            info["cpu_stepping"] = cpu_data.get("stepping", 0)
            
        except ImportError:
            self._log_warning("py-cpuinfo not available, falling back to manual detection", "hardware")
            # If py-cpuinfo is not available, attempt manual detection
            os_name = platform.system().lower()
            
            if os_name == "darwin":
                self._detect_darwin_cpu_features(info)
            elif os_name == "linux":
                self._detect_linux_cpu_features(info)
            elif os_name == "windows":
                self._detect_windows_cpu_features(info)
                
        return info
    
    def _detect_darwin_cpu_features(self, info: Dict[str, Any]) -> None:
        """Detect CPU features on macOS/Darwin"""
        try:
            # Check for AVX on macOS
            sysctl_output = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.features"]
            ).decode("utf-8").strip()
            features = sysctl_output.lower().split()
            
            # Check for various SIMD instruction sets
            info["has_sse"] = "sse" in features
            info["has_sse2"] = "sse2" in features
            info["has_sse3"] = "sse3" in features
            info["has_ssse3"] = "ssse3" in features
            info["has_sse4_1"] = "sse4.1" in features or "sse41" in features
            info["has_sse4_2"] = "sse4.2" in features or "sse42" in features
            info["has_avx"] = "avx" in features
            info["has_avx2"] = "avx2" in features
            info["has_avx512f"] = "avx512f" in features
            
            # Create SIMD support list
            for feature in ["sse", "sse2", "sse3", "ssse3"]:
                if feature in features:
                    info["simd_support"].append(feature)
            
            # Special handling for features with periods
            if "sse4.1" in features or "sse41" in features:
                info["simd_support"].append("sse4_1")
            if "sse4.2" in features or "sse42" in features:
                info["simd_support"].append("sse4_2")
                
            for feature in ["avx", "avx2", "avx512f"]:
                if feature in features:
                    info["simd_support"].append(feature)
            
            # Store all flags
            info["cpu_flags"] = features
            
            # Get CPU brand string
            brand = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"]
            ).decode("utf-8").strip()
            info["cpu_brand"] = brand
            
            # Get additional CPU info
            info["cpu_vendor"] = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.vendor"]
            ).decode("utf-8").strip()
            
            # Try to get CPU model information
            try:
                info["cpu_family"] = int(subprocess.check_output(
                    ["sysctl", "-n", "machdep.cpu.family"]
                ).decode("utf-8").strip())
                
                info["cpu_model"] = int(subprocess.check_output(
                    ["sysctl", "-n", "machdep.cpu.model"]
                ).decode("utf-8").strip())
                
                info["cpu_stepping"] = int(subprocess.check_output(
                    ["sysctl", "-n", "machdep.cpu.stepping"]
                ).decode("utf-8").strip())
            except (subprocess.SubprocessError, ValueError):
                pass
            
            # Check for ARM/Apple Silicon
            if platform.machine() == "arm64":
                info["has_neon"] = True
                info["simd_support"].append("neon")
            
        except Exception as e:
            self._log_warning(f"Could not detect CPU features on macOS: {str(e)}", "hardware")
            
    def _detect_linux_cpu_features(self, info: Dict[str, Any]) -> None:
        """Detect CPU features on Linux"""
        try:
            # Check CPU info on Linux
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo_content = f.read()
                
            # Get CPU model name
            for line in cpuinfo_content.split("\n"):
                if "model name" in line:
                    info["cpu_brand"] = line.split(":")[1].strip()
                    break
                    
            # Get CPU flags
            for line in cpuinfo_content.split("\n"):
                if "flags" in line or "Features" in line:
                    flags = line.split(":")[1].strip().split()
                    info["cpu_flags"] = flags
                    
                    # Check for SIMD instructions
                    info["has_sse"] = "sse" in flags
                    info["has_sse2"] = "sse2" in flags
                    info["has_sse3"] = "sse3" in flags
                    info["has_ssse3"] = "ssse3" in flags
                    info["has_sse4_1"] = "sse4_1" in flags
                    info["has_sse4_2"] = "sse4_2" in flags
                    info["has_avx"] = "avx" in flags
                    info["has_avx2"] = "avx2" in flags
                    info["has_avx512f"] = "avx512f" in flags
                    info["has_neon"] = "neon" in flags
                    
                    # Build SIMD support list
                    for feature in ["sse", "sse2", "sse3", "ssse3", "sse4_1", "sse4_2", 
                                  "avx", "avx2", "avx512f", "neon"]:
                        if feature in flags:
                            info["simd_support"].append(feature)
                    
                    break
                    
        except Exception as e:
            self._log_warning(f"Could not read /proc/cpuinfo: {str(e)}", "hardware")
            
    def _detect_windows_cpu_features(self, info: Dict[str, Any]) -> None:
        """Detect CPU features on Windows"""
        # On Windows, we have limited options without additional libraries
        info["cpu_brand"] = platform.processor()
        
        # Try to use wmic to get more CPU info
        try:
            cpu_id = subprocess.check_output(
                ["wmic", "cpu", "get", "ProcessorId"], 
                universal_newlines=True
            )
            if "ProcessorId" in cpu_id:
                processor_id = cpu_id.strip().split("\n")[1].strip()
                info["cpu_processor_id"] = processor_id
        except Exception:
            pass
            
    def _detect_gpu_capabilities(self) -> Dict[str, Any]:
        """Detect GPU capabilities"""
        info = {
            "has_gpu": False,
            "gpu_vendor": None,
            "gpu_name": None,
            "gpu_memory": None,
            "cuda_available": False,
            "opencl_available": False,
            "metal_available": False,
            "directx_available": False,
            "vulkan_available": False,
        }
        
        # Check for CUDA GPUs with CuPy
        try:
            import cupy
            info["has_gpu"] = True
            info["gpu_vendor"] = "nvidia"
            info["cuda_available"] = True
            
            # Get device information
            device = cupy.cuda.Device(0)
            info["gpu_memory"] = device.mem_info[1]  # Total memory
            info["gpu_name"] = device.name
            
            # More CUDA details
            info["cuda_version"] = cupy.cuda.runtime.getVersion()
            info["cuda_device_count"] = cupy.cuda.runtime.getDeviceCount()
            
            # Get all CUDA devices
            devices = []
            for i in range(info["cuda_device_count"]):
                device = cupy.cuda.Device(i)
                devices.append({
                    "id": i,
                    "name": device.name,
                    "memory": device.mem_info[1],
                    "compute_capability": device.compute_capability,
                    "multi_processor_count": device.attributes["MultiProcessorCount"],
                })
            info["cuda_devices"] = devices
            
        except ImportError:
            # No CuPy, try alternative GPU detection methods
            self._log_debug("CuPy not available, trying alternative GPU detection", "hardware")
            self._try_alternative_gpu_detection(info)
            
        return info
    
    def _try_alternative_gpu_detection(self, info: Dict[str, Any]) -> None:
        """Try alternative GPU detection methods"""
        # Try to use pycuda
        try:
            import pycuda.driver as drv
            drv.init()
            info["has_gpu"] = True
            info["gpu_vendor"] = "nvidia"
            info["cuda_available"] = True
            info["cuda_device_count"] = drv.Device.count()
            
            if info["cuda_device_count"] > 0:
                device = drv.Device(0)
                info["gpu_name"] = device.name()
                info["gpu_memory"] = device.total_memory()
                
                # Get all CUDA devices
                devices = []
                for i in range(info["cuda_device_count"]):
                    device = drv.Device(i)
                    devices.append({
                        "id": i,
                        "name": device.name(),
                        "memory": device.total_memory(),
                        "compute_capability": (
                            device.get_attribute(drv.device_attribute.COMPUTE_CAPABILITY_MAJOR),
                            device.get_attribute(drv.device_attribute.COMPUTE_CAPABILITY_MINOR)
                        ),
                        "multi_processor_count": device.get_attribute(
                            drv.device_attribute.MULTIPROCESSOR_COUNT),
                    })
                info["cuda_devices"] = devices
                
        except (ImportError, Exception):
            # Try PyTorch
            self._try_pytorch_detection(info)
    
    def _try_pytorch_detection(self, info: Dict[str, Any]) -> None:
        """Try GPU detection using PyTorch"""
        try:
            import torch
            if torch.cuda.is_available():
                info["has_gpu"] = True
                info["gpu_vendor"] = "nvidia"
                info["cuda_available"] = True
                info["cuda_device_count"] = torch.cuda.device_count()
                
                if torch.cuda.device_count() > 0:
                    info["gpu_name"] = torch.cuda.get_device_name(0)
                    # PyTorch doesn't provide easy memory info, but we can mark as available
                    
                    # Check for MPS (Metal Performance Shaders) on Mac
                    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                        info["metal_available"] = True
            else:
                # Try OpenCL detection if CUDA not available
                self._try_opencl_detection(info)
        except ImportError:
            # Try OpenCL detection if PyTorch not available
            self._try_opencl_detection(info)
    
    def _try_opencl_detection(self, info: Dict[str, Any]) -> None:
        """Try GPU detection using OpenCL"""
        try:
            import pyopencl as cl
            platforms = cl.get_platforms()
            if platforms:
                info["opencl_available"] = True
                info["opencl_platforms"] = []
                
                for platform_idx, platform in enumerate(platforms):
                    platform_info = {
                        "name": platform.name,
                        "vendor": platform.vendor,
                        "version": platform.version,
                        "devices": []
                    }
                    
                    # Get GPU devices
                    gpu_devices = platform.get_devices(device_type=cl.device_type.GPU)
                    if gpu_devices:
                        info["has_gpu"] = True
                        
                        # Use the first GPU as the default one
                        if info["gpu_vendor"] is None:
                            device = gpu_devices[0]
                            vendor_lower = device.vendor.lower()
                            if "nvidia" in vendor_lower:
                                info["gpu_vendor"] = "nvidia"
                            elif "amd" in vendor_lower or "advanced micro devices" in vendor_lower:
                                info["gpu_vendor"] = "amd"
                            elif "intel" in vendor_lower:
                                info["gpu_vendor"] = "intel"
                            elif "apple" in vendor_lower:
                                info["gpu_vendor"] = "apple"
                            else:
                                info["gpu_vendor"] = "other"
                                
                            info["gpu_memory"] = device.global_mem_size
                            info["gpu_name"] = device.name
                            info["opencl_device_vendor"] = device.vendor
                            info["opencl_version"] = device.version
                        
                        # Get all devices in this platform
                        for device in gpu_devices:
                            platform_info["devices"].append({
                                "name": device.name,
                                "vendor": device.vendor,
                                "version": device.version,
                                "type": "GPU",
                                "memory": device.global_mem_size,
                                "compute_units": device.max_compute_units,
                                "max_work_group_size": device.max_work_group_size,
                                "max_work_item_dimensions": device.max_work_item_dimensions,
                                "extensions": device.extensions.split(),
                            })
                    
                    # Get CPU OpenCL devices
                    cpu_devices = platform.get_devices(device_type=cl.device_type.CPU)
                    for device in cpu_devices:
                        platform_info["devices"].append({
                            "name": device.name,
                            "vendor": device.vendor,
                            "version": device.version,
                            "type": "CPU",
                            "memory": device.global_mem_size,
                            "compute_units": device.max_compute_units,
                            "max_work_group_size": device.max_work_group_size,
                            "max_work_item_dimensions": device.max_work_item_dimensions,
                        })
                        
                    info["opencl_platforms"].append(platform_info)
        except ImportError:
            # Try OS-specific GPU detection
            self._try_os_specific_gpu_detection(info)
    
    def _try_os_specific_gpu_detection(self, info: Dict[str, Any]) -> None:
        """Try OS-specific GPU detection methods"""
        os_name = platform.system().lower()
        
        if os_name == "darwin":
            self._try_darwin_gpu_detection(info)
        elif os_name == "linux":
            self._try_linux_gpu_detection(info)
        elif os_name == "windows":
            self._try_windows_gpu_detection(info)
    
    def _try_darwin_gpu_detection(self, info: Dict[str, Any]) -> None:
        """Try GPU detection on macOS/Darwin"""
        # Try using system_profiler to detect GPU
        try:
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                output = result.stdout
                if "Metal" in output:
                    info["metal_available"] = True
                    info["has_gpu"] = True
                    info["gpu_vendor"] = "apple"
                    
                    # Try to extract GPU model
                    for line in output.split("\n"):
                        if "Chipset Model:" in line:
                            info["gpu_name"] = line.split("Chipset Model:")[1].strip()
                            break
        except Exception as e:
            self._log_warning(f"Error detecting GPU on macOS: {str(e)}", "hardware")
    
    def _try_linux_gpu_detection(self, info: Dict[str, Any]) -> None:
        """Try GPU detection on Linux"""
        # Try to check for NVIDIA GPUs with nvidia-smi
        try:
            nvidia_smi = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=False
            )
            
            if nvidia_smi.returncode == 0 and nvidia_smi.stdout.strip():
                info["has_gpu"] = True
                info["gpu_vendor"] = "nvidia"
                
                # Parse the first GPU info
                first_gpu = nvidia_smi.stdout.strip().split("\n")[0].split(",")
                if len(first_gpu) >= 3:
                    info["gpu_name"] = first_gpu[0].strip()
                    
                    # Extract memory (usually in MiB format)
                    mem_str = first_gpu[1].strip()
                    if "MiB" in mem_str:
                        try:
                            mem_mb = float(mem_str.split()[0])
                            info["gpu_memory"] = mem_mb * 1024 * 1024  # Convert to bytes
                        except (ValueError, IndexError):
                            pass
                            
                    info["nvidia_driver_version"] = first_gpu[2].strip()
        except Exception as e:
            # If nvidia-smi fails, try lspci
            self._try_linux_lspci_detection(info)
    
    def _try_linux_lspci_detection(self, info: Dict[str, Any]) -> None:
        """Try GPU detection using lspci on Linux"""
        try:
            lspci_output = subprocess.run(
                ["lspci", "-v"],
                capture_output=True,
                text=True,
                check=False
            )
            
            if lspci_output.returncode == 0:
                lspci_text = lspci_output.stdout.lower()
                
                # Check for common GPU vendors
                if "nvidia" in lspci_text and "vga" in lspci_text:
                    info["has_gpu"] = True
                    info["gpu_vendor"] = "nvidia"
                    # Extract GPU name if possible
                    for line in lspci_output.stdout.split("\n"):
                        if "nvidia" in line.lower() and "vga" in line.lower():
                            info["gpu_name"] = line.split(":", 2)[-1].strip()
                            break
                elif "amd" in lspci_text and "vga" in lspci_text:
                    info["has_gpu"] = True
                    info["gpu_vendor"] = "amd"
                    for line in lspci_output.stdout.split("\n"):
                        if "amd" in line.lower() and "vga" in line.lower():
                            info["gpu_name"] = line.split(":", 2)[-1].strip()
                            break
                elif "intel" in lspci_text and "vga" in lspci_text:
                    info["has_gpu"] = True
                    info["gpu_vendor"] = "intel"
                    for line in lspci_output.stdout.split("\n"):
                        if "intel" in line.lower() and "vga" in line.lower():
                            info["gpu_name"] = line.split(":", 2)[-1].strip()
                            break
        except Exception as e:
            self._log_warning(f"Error detecting GPU with lspci: {str(e)}", "hardware")

    def _try_windows_gpu_detection(self, info: Dict[str, Any]) -> None:
        """Try GPU detection on Windows"""
        try:
            # Use wmic to detect GPU
            wmic_output = subprocess.run(
                ["wmic", "path", "win32_videocontroller", "get", "name"],
                capture_output=True,
                text=True,
                check=False
            )
            
            if wmic_output.returncode == 0:
                # Skip the header line
                lines = wmic_output.stdout.strip().split("\n")[1:]
                for line in lines:
                    name = line.strip()
                    if name:
                        info["has_gpu"] = True
                        name_lower = name.lower()
                        if "nvidia" in name_lower:
                            info["gpu_vendor"] = "nvidia"
                        elif "amd" in name_lower or "ati" in name_lower:
                            info["gpu_vendor"] = "amd"
                        elif "intel" in name_lower:
                            info["gpu_vendor"] = "intel"
                        else:
                            info["gpu_vendor"] = "other"
                        info["gpu_name"] = name
                        break
                        
            # Check for DirectX
            try:
                dxdiag_output = subprocess.run(
                    ["dxdiag", "/t", "dxdiag.txt"],
                    capture_output=True,
                    text=True,
                    check=False
                )
                if dxdiag_output.returncode == 0:
                    with open("dxdiag.txt", "r") as f:
                        dxdiag_text = f.read().lower()
                        if "directx" in dxdiag_text:
                            info["directx_available"] = True
                    os.remove("dxdiag.txt")
            except Exception:
                pass
                
        except Exception as e:
            self._log_warning(f"Error detecting GPU on Windows: {str(e)}", "hardware")

    def _detect_memory_info(self) -> Dict[str, Any]:
        """Detect system memory information"""
        info = {
            "memory_total": None,
            "memory_available": None,
            "memory_used": None,
            "memory_percent": None,
        }
        
        try:
            import psutil
            mem = psutil.virtual_memory()
            info["memory_total"] = mem.total
            info["memory_available"] = mem.available
            info["memory_used"] = mem.used
            info["memory_percent"] = mem.percent
            
            # Get swap memory if available
            try:
                swap = psutil.swap_memory()
                info["swap_total"] = swap.total
                info["swap_used"] = swap.used
                info["swap_free"] = swap.free
                info["swap_percent"] = swap.percent
            except Exception:
                pass
                
        except ImportError:
            self._log_warning("psutil not available, attempting OS-specific memory detection", "hardware")
            os_name = platform.system().lower()
            
            if os_name == "darwin":
                self._detect_darwin_memory(info)
            elif os_name == "linux":
                self._detect_linux_memory(info)
            elif os_name == "windows":
                self._detect_windows_memory(info)
                
        return info

    def _detect_darwin_memory(self, info: Dict[str, Any]) -> None:
        """Detect memory information on macOS"""
        try:
            # Get total memory
            vm_stat = subprocess.check_output(["vm_stat"]).decode("utf-8")
            pages_free = 0
            pages_active = 0
            pages_inactive = 0
            pages_wired = 0
            page_size = 4096  # Default page size
            
            for line in vm_stat.split("\n"):
                if "page size of" in line:
                    try:
                        page_size = int(line.split()[-2])
                    except (ValueError, IndexError):
                        pass
                elif "Pages free" in line:
                    pages_free = int(line.split(":")[1].strip().rstrip("."))
                elif "Pages active" in line:
                    pages_active = int(line.split(":")[1].strip().rstrip("."))
                elif "Pages inactive" in line:
                    pages_inactive = int(line.split(":")[1].strip().rstrip("."))
                elif "Pages wired down" in line:
                    pages_wired = int(line.split(":")[1].strip().rstrip("."))
                    
            # Get total memory
            hw_memsize = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"]
            ).decode("utf-8").strip()
            
            info["memory_total"] = int(hw_memsize)
            info["memory_used"] = (pages_active + pages_wired) * page_size
            info["memory_available"] = (pages_free + pages_inactive) * page_size
            if info["memory_total"] > 0:
                info["memory_percent"] = (info["memory_used"] / info["memory_total"]) * 100
                
        except Exception as e:
            self._log_warning(f"Error detecting memory on macOS: {str(e)}", "hardware")

    def _detect_linux_memory(self, info: Dict[str, Any]) -> None:
        """Detect memory information on Linux"""
        try:
            with open("/proc/meminfo", "r") as f:
                meminfo = f.read()
                
            for line in meminfo.split("\n"):
                if "MemTotal" in line:
                    info["memory_total"] = int(line.split()[1]) * 1024  # Convert kB to bytes
                elif "MemFree" in line:
                    mem_free = int(line.split()[1]) * 1024
                elif "MemAvailable" in line:
                    info["memory_available"] = int(line.split()[1]) * 1024
                elif "SwapTotal" in line:
                    info["swap_total"] = int(line.split()[1]) * 1024
                elif "SwapFree" in line:
                    info["swap_free"] = int(line.split()[1]) * 1024
                    
            if info.get("memory_total") and info.get("memory_available"):
                info["memory_used"] = info["memory_total"] - info["memory_available"]
                info["memory_percent"] = (info["memory_used"] / info["memory_total"]) * 100
                
            if info.get("swap_total") and info.get("swap_free"):
                info["swap_used"] = info["swap_total"] - info["swap_free"]
                info["swap_percent"] = (
                    (info["swap_used"] / info["swap_total"]) * 100 if info["swap_total"] > 0 else 0
                )
                
        except Exception as e:
            self._log_warning(f"Error reading /proc/meminfo: {str(e)}", "hardware")

    def _detect_windows_memory(self, info: Dict[str, Any]) -> None:
        """Detect memory information on Windows"""
        try:
            # Use wmic to get memory info
            mem_output = subprocess.check_output(
                ["wmic", "OS", "get", "TotalVisibleMemorySize,FreePhysicalMemory"],
                universal_newlines=True
            )
            
            lines = mem_output.strip().split("\n")
            if len(lines) > 1:
                values = lines[1].strip().split()
                if len(values) >= 2:
                    info["memory_total"] = int(values[1]) * 1024  # Convert KB to bytes
                    info["memory_available"] = int(values[0]) * 1024
                    info["memory_used"] = info["memory_total"] - info["memory_available"]
                    info["memory_percent"] = (info["memory_used"] / info["memory_total"]) * 100
                    
            # Get swap memory
            swap_output = subprocess.check_output(
                ["wmic", "OS", "get", "TotalVirtualMemorySize,FreeVirtualMemory"],
                universal_newlines=True
            )
            
            lines = swap_output.strip().split("\n")
            if len(lines) > 1:
                values = lines[1].strip().split()
                if len(values) >= 2:
                    info["swap_total"] = int(values[1]) * 1024
                    info["swap_free"] = int(values[0]) * 1024
                    info["swap_used"] = info["swap_total"] - info["swap_free"]
                    info["swap_percent"] = (
                        (info["swap_used"] / info["swap_total"]) * 100 if info["swap_total"] > 0 else 0
                    )
                    
        except Exception as e:
            self._log_warning(f"Error detecting memory on Windows: {str(e)}", "hardware")

    def get_summary(self, capabilities: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get a summary of hardware capabilities"""
        if capabilities is None:
            capabilities = self.capabilities if self.capabilities else self.detect()
            
        summary = {
            "cpu": {
                "cores": capabilities.get("cpu_count", 0),
                "brand": capabilities.get("cpu_brand", "Unknown"),
                "architecture": capabilities.get("cpu_arch", capabilities.get("arch", "unknown")),
                "simd_support": capabilities.get("simd_support", []),
                "vendor": capabilities.get("cpu_vendor", "unknown"),
            },
            "gpu": {
                "available": capabilities.get("has_gpu", False),
                "vendor": capabilities.get("gpu_vendor", None),
                "name": capabilities.get("gpu_name", None),
                "memory": capabilities.get("gpu_memory", None),
            },
            "memory": {
                "total": capabilities.get("memory_total", None),
                "available": capabilities.get("memory_available", None),
                "used": capabilities.get("memory_used", None),
                "percent": capabilities.get("memory_percent", None),
            }
        }
        
        # Add accelerator support
        accelerators = []
        for accel in ["cuda", "opencl", "metal", "directx", "vulkan"]:
            if capabilities.get(f"{accel}_available", False):
                accelerators.append(accel)
        summary["accelerators"] = accelerators
        
        return summary