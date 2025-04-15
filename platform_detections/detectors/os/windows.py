"""
Platform Detection Framework - Windows Detection Module

This module provides specialized detection for Windows systems,
including version details, available libraries, and optimizations.
"""

import platform
import subprocess
import os
import sys
import ctypes
import warnings
from typing import Dict, Any, List, Optional

from ..base import BaseDetector
from ...utils import ErrorHandler
from ...constants import (
    FLAG_MSVC, FLAG_MINGW, FLAG_VISUAL_STUDIO, FLAG_WINSDK,
    FLAG_WSL, FLAG_DIRECTX, FLAG_VULKAN, FLAG_OPENCL
)


class WindowsDetector(BaseDetector):
    """
    Detects Windows-specific hardware and software capabilities, with special 
    focus on Windows version, DirectX, and available optimizations.
    """
    
    def __init__(self, error_handler: Optional[ErrorHandler] = None):
        """Initialize the Windows detector"""
        super().__init__(error_handler)
        self.capabilities = {}
        
    def detect(self) -> Dict[str, Any]:
        """
        Perform Windows-specific detection
        
        Returns:
            Dictionary with Windows capabilities
        """
        self._log_info("Starting Windows detection", "windows")
        
        # Basic Windows information
        info = {
            "windows_version": platform.version(),
            "windows_release": platform.release(),
            "windows_edition": "",
            "has_directx": False,
            "directx_version": "",
            "has_wsl": False,
            "wsl_version": "",
            "gpu_driver_version": "",
        }
        
        # Detect Windows version and edition
        self._log_info("Detecting Windows version and edition", "windows")
        info.update(self._detect_windows_version())
        
        # Detect DirectX and other graphics capabilities
        self._log_info("Detecting graphics capabilities", "windows")
        info.update(self._detect_graphics_capabilities())
        
        # Detect C libraries and compilers
        self._log_info("Detecting libraries and compilers", "windows")
        info.update(self._detect_libraries_and_compilers())
        
        # Detect WSL (Windows Subsystem for Linux)
        self._log_info("Detecting WSL (Windows Subsystem for Linux)", "windows")
        info.update(self._detect_wsl())
        
        # Detect hardware-specific optimizations
        self._log_info("Detecting hardware optimizations", "windows")
        info.update(self._detect_hardware_optimizations())
        
        # Normalize capability flags
        normalized_info = self._normalize_flags(info)
        
        # Store capabilities
        self.capabilities = normalized_info
        
        self._log_info("Windows detection complete", "windows")
        return normalized_info
    
    def _detect_windows_version(self) -> Dict[str, Any]:
        """
        Detect Windows version and edition information
        
        Returns:
            Dictionary with Windows version information
        """
        info = {
            "windows_edition": "",
            "windows_name": "",
            "windows_build": "",
            "is_server": False,
            "is_64bit": platform.machine().endswith('64'),
        }
        
        # Get Windows name mapping
        windows_names = {
            "10.0": "Windows 10/11",  # Need additional check for Windows 11
            "6.3": "Windows 8.1",
            "6.2": "Windows 8",
            "6.1": "Windows 7",
            "6.0": "Windows Vista",
            "5.2": "Windows XP 64-bit/Server 2003",
            "5.1": "Windows XP",
        }
        
        # Get major.minor version
        win_version = platform.version()
        
        # Try to extract major.minor
        version_parts = win_version.split('.')
        if len(version_parts) >= 2:
            major_minor = f"{version_parts[0]}.{version_parts[1]}"
            info["windows_name"] = windows_names.get(major_minor, f"Windows (version {win_version})")
            
            # Check if it's Windows 11 (Windows 10 version but build >= 22000)
            if major_minor == "10.0" and len(version_parts) >= 3:
                try:
                    build = int(version_parts[2])
                    if build >= 22000:
                        info["windows_name"] = "Windows 11"
                    info["windows_build"] = build
                except ValueError:
                    pass
        else:
            info["windows_name"] = f"Windows (version {win_version})"
            
        # Try to get edition information
        try:
            import winreg
            reg_path = r"SOFTWARE\Microsoft\Windows NT\CurrentVersion"
            reg_key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, reg_path)
            
            # Try to get ProductName
            try:
                product_name, _ = winreg.QueryValueEx(reg_key, "ProductName")
                info["windows_edition"] = product_name
                
                # Check if it's server edition
                info["is_server"] = "Server" in product_name
            except FileNotFoundError:
                pass
                
            # Try to get more specific version info
            try:
                build_number, _ = winreg.QueryValueEx(reg_key, "CurrentBuildNumber")
                info["windows_build"] = build_number
            except FileNotFoundError:
                pass
                
            winreg.CloseKey(reg_key)
        except (ImportError, FileNotFoundError, OSError):
            # Alternative method using WMI
            self._try_wmi_system_info(info)
                
        # If we still couldn't detect the edition, try using systeminfo
        if not info["windows_edition"]:
            self._try_systeminfo(info)
                
        return info
    
    def _try_wmi_system_info(self, info: Dict[str, Any]) -> None:
        """Try to get system information using WMI"""
        try:
            import wmi
            c = wmi.WMI()
            
            for os_info in c.Win32_OperatingSystem():
                info["windows_edition"] = os_info.Caption
                info["is_server"] = "Server" in os_info.Caption
                
                # Try to get build number
                if hasattr(os_info, "BuildNumber"):
                    info["windows_build"] = os_info.BuildNumber
        except ImportError:
            self._log_debug("WMI not available for Windows detection", "windows")
    
    def _try_systeminfo(self, info: Dict[str, Any]) -> None:
        """Try to get system information using systeminfo command"""
        try:
            si_output = subprocess.run(
                ["systeminfo"], 
                capture_output=True, 
                text=True, 
                check=False
            )
            if si_output.returncode == 0:
                for line in si_output.stdout.split("\n"):
                    if "OS Name:" in line:
                        info["windows_edition"] = line.split(":", 1)[1].strip()
                        info["is_server"] = "Server" in info["windows_edition"]
                        break
        except (FileNotFoundError, subprocess.SubprocessError):
            pass
    
    def _detect_graphics_capabilities(self) -> Dict[str, Any]:
        """
        Detect Windows-specific graphics capabilities
        
        Returns:
            Dictionary with graphics capability information
        """
        info = {
            "has_directx": False,
            "directx_version": "",
            "has_vulkan": False,
            "vulkan_version": "",
            "gpu_driver_vendor": "",
            "gpu_driver_version": "",
        }
        
        # Check for DirectX
        try:
            # Use dxdiag to detect DirectX version
            dxdiag_path = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "dxdiag.exe")
            tmp_file = os.path.join(os.environ.get("TEMP", "."), "dxdiag_output.txt")
            
            if os.path.exists(dxdiag_path):
                info["has_directx"] = True
                
                # Run dxdiag to get detailed DirectX information
                try:
                    subprocess.run([dxdiag_path, "/t", tmp_file], check=False, timeout=15)
                    
                    # Wait for the file to be created
                    import time
                    for _ in range(10):
                        if os.path.exists(tmp_file):
                            break
                        time.sleep(0.5)
                    
                    if os.path.exists(tmp_file):
                        with open(tmp_file, "r", encoding="utf-8", errors="ignore") as f:
                            dxdiag_output = f.read()
                        
                        # Extract DirectX version
                        for line in dxdiag_output.split("\n"):
                            if "DirectX Version:" in line:
                                info["directx_version"] = line.split(":", 1)[1].strip()
                                break
                                
                        # Extract GPU information
                        in_display_section = False
                        for line in dxdiag_output.split("\n"):
                            if "Display Devices" in line or "Display Device" in line:
                                in_display_section = True
                                
                            if in_display_section:
                                if "Card name:" in line:
                                    gpu_name = line.split(":", 1)[1].strip()
                                    info["gpu_name"] = gpu_name
                                    
                                    # Try to determine GPU vendor
                                    if "NVIDIA" in gpu_name:
                                        info["gpu_driver_vendor"] = "nvidia"
                                    elif "AMD" in gpu_name or "Radeon" in gpu_name:
                                        info["gpu_driver_vendor"] = "amd"
                                    elif "Intel" in gpu_name:
                                        info["gpu_driver_vendor"] = "intel"
                                        
                                elif "Driver Version:" in line:
                                    info["gpu_driver_version"] = line.split(":", 1)[1].strip()
                                    
                        # Clean up temporary file
                        try:
                            os.remove(tmp_file)
                        except (PermissionError, OSError):
                            pass
                except (subprocess.SubprocessError, OSError, IOError) as e:
                    self._log_warning(f"Error running dxdiag: {str(e)}", "windows")
        except Exception as e:
            self._log_warning(f"Error detecting DirectX: {str(e)}", "windows")
            
        # Check for Vulkan support
        try:
            # Check for vulkaninfo utility
            vulkan_info = subprocess.run(
                ["vulkaninfo"], 
                capture_output=True, 
                text=True, 
                check=False
            )
            
            if vulkan_info.returncode == 0:
                info["has_vulkan"] = True
                
                # Try to extract version information
                for line in vulkan_info.stdout.split("\n"):
                    if "Vulkan Instance Version:" in line:
                        info["vulkan_version"] = line.split(":", 1)[1].strip()
                        break
        except (FileNotFoundError, subprocess.SubprocessError):
            # Alternative checks for Vulkan
            vulkan_paths = [
                os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "vulkan-1.dll"),
                os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "SysWOW64", "vulkan-1.dll"),
            ]
            
            for path in vulkan_paths:
                if os.path.exists(path):
                    info["has_vulkan"] = True
                    break
                    
        return info
    
    def _detect_libraries_and_compilers(self) -> Dict[str, Any]:
        """
        Detect available compilers and libraries on Windows
        
        Returns:
            Dictionary with compiler and library information
        """
        info = {
            "has_msvc": False,
            "msvc_version": "",
            "has_mingw": False,
            "mingw_version": "",
            "has_visual_studio": False,
            "visual_studio_version": "",
            "has_cmake": False,
            "cmake_version": "",
            "has_vcpkg": False,
            "has_winsdk": False,
            "winsdk_version": "",
        }
        
        # Check for MSVC (Microsoft Visual C++ Compiler)
        try:
            msvc_cl = subprocess.run(
                ["cl"], 
                capture_output=True, 
                text=True, 
                check=False
            )
            
            if "Microsoft" in msvc_cl.stderr:
                info["has_msvc"] = True
                
                # Extract version information
                for line in msvc_cl.stderr.split("\n"):
                    if "Version" in line:
                        info["msvc_version"] = line.strip()
                        break
        except (FileNotFoundError, subprocess.SubprocessError):
            pass
            
        # Check for MinGW (GCC for Windows)
        try:
            mingw_gcc = subprocess.run(
                ["gcc", "--version"], 
                capture_output=True, 
                text=True, 
                check=False
            )
            
            if mingw_gcc.returncode == 0 and "MinGW" in mingw_gcc.stdout:
                info["has_mingw"] = True
                
                # Extract version
                first_line = mingw_gcc.stdout.split("\n")[0]
                import re
                version_match = re.search(r'(\d+\.\d+\.\d+)', first_line)
                if version_match:
                    info["mingw_version"] = version_match.group(1)
                else:
                    info["mingw_version"] = first_line
        except (FileNotFoundError, subprocess.SubprocessError):
            pass
            
        # Check for Visual Studio installation
        vs_paths = [
            r"C:\Program Files\Microsoft Visual Studio",
            r"C:\Program Files (x86)\Microsoft Visual Studio",
        ]
        
        for vs_path in vs_paths:
            if os.path.exists(vs_path):
                info["has_visual_studio"] = True
                
                # Try to determine version
                vs_versions = ["2022", "2019", "2017", "2015", "2013", "2012", "2010"]
                for version in vs_versions:
                    if os.path.exists(os.path.join(vs_path, version)):
                        info["visual_studio_version"] = version
                        break
                        
                # If we found a version, no need to check further
                if info["visual_studio_version"]:
                    break
                    
        # Check for CMake
        try:
            cmake_version = subprocess.run(
                ["cmake", "--version"], 
                capture_output=True, 
                text=True, 
                check=False
            )
            
            if cmake_version.returncode == 0:
                info["has_cmake"] = True
                
                # Extract version
                first_line = cmake_version.stdout.split("\n")[0]
                import re
                version_match = re.search(r'(\d+\.\d+\.\d+)', first_line)
                if version_match:
                    info["cmake_version"] = version_match.group(1)
                else:
                    info["cmake_version"] = first_line
        except (FileNotFoundError, subprocess.SubprocessError):
            pass
            
        # Check for vcpkg
        vcpkg_paths = [
            os.path.expanduser("~") + r"\vcpkg",
            r"C:\vcpkg",
            r"C:\dev\vcpkg",
        ]
        
        for path in vcpkg_paths:
            if os.path.exists(path) and os.path.exists(os.path.join(path, "vcpkg.exe")):
                info["has_vcpkg"] = True
                break
                
        # Check for Windows SDK
        winsdk_paths = [
            r"C:\Program Files (x86)\Windows Kits\10",
            r"C:\Program Files\Windows Kits\10",
            r"C:\Program Files (x86)\Windows Kits\8.1",
            r"C:\Program Files\Windows Kits\8.1",
        ]
        
        for path in winsdk_paths:
            if os.path.exists(path):
                info["has_winsdk"] = True
                
                # Try to determine version
                include_path = os.path.join(path, "Include")
                if os.path.exists(include_path):
                    try:
                        versions = [v for v in os.listdir(include_path) if os.path.isdir(os.path.join(include_path, v))]
                        if versions:
                            # Sort versions to get the latest
                            versions.sort(reverse=True)
                            info["winsdk_version"] = versions[0]
                    except (PermissionError, OSError):
                        pass
                        
                # If we found a version, no need to check further
                if info["winsdk_version"]:
                    break
                    
        return info
    
    def _detect_wsl(self) -> Dict[str, Any]:
        """
        Detect Windows Subsystem for Linux (WSL)
        
        Returns:
            Dictionary with WSL information
        """
        info = {
            "has_wsl": False,
            "wsl_version": "",
            "wsl_distros": [],
        }
        
        try:
            # Check if wsl.exe exists
            wsl_command = subprocess.run(
                ["wsl", "--status"], 
                capture_output=True, 
                text=True, 
                check=False
            )
            
            if wsl_command.returncode == 0:
                info["has_wsl"] = True
                
                # Try to determine WSL version
                for line in wsl_command.stdout.split("\n"):
                    if "WSL version:" in line:
                        info["wsl_version"] = line.split(":", 1)[1].strip()
                        break
                        
                # If WSL version wasn't found in --status output, check if it's WSL 1 or 2
                if not info["wsl_version"]:
                    wsl_version_check = subprocess.run(
                        ["wsl", "--list", "--verbose"], 
                        capture_output=True, 
                        text=True, 
                        check=False
                    )
                    
                    if "2" in wsl_version_check.stdout:
                        info["wsl_version"] = "2"
                    else:
                        info["wsl_version"] = "1"
                        
                # Get available WSL distributions
                wsl_distros = subprocess.run(
                    ["wsl", "--list"], 
                    capture_output=True, 
                    text=True, 
                    check=False
                )
                
                if wsl_distros.returncode == 0:
                    # Parse distro list, skipping the header line
                    lines = [line.strip() for line in wsl_distros.stdout.split("\n")[1:] if line.strip()]
                    info["wsl_distros"] = lines
        except (FileNotFoundError, subprocess.SubprocessError):
            pass
            
        return info
    
    def _detect_hardware_optimizations(self) -> Dict[str, Any]:
        """
        Detect hardware-specific optimizations on Windows
        
        Returns:
            Dictionary with optimization information
        """
        info = {
            "has_oneapi": False,
            "has_cuda": False,
            "has_rocm": False,
            "has_avx": False,
            "has_avx2": False,
            "has_avx512": False,
            "has_opencl": False,
            "has_openvino": False,
        }
        
        # Check for Intel oneAPI
        oneapi_paths = [
            r"C:\Program Files (x86)\Intel\oneAPI",
            r"C:\Program Files\Intel\oneAPI",
        ]
        
        for path in oneapi_paths:
            if os.path.exists(path):
                info["has_oneapi"] = True
                break
                
        # Check for NVIDIA CUDA Toolkit
        cuda_paths = [
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
            r"C:\CUDA",
        ]
        
        for path in cuda_paths:
            if os.path.exists(path):
                info["has_cuda"] = True
                
                # Try to determine CUDA version
                try:
                    cuda_dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
                    cuda_versions = [d for d in cuda_dirs if d.startswith("v")]
                    if cuda_versions:
                        # Sort versions to get the latest
                        cuda_versions.sort(reverse=True)
                        info["cuda_toolkit_version"] = cuda_versions[0].replace("v", "")
                except (PermissionError, OSError):
                    pass
                    
                break
                
        # Check for AMD ROCm
        rocm_paths = [
            r"C:\Program Files\AMD\ROCm",
            r"C:\ROCm",
        ]
        
        for path in rocm_paths:
            if os.path.exists(path):
                info["has_rocm"] = True
                break
                
        # Check for OpenCL
        opencl_paths = [
            os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "OpenCL.dll"),
            os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "SysWOW64", "OpenCL.dll"),
        ]
        
        for path in opencl_paths:
            if os.path.exists(path):
                info["has_opencl"] = True
                break
                
        # Check for Intel OpenVINO
        openvino_paths = [
            r"C:\Program Files (x86)\Intel\openvino",
            r"C:\Program Files\Intel\openvino",
        ]
        
        for path in openvino_paths:
            if os.path.exists(path):
                info["has_openvino"] = True
                break
                
        # Check CPU features (AVX/AVX2/AVX512 support)
        try:
            import cpuinfo
            
            cpu_info = cpuinfo.get_cpu_info()
            if "flags" in cpu_info:
                info["has_avx"] = "avx" in cpu_info["flags"]
                info["has_avx2"] = "avx2" in cpu_info["flags"]
                info["has_avx512"] = "avx512f" in cpu_info["flags"] or any(f.startswith("avx512") for f in cpu_info["flags"])
        except ImportError:
            # Alternative method using CPUID via ctypes
            self._try_cpuid_detection(info)
                
        return info
    
    def _try_cpuid_detection(self, info: Dict[str, Any]) -> None:
        """Try to detect CPU features using CPUID"""
        try:
            # Simple detection based on processor name
            if platform.processor():
                model_info = platform.processor()
                # Extract generation from model (e.g., i7-9700K is 9th gen)
                import re
                gen_match = re.search(r'i[357]-(\d)(\d+)', model_info)
                if gen_match:
                    gen = int(gen_match.group(1))
                    if gen >= 6:  # 6th gen or newer has AVX2
                        info["has_avx"] = True
                        info["has_avx2"] = True
                    if gen >= 3:  # 3rd gen or newer has AVX
                        info["has_avx"] = True
                    if gen >= 10:  # 10th gen Ice Lake or newer might have AVX-512
                        # Need more specific detection
                        if "Ice Lake" in model_info or "Tiger Lake" in model_info:
                            info["has_avx512"] = True
        except Exception as e:
            self._log_warning(f"Error detecting CPU features: {str(e)}", "windows")
    
    def get_summary(self, capabilities=None) -> Dict[str, Any]:
        """
        Get a summary of Windows-specific capabilities
        
        Args:
            capabilities: Optional pre-detected capabilities
            
        Returns:
            Dictionary with Windows capability summary
        """
        if capabilities is None:
            capabilities = self.detect() if not self.capabilities else self.capabilities
        
        summary = {
            "windows": {
                "name": capabilities.get("windows_name", "Unknown"),
                "edition": capabilities.get("windows_edition", "Unknown"),
                "build": capabilities.get("windows_build", "Unknown"),
                "is_server": capabilities.get("is_server", False),
                "is_64bit": capabilities.get("is_64bit", False),
            },
            "graphics": {
                "directx": {
                    "available": capabilities.get("has_directx", False),
                    "version": capabilities.get("directx_version", "Unknown"),
                },
                "vulkan": {
                    "available": capabilities.get("has_vulkan", False),
                    "version": capabilities.get("vulkan_version", "Unknown"),
                },
                "gpu_driver": {
                    "vendor": capabilities.get("gpu_driver_vendor", "Unknown"),
                    "version": capabilities.get("gpu_driver_version", "Unknown"),
                },
            },
            "development": {
                "msvc": {
                    "available": capabilities.get("has_msvc", False),
                    "version": capabilities.get("msvc_version", "Unknown"),
                },
                "mingw": {
                    "available": capabilities.get("has_mingw", False),
                    "version": capabilities.get("mingw_version", "Unknown"),
                },
                "visual_studio": {
                    "available": capabilities.get("has_visual_studio", False),
                    "version": capabilities.get("visual_studio_version", "Unknown"),
                },
                "winsdk": {
                    "available": capabilities.get("has_winsdk", False),
                    "version": capabilities.get("winsdk_version", "Unknown"),
                },
            },
            "wsl": {
                "available": capabilities.get("has_wsl", False),
                "version": capabilities.get("wsl_version", "Unknown"),
                "distros": capabilities.get("wsl_distros", []),
            },
            "optimizations": {
                "avx": capabilities.get("has_avx", False),
                "avx2": capabilities.get("has_avx2", False),
                "avx512": capabilities.get("has_avx512", False),
                "oneapi": capabilities.get("has_oneapi", False),
                "cuda": capabilities.get("has_cuda", False),
                "rocm": capabilities.get("has_rocm", False),
                "opencl": capabilities.get("has_opencl", False),
                "openvino": capabilities.get("has_openvino", False),
            }
        }
        
        return summary