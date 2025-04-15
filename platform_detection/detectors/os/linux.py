"""
Platform Detection Framework - Linux Detection Module

This module provides specialized detection for Linux systems,
including distribution details, desktop environments, and optimizations.
"""

import platform
import subprocess
import os
import sys
import warnings
from typing import Dict, Any, List, Optional

from ...base import BaseDetector
from ...utils import ErrorHandler
from ...constants import (
    FLAG_GCC, FLAG_CLANG, FLAG_OPENMP, FLAG_MKL, FLAG_OPENBLAS, FLAG_ATLAS
)


class LinuxDetector(BaseDetector):
    """
    Detects Linux-specific hardware and software capabilities, with special 
    focus on distribution, compiler availability, and library optimizations.
    """
    
    def __init__(self, error_handler: Optional[ErrorHandler] = None):
        """Initialize the Linux detector"""
        super().__init__(error_handler)
        self.capabilities = {}
        
    def detect(self) -> Dict[str, Any]:
        """
        Perform Linux-specific detection
        
        Returns:
            Dictionary with Linux capabilities
        """
        self._log_info("Starting Linux detection", "linux")
        
        # Basic Linux information
        info = {
            "linux_version": platform.release(),
            "linux_dist": "",
            "linux_dist_version": "",
            "desktop_environment": "",
            "has_gcc": False,
            "has_clang": False,
        }
        
        # Detect Linux distribution
        self._log_info("Detecting Linux distribution", "linux")
        info.update(self._detect_linux_distribution())
        
        # Detect desktop environment
        self._log_info("Detecting desktop environment", "linux")
        info.update(self._detect_desktop_environment())
        
        # Detect compiler and library availability
        self._log_info("Detecting compilers and libraries", "linux")
        info.update(self._detect_compilers_and_libraries())
        
        # Detect BLAS and LAPACK implementations
        self._log_info("Detecting BLAS and LAPACK implementations", "linux")
        info.update(self._detect_blas_lapack())
        
        # Detect filesystem info
        self._log_info("Detecting filesystem information", "linux")
        info.update(self._detect_filesystem_info())
        
        # Normalize capability flags
        normalized_info = self._normalize_flags(info)
        
        # Store capabilities
        self.capabilities = normalized_info
        
        self._log_info("Linux detection complete", "linux")
        return normalized_info
    
    def _detect_linux_distribution(self) -> Dict[str, Any]:
        """
        Detect Linux distribution information
        
        Returns:
            Dictionary with distribution information
        """
        info = {
            "linux_dist": "",
            "linux_dist_version": "",
            "linux_dist_codename": "",
            "is_container": False,
            "is_wsl": False,
        }
        
        # Try using /etc/os-release - modern and standard method
        if os.path.exists("/etc/os-release"):
            try:
                with open("/etc/os-release", "r") as f:
                    os_release = f.readlines()
                
                for line in os_release:
                    if "=" not in line:
                        continue
                    
                    key, value = line.strip().split("=", 1)
                    value = value.strip('"')
                    
                    if key == "NAME":
                        info["linux_dist"] = value
                    elif key == "VERSION_ID":
                        info["linux_dist_version"] = value
                    elif key == "VERSION_CODENAME":
                        info["linux_dist_codename"] = value
                        
                # If we got the NAME but not version, try different parsing
                if info["linux_dist"] and not info["linux_dist_version"]:
                    for line in os_release:
                        if "VERSION=" in line:
                            version = line.split("=", 1)[1].strip('"')
                            # Try to extract version number from string
                            import re
                            version_match = re.search(r'(\d+\.\d+(\.\d+)?)', version)
                            if version_match:
                                info["linux_dist_version"] = version_match.group(1)
            except Exception as e:
                self._log_warning(f"Error reading /etc/os-release: {str(e)}", "linux")
        
        # If we didn't get enough info, try other methods
        if not info["linux_dist"]:
            # Check for other distribution files
            distro_files = [
                ("/etc/lsb-release", "DISTRIB_ID", "DISTRIB_RELEASE"),
                ("/etc/debian_version", None, None),
                ("/etc/redhat-release", None, None),
                ("/etc/SuSE-release", None, None),
                ("/etc/arch-release", None, None),
            ]
            
            for file_path, id_key, version_key in distro_files:
                if os.path.exists(file_path):
                    try:
                        if id_key and version_key:
                            # Format with key=value pairs
                            with open(file_path, "r") as f:
                                content = f.readlines()
                            
                            for line in content:
                                if "=" not in line:
                                    continue
                                
                                key, value = line.strip().split("=", 1)
                                value = value.strip('"')
                                
                                if key == id_key:
                                    info["linux_dist"] = value
                                elif key == version_key:
                                    info["linux_dist_version"] = value
                        else:
                            # Format with just content
                            with open(file_path, "r") as f:
                                content = f.read().strip()
                            
                            if "debian" in file_path:
                                info["linux_dist"] = "Debian"
                                info["linux_dist_version"] = content
                            elif "redhat" in file_path:
                                if "CentOS" in content:
                                    info["linux_dist"] = "CentOS"
                                else:
                                    info["linux_dist"] = "Red Hat"
                                # Try to extract version number
                                import re
                                version_match = re.search(r'release\s+(\d+\.\d+(\.\d+)?)', content)
                                if version_match:
                                    info["linux_dist_version"] = version_match.group(1)
                            elif "SuSE" in file_path:
                                info["linux_dist"] = "SuSE"
                                # Try to find version in content
                                for line in content.split("\n"):
                                    if "VERSION" in line:
                                        info["linux_dist_version"] = line.split("=", 1)[1].strip()
                                        break
                            elif "arch" in file_path:
                                info["linux_dist"] = "Arch Linux"
                                # Arch is rolling release, no specific version
                                info["linux_dist_version"] = "rolling"
                    except Exception as e:
                        self._log_warning(f"Error reading {file_path}: {str(e)}", "linux")
                
                # If we got distribution info, break out of the loop
                if info["linux_dist"]:
                    break
        
        # Check if running in a container
        container_indicators = [
            "/.dockerenv",
            "/run/.containerenv",
        ]
        for indicator in container_indicators:
            if os.path.exists(indicator):
                info["is_container"] = True
                break
                
        # Check if running in WSL (Windows Subsystem for Linux)
        try:
            with open("/proc/version", "r") as f:
                proc_version = f.read()
                
            info["is_wsl"] = "Microsoft" in proc_version or "WSL" in proc_version
            
            # Try to determine WSL version
            if info["is_wsl"]:
                if "WSL2" in proc_version:
                    info["wsl_version"] = "2"
                else:
                    info["wsl_version"] = "1"
        except Exception:
            pass
            
        return info
    
    def _detect_desktop_environment(self) -> Dict[str, Any]:
        """
        Detect Linux desktop environment
        
        Returns:
            Dictionary with desktop environment information
        """
        info = {
            "desktop_environment": "",
            "display_server": "",
            "window_manager": "",
        }
        
        # First check the environment variables
        desktop_env = os.environ.get("XDG_CURRENT_DESKTOP", "")
        if desktop_env:
            info["desktop_environment"] = desktop_env
        else:
            # Try other environment variables
            for env_var in ["DESKTOP_SESSION", "GNOME_DESKTOP_SESSION_ID", "KDE_FULL_SESSION"]:
                if env_var in os.environ:
                    if env_var == "GNOME_DESKTOP_SESSION_ID":
                        info["desktop_environment"] = "GNOME"
                    elif env_var == "KDE_FULL_SESSION":
                        info["desktop_environment"] = "KDE"
                    else:
                        info["desktop_environment"] = os.environ[env_var]
                    break
        
        # Determine display server (X11 or Wayland)
        if "WAYLAND_DISPLAY" in os.environ:
            info["display_server"] = "Wayland"
        elif "DISPLAY" in os.environ:
            info["display_server"] = "X11"
            
            # Try to get window manager in X11
            try:
                wmctrl_output = subprocess.run(
                    ["wmctrl", "-m"], 
                    capture_output=True, 
                    text=True, 
                    check=False
                )
                
                if wmctrl_output.returncode == 0:
                    for line in wmctrl_output.stdout.split("\n"):
                        if "Name:" in line:
                            info["window_manager"] = line.split(":", 1)[1].strip()
                            break
            except (FileNotFoundError, subprocess.SubprocessError):
                pass
        
        return info
    
    def _detect_compilers_and_libraries(self) -> Dict[str, Any]:
        """
        Detect available compilers and libraries on Linux
        
        Returns:
            Dictionary with compiler and library information
        """
        info = {
            "has_gcc": False,
            "gcc_version": "",
            "has_clang": False,
            "clang_version": "",
            "has_openmp": False,
            "has_fortran": False,
            "fortran_version": "",
            "has_cmake": False,
            "cmake_version": "",
            "has_make": False,
            "make_version": "",
        }
        
        # Check for GCC (GNU Compiler Collection)
        try:
            gcc_version = subprocess.run(
                ["gcc", "--version"], 
                capture_output=True, 
                text=True, 
                check=False
            )
            
            if gcc_version.returncode == 0:
                info["has_gcc"] = True
                
                # Extract version
                first_line = gcc_version.stdout.split("\n")[0]
                import re
                version_match = re.search(r'(\d+\.\d+\.\d+)', first_line)
                if version_match:
                    info["gcc_version"] = version_match.group(1)
                else:
                    info["gcc_version"] = first_line
                    
                # Check for OpenMP
                try:
                    # Create a simple test file
                    with open("/tmp/openmp_test.c", "w") as f:
                        f.write("""
                        #include <omp.h>
                        int main() {
                            return 0;
                        }
                        """)
                    
                    # Try to compile with OpenMP
                    openmp_test = subprocess.run(
                        ["gcc", "-fopenmp", "-o", "/tmp/openmp_test", "/tmp/openmp_test.c"],
                        capture_output=True,
                        check=False
                    )
                    
                    info["has_openmp"] = openmp_test.returncode == 0
                    
                    # Clean up test files
                    try:
                        os.remove("/tmp/openmp_test.c")
                        if os.path.exists("/tmp/openmp_test"):
                            os.remove("/tmp/openmp_test")
                    except (PermissionError, OSError):
                        pass
                except Exception:
                    pass
        except (FileNotFoundError, subprocess.SubprocessError):
            pass
            
        # Check for Clang
        try:
            clang_version = subprocess.run(
                ["clang", "--version"], 
                capture_output=True, 
                text=True, 
                check=False
            )
            
            if clang_version.returncode == 0:
                info["has_clang"] = True
                
                # Extract version
                first_line = clang_version.stdout.split("\n")[0]
                import re
                version_match = re.search(r'version\s+(\d+\.\d+\.\d+)', first_line)
                if version_match:
                    info["clang_version"] = version_match.group(1)
                else:
                    info["clang_version"] = first_line
        except (FileNotFoundError, subprocess.SubprocessError):
            pass
            
        # Check for Fortran
        for fortran_cmd in ["gfortran", "f95", "f90", "f77"]:
            try:
                fortran_version = subprocess.run(
                    [fortran_cmd, "--version"], 
                    capture_output=True, 
                    text=True, 
                    check=False
                )
                
                if fortran_version.returncode == 0:
                    info["has_fortran"] = True
                    info["fortran_compiler"] = fortran_cmd
                    
                    # Extract version
                    first_line = fortran_version.stdout.split("\n")[0]
                    import re
                    version_match = re.search(r'(\d+\.\d+\.\d+)', first_line)
                    if version_match:
                        info["fortran_version"] = version_match.group(1)
                    else:
                        info["fortran_version"] = first_line
                    
                    break
            except (FileNotFoundError, subprocess.SubprocessError):
                continue
                
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
                version_match = re.search(r'version\s+(\d+\.\d+\.\d+)', first_line)
                if version_match:
                    info["cmake_version"] = version_match.group(1)
                else:
                    info["cmake_version"] = first_line
        except (FileNotFoundError, subprocess.SubprocessError):
            pass
            
        # Check for Make
        try:
            make_version = subprocess.run(
                ["make", "--version"], 
                capture_output=True, 
                text=True, 
                check=False
            )
            
            if make_version.returncode == 0:
                info["has_make"] = True
                
                # Extract version
                first_line = make_version.stdout.split("\n")[0]
                import re
                version_match = re.search(r'(\d+\.\d+(\.\d+)?)', first_line)
                if version_match:
                    info["make_version"] = version_match.group(1)
                else:
                    info["make_version"] = first_line
        except (FileNotFoundError, subprocess.SubprocessError):
            pass
            
        return info
    
    def _detect_blas_lapack(self) -> Dict[str, Any]:
        """
        Detect BLAS and LAPACK implementations
        
        Returns:
            Dictionary with BLAS and LAPACK information
        """
        info = {
            "has_mkl": False,
            "has_openblas": False,
            "has_atlas": False,
            "has_lapack": False,
        }
        
        # Check for shared libraries
        library_paths = [
            "/usr/lib",
            "/usr/lib64",
            "/usr/local/lib",
            "/usr/local/lib64",
            "/lib",
            "/lib64",
        ]
        
        mkl_libraries = ["libmkl_rt.so", "libmkl_core.so"]
        openblas_libraries = ["libopenblas.so", "libopenblasp.so"]
        atlas_libraries = ["libatlas.so", "libsatlas.so"]
        lapack_libraries = ["liblapack.so"]
        
        # Check libraries in all library paths
        for lib_path in library_paths:
            if not os.path.exists(lib_path):
                continue
                
            try:
                files = os.listdir(lib_path)
                
                # Check for MKL
                if not info["has_mkl"]:
                    for lib in mkl_libraries:
                        if any(f.startswith(lib) for f in files):
                            info["has_mkl"] = True
                            break
                
                # Check for OpenBLAS
                if not info["has_openblas"]:
                    for lib in openblas_libraries:
                        if any(f.startswith(lib) for f in files):
                            info["has_openblas"] = True
                            break
                
                # Check for ATLAS
                if not info["has_atlas"]:
                    for lib in atlas_libraries:
                        if any(f.startswith(lib) for f in files):
                            info["has_atlas"] = True
                            break
                
                # Check for LAPACK
                if not info["has_lapack"]:
                    for lib in lapack_libraries:
                        if any(f.startswith(lib) for f in files):
                            info["has_lapack"] = True
                            break
            except (PermissionError, OSError):
                continue
                
        # Try an alternative approach with scipy
        try:
            import scipy
            import numpy
            
            # Check numpy config to see what BLAS is being used
            try:
                blas_info = scipy.__config__.blas_opt_info
                if blas_info:
                    if any("mkl" in str(value).lower() for value in blas_info.values()):
                        info["has_mkl"] = True
                    if any("openblas" in str(value).lower() for value in blas_info.values()):
                        info["has_openblas"] = True
                    if any("atlas" in str(value).lower() for value in blas_info.values()):
                        info["has_atlas"] = True
                
                # Check lapack config
                lapack_info = scipy.__config__.lapack_opt_info
                if lapack_info:
                    info["has_lapack"] = True
                    if any("mkl" in str(value).lower() for value in lapack_info.values()):
                        info["has_mkl"] = True
            except AttributeError:
                # If scipy.__config__ attribute doesn't exist or doesn't have blas_opt_info
                # Try numpy instead
                try:
                    numpy_config = numpy.__config__.show()
                    info["has_mkl"] = "mkl" in numpy_config.lower()
                    info["has_openblas"] = "openblas" in numpy_config.lower()
                    info["has_atlas"] = "atlas" in numpy_config.lower()
                    info["has_lapack"] = "lapack" in numpy_config.lower()
                except (AttributeError, NameError):
                    pass
        except ImportError:
            pass
            
        return info
    
    def _detect_filesystem_info(self) -> Dict[str, Any]:
        """
        Detect filesystem information
        
        Returns:
            Dictionary with filesystem information
        """
        info = {
            "filesystem_type": {},
            "storage_info": {},
        }
        
        # Try to use df to get filesystem information
        try:
            df_output = subprocess.run(
                ["df", "-T"], 
                capture_output=True, 
                text=True, 
                check=False
            )
            
            if df_output.returncode == 0:
                lines = df_output.stdout.strip().split("\n")[1:]  # Skip header
                
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 7:
                        device = parts[0]
                        fs_type = parts[1]
                        mount_point = parts[6]
                        
                        # Only store info for non-special filesystems
                        if not (mount_point.startswith("/dev") or mount_point.startswith("/proc") or 
                                mount_point.startswith("/sys") or mount_point.startswith("/run")):
                            info["filesystem_type"][mount_point] = fs_type
        except (FileNotFoundError, subprocess.SubprocessError):
            pass
            
        # Try to get disk usage with psutil for more detailed info
        try:
            import psutil
            
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    info["storage_info"][partition.mountpoint] = {
                        "device": partition.device,
                        "fstype": partition.fstype,
                        "total": usage.total,
                        "used": usage.used,
                        "free": usage.free,
                        "percent": usage.percent,
                    }
                except (PermissionError, OSError):
                    continue
        except ImportError:
            pass
            
        return info
    
    def get_summary(self, capabilities=None) -> Dict[str, Any]:
        """
        Get a summary of Linux-specific capabilities
        
        Args:
            capabilities: Optional pre-detected capabilities
            
        Returns:
            Dictionary with Linux capability summary
        """
        if capabilities is None:
            capabilities = self.detect() if not self.capabilities else self.capabilities
        
        summary = {
            "distribution": {
                "name": capabilities.get("linux_dist", "Unknown"),
                "version": capabilities.get("linux_dist_version", "Unknown"),
                "codename": capabilities.get("linux_dist_codename", ""),
            },
            "desktop": {
                "environment": capabilities.get("desktop_environment", "Unknown"),
                "display_server": capabilities.get("display_server", "Unknown"),
                "window_manager": capabilities.get("window_manager", ""),
            },
            "development": {
                "gcc": {
                    "available": capabilities.get("has_gcc", False),
                    "version": capabilities.get("gcc_version", "Unknown"),
                },
                "clang": {
                    "available": capabilities.get("has_clang", False),
                    "version": capabilities.get("clang_version", "Unknown"),
                },
                "fortran": {
                    "available": capabilities.get("has_fortran", False),
                    "version": capabilities.get("fortran_version", "Unknown"),
                },
                "openmp": capabilities.get("has_openmp", False),
            },
            "optimizations": {
                "mkl": capabilities.get("has_mkl", False),
                "openblas": capabilities.get("has_openblas", False),
                "atlas": capabilities.get("has_atlas", False),
                "lapack": capabilities.get("has_lapack", False),
            },
            "container": {
                "is_container": capabilities.get("is_container", False),
                "is_wsl": capabilities.get("is_wsl", False),
                "wsl_version": capabilities.get("wsl_version", ""),
            }
        }
        
        return summary