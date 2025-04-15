"""
Platform Detection Framework - Detectors Package

Part of QuantJourney, a project for high-performance investment research and
quantitative finance, build on AI infrastructure for Investors and Quantitative
Researchers available at https://quantjourney.substack.com/

This package contains modules for detecting various aspects of the platform,
including hardware, software, and OS-specific capabilities.

This code is licensed under the MIT License.
Copyright (c) 2023 Jakub Polec
"""

from .base import BaseDetector
from .hardware import HardwareDetector
from .software import SoftwareDetector

# Import OS-specific detectors if available
import platform
import sys
import importlib.util

os_name = platform.system().lower()

# Initialize OS detector variables
DarwinDetector = None
LinuxDetector = None
WindowsDetector = None

# Dynamically import the appropriate OS-specific detector
if os_name == "darwin":
    spec = importlib.util.find_spec('.darwin', package=__name__)
    if spec is not None:
        from .darwin import DarwinDetector
elif os_name == "linux":
    spec = importlib.util.find_spec('.linux', package=__name__)
    if spec is not None:
        from .linux import LinuxDetector
elif os_name == "windows":
    spec = importlib.util.find_spec('.windows', package=__name__)
    if spec is not None:
        from .windows import WindowsDetector

__all__ = [
    'BaseDetector',
    'HardwareDetector',
    'SoftwareDetector',
    'DarwinDetector',
    'LinuxDetector',
    'WindowsDetector',
]