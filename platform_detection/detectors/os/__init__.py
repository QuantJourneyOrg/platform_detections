"""
Platform Detection Framework - OS-Specific Detectors Package

Part of QuantJourney, a project for high-performance investment research and
quantitative finance, build on AI infrastructure for Investors and Quantitative
Researchers available at https://quantjourney.substack.com/

This package contains modules for detecting OS-specific capabilities
for different operating systems.

This code is licensed under the MIT License.
Copyright (c) 2023 Jakub Polec

"""

from .darwin import DarwinDetector
from .linux import LinuxDetector
from .windows import WindowsDetector

__all__ = [
    'DarwinDetector',
    'LinuxDetector', 
    'WindowsDetector',
]