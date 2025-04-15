"""
Setup script for the Platform Detection Framework

Part of QuantJourney, a project for high-performance investment research and
quantitative finance, build on AI infrastructure for Investors and Quantitative
Researchers available at https://quantjourney.substack.com/

This script sets up the package for distribution, including metadata,
dependencies, and entry points for command-line interfaces.

This code is licensed under the MIT License.
Copyright (c) 2023 Jakub Polec
"""

from setuptools import setup, find_packages
import os
import re

# Read version from __init__.py
with open(os.path.join('platform_detection', '__init__.py'), 'r') as f:
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if version_match:
        version = version_match.group(1)
    else:
        version = '0.1.0'

# Read long description from README.md
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='platform_detection',
    version=version,
    description='A comprehensive framework for detecting and optimizing computational workloads across platforms',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Jakub Polec',
    author_email='jakub@quantjourney.com',
    url='https://github.com/QuantJourneyOrg/platform_detection',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Financial and Insurance Industry',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: System :: Hardware',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
    install_requires=[
        'psutil>=5.8.0',
    ],
    extras_require={
        'full': [
            'numpy>=1.19.0',
            'pandas>=1.0.0',
            'numba>=0.50.0',
            'cython>=0.29.0',
            'cpuinfo>=8.0.0',
        ],
        'dev': [
            'pytest>=6.0.0',
            'pytest-cov>=2.10.0',
            'black>=21.5b2',
            'isort>=5.9.1',
            'flake8>=3.9.2',
            'mypy>=0.812',
            'sphinx>=4.0.0',
            'sphinx-rtd-theme>=0.5.2',
        ],
    },
    entry_points={
        'console_scripts': [
            'platform-detect=platform_detection.cli:main',
        ],
    },
)