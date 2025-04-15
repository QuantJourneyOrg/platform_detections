"""
Platform Detection Framework - Constants Module

Part of QuantJourney, a project for high-performance investment research and
quantitative finance, build on AI infrastructure for Investors and Quantitative
Researchers available at https://quantjourney.substack.com/

This module provides standardized constants for capability flags,
error handling, and other shared resources across the framework.

This code is licensed under the MIT License.
Copyright (c) 2023 Jakub Polec

"""

# Flag prefixes for consistent naming
HAS_PREFIX = "has_"  # For binary feature flags (e.g., has_gpu)
VERSION_PREFIX = "_version"  # For version information (e.g., cuda_version)

# Common capability flags used across multiple detectors
# Hardware capabilities
FLAG_GPU = "has_gpu"
FLAG_CPU_CORES = "cpu_count"
FLAG_CPU_BRAND = "cpu_brand"
FLAG_MEMORY_TOTAL = "memory_total"
FLAG_MEMORY_AVAILABLE = "memory_available"
FLAG_MEMORY_USED = "memory_used"
FLAG_MEMORY_PERCENT = "memory_percent"

# SIMD capabilities
FLAG_AVX = "has_avx"
FLAG_AVX2 = "has_avx2"
FLAG_AVX512 = "has_avx512"
FLAG_SSE = "has_sse"
FLAG_SSE2 = "has_sse2"
FLAG_SSE3 = "has_sse3"
FLAG_SSE4_1 = "has_sse4_1"
FLAG_SSE4_2 = "has_sse4_2"
FLAG_NEON = "has_neon"

# GPU and accelerator capabilities
FLAG_CUDA = "has_cuda"
FLAG_METAL = "has_metal"
FLAG_OPENCL = "has_opencl"
FLAG_VULKAN = "has_vulkan"
FLAG_DIRECTX = "has_directx"
FLAG_ROCM = "has_rocm"
FLAG_ONEAPI = "has_oneapi"

# Software capabilities
FLAG_NUMBA = "has_numba"
FLAG_CYTHON = "has_cython"
FLAG_PYTHON_VERSION = "python_version"
FLAG_PYTHON_BITS = "python_bits"
FLAG_PYTHON_IMPL = "python_implementation"

# Machine learning framework capabilities
FLAG_TENSORFLOW = "has_tensorflow"
FLAG_PYTORCH = "has_pytorch"
FLAG_JAX = "has_jax"
FLAG_SKLEARN = "has_sklearn"

# Data processing capabilities
FLAG_NUMPY = "has_numpy"
FLAG_PANDAS = "has_pandas"
FLAG_POLARS = "has_polars"
FLAG_DASK = "has_dask"
FLAG_BOTTLENECK = "has_bottleneck"

# OS-specific capabilities
# Darwin (macOS)
FLAG_APPLE_SILICON = "has_apple_silicon"
FLAG_ACCELERATE = "has_accelerate"
FLAG_COREML = "has_coreml"
FLAG_NEURAL_ENGINE = "has_neural_engine"

# Windows-specific
FLAG_WSL = "has_wsl"
FLAG_MSVC = "has_msvc"
FLAG_MINGW = "has_mingw"
FLAG_VISUAL_STUDIO = "has_visual_studio"
FLAG_WINSDK = "has_winsdk"

# Linux-specific
FLAG_GLIBC = "has_glibc"
FLAG_GCC = "has_gcc"
FLAG_CLANG = "has_clang"
FLAG_OPENMP = "has_openmp"
FLAG_MKL = "has_mkl"
FLAG_OPENBLAS = "has_openblas"
FLAG_ATLAS = "has_atlas"

# Error levels for unified error handling
ERROR_LEVEL_DEBUG = 0
ERROR_LEVEL_INFO = 1
ERROR_LEVEL_WARNING = 2
ERROR_LEVEL_ERROR = 3
ERROR_LEVEL_CRITICAL = 4

# Error categories for organizing error messages
ERROR_CATEGORY_HARDWARE = "hardware"
ERROR_CATEGORY_SOFTWARE = "software"
ERROR_CATEGORY_OS = "os"
ERROR_CATEGORY_GENERAL = "general"
ERROR_CATEGORY_DETECTION = "detection"

# JSON serialization keys and special values
JSON_NULL_VALUE = "null"  # For representing null values in JSON
JSON_METADATA_KEY = "platform_detection_metadata"  # For metadata in JSON output
JSON_TIMESTAMP_KEY = "timestamp"  # For recording when detection was performed
JSON_VERSION_KEY = "framework_version"  # Framework version tracking

# Framework version information
FRAMEWORK_VERSION = "1.0.0"

# Operation types for backend selection
OPERATION_TYPE_MATRIX = "matrix"
OPERATION_TYPE_STAT = "stat"
OPERATION_TYPE_ML = "ml"
OPERATION_TYPE_DATA = "data"
OPERATION_TYPE_GRAPHICS = "graphics"
OPERATION_TYPE_SIGNAL = "signal"
OPERATION_TYPE_FINANCE = "finance"
OPERATION_TYPE_GENERAL = "general"

# Data size thresholds for optimization decisions
DATA_SIZE_SMALL = 10000       # Below this, CPU is typically better
DATA_SIZE_MEDIUM = 1000000    # Medium-sized data
DATA_SIZE_LARGE = 100000000   # Large data, typically needs distributed processing

# Backend capabilities flags
FLAG_BACKEND_PARALLEL = "parallel"
FLAG_BACKEND_VECTORIZED = "vectorized"
FLAG_BACKEND_DISTRIBUTED = "distributed"
FLAG_BACKEND_GPU = "gpu"
FLAG_BACKEND_OPTIMIZED = "optimized"