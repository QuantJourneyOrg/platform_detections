"""
Platform Detection Framework - Software Detection Module

This module provides detection of Python packages and libraries 
for identifying available computation and optimization capabilities.

Part of QuantJourney, a project for high-performance investment research and
quantitative finance, build on AI infrastructure for Investors and Quantitative
Researchers available at https://quantjourney.substack.com/

This code is licensed under the MIT License.
Copyright (c) 2023 Jakub Polec

"""

import platform
import importlib
import sys
import warnings
from typing import Dict, Any, List, Optional, Set

from .base import BaseDetector
from ..utils import ErrorHandler
from ..constants import (
    FLAG_NUMPY, FLAG_PANDAS, FLAG_POLARS, FLAG_DASK, FLAG_NUMBA,
    FLAG_CYTHON, FLAG_TENSORFLOW, FLAG_PYTORCH, FLAG_JAX, FLAG_BOTTLENECK
)


class SoftwareDetector(BaseDetector):
    """
    Detects available Python packages, libraries, and software capabilities
    for computational optimization.
    """
    
    def __init__(self, error_handler: Optional[ErrorHandler] = None):
        """Initialize the software detector"""
        super().__init__(error_handler)
        self.packages_info = {}
        self.ml_info = {}
        self.backends_info = {}
        
    def detect(self) -> Dict[str, Any]:
        """
        Detect available Python packages and computational libraries
        
        Returns:
            Dictionary with software capabilities
        """
        self._log_info("Starting software detection", "software")
        
        info = {}
        
        # Detect general computational packages
        self._log_info("Detecting computational packages", "software")
        packages_info = self._detect_packages()
        info.update(packages_info)
        self.packages_info = packages_info
        
        # Detect machine learning packages
        self._log_info("Detecting machine learning packages", "software")
        ml_info = self._detect_ml_packages()
        info.update(ml_info)
        self.ml_info = ml_info
        
        # Detect specialized backends
        self._log_info("Detecting specialized backends", "software")
        backends_info = self._detect_backend_support()
        info.update(backends_info)
        self.backends_info = backends_info
        
        # Normalize capability flags
        normalized_info = self._normalize_flags(info)
        
        self._log_info("Software detection complete", "software")
        return normalized_info
    
    def _detect_packages(self) -> Dict[str, Any]:
        """
        Detect general computational packages
        
        Returns:
            Dictionary with package information
        """
        info = {}
        
        # Core computational packages
        packages_to_check = [
            # Essential data packages
            ("numpy", "numpy"),
            ("pandas", "pandas"),
            ("polars", "polars"),
            ("dask", "dask"),
            ("pyarrow", "pyarrow"),
            
            # Scientific computing
            ("scipy", "scipy"),
            ("statsmodels", "statsmodels"),
            ("sympy", "sympy"),
            
            # Optimization packages
            ("numba", "numba"),
            ("Cython", "cython"),
            ("bottleneck", "bottleneck"),
            
            # Visualization
            ("matplotlib", "matplotlib"),
            ("seaborn", "seaborn"),
            ("plotly", "plotly"),
            ("bokeh", "bokeh"),
            
            # IO and data formats
            ("h5py", "h5py"),
            ("netCDF4", "netcdf4"),
            ("zarr", "zarr"),
            ("tables", "pytables"),
            
            # Domain-specific
            ("astropy", "astropy"),
            ("biopython", "biopython"),
            ("nilearn", "nilearn"),
        ]
        
        # Check each package
        for package_name, key_prefix in packages_to_check:
            self._check_package_availability(package_name, key_prefix, info)
            
        # NumPy-specific checks for optimized backends
        if info.get("numpy_available", False):
            try:
                import numpy as np
                
                # Check if NumPy is using MKL
                config_info = str(np.__config__.show())
                info["numpy_using_mkl"] = "mkl" in config_info.lower()
                info["numpy_using_openblas"] = "openblas" in config_info.lower()
                info["numpy_using_blis"] = "blis" in config_info.lower()
                info["numpy_blas_info"] = config_info
                
                # More detailed MKL detection
                if info["numpy_using_mkl"]:
                    try:
                        import mkl
                        info["mkl_version"] = mkl.get_version_string()
                        info["mkl_num_threads"] = mkl.get_max_threads()
                    except ImportError:
                        try:
                            # Alternative way to get MKL info
                            from ctypes import cdll, c_char_p, byref, create_string_buffer
                            mkl_rt = cdll.LoadLibrary("libmkl_rt.so")
                            buf = create_string_buffer(1024)
                            mkl_rt.MKL_Get_Version_String(byref(buf), 1024)
                            info["mkl_version"] = buf.value.decode('utf-8')
                        except Exception as e:
                            self._log_debug(f"Could not get detailed MKL info: {str(e)}", "software")
            except ImportError as e:
                self._log_warning(f"Error during NumPy config detection: {str(e)}", "software")
        
        # Check for Numba capabilities if available
        if info.get("numba_available", False):
            try:
                import numba
                
                info["numba_threading_layer"] = numba.config.THREADING_LAYER
                
                # Check if Numba can use CUDA
                try:
                    from numba import cuda
                    info["numba_cuda_available"] = cuda.is_available()
                    
                    if info["numba_cuda_available"]:
                        # Get CUDA devices
                        info["numba_cuda_devices"] = cuda.gpus.lst
                        
                        # Get compute capability for first device
                        if len(cuda.gpus) > 0:
                            cc_major, cc_minor = cuda.gpus[0].compute_capability
                            info["numba_cuda_compute_capability"] = f"{cc_major}.{cc_minor}"
                except (ImportError, AttributeError) as e:
                    info["numba_cuda_available"] = False
                    self._log_debug(f"Numba CUDA not available: {str(e)}", "software")
                    
                # Check Numba vectorization capabilities
                info["numba_has_svml"] = numba.config.USING_SVML
            except ImportError as e:
                self._log_warning(f"Error during Numba capability detection: {str(e)}", "software")
                
        # Check for Cython if available
        if info.get("cython_available", False):
            try:
                from Cython.Compiler.Version import version as cython_compiler_version
                info["cython_compiler_version"] = cython_compiler_version
            except ImportError:
                pass
        
        return info
    
    def _detect_ml_packages(self) -> Dict[str, Any]:
        """
        Detect machine learning packages and capabilities
        
        Returns:
            Dictionary with ML package information
        """
        info = {}
        
        # ML frameworks to check
        ml_packages = [
            # Deep Learning frameworks
            ("tensorflow", "tensorflow"),
            ("torch", "pytorch"),
            ("jax", "jax"),
            ("mxnet", "mxnet"),
            ("tvm", "tvm"),
            ("onnx", "onnx"),
            ("onnxruntime", "onnxruntime"),
            
            # ML libraries
            ("sklearn", "scikit_learn"),
            ("xgboost", "xgboost"),
            ("lightgbm", "lightgbm"),
            ("catboost", "catboost"),
            ("fastai", "fastai"),
            ("transformers", "huggingface_transformers"),
            
            # NLP libraries
            ("spacy", "spacy"),
            ("gensim", "gensim"),
            ("nltk", "nltk"),
            
            # Image processing
            ("cv2", "opencv"),
            ("skimage", "scikit_image"),
            ("PIL", "pillow"),
            
            # Reinforcement learning
            ("gym", "gym"),
            ("stable_baselines3", "stable_baselines3"),
        ]
        
        # Check each package
        for package_name, key_prefix in ml_packages:
            self._check_package_availability(package_name, key_prefix, info)
            
        # TensorFlow-specific checks
        if info.get("tensorflow_available", False):
            try:
                import tensorflow as tf
                
                # Check TensorFlow GPU support
                try:
                    info["tensorflow_gpu_available"] = len(tf.config.list_physical_devices("GPU")) > 0
                    info["tensorflow_gpu_devices"] = [d.name for d in tf.config.list_physical_devices("GPU")]
                    
                    # Get more device info
                    if info["tensorflow_gpu_available"]:
                        gpu_details = []
                        for i, device in enumerate(tf.config.list_physical_devices("GPU")):
                            try:
                                details = tf.config.experimental.get_device_details(device)
                                gpu_details.append(details)
                            except Exception:
                                gpu_details.append({"name": device.name, "index": i})
                        info["tensorflow_gpu_details"] = gpu_details
                except Exception as e:
                    info["tensorflow_gpu_available"] = False
                    self._log_debug(f"TensorFlow GPU detection error: {str(e)}", "software")
                    
                # Check for TF optimization flags
                info["tensorflow_xla_enabled"] = hasattr(tf, "function") and hasattr(tf.function, "experimental_compile")
                
                # Check for TensorFlow Lite
                try:
                    import tensorflow.lite
                    info["tensorflow_lite_available"] = True
                except (ImportError, AttributeError):
                    info["tensorflow_lite_available"] = False
                    
                # Check for TensorFlow Probability
                try:
                    import tensorflow_probability
                    info["tensorflow_probability_available"] = True
                    info["tensorflow_probability_version"] = tensorflow_probability.__version__
                except ImportError:
                    info["tensorflow_probability_available"] = False
            except Exception as e:
                self._log_warning(f"Error during TensorFlow capability detection: {str(e)}", "software")
                
        # PyTorch-specific checks
        if info.get("pytorch_available", False):
            try:
                import torch
                
                # Check PyTorch GPU support
                info["pytorch_gpu_available"] = torch.cuda.is_available()
                if info["pytorch_gpu_available"]:
                    info["pytorch_gpu_count"] = torch.cuda.device_count()
                    info["pytorch_gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None
                    info["pytorch_cuda_version"] = torch.version.cuda
                    
                    # Create list of available GPUs
                    pytorch_devices = []
                    for i in range(torch.cuda.device_count()):
                        pytorch_devices.append({
                            "index": i,
                            "name": torch.cuda.get_device_name(i),
                            "memory_allocated": torch.cuda.memory_allocated(i),
                            "memory_reserved": torch.cuda.memory_reserved(i),
                        })
                    info["pytorch_gpu_devices"] = pytorch_devices
                    
                # Check for MPS (Metal Performance Shaders) support on Apple Silicon
                try:
                    info["pytorch_mps_available"] = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                except Exception:
                    info["pytorch_mps_available"] = False
                    
                # Check for PyTorch XLA support
                try:
                    import torch_xla
                    info["pytorch_xla_available"] = True
                    info["pytorch_xla_version"] = torch_xla.__version__
                except ImportError:
                    info["pytorch_xla_available"] = False
                
                # Check for other PyTorch extensions
                try:
                    import torchvision
                    info["torchvision_available"] = True
                    info["torchvision_version"] = torchvision.__version__
                except ImportError:
                    info["torchvision_available"] = False
                    
                try:
                    import torchaudio
                    info["torchaudio_available"] = True
                    info["torchaudio_version"] = torchaudio.__version__
                except ImportError:
                    info["torchaudio_available"] = False
                    
                try:
                    import torchtext
                    info["torchtext_available"] = True
                    info["torchtext_version"] = torchtext.__version__
                except ImportError:
                    info["torchtext_available"] = False
            except Exception as e:
                self._log_warning(f"Error during PyTorch capability detection: {str(e)}", "software")
        
        # JAX-specific checks
        if info.get("jax_available", False):
            try:
                import jax
                
                # Check JAX backends
                try:
                    backend_name = jax.lib.xla_bridge.get_backend().platform
                    info["jax_backend"] = backend_name
                    info["jax_gpu_available"] = backend_name != "cpu"
                    
                    # Get device info
                    devices = jax.devices()
                    info["jax_device_count"] = len(devices)
                    info["jax_devices"] = [str(d) for d in devices]
                except Exception as e:
                    info["jax_backend"] = "unknown"
                    info["jax_gpu_available"] = False
                    self._log_debug(f"JAX backend detection error: {str(e)}", "software")
                
                # Check for JAX extensions
                try:
                    import flax
                    info["flax_available"] = True
                    info["flax_version"] = flax.__version__
                except ImportError:
                    info["flax_available"] = False
                    
                try:
                    import optax
                    info["optax_available"] = True
                    info["optax_version"] = optax.__version__
                except ImportError:
                    info["optax_available"] = False
                    
                try:
                    import haiku
                    info["haiku_available"] = True
                    info["haiku_version"] = haiku.__version__
                except ImportError:
                    info["haiku_available"] = False
            except Exception as e:
                self._log_warning(f"Error during JAX capability detection: {str(e)}", "software")
        
        return info
    
    def _detect_backend_support(self) -> Dict[str, Any]:
        """
        Detect specialized computational backends and libraries
        
        Returns:
            Dictionary with backend capabilities
        """
        info = {}
        
        # Check for BLAS/LAPACK libraries
        try:
            import scipy.linalg
            info["lapack_available"] = True
            info["lapack_opt_info"] = scipy.__config__.lapack_opt_info
            info["blas_opt_info"] = scipy.__config__.blas_opt_info
        except (ImportError, AttributeError):
            info["lapack_available"] = False
            info["blas_available"] = False
            
        # Check for specialized data processing libraries
        try:
            import datatable
            info["datatable_available"] = True
            info["datatable_version"] = datatable.__version__
        except ImportError:
            info["datatable_available"] = False
            
        # Check for modin (distributed pandas)
        try:
            import modin
            info["modin_available"] = True
            info["modin_version"] = modin.__version__
            
            # Check which engine modin is using
            try:
                import modin.config as cfg
                info["modin_engine"] = cfg.get_execution_engine()
            except Exception:
                pass
        except ImportError:
            info["modin_available"] = False
            
        # Check for cuDF (RAPIDS)
        try:
            import cudf
            info["cudf_available"] = True
            info["cudf_version"] = cudf.__version__
            
            # Check for other RAPIDS components
            try:
                import cuml
                info["cuml_available"] = True
                info["cuml_version"] = cuml.__version__
            except ImportError:
                info["cuml_available"] = False
                
            try:
                import cugraph
                info["cugraph_available"] = True
                info["cugraph_version"] = cugraph.__version__
            except ImportError:
                info["cugraph_available"] = False
        except ImportError:
            info["cudf_available"] = False
            
        # Check for Vaex (lazy out-of-core dataframes)
        try:
            import vaex
            info["vaex_available"] = True
            info["vaex_version"] = vaex.__version__
        except ImportError:
            info["vaex_available"] = False
            
        # Check for distributed computing frameworks
        try:
            import ray
            info["ray_available"] = True
            info["ray_version"] = ray.__version__
        except ImportError:
            info["ray_available"] = False
            
        # Check for CuPy
        try:
            import cupy
            info["cupy_available"] = True
            info["cupy_version"] = cupy.__version__
        except ImportError:
            info["cupy_available"] = False
            
        # Check for deep learning compiler frameworks
        try:
            import tvm
            info["tvm_available"] = True
            info["tvm_version"] = tvm.__version__
        except ImportError:
            info["tvm_available"] = False
            
        # Check for specialized visualization libraries
        try:
            import plotly
            info["plotly_available"] = True
            info["plotly_version"] = plotly.__version__
        except ImportError:
            info["plotly_available"] = False
            
        try:
            import bokeh
            info["bokeh_available"] = True
            info["bokeh_version"] = bokeh.__version__
        except ImportError:
            info["bokeh_available"] = False
            
        # Check for specialized scientific computing libraries
        try:
            import pyfftw
            info["pyfftw_available"] = True
            info["pyfftw_version"] = pyfftw.__version__
        except ImportError:
            info["pyfftw_available"] = False
            info["pyfftw_version"] = None

        # Check for GPU-accelerated FFT libraries
        try:
            import cufft
            info["cufft_available"] = True
        except ImportError:
            info["cufft_available"] = False

        # Check for MPI support
        try:
            import mpi4py
            info["mpi4py_available"] = True
            info["mpi4py_version"] = mpi4py.__version__
        except ImportError:
            info["mpi4py_available"] = False

        # Check for parallel processing libraries
        try:
            import joblib
            info["joblib_available"] = True
            info["joblib_version"] = joblib.__version__
        except ImportError:
            info["joblib_available"] = False

        # Check for symbolic computation libraries
        try:
            import sympy
            info["sympy_available"] = True
            info["sympy_version"] = sympy.__version__
        except ImportError:
            info["sympy_available"] = False

        # Check for probabilistic programming frameworks
        try:
            import pymc
            info["pymc_available"] = True
            info["pymc_version"] = pymc.__version__
        except ImportError:
            # Try older PyMC3
            try:
                import pymc3
                info["pymc_available"] = True
                info["pymc_version"] = "pymc3-" + pymc3.__version__
            except ImportError:
                info["pymc_available"] = False

        try:
            import numpyro
            info["numpyro_available"] = True
            info["numpyro_version"] = numpyro.__version__
        except ImportError:
            info["numpyro_available"] = False

        # Check for tensor optimization libraries
        try:
            import opt_einsum
            info["opt_einsum_available"] = True
            info["opt_einsum_version"] = opt_einsum.__version__
        except ImportError:
            info["opt_einsum_available"] = False

        # Check for caching and memoization libraries
        try:
            import functools
            info["functools_lru_cache_available"] = hasattr(functools, "lru_cache")
        except ImportError:
            info["functools_lru_cache_available"] = False

        # Check for quantization support
        if info.get("pytorch_available", False):
            try:
                import torch.quantization
                info["pytorch_quantization_available"] = True
            except ImportError:
                info["pytorch_quantization_available"] = False

        if info.get("tensorflow_available", False):
            try:
                # TensorFlow lite typically handles quantization
                import tensorflow.lite
                info["tensorflow_quantization_available"] = True
            except ImportError:
                info["tensorflow_quantization_available"] = False

        # Check for high-performance dataframe implementations
        try:
            import pyarrow.compute
            info["pyarrow_compute_available"] = True
        except ImportError:
            info["pyarrow_compute_available"] = False

        # Check for graph analytics libraries
        try:
            import networkx
            info["networkx_available"] = True
            info["networkx_version"] = networkx.__version__
        except ImportError:
            info["networkx_available"] = False

        # Check for high-performance statistics libraries
        try:
            import pingouin
            info["pingouin_available"] = True
            info["pingouin_version"] = pingouin.__version__
        except ImportError:
            info["pingouin_available"] = False

        return info

    def _check_package_availability(self, package_name: str, key_prefix: str, info: Dict[str, Any]) -> None:
        """
        Check if a package is available and get its version
        
        Args:
            package_name: Name of the package to import
            key_prefix: Prefix for the keys in the info dictionary
            info: Dictionary to update with package information
        """
        try:
            module = importlib.import_module(package_name)
            info[f"{key_prefix}_available"] = True
            
            # Try to get version
            try:
                version = getattr(module, "__version__", "unknown")
                info[f"{key_prefix}_version"] = version
            except AttributeError:
                info[f"{key_prefix}_version"] = "unknown"
                
        except ImportError:
            info[f"{key_prefix}_available"] = False
            info[f"{key_prefix}_version"] = None

    def get_summary(self, capabilities: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get a summary of software capabilities"""
        if capabilities is None:
            capabilities = self.capabilities if self.capabilities else self.detect()
            
        summary = {
            "data_processing": {
                "numpy": capabilities.get("numpy_available", False),
                "pandas": capabilities.get("pandas_available", False),
                "dask": capabilities.get("dask_available", False),
                "polars": capabilities.get("polars_available", False),
                "pyarrow": capabilities.get("pyarrow_available", False),
            },
            "machine_learning": {
                "tensorflow": capabilities.get("tensorflow_available", False),
                "pytorch": capabilities.get("pytorch_available", False),
                "sklearn": capabilities.get("scikit_learn_available", False),
                "xgboost": capabilities.get("xgboost_available", False),
                "jax": capabilities.get("jax_available", False),
            },
            "optimization": {
                "numba": capabilities.get("numba_available", False),
                "cython": capabilities.get("cython_available", False),
                "mkl": capabilities.get("numpy_using_mkl", False),
                "cuda": capabilities.get("cuda_available", False) or 
                        capabilities.get("pytorch_gpu_available", False) or 
                        capabilities.get("tensorflow_gpu_available", False),
            },
            "gpu_acceleration": {
                "cuda_available": capabilities.get("cuda_available", False) or 
                                  capabilities.get("pytorch_gpu_available", False) or 
                                  capabilities.get("tensorflow_gpu_available", False),
                "tensorflow_gpu": capabilities.get("tensorflow_gpu_available", False),
                "pytorch_gpu": capabilities.get("pytorch_gpu_available", False),
                "jax_gpu": capabilities.get("jax_gpu_available", False),
                "cupy": capabilities.get("cupy_available", False),
                "rapids": capabilities.get("cudf_available", False),
            }
        }
        
        # Calculate capability scores (simplified metric)
        has_gpu_ml = (capabilities.get("tensorflow_gpu_available", False) or 
                     capabilities.get("pytorch_gpu_available", False) or 
                     capabilities.get("jax_gpu_available", False))
                     
        has_advanced_cpu = (capabilities.get("numpy_using_mkl", False) or 
                           capabilities.get("numpy_using_openblas", False))
                           
        has_optimization = (capabilities.get("numba_available", False) or 
                           capabilities.get("cython_available", False))
        
        # Simple capability score from 0-10
        capability_score = 0
        
        # Basic packages
        if capabilities.get("numpy_available", False): capability_score += 1
        if capabilities.get("pandas_available", False): capability_score += 1
        
        # Advanced optimized packages
        if has_advanced_cpu: capability_score += 1
        if has_optimization: capability_score += 1
        
        # ML frameworks
        if capabilities.get("scikit_learn_available", False): capability_score += 1
        if capabilities.get("tensorflow_available", False): capability_score += 1
        if capabilities.get("pytorch_available", False): capability_score += 1
        
        # GPU acceleration
        if has_gpu_ml: capability_score += 2
        if capabilities.get("cudf_available", False): capability_score += 1
        
        # Distributed computing
        if capabilities.get("dask_available", False) or capabilities.get("ray_available", False): 
            capability_score += 1
        
        summary["capability_score"] = capability_score
        
        return summary