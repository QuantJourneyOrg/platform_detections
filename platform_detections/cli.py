"""
Platform Detection Framework - Command Line Interface

This module provides a command-line interface to the platform detection
framework, allowing users to detect and report on platform capabilities.
"""

import argparse
import logging
import sys
import json
from typing import Dict, Any, Optional

from . import get_detector
from .orchestrator import PlatformOrchestrator


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Platform Detection Framework - Detect and optimize for platform capabilities"
    )
    
    parser.add_argument(
        "--json", 
        action="store_true", 
        help="Output as JSON"
    )
    
    parser.add_argument(
        "--file", 
        type=str, 
        help="Save output to specified file"
    )
    
    parser.add_argument(
        "--summary", 
        action="store_true", 
        help="Show summary instead of full details"
    )
    
    parser.add_argument(
        "--quiet", 
        action="store_true", 
        help="Suppress warnings and info messages"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Show verbose debug output"
    )
    
    parser.add_argument(
        "--format", 
        choices=["text", "json", "yaml"], 
        default="text",
        help="Output format (default: text)"
    )
    
    parser.add_argument(
        "--backends", 
        action="store_true", 
        help="Show only available computation backends"
    )
    
    parser.add_argument(
        "--recommend", 
        type=str,
        help="Recommend backend for operation type (e.g., matrix, stat, ml)"
    )
    
    parser.add_argument(
        "--version", 
        action="store_true", 
        help="Show version information"
    )
    
    return parser.parse_args()


def format_backends(detector: PlatformOrchestrator) -> str:
    """
    Format available backends information
    
    Args:
        detector: PlatformOrchestrator instance
        
    Returns:
        Formatted string with backends information
    """
    from platform_detection.orchestrator import ComputeBackend
    
    # Get the optimal backend
    optimal = detector.get_optimal_backend()
    
    # Get recommendations for different operation types
    recommendations = {
        "matrix": detector.get_backend_for_operation("matrix", 1000000),
        "stat": detector.get_backend_for_operation("stat", 100000),
        "ml": detector.get_backend_for_operation("ml", 1000000),
        "data": detector.get_backend_for_operation("data", 1000000),
        "finance": detector.get_backend_for_operation("finance", 100000),
    }
    
    # Format the output
    output = ["Available Computation Backends", "-----------------------------"]
    output.append(f"Optimal general backend: {optimal.value}")
    output.append("\nRecommended backends by operation type:")
    
    for op_type, backend in recommendations.items():
        output.append(f"  {op_type.ljust(10)}: {backend.value}")
        
    return "\n".join(output)


def show_summary(detector: PlatformOrchestrator) -> None:
    """
    Show a summary of detected capabilities
    
    Args:
        detector: PlatformOrchestrator instance
    """
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
        macos = os_summary.get("macos", {})
        if macos:
            print(f"  macOS: {macos.get('name', 'Unknown')} {macos.get('version', '')}")
            if macos.get("is_apple_silicon", False):
                print(f"  Chip: {macos.get('apple_chip', 'Apple Silicon')}")
                cores = macos.get("cores", {})
                print(f"  Cores: {cores.get('performance', 0)} performance, {cores.get('efficiency', 0)} efficiency")
            frameworks = os_summary.get("frameworks", {})
            if frameworks:
                print(f"  Frameworks: {'Metal, ' if frameworks.get('metal', False) else ''}{'Accelerate' if frameworks.get('accelerate', False) else ''}")
        
    elif os_name == "Linux":
        distro = os_summary.get("distribution", {})
        if distro:
            print(f"  Distribution: {distro.get('name', 'Unknown')} {distro.get('version', '')}")
            desktop = os_summary.get("desktop", {})
            if desktop:
                print(f"  Desktop: {desktop.get('environment', 'Unknown')}")
        
        # Print available BLAS implementations
        optimizations = os_summary.get("optimizations", {})
        if optimizations:
            blas_impls = []
            for blas, available in optimizations.items():
                if available and blas in ["mkl", "openblas", "atlas"]:
                    blas_impls.append(blas.upper())
            if blas_impls:
                print(f"  BLAS: {', '.join(blas_impls)}")
        
    elif os_name == "Windows":
        windows = os_summary.get("windows", {})
        if windows:
            print(f"  Windows: {windows.get('name', 'Unknown')} {windows.get('edition', '')}")
            print(f"  Build: {windows.get('build', 'Unknown')}")
        
        graphics = os_summary.get("graphics", {})
        if graphics:
            directx = graphics.get("directx", {})
            if directx.get("available", False):
                print(f"  DirectX: {directx.get('version', 'Unknown')}")
        
        wsl = os_summary.get("wsl", {})
        if wsl and wsl.get("available", False):
            print(f"  WSL: Version {wsl.get('version', 'Unknown')}")
            if wsl.get("distros"):
                print(f"  WSL Distros: {', '.join(wsl.get('distros', []))}")
                
    # Optimal backend
    print(f"\nOptimal computation backend: {summary['optimal_backend']}")


def main() -> int:
    """
    Main entry point for the command-line interface
    
    Returns:
        Exit code
    """
    import platform_detection
    
    # Parse arguments
    args = parse_args()
    
    # Show version and exit if requested
    if args.version:
        print(f"Platform Detection Framework v{platform_detection.__version__}")
        return 0
    
    # Set log level based on arguments
    if args.verbose:
        log_level = logging.DEBUG
    elif args.quiet:
        log_level = logging.ERROR
    else:
        log_level = logging.INFO
    
    # Initialize detector
    detector = get_detector(log_level=log_level, enable_warnings=not args.quiet)
    
    # Show only backends if requested
    if args.backends:
        print(format_backends(detector))
        return 0
    
    # Recommend backend for specific operation
    if args.recommend:
        backend = detector.get_backend_for_operation(args.recommend, 100000)
        print(f"Recommended backend for {args.recommend} operations: {backend.value}")
        return 0
    
    # Determine output format
    if args.json or args.format == "json":
        # Output as JSON
        if args.file:
            detector.json_dump(args.file)
            print(f"Platform capabilities saved to {args.file}")
        else:
            # Print JSON to stdout
            print(detector.json_dump())
    elif args.format == "yaml":
        # Output as YAML if PyYAML is available
        try:
            import yaml
            if args.file:
                with open(args.file, 'w') as f:
                    yaml.dump(detector.capabilities, f, default_flow_style=False)
                print(f"Platform capabilities saved to {args.file}")
            else:
                print(yaml.dump(detector.capabilities, default_flow_style=False))
        except ImportError:
            print("PyYAML is required for YAML output. Install with 'pip install PyYAML'.")
            return 1
    elif args.summary:
        # Show summary
        show_summary(detector)
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
    
    return 0


if __name__ == "__main__":
    sys.exit(main())