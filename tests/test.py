#!/usr/bin/env python3
"""
Test script for Platform Detection Framework
"""

import os
import sys

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from platform_detections import get_detector

def main():
    # Create a detector with more verbose output
    print("Initializing Platform Detection Framework...")
    detector = get_detector(force_detect=True, enable_warnings=True)
    
    # Get basic capabilities
    print("\nDetecting platform capabilities...")
    capabilities = detector.detect()
    
    # Show a summary of what was detected
    print("\n--- Platform Summary ---")
    print(f"OS: {detector.os_name.capitalize()}")
    
    # Get hardware summary
    hw_summary = detector.get_hardware_summary()
    if hw_summary:
        print("\n--- Hardware ---")
        print(f"CPU: {hw_summary.get('cpu', {}).get('brand', 'Unknown')}")
        print(f"Cores: {hw_summary.get('cpu', {}).get('cores', 0)}")
        
        if hw_summary.get('gpu', {}).get('available', False):
            print(f"GPU: {hw_summary.get('gpu', {}).get('vendor', 'Unknown')} {hw_summary.get('gpu', {}).get('name', 'Unknown')}")
        else:
            print("GPU: None detected")
    
    # Get software summary
    sw_summary = detector.get_software_summary()
    if sw_summary:
        print("\n--- Software ---")
        data_packages = sw_summary.get('data_processing', {})
        ml_packages = sw_summary.get('machine_learning', {})
        
        data_pkgs_available = [pkg for pkg, available in data_packages.items() if available]
        ml_pkgs_available = [pkg for pkg, available in ml_packages.items() if available]
        
        if data_pkgs_available:
            print(f"Data packages: {', '.join(data_pkgs_available)}")
        if ml_pkgs_available:
            print(f"ML packages: {', '.join(ml_pkgs_available)}")
    
    # Get OS-specific summary
    os_summary = detector.get_os_summary()
    if os_summary:
        print(f"\n--- {detector.os_name.capitalize()} Specific ---")
        if detector.os_name == "darwin":
            macos = os_summary.get('macos', {})
            if macos:
                print(f"macOS: {macos.get('name', '')} {macos.get('version', '')}")
                if os_summary.get('has_apple_silicon', False):
                    print(f"Chip: {os_summary.get('apple_chip', 'Apple Silicon')}")
        elif detector.os_name == "linux":
            distro = os_summary.get('distribution', {})
            if distro:
                print(f"Distribution: {distro.get('name', '')} {distro.get('version', '')}")
                print(f"Desktop: {os_summary.get('desktop', {}).get('environment', '')}")
        elif detector.os_name == "windows":
            windows = os_summary.get('windows', {})
            if windows:
                print(f"Windows: {windows.get('name', '')} {windows.get('edition', '')}")
                print(f"Build: {windows.get('build', '')}")
    
    # Get optimal backend
    print("\n--- Optimization ---")
    backend = detector.get_optimal_backend()
    print(f"Optimal backend: {backend.value}")
    
    # Save results to a file
    output_file = "platform_detection_results.json"
    detector.json_dump(output_file)
    print(f"\nDetailed results saved to: {output_file}")

if __name__ == "__main__":
    main()