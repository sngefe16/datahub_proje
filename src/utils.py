"""
Utility functions for cross-platform compatibility.
Handles OS-specific paths and configurations.
"""

import os
import sys
from pathlib import Path
from typing import Optional


def get_project_root() -> Path:
    """
    Get the project root directory.
    Works on Windows, Linux, and macOS.
    
    Returns
    -------
    Path
        Project root directory path
    """
    # Get the directory of this file
    current_file = Path(__file__).resolve()
    # Go up to project root (src -> project root)
    project_root = current_file.parent.parent
    return project_root


def get_data_dir() -> Path:
    """
    Get the data directory path.
    
    Returns
    -------
    Path
        Data directory path
    """
    return get_project_root() / "data"


def get_models_dir() -> Path:
    """
    Get the models directory path.
    Creates directory if it doesn't exist.
    
    Returns
    -------
    Path
        Models directory path
    """
    models_dir = get_project_root() / "models"
    models_dir.mkdir(exist_ok=True)
    return models_dir


def get_results_dir() -> Path:
    """
    Get the results directory path.
    Creates directory if it doesn't exist.
    
    Returns
    -------
    Path
        Results directory path
    """
    results_dir = get_project_root() / "results"
    results_dir.mkdir(exist_ok=True)
    return results_dir


def get_reports_dir() -> Path:
    """
    Get the reports directory path.
    Creates directory if it doesn't exist.
    
    Returns
    -------
    Path
        Reports directory path
    """
    reports_dir = get_project_root() / "reports"
    reports_dir.mkdir(exist_ok=True)
    return reports_dir


def get_platform_info() -> dict:
    """
    Get platform information.
    
    Returns
    -------
    dict
        Platform information including OS, Python version, etc.
    """
    return {
        'os': sys.platform,
        'os_name': os.name,
        'python_version': sys.version,
        'python_version_info': sys.version_info,
        'is_windows': sys.platform.startswith('win'),
        'is_linux': sys.platform.startswith('linux'),
        'is_macos': sys.platform == 'darwin',
    }


def ensure_dir(path: Path) -> Path:
    """
    Ensure a directory exists, create if it doesn't.
    
    Parameters
    ----------
    path : Path
        Directory path
        
    Returns
    -------
    Path
        The directory path
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_separator() -> str:
    """
    Get the path separator for the current platform.
    
    Returns
    -------
    str
        Path separator ('/' for Unix, '\\' for Windows)
    """
    return os.sep


if __name__ == "__main__":
    # Test utility functions
    print("Platform Info:")
    info = get_platform_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print(f"\nProject Root: {get_project_root()}")
    print(f"Data Dir: {get_data_dir()}")
    print(f"Models Dir: {get_models_dir()}")
    print(f"Results Dir: {get_results_dir()}")
    print(f"Path Separator: '{get_separator()}'")









