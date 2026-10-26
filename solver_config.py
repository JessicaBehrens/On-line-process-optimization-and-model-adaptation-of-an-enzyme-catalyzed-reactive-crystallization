"""
Centralized configuration, directory setup, and dynamic HSL library path resolution for IPOPT.
"""
import os  
import sys  
from pathlib import Path  

# Dynamically resolve project directory at runtime without hardcoding folder names  
PROJECT_DIR = Path(__file__).resolve().parent  

# Ensure project directory is in sys.path 
if str(PROJECT_DIR) not in sys.path:  
    sys.path.insert(0, str(PROJECT_DIR))  

# Candidate search locations for libhsl.dll  
_env_hsl = os.environ.get("IPOPT_HSL_LIB")  
_local_hsl = PROJECT_DIR / "HSL_coin_library" / "libhsl.dll"  
_parent_hsl = PROJECT_DIR.parent / "HSL_coin_library" / "libhsl.dll"  
import ctypes  

_candidate_hsl = None  
if _env_hsl and os.path.exists(_env_hsl):  
    _candidate_hsl = str(Path(_env_hsl).resolve())  
elif _local_hsl.exists():  
    _candidate_hsl = str(_local_hsl.resolve())  
elif _parent_hsl.exists():  
    _candidate_hsl = str(_parent_hsl.resolve())  

HSL_LIB_PATH = None 
if _candidate_hsl: 
    _hsl_dir = str(Path(_candidate_hsl).parent) 
    # Register directory with Windows DLL loader for HSL dependencies (e.g. openblas, gfortran) 
    if hasattr(os, "add_dll_directory"): 
        try:  
            os.add_dll_directory(_hsl_dir)  
        except Exception:  
            pass  
    # Prepend to PATH environment variable as fallback for C dynamic linkers  
    os.environ["PATH"] = _hsl_dir + os.pathsep + os.environ.get("PATH", "")  
    # Verify library loadability via ctypes; fallback to MUMPS if missing dependencies  
    try:  
        ctypes.CDLL(_candidate_hsl)  
        HSL_LIB_PATH = _candidate_hsl  
    except Exception:  
        HSL_LIB_PATH = None  
