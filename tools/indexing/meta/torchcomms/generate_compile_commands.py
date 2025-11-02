#!/usr/bin/env python3
"""
Generate compile_commands.json for Meta TorchComms source code without building.
This enables code navigation features in editors using clangd.

Usage:
    python3 generate_compile_commands.py [TORCHCOMMS_ROOT_DIR]

If TORCHCOMMS_ROOT_DIR is not provided, it defaults to ../../../../meta/torchcomms relative to this script.
"""

import json
import os
import sys
from pathlib import Path

def find_torchcomms_root():
    """Find TorchComms root directory"""
    if len(sys.argv) > 1:
        tc_root = Path(sys.argv[1]).absolute()
    else:
        # Default: assume script is in tools/indexing/meta/torchcomms/
        script_dir = Path(__file__).parent.absolute()
        tc_root = (script_dir / "../../../../meta/torchcomms").resolve()
    
    if not tc_root.exists():
        print(f"❌ Error: TorchComms root directory not found: {tc_root}")
        print(f"   Usage: python3 {sys.argv[0]} [TORCHCOMMS_ROOT_DIR]")
        sys.exit(1)
    
    comms_dir = tc_root / "comms"
    if not comms_dir.exists():
        print(f"❌ Error: TorchComms comms directory not found: {comms_dir}")
        print(f"   Make sure {tc_root} is the correct TorchComms root directory")
        sys.exit(1)
    
    return tc_root

# Base directory for TorchComms
TC_ROOT = find_torchcomms_root()
COMMS_DIR = TC_ROOT / "comms"

# Detect CUDA installation
def find_cuda_path():
    """Try to find CUDA installation"""
    candidates = [
        Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")),
        Path("/usr/local/cuda"),
        Path("/usr/local/cuda-12.0"),
        Path("/usr/local/cuda-11.8"),
        Path("/opt/cuda"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path("/usr/local/cuda")  # fallback

CUDA_PATH = find_cuda_path()

# Detect ROCm installation (for HIP support)
def find_rocm_path():
    """Try to find ROCm installation"""
    candidates = [
        Path(os.environ.get("ROCM_PATH", "/opt/rocm")),
        Path("/opt/rocm"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None

ROCM_PATH = find_rocm_path()

# Common compile flags for C++ files
BASE_CXX_FLAGS = [
    "-std=c++17",
    "-Wall",
    "-Wno-unused-function",
    "-Wno-unused-variable",
    "-fPIC",
    "-D__CUDA_RUNTIME_API_H__",  # Avoid some CUDA header issues
]

# CUDA-specific flags
CUDA_FLAGS = [
    "-x", "cuda",
    "--cuda-gpu-arch=sm_80",  # Adjust for your GPU
    "-D__CUDACC__",
    "-D__NVCC__",
]

# Include directories
def get_include_dirs():
    """Generate list of include directories"""
    includes = [
        str(TC_ROOT),
        str(COMMS_DIR),
        str(COMMS_DIR / "ctran"),
        str(COMMS_DIR / "ctran" / "algos"),
        str(COMMS_DIR / "ctran" / "backends"),
        str(COMMS_DIR / "ctran" / "utils"),
        str(COMMS_DIR / "torchcomms"),
        str(COMMS_DIR / "utils"),
        str(COMMS_DIR / "common"),
        str(COMMS_DIR / "ncclx" / "headers"),
        str(CUDA_PATH / "include"),
        "/usr/include",
        "/usr/local/include",
    ]
    
    # Add ROCm includes if available
    if ROCM_PATH:
        includes.extend([
            str(ROCM_PATH / "include"),
            str(ROCM_PATH / "include" / "hip"),
        ])
    
    # Add common third-party locations
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        includes.append(str(Path(conda_prefix) / "include"))
    
    return [i for i in includes if Path(i).exists() or i.startswith("/usr")]

INCLUDE_DIRS = get_include_dirs()

def get_source_files():
    """Collect all C/C++/CUDA source files from the comms directory"""
    source_files = []
    
    # Recursively find all source files AND header files
    for ext in ['.cc', '.cpp', '.c', '.cu', '.cuh', '.h', '.hpp']:
        source_files.extend(COMMS_DIR.rglob(f"*{ext}"))
    
    # Filter out test files (optional, comment out if you want to index tests)
    # source_files = [f for f in source_files if '/tests/' not in str(f)]
    
    # Filter out benchmarks (optional)
    # source_files = [f for f in source_files if '/benchmarks/' not in str(f)]
    
    return source_files

def generate_compile_command(source_file):
    """Generate a compile command entry for a source file"""
    
    # Determine compiler and flags based on file type
    if source_file.suffix in ['.cu', '.cuh']:
        # CUDA files
        compiler = str(CUDA_PATH / "bin" / "nvcc")
        if not Path(compiler).exists():
            compiler = "clang++"  # clang++ can handle CUDA with appropriate flags
        arguments = [compiler] + BASE_CXX_FLAGS.copy()
        if compiler.endswith("clang++"):
            arguments.extend(CUDA_FLAGS)
    elif source_file.suffix == '.c':
        compiler = "clang"
        arguments = [compiler, "-std=c11"] + [f for f in BASE_CXX_FLAGS if f != "-std=c++17"]
    elif source_file.suffix == '.h':
        # .h files - treat as C header (clangd will determine from context)
        compiler = "clang++"
        arguments = [compiler] + BASE_CXX_FLAGS.copy()
    else:  # .cpp, .cc, .hpp files
        compiler = "clang++"
        arguments = [compiler] + BASE_CXX_FLAGS.copy()
    
    # Add include directories
    for inc_dir in INCLUDE_DIRS:
        arguments.extend(["-I", inc_dir])
    
    # Add source file
    arguments.extend(["-c", str(source_file)])
    
    return {
        "directory": str(TC_ROOT),
        "command": " ".join(arguments),
        "file": str(source_file),
        "arguments": arguments,
    }

def main():
    """Generate compile_commands.json"""
    print(f"🔍 Scanning TorchComms source tree...")
    print(f"   TorchComms root: {TC_ROOT}")
    print(f"   CUDA path: {CUDA_PATH}")
    if ROCM_PATH:
        print(f"   ROCm path: {ROCM_PATH}")
    
    source_files = get_source_files()
    
    if not source_files:
        print(f"❌ Error: No source files found in {COMMS_DIR}")
        sys.exit(1)
    
    print(f"   Found {len(source_files)} source files")
    
    compile_commands = []
    for source_file in sorted(source_files):
        compile_commands.append(generate_compile_command(source_file))
    
    output_file = TC_ROOT / "compile_commands.json"
    with open(output_file, 'w') as f:
        json.dump(compile_commands, f, indent=2)
    
    print(f"\n✅ Generated compile_commands.json with {len(compile_commands)} entries")
    print(f"✅ Output: {output_file}")
    print(f"\n💡 You can now use clangd for code navigation!")
    print(f"   Try opening {TC_ROOT} in your editor with clangd support.")

if __name__ == "__main__":
    main()



