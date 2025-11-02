#!/usr/bin/env python3
"""
Generate compile_commands.json for RCCL-tests source code without building.
This enables code navigation features in editors using clangd.

Usage:
    python3 generate_compile_commands.py [RCCL_TESTS_ROOT_DIR]

If RCCL_TESTS_ROOT_DIR is not provided, it defaults to ../../../amd/rccl-tests relative to this script.
"""

import json
import os
import sys
from pathlib import Path

def find_rccl_tests_root():
    """Find RCCL-tests root directory"""
    if len(sys.argv) > 1:
        rt_root = Path(sys.argv[1]).absolute()
    else:
        # Default: assume script is in tools/indexing/rccl-tests/
        script_dir = Path(__file__).parent.absolute()
        rt_root = (script_dir / "../../../amd/rccl-tests").resolve()
    
    if not rt_root.exists():
        print(f"❌ Error: RCCL-tests root directory not found: {rt_root}")
        print(f"   Usage: python3 {sys.argv[0]} [RCCL_TESTS_ROOT_DIR]")
        sys.exit(1)
    
    src_dir = rt_root / "src"
    if not src_dir.exists():
        print(f"❌ Error: RCCL-tests src directory not found: {src_dir}")
        print(f"   Make sure {rt_root} is the correct RCCL-tests root directory")
        sys.exit(1)
    
    return rt_root

# Base directory for RCCL-tests
RT_ROOT = find_rccl_tests_root()
SRC_DIR = RT_ROOT / "src"

# Detect ROCm installation
def find_rocm_path():
    """Try to find ROCm installation"""
    candidates = [
        Path(os.environ.get("ROCM_PATH", "/opt/rocm")),
        Path("/opt/rocm"),
        Path("/opt/rocm-6.2.0"),
        Path("/opt/rocm-6.1.0"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path("/opt/rocm")  # fallback

ROCM_PATH = find_rocm_path()

# Try to find RCCL installation
def find_rccl_path():
    """Try to find RCCL installation"""
    candidates = [
        RT_ROOT.parent / "rccl",  # Sibling directory
        Path("/opt/rocm/rccl"),
        ROCM_PATH / "rccl",
    ]
    for candidate in candidates:
        if candidate.exists() and (candidate / "src").exists():
            return candidate
    return RT_ROOT.parent / "rccl"  # fallback to sibling

RCCL_PATH = find_rccl_path()

# Common compile flags for C++ files
BASE_CXX_FLAGS = [
    "-std=c++17",
    "-Wall",
    "-Wno-unused-function",
    "-fPIC",
    "-D__HIP_PLATFORM_AMD__",
    "-D__HIP_PLATFORM_HCC__",
    "-DROCM_VERSION=60200",
]

# HIP/CUDA flags for .cu files
CUDA_FLAGS = [
    "-x", "hip",
    "--cuda-gpu-arch=gfx90a",
    "-D__CUDACC__",
]

# Include directories
def get_include_dirs():
    """Generate list of include directories"""
    includes = [
        str(SRC_DIR),
        str(RT_ROOT),
        str(RCCL_PATH / "src"),
        str(RCCL_PATH / "src" / "include"),
        str(ROCM_PATH / "include"),
        str(ROCM_PATH / "include" / "hip"),
        str(ROCM_PATH / "include" / "hsa"),
        str(ROCM_PATH / "rccl" / "include"),
        "/usr/include",
        "/usr/local/include",
    ]
    return [i for i in includes if Path(i).exists() or i.startswith("/usr")]

INCLUDE_DIRS = get_include_dirs()

def get_source_files():
    """Collect all C/C++/CUDA source files"""
    source_files = []
    
    # Get files from src/ and verifiable/
    for directory in [SRC_DIR, RT_ROOT / "verifiable"]:
        if directory.exists():
            for ext in ['.cu', '.cc', '.cpp', '.c', '.h']:
                source_files.extend(directory.glob(f"*{ext}"))
    
    return source_files

def generate_compile_command(source_file):
    """Generate a compile command entry for a source file"""
    
    # Determine compiler based on file type
    if source_file.suffix in ['.cu', '.cuh']:
        compiler = str(ROCM_PATH / "bin" / "hipcc")
        if not Path(compiler).exists():
            compiler = str(ROCM_PATH / "llvm" / "bin" / "clang++")
            if not Path(compiler).exists():
                compiler = "clang++"
        arguments = [compiler] + BASE_CXX_FLAGS.copy()
        if compiler.endswith("clang++"):
            arguments.extend(CUDA_FLAGS)
    elif source_file.suffix == '.c':
        compiler = str(ROCM_PATH / "llvm" / "bin" / "clang")
        if not Path(compiler).exists():
            compiler = "clang"
        arguments = [compiler, "-std=c11"] + [f for f in BASE_CXX_FLAGS if f != "-std=c++17"]
    else:
        compiler = str(ROCM_PATH / "llvm" / "bin" / "clang++")
        if not Path(compiler).exists():
            compiler = "clang++"
        arguments = [compiler] + BASE_CXX_FLAGS.copy()
    
    # Add include directories
    for inc_dir in INCLUDE_DIRS:
        arguments.extend(["-I", inc_dir])
    
    # Add source file
    arguments.extend(["-c", str(source_file)])
    
    return {
        "directory": str(RT_ROOT),
        "command": " ".join(arguments),
        "file": str(source_file),
        "arguments": arguments,
    }

def main():
    """Generate compile_commands.json"""
    print(f"🔍 Scanning RCCL-tests source tree...")
    print(f"   RCCL-tests root: {RT_ROOT}")
    print(f"   ROCm path: {ROCM_PATH}")
    print(f"   RCCL path: {RCCL_PATH}")
    
    source_files = get_source_files()
    
    if not source_files:
        print(f"❌ Error: No source files found in {SRC_DIR}")
        sys.exit(1)
    
    print(f"   Found {len(source_files)} source files")
    
    compile_commands = []
    for source_file in sorted(source_files):
        compile_commands.append(generate_compile_command(source_file))
    
    output_file = RT_ROOT / "compile_commands.json"
    with open(output_file, 'w') as f:
        json.dump(compile_commands, f, indent=2)
    
    print(f"\n✅ Generated compile_commands.json with {len(compile_commands)} entries")
    print(f"✅ Output: {output_file}")
    print(f"\n💡 You can now use clangd for code navigation!")
    print(f"   Try opening {RT_ROOT} in your editor with clangd support.")

if __name__ == "__main__":
    main()

