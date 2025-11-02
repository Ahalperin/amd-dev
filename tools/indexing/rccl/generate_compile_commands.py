#!/usr/bin/env python3
"""
Generate compile_commands.json for RCCL source code without building.
This enables code navigation features in editors using clangd.

Usage:
    python3 generate_compile_commands.py [RCCL_ROOT_DIR]

If RCCL_ROOT_DIR is not provided, it defaults to ../../../amd/rccl relative to this script.
"""

import json
import os
import sys
from pathlib import Path

def find_rccl_root():
    """Find RCCL root directory"""
    if len(sys.argv) > 1:
        rccl_root = Path(sys.argv[1]).absolute()
    else:
        # Default: assume script is in tools/indexing/rccl/
        script_dir = Path(__file__).parent.absolute()
        rccl_root = (script_dir / "../../../amd/rccl").resolve()
    
    if not rccl_root.exists():
        print(f"❌ Error: RCCL root directory not found: {rccl_root}")
        print(f"   Usage: python3 {sys.argv[0]} [RCCL_ROOT_DIR]")
        sys.exit(1)
    
    src_dir = rccl_root / "src"
    if not src_dir.exists():
        print(f"❌ Error: RCCL src directory not found: {src_dir}")
        print(f"   Make sure {rccl_root} is the correct RCCL root directory")
        sys.exit(1)
    
    return rccl_root

# Base directory for RCCL
RCCL_ROOT = find_rccl_root()
SRC_DIR = RCCL_ROOT / "src"

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

# Common compile flags for C++ files
BASE_CXX_FLAGS = [
    "-std=c++17",
    "-Wall",
    "-Wno-format-nonliteral",
    "-Wno-unused-function",
    "-fPIC",
    "-D__HIP_PLATFORM_AMD__",
    "-D__HIP_PLATFORM_HCC__",
    "-DROCM_VERSION=60200",
    "-DENABLE_COLLTRACE",
    "-DCOMPILE_MSCCL_KERNEL",
    "-DHIP_UNCACHED_MEMORY",
    "-DENABLE_LL128",
    "-DROCTX_ENABLE",
    "-DENABLE_PROFILING",
]

# Include directories
def get_include_dirs():
    """Generate list of include directories"""
    includes = [
        str(SRC_DIR),
        str(SRC_DIR / "include"),
        str(SRC_DIR / "device"),
        str(SRC_DIR / "device" / "network" / "unpack"),
        str(SRC_DIR / "include" / "mlx5"),
        str(SRC_DIR / "include" / "plugin"),
        str(SRC_DIR / "include" / "nvtx3"),
        str(SRC_DIR / "include" / "msccl"),
        str(SRC_DIR / "include" / "npkit"),
        str(SRC_DIR / "include" / "latency_profiler"),
        str(SRC_DIR / "include" / "proxy_trace"),
        str(SRC_DIR / "include" / "mscclpp"),
        str(ROCM_PATH / "include"),
        str(ROCM_PATH / "include" / "hip"),
        str(ROCM_PATH / "include" / "hsa"),
        str(ROCM_PATH / "include" / "rocm_smi"),
        str(ROCM_PATH / "rocm_smi" / "include"),
        "/usr/include",
        "/usr/local/include",
    ]
    return [i for i in includes if Path(i).exists() or i.startswith("/usr")]

INCLUDE_DIRS = get_include_dirs()

def get_source_files():
    """Collect all C/C++ source files AND header files from the src directory"""
    source_files = []
    
    # Include source files AND header files for complete indexing
    for ext in ['.cc', '.cpp', '.c', '.cu', '.cuh', '.h', '.hpp']:
        source_files.extend(SRC_DIR.rglob(f"*{ext}"))
    
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
    elif source_file.suffix == '.c':
        compiler = str(ROCM_PATH / "llvm" / "bin" / "clang")
        if not Path(compiler).exists():
            compiler = "clang"
    elif source_file.suffix == '.h':
        # .h files - treat as C++ header (clangd will determine from context)
        compiler = str(ROCM_PATH / "llvm" / "bin" / "clang++")
        if not Path(compiler).exists():
            compiler = "clang++"
    else:  # .cpp, .cc, .hpp files
        compiler = str(ROCM_PATH / "llvm" / "bin" / "clang++")
        if not Path(compiler).exists():
            compiler = "clang++"
    
    # Build command arguments
    arguments = [compiler] + BASE_CXX_FLAGS.copy()
    
    # Add include directories
    for inc_dir in INCLUDE_DIRS:
        arguments.extend(["-I", inc_dir])
    
    # Add source file
    arguments.extend(["-c", str(source_file)])
    
    return {
        "directory": str(RCCL_ROOT),
        "command": " ".join(arguments),
        "file": str(source_file),
        "arguments": arguments,
    }

def main():
    """Generate compile_commands.json"""
    print(f"🔍 Scanning RCCL source tree...")
    print(f"   RCCL root: {RCCL_ROOT}")
    print(f"   ROCm path: {ROCM_PATH}")
    
    source_files = get_source_files()
    
    if not source_files:
        print(f"❌ Error: No source files found in {SRC_DIR}")
        sys.exit(1)
    
    print(f"   Found {len(source_files)} source files")
    
    compile_commands = []
    for source_file in sorted(source_files):
        compile_commands.append(generate_compile_command(source_file))
    
    output_file = RCCL_ROOT / "compile_commands.json"
    with open(output_file, 'w') as f:
        json.dump(compile_commands, f, indent=2)
    
    print(f"\n✅ Generated compile_commands.json with {len(compile_commands)} entries")
    print(f"✅ Output: {output_file}")
    print(f"\n💡 You can now use clangd for code navigation!")
    print(f"   Try opening {RCCL_ROOT} in your editor with clangd support.")

if __name__ == "__main__":
    main()





