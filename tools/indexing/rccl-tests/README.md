# RCCL-tests Code Indexing (No Build Required)

This directory contains tools to enable code navigation in RCCL-tests without requiring a full build. These tools generate a compilation database (`compile_commands.json`) that language servers like clangd can use to understand your code.

## Quick Start

```bash
# From this directory
./setup.sh

# Or specify a custom RCCL-tests directory
./setup.sh /path/to/rccl-tests
```

That's it! The script will deploy all necessary files and generate the compilation database.

## What Gets Installed

The setup script will create/copy these files to your RCCL-tests directory:

- `compile_commands.json` - Compilation database for clangd
- `.clangd` - clangd configuration
- `.vscode/settings.json` - VSCode settings for clangd
- `README-INDEXING.md` - User documentation

## About RCCL-tests

RCCL-tests is the official test suite for RCCL (ROCm Communication Collectives Library). It includes:
- Performance tests for all collective operations
- Correctness verification tests
- Bandwidth and latency benchmarks

This indexing setup covers all C++/CUDA test files in the `src/` and `verifiable/` directories.

## Prerequisites

### 1. clangd (Required)

**Install on macOS:**
```bash
brew install llvm
```

**Install on Ubuntu/Debian:**
```bash
sudo apt install clangd-15
```

**Verify installation:**
```bash
which clangd
clangd --version
```

### 2. Editor Configuration

#### VSCode (Recommended)

1. Install the **clangd** extension by LLVM
2. Disable the C/C++ extension (or disable for this workspace)
3. Reload the window

## Available Features

✅ **Jump to Definition** (`F12`)
- Jump from test code to RCCL functions
- Navigate between test files

✅ **Find All References** (`Shift+F12`)
- See all usages of test functions
- Find where collectives are tested

✅ **Hover Documentation**
- See function signatures and types
- Understand test parameters

✅ **Code Completion** (`Ctrl+Space`)
- Autocomplete RCCL API calls
- Suggest test helper functions

✅ **Symbol Search** (`Cmd+T`)
- Quickly find any test
- Search for collective operations

## RCCL-tests-Specific Features

### Navigate Test Files

The tests are organized by collective operation:

- `all_reduce.cu` - AllReduce tests
- `all_gather.cu` - AllGather tests
- `reduce_scatter.cu` - ReduceScatter tests
- `broadcast.cu` - Broadcast tests
- `alltoall.cu` - AllToAll tests
- `sendrecv.cu` - Send/Recv tests

Use F12 to jump between test implementations and RCCL API calls.

### Understand Test Parameters

Hover over test functions to see:
- Data types being tested
- Operation types (sum, prod, min, max)
- Communication patterns
- Performance metrics

### Trace RCCL API Usage

From test files, use F12 on RCCL functions like:
- `ncclAllReduce`
- `ncclAllGather`
- `ncclBroadcast`

This jumps to RCCL implementation (if RCCL is indexed too).

## Manual Setup (Advanced)

If you prefer not to use the setup script:

```bash
# 1. Generate compile_commands.json
python3 generate_compile_commands.py /path/to/rccl-tests

# 2. Copy configuration files
cp clangd-config /path/to/rccl-tests/.clangd
cp vscode-settings.json /path/to/rccl-tests/.vscode/settings.json
```

## Troubleshooting

### clangd not finding RCCL headers

The script automatically looks for RCCL in:
1. Sibling directory (`../rccl`)
2. `/opt/rocm/rccl`

If RCCL is elsewhere, edit `generate_compile_commands.py`:

```python
RCCL_PATH = Path("/your/rccl/path")
```

Then regenerate:
```bash
python3 generate_compile_commands.py
```

### Headers not found

Check ROCm installation:
```bash
echo $ROCM_PATH
ls /opt/rocm/include
```

Update `ROCM_PATH` in `generate_compile_commands.py` if needed.

### Slow initial indexing

First-time indexing takes 1-3 minutes. Subsequent opens are much faster (cached).

## Updating the Index

When RCCL-tests source changes:

```bash
cd /path/to/tools/indexing/rccl-tests
python3 generate_compile_commands.py
```

clangd will automatically detect the change and re-index.

## Files in This Directory

- `generate_compile_commands.py` - Main script to generate compilation database
- `clangd-config` - Configuration for clangd
- `vscode-settings.json` - VSCode settings
- `setup.sh` - One-command setup script
- `verify.sh` - Verification script
- `README.md` - This file
- `QUICK-START.md` - Quick reference

## Common Use Cases

### 1. Understanding Test Structure

```bash
# Open a test file
code src/all_reduce.cu

# Navigate to test setup
# Press F12 on testCollective()
# See how tests are structured
```

### 2. Finding Specific Tests

```bash
# Press Cmd+T in VSCode
# Type "test" or operation name
# Jump to specific test implementation
```

### 3. Tracing RCCL API Calls

```bash
# In any test file
# Click on ncclAllReduce, ncclAllGather, etc.
# Press F12
# (Jumps to RCCL implementation if indexed)
```

## See Also

- [RCCL indexing](../rccl/README.md)
- [RCCL-tests GitHub](https://github.com/ROCmSoftwarePlatform/rccl-tests)
- [clangd documentation](https://clangd.llvm.org/)

