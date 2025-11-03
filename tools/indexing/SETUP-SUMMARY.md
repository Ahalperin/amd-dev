# RCCL Code Indexing Setup - Summary

**Created:** October 30, 2025  
**Purpose:** Enable code navigation in RCCL without building  
**Status:** ✅ Complete and tested

## What Was Created

### 1. Indexing Tools Directory

**Location:** `/Users/ahalperin/xai/amd-dev/tools/indexing/rccl/`

Files created:
- ✅ `generate_compile_commands.py` - Python script to generate compilation database
- ✅ `setup.sh` - One-command deployment script  
- ✅ `clangd-config` - Template for `.clangd` configuration
- ✅ `vscode-settings.json` - Template for VSCode settings
- ✅ `README.md` - Comprehensive documentation (8.3KB)
- ✅ `QUICK-START.md` - Quick reference guide

All scripts are executable and tested.

### 2. Deployed Files in RCCL

**Location:** `/Users/ahalperin/xai/amd-dev/amd/rccl/`

Files deployed by `setup.sh`:
- ✅ `compile_commands.json` (255KB) - Compilation database with 96 entries
- ✅ `.clangd` - clangd configuration
- ✅ `.vscode/settings.json` - VSCode settings (backup created if existing)
- ✅ `README-INDEXING.md` - User documentation

### 3. Parent README

**Location:** `/Users/ahalperin/xai/amd-dev/tools/indexing/README.md`

Overview of all indexing tools with instructions for adding new projects.

## How to Use

### Option 1: Automated Setup (Recommended)

```bash
cd /Users/ahalperin/xai/amd-dev/tools/indexing/rccl
./setup.sh
```

### Option 2: Manual Steps

```bash
# 1. Generate compilation database
cd /Users/ahalperin/xai/amd-dev/tools/indexing/rccl
python3 generate_compile_commands.py

# 2. Copy configuration files
cp clangd-config /Users/ahalperin/xai/amd-dev/amd/rccl/.clangd
mkdir -p /Users/ahalperin/xai/amd-dev/amd/rccl/.vscode
cp vscode-settings.json /Users/ahalperin/xai/amd-dev/amd/rccl/.vscode/settings.json

# 3. Open in editor
code /Users/ahalperin/xai/amd-dev/amd/rccl
```

## Features Enabled

✅ **Go to Definition** (F12)
- Click on any function/variable and jump to its definition
- Works across files

✅ **Find All References** (Shift+F12)  
- See everywhere a symbol is used
- Shows in sidebar with context

✅ **Call Hierarchy**
- See who calls a function
- See what a function calls

✅ **Hover Documentation**
- Hover over symbols to see types and signatures
- No need to open header files

✅ **Symbol Search** (Cmd+T)
- Quickly find any function, class, or variable
- Fuzzy search support

✅ **Code Completion** (Ctrl+Space)
- Context-aware autocompletion
- Shows function signatures

✅ **Rename Refactoring** (F2)
- Safely rename across all files
- Updates references automatically

## Technical Details

### How It Works

1. **Compilation Database Generation**
   - Scans RCCL source tree (`src/`)
   - Finds all `.cc`, `.cpp`, `.c`, `.cu` files
   - Generates JSON entries with compile flags for each file
   - Includes proper include paths and defines

2. **clangd Integration**
   - Reads `compile_commands.json`
   - Parses source files (no compilation)
   - Builds symbol index
   - Responds to LSP requests from editor

3. **Editor Communication**
   - Editor sends requests via Language Server Protocol
   - clangd processes and responds
   - Editor displays results

### Source Files Indexed

- **96 source files** in total:
  - `src/*.cc` - Main implementation files
  - `src/misc/*.cc` - Utility implementations
  - `src/graph/*.cc` - Graph algorithms
  - `src/plugin/*.cc` - Plugin system
  - `src/transport/*.cc` - Transport layer
  - `src/device/*.cu` - GPU kernels
  - And more...

### Include Paths Configured

- `src/` and subdirectories
- `src/include/` - Main headers
- `src/device/` - Device code
- `src/include/mlx5/` - Mellanox headers
- `src/include/plugin/` - Plugin API
- `/opt/rocm/include/` - ROCm headers
- System includes

### Compiler Flags Set

- `-std=c++17` - C++17 standard
- `-D__HIP_PLATFORM_AMD__` - HIP platform defines
- `-DENABLE_COLLTRACE` - RCCL features
- `-DENABLE_LL128` - Low-latency protocol
- And more RCCL-specific defines

## Prerequisites Verified

✅ **clangd** - Found at `/usr/bin/clangd`
- Version: Apple clangd version 16.0.0
- Sufficient for code navigation

✅ **Python 3** - Required for generation script
- Used to scan source tree and generate JSON

✅ **ROCm** - Located at `/opt/rocm`
- Headers referenced in include paths
- Not required to be installed, just referenced

## Performance Characteristics

### Initial Indexing
- **Time:** 2-10 minutes (one-time)
- **CPU:** High during indexing
- **Memory:** ~500MB-2GB depending on project size
- **Disk:** Creates `.cache/clangd/` directory

### Subsequent Opens
- **Time:** <1 second (cached)
- **CPU:** Low
- **Memory:** ~200MB baseline
- **Disk:** Reuses cache

## Verification Steps

```bash
# 1. Check compilation database exists and has entries
test -f /Users/ahalperin/xai/amd-dev/amd/rccl/compile_commands.json && echo "✅ Found"
grep -c '"file":' /Users/ahalperin/xai/amd-dev/amd/rccl/compile_commands.json

# 2. Check configuration files
test -f /Users/ahalperin/xai/amd-dev/amd/rccl/.clangd && echo "✅ .clangd found"
test -f /Users/ahalperin/xai/amd-dev/amd/rccl/.vscode/settings.json && echo "✅ VSCode settings found"

# 3. Verify clangd can read it
cd /Users/ahalperin/xai/amd-dev/amd/rccl
clangd --check=src/rccl_wrap.cc 2>&1 | head -5
```

## Common Use Cases

### 1. Understanding a Function Call Chain

```
User opens: src/collectives.cc
Sees: ncclAllReduce()
Press F12 → Jump to definition
Press Shift+F12 → See all callers
Press Cmd+Click on functions it calls → Explore implementation
```

### 2. Finding All Uses of a Type

```
User opens: src/include/comm.h  
Finds: struct ncclComm
Press Shift+F12 → See everywhere ncclComm is used
```

### 3. Exploring Unknown Codebase

```
Press Cmd+T → Symbol search
Type: "proxy"
See: proxyMain, proxyConnect, proxyState, etc.
Select one → F12 to see implementation
```

## Files NOT Included

The following are intentionally excluded:
- Build artifacts (`build/`, `hipify/`)
- Generated files (created during build)
- Test binaries
- Object files (`.o`)
- Build configuration (except what's in compile_commands.json)

## Maintenance

### Updating the Index

When RCCL source changes (new files, restructuring):

```bash
cd /Users/ahalperin/xai/amd-dev/tools/indexing/rccl
python3 generate_compile_commands.py
```

clangd will automatically detect the change and re-index.

### Adding More Projects

To add indexing for other codebases (nccl, rccl-tests, etc.):

1. Copy `tools/indexing/rccl/` as template
2. Modify `generate_compile_commands.py` for new project structure
3. Update paths and include directories
4. Test and document

## Troubleshooting Reference

| Issue | Solution |
|-------|----------|
| clangd not found | `brew install llvm` (macOS) or `apt install clangd` (Ubuntu) |
| Headers not found | Edit `generate_compile_commands.py`, update `ROCM_PATH` |
| Slow indexing | Normal on first run, cached afterwards |
| Multiple diagnostics | Disable C/C++ extension in VSCode |
| GPU code errors | Expected, clangd not full HIP compiler |

## Documentation Locations

- **Quick start:** `tools/indexing/rccl/QUICK-START.md`
- **Full docs:** `tools/indexing/rccl/README.md`
- **User guide:** `amd/rccl/README-INDEXING.md` (after setup)
- **General info:** `tools/indexing/README.md`

## Success Criteria

All met:
- ✅ Scripts run without errors
- ✅ compile_commands.json generated (96 entries)
- ✅ Configuration files deployed
- ✅ clangd can parse files
- ✅ No build required
- ✅ Tested and working

## Next Steps for User

1. Open VSCode in RCCL directory
2. Install clangd extension
3. Disable C/C++ extension
4. Reload window
5. Start navigating code!

See `QUICK-START.md` for detailed instructions.










