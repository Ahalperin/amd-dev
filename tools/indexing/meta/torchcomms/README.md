# Meta TorchComms Code Indexing (No Build Required)

This directory contains tools to enable code navigation in Meta TorchComms without requiring a full build. These tools generate a compilation database (`compile_commands.json`) that language servers like clangd can use to understand your code.

## Quick Start

```bash
# From this directory
./setup.sh

# Or specify a custom TorchComms directory
./setup.sh /path/to/torchcomms
```

That's it! The script will deploy all necessary files and generate the compilation database.

## What Gets Installed

The setup script will create/copy these files to your TorchComms directory:

- `compile_commands.json` - Compilation database for clangd
- `.clangd` - clangd configuration
- `.vscode/settings.json` - VSCode settings for clangd
- `README-INDEXING.md` - User documentation

## About TorchComms

TorchComms is Meta's communication library that includes:
- **ctran** - Communication transport layer
- **torchcomms** - PyTorch bindings and API
- **ncclx** - Extended NCCL functionality
- **utils** - Logging, tracing, and utilities

This indexing setup covers all C++/CUDA code in the `comms/` directory.

## Prerequisites

### 1. clangd (Required)

clangd is a language server that provides IDE features.

**Install on macOS:**
```bash
brew install llvm
```

**Install on Ubuntu/Debian:**
```bash
sudo apt install clangd-15
```

**Install on RHEL/CentOS:**
```bash
sudo yum install clang-tools-extra
```

**Verify installation:**
```bash
which clangd
clangd --version
```

### 2. CUDA (Optional but Recommended)

If you have CUDA installed, the script will detect it automatically:

```bash
export CUDA_HOME=/usr/local/cuda
# or it will search common locations
```

### 3. Editor Configuration

#### VSCode (Recommended)

1. Install the **clangd** extension:
   - Open Extensions (Cmd+Shift+X / Ctrl+Shift+X)
   - Search for "clangd"
   - Install the extension by LLVM

2. Disable the C/C++ extension:
   - Search for "C/C++" extension by Microsoft
   - Click the gear icon → "Disable (Workspace)"

3. Reload the window:
   - Press Cmd+Shift+P (macOS) or Ctrl+Shift+P (Linux)
   - Type "Reload Window" and select it

## Available Features

Once set up, you'll have:

✅ **Jump to Definition** (`F12` or `Cmd+Click`)
- Jump from function call to implementation
- Works across all files in comms/

✅ **Find All References** (`Shift+F12`)
- See everywhere a function/variable/class is used
- Includes callers and callees

✅ **Go to Implementation** (`Cmd+F12` / `Ctrl+F12`)
- Jump from declaration to implementation

✅ **Hover Documentation**
- Hover over any symbol to see:
  - Type information
  - Function signatures
  - Template parameters

✅ **Code Completion** (`Ctrl+Space`)
- Context-aware autocompletion
- Namespace-aware suggestions

✅ **Signature Help**
- See function parameters as you type
- Shows parameter types

✅ **Symbol Search** (`Cmd+T` / `Ctrl+T`)
- Quickly find any function, class, variable
- Search across entire codebase

✅ **Call Hierarchy**
- See complete call chains
- Find who calls a function

✅ **Rename Symbol** (`F2`)
- Safely rename across all files

## Manual Setup (Advanced)

If you prefer not to use the setup script:

### 1. Generate compile_commands.json

```bash
python3 generate_compile_commands.py /path/to/torchcomms
```

### 2. Copy configuration files

```bash
# Copy clangd config
cp clangd-config /path/to/torchcomms/.clangd

# Copy VSCode settings
mkdir -p /path/to/torchcomms/.vscode
cp vscode-settings.json /path/to/torchcomms/.vscode/settings.json
```

### 3. Open your editor in the TorchComms directory

The editor should automatically detect `compile_commands.json` and `.clangd`.

## TorchComms-Specific Features

### Navigate Communication Algorithms

Example: Understanding AllReduce:

1. Open `comms/ctran/algos/AllReduce/AllReduceDirect.cc`
2. Click on `AllReduceDirect` class
3. Press `Shift+F12` to see where it's used
4. Press `F12` on any method call to see implementation

### Explore CUDA Kernels

Navigate between host and device code:

1. Open any `.cc` file with CUDA kernel launches
2. Click on kernel name
3. Press `F12` to jump to `.cu` file with kernel implementation

### Trace Data Flow

Use call hierarchy to understand data flow:

1. Right-click on a function
2. Select "Show Call Hierarchy"
3. See callers (who calls this) and callees (what this calls)

## Troubleshooting

### clangd not finding headers

If you see "file not found" errors:

1. Check CUDA installation:
   ```bash
   echo $CUDA_HOME
   ls /usr/local/cuda/include
   ```

2. Edit `generate_compile_commands.py` and update paths:
   ```python
   CUDA_PATH = Path("/your/cuda/path")
   ```

3. Regenerate:
   ```bash
   python3 generate_compile_commands.py /path/to/torchcomms
   ```

### Slow initial indexing

First-time indexing can take 5-15 minutes depending on your machine (TorchComms is a large codebase). Progress shown in editor status bar. Subsequent opens are much faster (cached).

### CUDA errors

Some CUDA-specific syntax may not be fully understood. This is normal. Most functionality will still work:
- Host code: ✅ Perfect
- Device code: ✅ Mostly works
- CUDA templates: ⚠️ May show some errors (navigation still works)

### Python files

This indexing setup is for C++/CUDA only. For Python files in `comms/torchcomms/`:
- Use Python language server (Pylance, Pyright, or jedi)
- Set `python.analysis.extraPaths` in VSCode settings (already configured)

## Updating the Index

When TorchComms source changes (new files, restructuring):

```bash
cd /path/to/tools/indexing/meta/torchcomms
python3 generate_compile_commands.py
```

clangd will automatically detect the change and re-index.

## Files in This Directory

- `generate_compile_commands.py` - Main script to generate compilation database
- `clangd-config` - Configuration for clangd (deployed as `.clangd`)
- `vscode-settings.json` - VSCode settings (deployed to `.vscode/settings.json`)
- `setup.sh` - One-command setup script
- `verify.sh` - Verification script
- `README.md` - This file
- `QUICK-START.md` - Quick reference

## How It Works

1. **generate_compile_commands.py** scans the TorchComms source tree and creates a JSON file that describes how each source file would be compiled.

2. **clangd** reads this file and uses it to understand:
   - Which headers to include
   - What preprocessor defines are active
   - What language standard to use
   - Where to find CUDA/system dependencies

3. Your editor communicates with clangd using the Language Server Protocol (LSP).

**No build required!** We just tell clangd what the compile commands *would* be.

## Performance Tips

- **Exclude tests**: Edit `generate_compile_commands.py` to exclude `/tests/` directories
- **Use SSD**: Indexing is I/O intensive
- **Close unused files**: Each open file uses memory
- **Increase memory**: clangd caches parsed files

## Common Use Cases

### 1. Understanding Ctran Architecture

```bash
# Open the main interface
code comms/ctran/interfaces/ICtran.h

# Navigate to implementations
# - F12 on any method
# - Shift+F12 to see all implementations
```

### 2. Tracing AllReduce Implementation

```bash
# Start at the API
code comms/torchcomms/TorchComm.cpp
# Find allreduce method
# F12 → jumps to implementation
# Continue F12 to go deeper
```

### 3. Finding Transport Backends

```bash
# Cmd+T → "IbBackend"
# See all IB-related code
# Explore implementations
```

## See Also

- [TorchComms README](../../comms/README.md)
- [clangd documentation](https://clangd.llvm.org/)
- [Language Server Protocol](https://microsoft.github.io/language-server-protocol/)
- Parent indexing tools: `../../tools/indexing/README.md`






