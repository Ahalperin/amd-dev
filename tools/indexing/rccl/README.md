# RCCL Code Indexing (No Build Required)

This directory contains tools to enable code navigation in RCCL without requiring a full build. These tools generate a compilation database (`compile_commands.json`) that language servers like clangd can use to understand your code.

## Quick Start

```bash
# From this directory
./setup.sh

# Or specify a custom RCCL directory
./setup.sh /path/to/rccl
```

That's it! The script will deploy all necessary files and generate the compilation database.

## What Gets Installed

The setup script will create/copy these files to your RCCL directory:

- `compile_commands.json` - Compilation database for clangd
- `.clangd` - clangd configuration
- `.vscode/settings.json` - VSCode settings for clangd
- `README-INDEXING.md` - User documentation

## Prerequisites

### 1. clangd (Required)

clangd is a language server that provides IDE features like go-to-definition, find-references, etc.

**Install on macOS:**
```bash
brew install llvm
# clangd will be at /opt/homebrew/opt/llvm/bin/clangd
# Add to PATH or the system will use /usr/bin/clangd
```

**Install on Ubuntu/Debian:**
```bash
sudo apt install clangd-15
# Or get the latest version
wget https://github.com/clangd/clangd/releases/download/15.0.0/clangd-linux-15.0.0.zip
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

### 2. Editor Configuration

#### VSCode (Recommended)

1. Install the **clangd** extension:
   - Open Extensions (Cmd+Shift+X / Ctrl+Shift+X)
   - Search for "clangd"
   - Install the extension by LLVM

2. Disable the C/C++ extension:
   - Search for "C/C++" extension by Microsoft
   - Click the gear icon → "Disable (Workspace)"
   - Or uninstall it if you don't need it

3. Reload the window:
   - Press Cmd+Shift+P (macOS) or Ctrl+Shift+P (Linux)
   - Type "Reload Window" and select it

#### Neovim with LSP

Add to your init.lua or lsp config:

```lua
require'lspconfig'.clangd.setup{
  cmd = {
    "clangd",
    "--background-index",
    "--clang-tidy",
    "--completion-style=detailed",
    "--header-insertion=iwyu",
  },
  root_dir = require'lspconfig'.util.root_pattern(
    "compile_commands.json",
    ".clangd",
    ".git"
  ),
}
```

#### Emacs with lsp-mode

```elisp
(use-package lsp-mode
  :hook ((c-mode c++-mode) . lsp)
  :config
  (setq lsp-clients-clangd-args
        '("--background-index"
          "--clang-tidy"
          "--completion-style=detailed"
          "--header-insertion=iwyu")))
```

#### Vim with coc.nvim

In your `:CocConfig`:

```json
{
  "languageserver": {
    "clangd": {
      "command": "clangd",
      "args": ["--background-index", "--clang-tidy"],
      "rootPatterns": ["compile_commands.json", ".git/"],
      "filetypes": ["c", "cpp", "cc", "objc", "objcpp"]
    }
  }
}
```

## Available Features

Once set up, you'll have:

✅ **Jump to Definition** (`F12` or `Cmd+Click`)
- Jump from function call to implementation
- Works across files

✅ **Find All References** (`Shift+F12`)
- See everywhere a function/variable is used
- Includes callers and callees

✅ **Go to Implementation** (`Cmd+F12` / `Ctrl+F12`)
- Jump from declaration to implementation

✅ **Hover Documentation**
- Hover over any symbol to see:
  - Type information
  - Function signatures
  - Documentation comments

✅ **Code Completion** (`Ctrl+Space`)
- IntelliSense-style autocompletion
- Context-aware suggestions

✅ **Signature Help**
- See function parameters as you type
- Shows parameter types and documentation

✅ **Symbol Search** (`Cmd+T` / `Ctrl+T`)
- Quickly find any function, class, variable
- Fuzzy search support

✅ **Call Hierarchy**
- See the complete call chain
- Find who calls a function and what it calls

✅ **Rename Symbol** (`F2`)
- Safely rename across all files

✅ **Code Formatting**
- Format code according to style guides

## Manual Setup (Advanced)

If you prefer not to use the setup script:

### 1. Generate compile_commands.json

```bash
python3 generate_compile_commands.py /path/to/rccl
```

### 2. Copy configuration files

```bash
# Copy clangd config
cp clangd-config /path/to/rccl/.clangd

# Copy VSCode settings
mkdir -p /path/to/rccl/.vscode
cp vscode-settings.json /path/to/rccl/.vscode/settings.json
```

### 3. Open your editor in the RCCL directory

The editor should automatically detect `compile_commands.json` and `.clangd`.

## Troubleshooting

### clangd not finding headers

If you see "file not found" errors for system headers:

1. Check your ROCm installation path:
   ```bash
   echo $ROCM_PATH
   ls /opt/rocm/include
   ```

2. Edit `generate_compile_commands.py` and update `ROCM_PATH`:
   ```python
   ROCM_PATH = Path("/your/rocm/path")
   ```

3. Regenerate:
   ```bash
   python3 generate_compile_commands.py /path/to/rccl
   ```

### Slow initial indexing

The first time clangd runs, it indexes all files. This can take 2-10 minutes depending on your machine. Progress is shown in the editor status bar. Subsequent opens are much faster (cached).

### "Multiple definitions" warnings

This is expected - we're setting up indexing, not building. clangd will still provide navigation despite these warnings.

### GPU kernel code not recognized

Some GPU-specific syntax may not be fully understood by clangd since it's not a full HIP/CUDA compiler. However:
- Host code works perfectly
- Most device code works well enough for navigation
- You can suppress specific warnings in `.clangd`

### VSCode shows two sets of diagnostics

This happens if both clangd and the C/C++ extension are active. Disable the C/C++ extension for the workspace.

## Alternative Tools

If clangd doesn't work for your use case:

### GNU Global (gtags)

```bash
# Install
brew install global  # macOS
sudo apt install global  # Ubuntu

# Generate tags
cd /path/to/rccl
gtags -v

# Use
global -x symbol_name      # Find references
global -r symbol_name      # Find callers
global -s symbol_name      # Find definition
```

### Universal ctags

```bash
# Install
brew install universal-ctags  # macOS
sudo apt install universal-ctags  # Ubuntu

# Generate tags
cd /path/to/rccl
ctags -R --c++-kinds=+p --fields=+iaS --extras=+q .

# Use with ctags-aware editors/plugins
```

### cscope

```bash
# Install
brew install cscope  # macOS
sudo apt install cscope  # Ubuntu

# Generate database
cd /path/to/rccl
find src -name "*.c" -o -name "*.cc" -o -name "*.cpp" -o -name "*.h" > cscope.files
cscope -b -q -k

# Use
cscope -d
```

## Updating the Index

When you pull new code or files change:

```bash
cd /path/to/rccl
python3 /path/to/tools/indexing/rccl/generate_compile_commands.py .
```

clangd will automatically detect the change and re-index.

## Files in This Directory

- `generate_compile_commands.py` - Main script to generate compilation database
- `clangd-config` - Configuration for clangd (deployed as `.clangd`)
- `vscode-settings.json` - VSCode settings (deployed to `.vscode/settings.json`)
- `setup.sh` - One-command setup script
- `README.md` - This file

## How It Works

1. **generate_compile_commands.py** scans the RCCL source tree and creates a JSON file (`compile_commands.json`) that describes how each source file would be compiled.

2. **clangd** reads this file and uses it to understand:
   - Which headers to include
   - What preprocessor defines are active
   - What language standard to use
   - Where to find dependencies

3. Your editor communicates with clangd using the Language Server Protocol (LSP) to provide IDE features.

The beauty of this approach: **no build required**! We just tell clangd what the compile commands *would* be, and it figures out the rest.

## Performance Tips

- **Exclude build directories**: The `.clangd` config already excludes `build/` and `hipify/`
- **Use SSD**: Indexing is I/O intensive
- **Increase memory**: clangd caches parsed files in memory
- **Close unused files**: Each open file uses memory

## Contributing

Found an issue or improvement? Update the scripts in:
```
/Users/ahalperin/xai/amd-dev/tools/indexing/rccl/
```

## See Also

- [clangd documentation](https://clangd.llvm.org/)
- [Language Server Protocol](https://microsoft.github.io/language-server-protocol/)
- [RCCL GitHub](https://github.com/ROCmSoftwarePlatform/rccl)



