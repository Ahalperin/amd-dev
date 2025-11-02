# Code Indexing Tools for AMD Development

This directory contains tools to set up code indexing and navigation for various codebases without requiring a full build. These tools generate compilation databases that language servers can use to provide IDE features like go-to-definition, find-references, and more.

## Available Tools

### RCCL (ROCm Communication Collectives Library)

Setup code navigation for RCCL using clangd.

```bash
cd rccl/
./setup.sh
```

See [rccl/README.md](rccl/README.md) for detailed documentation.

### Meta TorchComms

Setup code navigation for Meta's TorchComms library.

```bash
cd meta/torchcomms/
./setup.sh
```

See [meta/torchcomms/README.md](meta/torchcomms/README.md) for detailed documentation.

### RCCL-tests

Setup code navigation for RCCL test suite.

```bash
cd rccl-tests/
./setup.sh
```

See [rccl-tests/README.md](rccl-tests/README.md) for detailed documentation.

## Why Use These Tools?

Building large projects like RCCL can be:
- **Time-consuming**: Full builds can take 30+ minutes
- **Resource-intensive**: Requires significant memory and CPU
- **Complex**: May require specific hardware or dependencies
- **Unnecessary**: You just want to browse and understand code

These tools provide:
- ✅ **No build required**: Set up indexing in seconds
- ✅ **Full IDE features**: Go-to-definition, find-references, hover docs
- ✅ **Cross-file navigation**: Jump between headers and implementations
- ✅ **Call hierarchy**: See who calls what
- ✅ **Symbol search**: Find any function/class/variable quickly

## How It Works

1. **Generate compilation database** (`compile_commands.json`)
   - Describes how each file would be compiled
   - Includes compiler flags, include paths, defines

2. **Configure language server** (e.g., clangd)
   - Reads the compilation database
   - Parses code without actually compiling
   - Provides IDE features via Language Server Protocol (LSP)

3. **Editor integration**
   - VSCode, Neovim, Emacs, etc.
   - Communicate with language server
   - Show results in the editor UI

## Supported Editors

These tools work with any editor that supports LSP and clangd:

- **VSCode**: Install "clangd" extension by LLVM
- **Neovim**: Built-in LSP support (nvim-lspconfig)
- **Emacs**: lsp-mode or eglot
- **Vim**: coc.nvim, vim-lsp, or ALE
- **Sublime Text**: LSP package
- **Kate/KDevelop**: Built-in clangd support
- **QtCreator**: Built-in clangd support

## Prerequisites

### clangd

The language server that powers code navigation.

**macOS:**
```bash
brew install llvm
```

**Ubuntu/Debian:**
```bash
sudo apt install clangd-15
```

**RHEL/CentOS:**
```bash
sudo yum install clang-tools-extra
```

**From source/binary:**
Download from https://clangd.llvm.org/installation.html

### Python 3

Required for generation scripts (usually pre-installed on most systems).

```bash
python3 --version
```

## Quick Start Example

```bash
# Set up indexing for RCCL
cd /Users/ahalperin/xai/amd-dev/tools/indexing/rccl
./setup.sh

# Open RCCL in your editor
code /Users/ahalperin/xai/amd-dev/amd/rccl

# Start navigating!
# - F12: Go to definition
# - Shift+F12: Find references
# - Cmd+Click: Jump to symbol
```

## Adding New Projects

To add indexing support for a new project:

1. Create a directory: `tools/indexing/project-name/`
2. Create `generate_compile_commands.py` script
3. Create configuration files (`.clangd`, editor settings)
4. Create `setup.sh` for one-command deployment
5. Document in `README.md`

Use the RCCL tools as a template.

## Troubleshooting

### clangd not found

Make sure clangd is installed and in your PATH:

```bash
which clangd
clangd --version
```

### Headers not found

Check the include paths in `generate_compile_commands.py` and adjust for your system.

### Slow indexing

First-time indexing can take several minutes. Subsequent opens are cached and much faster.

### Multiple errors shown

If using VSCode, disable the "C/C++" extension to avoid conflicts with clangd.

## Alternative Approaches

If clangd doesn't work for you:

- **ctags/gtags**: Tag-based navigation (faster but less accurate)
- **cscope**: Text-based code browser
- **LSP alternatives**: ccls (another C++ language server)
- **IDE-specific**: Use built-in parsers (CLion, Visual Studio)

## Contributing

To improve or add new indexing tools:

1. Follow the existing structure
2. Document thoroughly
3. Test with multiple editors
4. Include troubleshooting section

## Resources

- [clangd documentation](https://clangd.llvm.org/)
- [Language Server Protocol](https://microsoft.github.io/language-server-protocol/)
- [compile_commands.json spec](https://clang.llvm.org/docs/JSONCompilationDatabase.html)

## License

These tools follow the same license as the parent project.

