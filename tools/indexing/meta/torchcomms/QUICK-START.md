# TorchComms Code Indexing - Quick Start Guide

## One-Line Setup

```bash
cd /Users/ahalperin/xai/amd-dev/tools/indexing/meta/torchcomms && ./setup.sh
```

## What This Does

✅ Generates `compile_commands.json` with hundreds of source files  
✅ Deploys `.clangd` configuration  
✅ Configures VSCode settings  
✅ Creates documentation  

## Next Steps (VSCode)

1. **Install clangd extension**
   - Open Extensions (⌘⇧X)
   - Search "clangd"
   - Install by LLVM

2. **Disable C/C++ extension**
   - Find "C/C++" extension
   - Gear icon → "Disable (Workspace)"

3. **Reload window**
   - Press ⌘⇧P
   - Type "Reload Window"

4. **Start navigating!**
   - Open `comms/ctran/algos/AllReduce/AllReduceDirect.cc`
   - Try F12 (Go to Definition)
   - Try ⇧F12 (Find References)

## Keyboard Shortcuts (macOS)

| Action | Shortcut | Description |
|--------|----------|-------------|
| Go to Definition | `F12` | Jump to where symbol is defined |
| Find References | `⇧F12` | Show all uses of symbol |
| Go to Implementation | `⌘F12` | Jump from declaration to implementation |
| Symbol Search | `⌘T` | Search for any symbol by name |
| Hover Info | `hover` | See type info and docs |
| Peek Definition | `⌥F12` | Show definition inline |
| Rename Symbol | `F2` | Rename across all files |
| Call Hierarchy | `right-click → Show Call Hierarchy` | See call chains |

## Command Line Usage

```bash
# Generate/regenerate compile_commands.json
cd /Users/ahalperin/xai/amd-dev/tools/indexing/meta/torchcomms
python3 generate_compile_commands.py

# Or specify custom TorchComms directory
python3 generate_compile_commands.py /path/to/torchcomms

# Run full setup on different directory
./setup.sh /path/to/torchcomms

# Verify setup
./verify.sh
```

## Verify Setup

```bash
# Check clangd is installed
which clangd
clangd --version

# Check files were created
ls -lh /Users/ahalperin/xai/amd-dev/meta/torchcomms/compile_commands.json
ls -lh /Users/ahalperin/xai/amd-dev/meta/torchcomms/.clangd

# Count entries in compilation database
grep -c "\"file\":" /Users/ahalperin/xai/amd-dev/meta/torchcomms/compile_commands.json
```

## Troubleshooting

### clangd not working
- Check if installed: `which clangd`
- Install on macOS: `brew install llvm`
- Install on Ubuntu: `sudo apt install clangd-15`

### Headers not found
- Set CUDA_HOME: `export CUDA_HOME=/usr/local/cuda`
- Edit `generate_compile_commands.py`
- Update `CUDA_PATH` variable
- Re-run: `python3 generate_compile_commands.py`

### Slow performance
- First-time indexing takes 5-15 minutes
- Subsequent opens are cached (fast)
- Close unused files to save memory

## Example: Exploring AllReduce

Let's trace how AllReduce works:

1. **Find the entry point**
   ```bash
   # Press Cmd+T in VSCode
   # Type "AllReduce"
   # Select AllReduceDirect class
   ```

2. **See all usages**
   ```bash
   # With AllReduceDirect selected
   # Press Shift+F12
   # View all references in sidebar
   ```

3. **Jump to implementation**
   ```bash
   # Click on any method
   # Press F12
   # Explore the code
   ```

4. **Understand call flow**
   ```bash
   # Right-click on a function
   # Select "Show Call Hierarchy"
   # See who calls it (incoming)
   # See what it calls (outgoing)
   ```

## TorchComms Code Structure

```
meta/torchcomms/
├── comms/
│   ├── ctran/              # Communication transport layer
│   │   ├── algos/          # Collective algorithms
│   │   │   ├── AllReduce/
│   │   │   ├── AllGather/
│   │   │   ├── AllToAll/
│   │   │   └── ...
│   │   ├── backends/       # Network backends (IB, socket, etc.)
│   │   ├── gpe/            # GPU execution engine
│   │   └── utils/          # Utilities
│   ├── torchcomms/         # PyTorch bindings
│   ├── ncclx/              # NCCL extensions
│   └── utils/              # Common utilities
└── compile_commands.json   # Generated index

After running setup.sh:
├── .clangd                 # clangd config
├── .vscode/settings.json   # VSCode config
└── README-INDEXING.md      # User guide
```

## Advanced: Neovim Setup

Add to your `init.lua`:

```lua
require'lspconfig'.clangd.setup{
  cmd = {
    "clangd",
    "--background-index",
    "--clang-tidy",
    "--header-insertion=iwyu",
  },
}
```

Open file in Neovim:
- `gd` - Go to definition
- `gr` - Find references  
- `K` - Hover info
- `<leader>ca` - Code actions

## For More Help

- Full docs: `cat README.md`
- User guide: `cat /Users/ahalperin/xai/amd-dev/meta/torchcomms/README-INDEXING.md`
- clangd docs: https://clangd.llvm.org/
- TorchComms docs: Check comms/README.md






