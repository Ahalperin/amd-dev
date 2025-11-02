# RCCL Code Indexing - Quick Start Guide

## One-Line Setup

```bash
cd /Users/ahalperin/xai/amd-dev/tools/indexing/rccl && ./setup.sh
```

## What This Does

✅ Creates `compile_commands.json` with 96+ source file entries  
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
   - Open `/Users/ahalperin/xai/amd-dev/amd/rccl/src/rccl_wrap.cc`
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

## Command Line Usage

```bash
# Generate/regenerate compile_commands.json
cd /Users/ahalperin/xai/amd-dev/tools/indexing/rccl
python3 generate_compile_commands.py

# Or specify custom RCCL directory
python3 generate_compile_commands.py /path/to/rccl

# Run full setup on different RCCL directory
./setup.sh /path/to/rccl
```

## Verify Setup

```bash
# Check clangd is installed
which clangd
clangd --version

# Check files were created
ls -lh /Users/ahalperin/xai/amd-dev/amd/rccl/compile_commands.json
ls -lh /Users/ahalperin/xai/amd-dev/amd/rccl/.clangd

# Count entries in compilation database
grep -c "\"file\":" /Users/ahalperin/xai/amd-dev/amd/rccl/compile_commands.json
```

## Troubleshooting

### clangd not working
- Check if installed: `which clangd`
- Install on macOS: `brew install llvm`
- Install on Ubuntu: `sudo apt install clangd-15`

### Headers not found
- Edit `generate_compile_commands.py`
- Update `ROCM_PATH = Path("/your/rocm/path")`
- Re-run: `python3 generate_compile_commands.py`

### Slow performance
- First-time indexing takes 2-10 minutes
- Subsequent opens are cached (fast)
- Close unused files to save memory

### VSCode shows errors but navigation works
- That's normal! We're not building, just indexing
- Navigation features work despite warnings

## Example: Finding a Function

Let's say you want to understand `ncclAllReduce`:

1. Open any file in VSCode
2. Press `⌘T` (symbol search)
3. Type "ncclAllReduce"
4. Select the result
5. Press `⇧F12` to see all callers

Or:

```bash
# Using grep to find it first
cd /Users/ahalperin/xai/amd-dev/amd/rccl
grep -r "ncclAllReduce" src/

# Then use clangd in your editor for detailed navigation
```

## File Structure

```
tools/indexing/rccl/
├── README.md                      # Full documentation
├── QUICK-START.md                 # This file
├── setup.sh                       # One-command setup
├── generate_compile_commands.py   # Generate compilation database
├── clangd-config                  # clangd configuration template
└── vscode-settings.json           # VSCode settings template

After running setup.sh:

amd/rccl/
├── compile_commands.json          # Compilation database (255KB)
├── .clangd                        # clangd config
├── .vscode/settings.json          # VSCode config
└── README-INDEXING.md             # User documentation
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

## For More Help

- Full docs: `cat README.md`
- RCCL docs: `cat /Users/ahalperin/xai/amd-dev/amd/rccl/README-INDEXING.md`
- clangd docs: https://clangd.llvm.org/



