# RCCL-tests Code Indexing - Quick Start Guide

## One-Line Setup

```bash
cd /Users/ahalperin/xai/amd-dev/tools/indexing/rccl-tests && ./setup.sh
```

## What This Does

✅ Generates `compile_commands.json` for test files  
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
   - Open `src/all_reduce.cu`
   - Try F12 (Go to Definition)
   - Try ⇧F12 (Find References)

## Keyboard Shortcuts (macOS)

| Action | Shortcut | Description |
|--------|----------|-------------|
| Go to Definition | `F12` | Jump to implementation |
| Find References | `⇧F12` | Show all uses |
| Symbol Search | `⌘T` | Search for any symbol |
| Hover Info | `hover` | See type info |

## Command Line Usage

```bash
# Generate/regenerate compile_commands.json
cd /Users/ahalperin/xai/amd-dev/tools/indexing/rccl-tests
python3 generate_compile_commands.py

# Or specify custom directory
python3 generate_compile_commands.py /path/to/rccl-tests

# Run full setup
./setup.sh /path/to/rccl-tests

# Verify setup
./verify.sh
```

## Verify Setup

```bash
# Check clangd is installed
which clangd
clangd --version

# Check files were created
ls -lh /Users/ahalperin/xai/amd-dev/amd/rccl-tests/compile_commands.json
ls -lh /Users/ahalperin/xai/amd-dev/amd/rccl-tests/.clangd

# Count entries
grep -c "\"file\":" /Users/ahalperin/xai/amd-dev/amd/rccl-tests/compile_commands.json
```

## RCCL-tests File Structure

```
amd/rccl-tests/
├── src/
│   ├── all_reduce.cu           # AllReduce tests
│   ├── all_gather.cu           # AllGather tests
│   ├── reduce_scatter.cu       # ReduceScatter tests
│   ├── broadcast.cu            # Broadcast tests
│   ├── alltoall.cu             # AllToAll tests
│   ├── sendrecv.cu             # SendRecv tests
│   ├── common.cu               # Common test utilities
│   └── common.h                # Test headers
└── verifiable/
    └── verifiable.cu           # Verification tests

After running setup.sh:
├── compile_commands.json       # Compilation database
├── .clangd                     # clangd config
├── .vscode/settings.json       # VSCode config
└── README-INDEXING.md          # User guide
```

## Example: Exploring AllReduce Test

```bash
# 1. Open test file
code src/all_reduce.cu

# 2. Find testCollective function
# Press Cmd+F → search "testCollective"

# 3. Navigate to RCCL API
# Click on ncclAllReduce
# Press F12 → jumps to RCCL implementation

# 4. See all test usages
# Click on any test function
# Press Shift+F12 → see all callers
```

## Troubleshooting

### clangd not working
```bash
# Install clangd
brew install llvm  # macOS
sudo apt install clangd-15  # Ubuntu
```

### RCCL headers not found
```bash
# Edit generate_compile_commands.py
# Update RCCL_PATH variable
# Re-run: python3 generate_compile_commands.py
```

### Slow performance
- First-time indexing: 1-3 minutes
- Subsequent opens: instant (cached)

## For More Help

- Full docs: `cat README.md`
- User guide: `cat /Users/ahalperin/xai/amd-dev/amd/rccl-tests/README-INDEXING.md`
- clangd docs: https://clangd.llvm.org/

