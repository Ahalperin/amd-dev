# RCCL-tests Code Indexing - Complete Package

## 📦 What's Included

This package provides everything needed to set up code navigation for RCCL-tests without building.

### Files

1. **setup.sh** ⭐ - One-command setup script
2. **generate_compile_commands.py** - Generate compilation database
3. **verify.sh** - Verify setup is working
4. **clangd-config** - clangd configuration template
5. **vscode-settings.json** - VSCode settings template
6. **README.md** - Full documentation
7. **QUICK-START.md** - Quick reference
8. **INDEX.md** - This file

## 🚀 Quick Start (30 seconds)

```bash
# Run setup
cd /Users/ahalperin/xai/amd-dev/tools/indexing/rccl-tests
./setup.sh

# Verify
./verify.sh

# Open in VSCode
code /Users/ahalperin/xai/amd-dev/amd/rccl-tests
```

Done! Install the clangd extension in VSCode and start navigating.

## 📚 Documentation

- **Quick Start**: `QUICK-START.md` - Essential commands and shortcuts
- **Full Guide**: `README.md` - Complete documentation with troubleshooting
- **Parent**: `../README.md` - Overview of all indexing tools

## ✅ What You Get

After running `./setup.sh`, RCCL-tests will have:

- ✅ **24 test files indexed**
- ✅ Full go-to-definition support
- ✅ Find all references
- ✅ Symbol search
- ✅ Hover documentation
- ✅ Code completion
- ✅ Navigate to RCCL API calls

All without building!

## 🎯 Common Tasks

### Setup RCCL-tests Indexing
```bash
./setup.sh
```

### Update After Code Changes
```bash
python3 generate_compile_commands.py
```

### Setup Different RCCL-tests Directory
```bash
./setup.sh /path/to/other/rccl-tests
```

### Verify Setup
```bash
./verify.sh
```

## 🔑 Key Shortcuts (VSCode)

| Action | Key |
|--------|-----|
| Go to Definition | `F12` |
| Find References | `Shift+F12` |
| Symbol Search | `Cmd+T` |
| Hover Info | `hover` |

## 📂 Directory Structure

```
tools/indexing/rccl-tests/        # This directory
├── setup.sh                      # Main setup script
├── generate_compile_commands.py  # Database generator
├── verify.sh                     # Verification script
├── clangd-config                 # Configuration template
├── vscode-settings.json          # VSCode template
├── README.md                     # Full documentation
├── QUICK-START.md                # Quick reference
└── INDEX.md                      # This file

After setup:
amd/rccl-tests/                   # RCCL-tests directory
├── compile_commands.json         # Generated database (31KB, 24 files)
├── .clangd                       # Deployed config
├── .vscode/settings.json         # Deployed settings
└── README-INDEXING.md            # User guide
```

## 🛠️ Requirements

- **clangd** - Language server (install: `brew install llvm`)
- **Python 3** - For generation script
- **ROCm** - Auto-detected if available
- **Editor** - VSCode, Neovim, Emacs, etc. with clangd support

## 📊 Status

✅ **Tested and working**
- 12/12 verification checks passed
- 24 source files indexed
- 31KB compilation database
- All configuration files deployed

## 🎓 Learning Path

1. **Start here**: `QUICK-START.md`
2. **Try it**: Open RCCL-tests in VSCode with clangd
3. **Experiment**: Navigate AllReduce test implementation
4. **Go deeper**: Read `README.md` for advanced features

## 🔄 Maintenance

When RCCL-tests source code changes:

```bash
cd /Users/ahalperin/xai/amd-dev/tools/indexing/rccl-tests
python3 generate_compile_commands.py
```

clangd will automatically re-index.

## 🎉 Success!

If you can:
- Press F12 on `ncclAllReduce` in a test and jump to RCCL
- Press Shift+F12 to see all test usages
- Use Cmd+T to search for test functions

Then it's working! Enjoy navigating the RCCL test suite.

## 📞 Support

- **Documentation**: See `README.md`
- **Quick help**: See `QUICK-START.md`
- **clangd docs**: https://clangd.llvm.org/

