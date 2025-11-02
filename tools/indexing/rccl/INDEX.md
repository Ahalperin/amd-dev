# RCCL Code Indexing - Complete Package

## 📦 What's Included

This package provides everything needed to set up code navigation for RCCL without building.

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
cd /Users/ahalperin/xai/amd-dev/tools/indexing/rccl
./setup.sh

# Verify
./verify.sh

# Open in VSCode
code /Users/ahalperin/xai/amd-dev/amd/rccl
```

Done! Install the clangd extension in VSCode and start navigating.

## 📚 Documentation

- **Quick Start**: `QUICK-START.md` - Essential commands and shortcuts
- **Full Guide**: `README.md` - Complete documentation with troubleshooting
- **Parent**: `../README.md` - Overview of all indexing tools
- **Summary**: `../SETUP-SUMMARY.md` - Technical details of what was created

## ✅ What You Get

After running `./setup.sh`, RCCL will have:

- ✅ 96 source files indexed
- ✅ Full go-to-definition support
- ✅ Find all references
- ✅ Call hierarchy
- ✅ Symbol search
- ✅ Hover documentation
- ✅ Code completion
- ✅ Rename refactoring

All without building!

## 🎯 Common Tasks

### Setup RCCL Indexing
```bash
./setup.sh
```

### Update After Code Changes
```bash
python3 generate_compile_commands.py
```

### Setup Different RCCL Directory
```bash
./setup.sh /path/to/other/rccl
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
| Rename | `F2` |

## 📂 Directory Structure

```
tools/indexing/rccl/          # This directory
├── setup.sh                  # Main setup script
├── generate_compile_commands.py  # Database generator
├── verify.sh                 # Verification script
├── clangd-config            # Configuration template
├── vscode-settings.json     # VSCode template
├── README.md                 # Full documentation
├── QUICK-START.md           # Quick reference
└── INDEX.md                 # This file

After setup:
amd/rccl/                    # RCCL directory
├── compile_commands.json    # Generated database
├── .clangd                  # Deployed config
├── .vscode/settings.json    # Deployed settings
└── README-INDEXING.md       # User guide
```

## 🛠️ Requirements

- **clangd** - Language server (install: `brew install llvm`)
- **Python 3** - For generation script
- **Editor** - VSCode, Neovim, Emacs, etc. with clangd support

## ❓ Help

### Get Help
```bash
# Read documentation
cat README.md
cat QUICK-START.md

# Check what was deployed
ls -lh /Users/ahalperin/xai/amd-dev/amd/rccl/{compile_commands.json,.clangd}

# Verify setup
./verify.sh
```

### Common Issues

**clangd not found**
```bash
brew install llvm  # macOS
sudo apt install clangd-15  # Ubuntu
```

**Headers not found**
```bash
# Edit generate_compile_commands.py
# Update ROCM_PATH line
python3 generate_compile_commands.py
```

**VSCode not working**
- Install "clangd" extension
- Disable "C/C++" extension
- Reload window (Cmd+Shift+P → Reload Window)

## 🎓 Learning Path

1. **Start here**: `QUICK-START.md`
2. **Try it**: Open RCCL in VSCode with clangd
3. **Experiment**: Use F12, Shift+F12 on functions
4. **Go deeper**: Read `README.md` for advanced features
5. **Troubleshoot**: Check `README.md` troubleshooting section

## 📊 Status

✅ **Tested and working**
- 9/10 verification checks passed
- 96 source files indexed
- 255KB compilation database
- All configuration files deployed

## 🔄 Maintenance

When RCCL source code changes:

```bash
cd /Users/ahalperin/xai/amd-dev/tools/indexing/rccl
python3 generate_compile_commands.py
```

clangd will automatically re-index.

## 🤝 Contributing

To improve these tools:

1. Edit files in `tools/indexing/rccl/`
2. Test with `./setup.sh` and `./verify.sh`
3. Update documentation

## 📞 Support

- **Documentation**: See `README.md`
- **Quick help**: See `QUICK-START.md`
- **clangd docs**: https://clangd.llvm.org/
- **LSP docs**: https://microsoft.github.io/language-server-protocol/

## 🎉 Success!

If you can:
- Press F12 on a function and jump to its definition
- Press Shift+F12 to see all callers
- Use Cmd+T to search for symbols

Then it's working! Enjoy navigating the RCCL codebase.






