# Meta Code Indexing Tools

This directory contains code indexing tools for Meta's open-source projects.

## Available Projects

### TorchComms

Communication library for distributed training.

**Location:** `torchcomms/`

**Setup:**
```bash
cd torchcomms/
./setup.sh
```

**What's Indexed:**
- 657 C++/CUDA source files
- Ctran communication transport layer
- TorchComms PyTorch API
- NCCLX extensions
- Utilities and backends

**Documentation:** See [torchcomms/README.md](torchcomms/README.md)

## Quick Start

Each subdirectory contains a complete indexing package with:
- `setup.sh` - One-command deployment
- `generate_compile_commands.py` - Database generator
- `verify.sh` - Verification script
- Full documentation

## See Also

- Parent indexing tools: `../README.md`
- RCCL indexing: `../rccl/`










