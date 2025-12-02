# TorchComms Installation Guide

This guide provides instructions for installing TorchComms with AMD GPU support.

## Overview

**TorchComms** is Meta's experimental communications library for PyTorch, designed for large-scale distributed training. It is **separate from PyTorch** and must be installed independently.

## Prerequisites

- ROCm 7.0.1 or 7.1.1 installed
- Python 3.10 or higher
- PyTorch 2.8 or higher with ROCm support
- Conda or Miniconda

## Installation Options

### Option 1: RCCLX Backend (Main Installation)

**Best for:** Users needing Meta's enhanced RCCL features and optimizations

RCCLX is Meta's enhanced version of RCCL with additional optimizations and features. This is the main installation script.

```bash
cd /home/amir/amd-dev/dn
bash create-torchcomms-dev-tools.sh
```

**Features:**
- ✨ Enhanced features and optimizations
- ✨ Used in Meta's production systems
- ✨ Better performance at scale
- ⚙️ Requires building RCCLX from source (automated in script)

**Backend name in code:** `"rcclx"`

```python
from torchcomms import new_comm, ReduceOp
import torch

comm = new_comm("rcclx", torch.device("cuda"))
```

### Option 2: Standard RCCL Backend (Simpler Alternative)

**Best for:** Quick testing, simpler installation, or troubleshooting

The standard RCCL backend uses AMD's ROCm Collective Communications Library that comes with ROCm.

```bash
cd /home/amir/amd-dev/dn
bash create-torchcomms-rccl.sh  # This is the simplified version
```

**Features:**
- ✅ Simpler installation
- ✅ Works out of the box with ROCm
- ✅ Stable and well-tested
- ✅ Good for getting started quickly

**Backend name in code:** `"rccl"`

```python
from torchcomms import new_comm, ReduceOp
import torch

comm = new_comm("rccl", torch.device("cuda"))
```

## Verification

After installation, verify it works:

```bash
conda activate torchcomms
python -c "from torchcomms import new_comm, ReduceOp; import torch; print('✓ TorchComms:', 'OK'); print('✓ PyTorch:', torch.__version__)"
```

## Example Usage

```python
#!/usr/bin/env python3
import torch
from torchcomms import new_comm, ReduceOp

def main():
    # Initialize TorchComm with RCCL backend
    device = torch.device("cuda")
    comm = new_comm("rccl", device, name="main_comm")
    
    # Get rank and world size
    rank = comm.get_rank()
    world_size = comm.get_size()
    
    # Calculate device ID
    device_id = rank % torch.cuda.device_count()
    target_device = torch.device(f"cuda:{device_id}")
    
    print(f"Rank {rank}/{world_size}: Running on device {device_id}")
    
    # Create a tensor
    tensor = torch.full((1024,), float(rank + 1), 
                       dtype=torch.float32, device=target_device)
    
    print(f"Rank {rank}: Before AllReduce: {tensor[0].item()}")
    
    # Perform AllReduce
    comm.all_reduce(tensor, ReduceOp.SUM, async_op=False)
    torch.cuda.current_stream().synchronize()
    
    print(f"Rank {rank}: After AllReduce: {tensor[0].item()}")
    
    # Cleanup
    comm.finalize()

if __name__ == "__main__":
    main()
```

Run with:
```bash
conda activate torchcomms
torchrun --nproc_per_node=2 your_script.py
```

## Comparison: TorchComms vs torch.distributed

| Feature | TorchComms | torch.distributed |
|---------|------------|-------------------|
| Installation | Separate library | Built into PyTorch |
| API Style | Object-oriented | Module-based |
| Scalability | 100k+ GPUs | Up to ~10k GPUs |
| Backends | NCCL, NCCLX, RCCL, RCCLX, GLOO | NCCL, GLOO, MPI |
| Use Case | Large-scale (Meta production) | General PyTorch distributed |

## Troubleshooting

### "ModuleNotFoundError: No module named 'torchcomms'"

TorchComms is not installed. Run the installation script.

### "undefined symbol" errors

This usually indicates ABI compatibility issues. Try:
1. Rebuilding with `USE_SYSTEM_LIBS=0` (already set in our script)
2. Ensuring ROCm paths are correctly set

### CMake can't find HIP

Make sure ROCm environment variables are set:
```bash
export ROCM_HOME=/opt/rocm-7.0.1  # or your ROCm version
export HIP_PATH=${ROCM_HOME}
export CMAKE_PREFIX_PATH="${ROCM_HOME}:$CMAKE_PREFIX_PATH"
```

### Backend not found (rcclx/nccl)

Make sure you're using the correct backend name for your build:
- If you installed with RCCL: use `"rccl"`
- If you installed with RCCLX: use `"rcclx"`

## Environment Details

After successful installation, you should see:

```bash
conda activate torchcomms
python -c "import torch; print('PyTorch:', torch.__version__); print('ROCm/HIP:', torch.version.hip)"
```

Expected output:
```
PyTorch: 2.10.0.dev20251124+rocm7.0
ROCm/HIP: 7.0.51831
```

## Additional Resources

- [TorchComms Documentation](https://meta-pytorch.org/torchcomms/)
- [TorchComms GitHub](https://github.com/meta-pytorch/torchcomms)
- [PyTorch Distributed](https://pytorch.org/docs/stable/distributed.html)

## Summary

1. ✅ **PyTorch** is installed separately (comes with ROCm support)
2. ✅ **TorchComms** must be built and installed as a separate library
3. ✨ Use `create-torchcomms-dev-tools.sh` for **RCCLX** (Meta's enhanced RCCL - recommended)
4. ✅ Use `create-torchcomms-rccl.sh` for **standard RCCL** (simpler alternative)

Your distributed training setup is ready! 🚀

