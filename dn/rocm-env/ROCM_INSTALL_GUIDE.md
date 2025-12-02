# ROCm 7.1.1 Side-by-Side Installation Guide

This guide will help you install ROCm 7.1.1 alongside your existing ROCm 7.0.1 installation on Ubuntu 22.04.

## Overview

Your system currently has:
- **ROCm 7.0.1** installed at `/opt/rocm-7.0.1`
- **8x AMD MI350X GPUs** (gfx950 architecture)
- **Ubuntu 22.04.5 LTS**

## Why Side-by-Side Installation?

Installing ROCm 7.1.1 side-by-side allows you to:
- ✅ Keep your working ROCm 7.0.1 setup
- ✅ Test ROCm 7.1.1 without breaking existing workflows
- ✅ Switch between versions easily
- ✅ Use PyTorch 2.9.1 with proper RCCL support

## Installation Steps

### Step 1: Install ROCm 7.1.1

Run the installation script with sudo:

```bash
cd /home/amir/amd-dev/dn
sudo ./install_rocm_7.1.1.sh
```

This script will:
1. Add the ROCm 7.1.1 repository
2. Install ROCm 7.1.1 to `/opt/rocm-7.1.1`
3. Install RCCL 7.1.1
4. Keep your existing ROCm 7.0.1 installation intact

**Installation takes approximately 10-15 minutes** depending on your internet connection.

### Step 2: Verify Installation

After installation completes, verify both ROCm versions exist:

```bash
ls -d /opt/rocm*
```

You should see:
```
/opt/rocm -> /etc/alternatives/rocm
/opt/rocm-7.0.1
/opt/rocm-7.1.1
```

Check ROCm 7.1.1 works:

```bash
/opt/rocm-7.1.1/bin/rocm-smi
/opt/rocm-7.1.1/bin/rocminfo | grep "Name:" | head -5
```

You should see your 8 MI350X GPUs listed.

## Using Different ROCm Versions

### Method 1: Source Environment Scripts (Recommended for Development)

**To use ROCm 7.1.1:**
```bash
source /home/amir/amd-dev/dn/use-rocm-7.1.sh
```

**To use ROCm 7.0.1:**
```bash
source /home/amir/amd-dev/dn/use-rocm-7.0.sh
```

**Verify which version is active:**
```bash
echo $ROCM_PATH
rocm-smi --version
```

### Method 2: Set Environment Variables Manually

```bash
# For ROCm 7.1.1
export ROCM_PATH=/opt/rocm-7.1.1
export PATH=$ROCM_PATH/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH
export HIP_PATH=$ROCM_PATH
```

### Method 3: Change System Default

```bash
# This affects all users and sessions
sudo update-alternatives --config rocm
```

## Reinstalling PyTorch for ROCm 7.1.1

Once ROCm 7.1.1 is installed, you can reinstall your PyTorch:

```bash
# Activate ROCm 7.1.1 environment
source /home/amir/amd-dev/dn/use-rocm-7.1.sh

# Activate your conda environment
conda activate torchcomms

# Install PyTorch 2.9.1 compatible with ROCm 7.1
pip install torch==2.9.1 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/rocm6.2

# Or if you have the custom build, reinstall it
# pip install torch==2.9.1+rocm7.1.1 torchvision torchaudio --find-links <your-custom-repo>
```

### Verify PyTorch Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'HIP: {torch.version.hip}'); print(f'GPUs: {torch.cuda.device_count()}')"
```

Expected output:
```
PyTorch: 2.9.1+rocm7.1
HIP: 6.2.xxxxx
GPUs: 8
```

## Building TorchComms with ROCm 7.1.1

After switching to ROCm 7.1.1:

```bash
# Activate ROCm 7.1.1
source /home/amir/amd-dev/dn/use-rocm-7.1.sh

# Activate conda environment
conda activate torchcomms

# Navigate to torchcomms
cd /home/amir/amd-dev/dn/torchcomms

# Install in development mode
pip install -v --no-build-isolation -e .
```

This should now work without the `ncclCommWindowRegister` error!

## Testing Your Setup

### Test 1: Verify GPU Access

```bash
rocm-smi
```

### Test 2: Run Simple HIP Test

```bash
/opt/rocm-7.1.1/bin/rocminfo | grep -A 10 "Agent 2"
```

### Test 3: Test PyTorch with AMD GPUs

```python
python << 'EOF'
import torch

print(f"PyTorch: {torch.__version__}")
print(f"HIP available: {torch.cuda.is_available()}")
print(f"HIP version: {torch.version.hip}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Test tensor operation on GPU
    x = torch.rand(1000, 1000).cuda()
    y = torch.rand(1000, 1000).cuda()
    z = x @ y
    print(f"\n✓ GPU computation successful! Result shape: {z.shape}")
EOF
```

### Test 4: Run Your AllReduce Example

```bash
cd /home/amir/amd-dev/dn
torchrun --nproc_per_node=2 allReduce_pytorch.py
```

## Troubleshooting

### Issue: "ncclCommWindowRegister" error persists

**Solution:** Make sure you've:
1. Activated ROCm 7.1.1 environment: `source use-rocm-7.1.sh`
2. Reinstalled PyTorch after activating ROCm 7.1.1
3. Rebuilt torchcomms after switching to ROCm 7.1.1

### Issue: Wrong ROCm version being used

**Check which ROCm is active:**
```bash
echo $ROCM_PATH
which rocm-smi
ldd /home/amir/miniconda/envs/torchcomms/lib/python3.10/site-packages/torch/lib/libtorch_hip.so | grep rccl
```

### Issue: PyTorch can't find GPUs

**Verify environment:**
```bash
echo $HIP_PATH
echo $LD_LIBRARY_PATH
rocm-smi  # Should show 8 GPUs
```

## Quick Reference

| Action | Command |
|--------|---------|
| Install ROCm 7.1.1 | `sudo ./install_rocm_7.1.1.sh` |
| Switch to ROCm 7.1.1 | `source use-rocm-7.1.sh` |
| Switch to ROCm 7.0.1 | `source use-rocm-7.0.sh` |
| Check active ROCm | `echo $ROCM_PATH` |
| List GPUs | `rocm-smi` |
| Verify PyTorch | `python -c "import torch; print(torch.version.hip)"` |

## Additional Resources

- [ROCm Documentation](https://rocm.docs.amd.com/)
- [PyTorch ROCm Guide](https://pytorch.org/get-started/locally/)
- [AMD MI300X Documentation](https://www.amd.com/en/products/accelerators/instinct/mi300.html)

## Notes

- Both ROCm versions can coexist without conflicts
- You can switch between versions anytime using the environment scripts
- No firmware updates are needed for MI350X GPUs
- The kernel driver (amdgpu) is shared between both ROCm versions

