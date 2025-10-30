# Installation Guide

## System Requirements

- **OS**: Linux (tested on Ubuntu 20.04+)
- **Python**: 3.7 or higher
- **ROCm**: 5.0+ with RCCL installed
- **MPI**: OpenMPI or MPICH
- **Hardware**: AMD GPU(s) with InfiniBand network (recommended)

## Step 1: Verify Prerequisites

### Check Python Version
```bash
python3 --version  # Should be 3.7+
```

### Check RCCL Installation
```bash
ls -la /opt/rocm/lib/librccl.so
# or your custom RCCL build path
ls -la /home/dn/amd-dev/amd/rccl/build/release/librccl.so
```

### Check rccl-tests
```bash
ls -la /home/dn/amd-dev/amd/rccl-tests/build/all_reduce_perf
```

### Check MPI
```bash
which mpirun
mpirun --version
```

## Step 2: Install Python Dependencies

### Option A: Using pip (Recommended)

```bash
cd /home/dn/amd-dev/tools/scripts/tests/optimizer
pip install -r requirements.txt
```

### Option B: Using conda/mamba

```bash
conda create -n rccl-opt python=3.9
conda activate rccl-opt
pip install -r requirements.txt
```

### Option C: Install Individually

```bash
pip install scikit-optimize numpy scipy pandas pyyaml matplotlib seaborn tqdm colorama tabulate
```

## Step 3: Verify Installation

### Test Python Imports
```bash
python3 -c "import skopt; import pandas; import matplotlib; print('All imports successful!')"
```

### Test Syntax
```bash
cd /home/dn/amd-dev/tools/scripts/tests/optimizer
python3 -m py_compile *.py
echo "All Python files compiled successfully!"
```

### Run Help
```bash
./optimize_rccl.py --help
./analyze.py --help
```

## Step 4: Configure for Your System

### Edit config.yaml

```bash
cp config.yaml config.yaml.backup  # Backup original
nano config.yaml
```

### Key Configuration Items

1. **Test Paths**: Update paths to your RCCL and rccl-tests builds
   ```yaml
   fixed_env_vars:
     LD_LIBRARY_PATH: "/YOUR/PATH/TO/rccl/build/release:..."
     LD_PRELOAD: "/YOUR/PATH/TO/librccl.so"
   ```

2. **MPI Hosts**: Your node IPs and process counts
   ```yaml
   test_config:
     mpi_hosts: "YOUR_IP1:8,YOUR_IP2:8"
     num_processes: 16
   ```

3. **Network Interfaces**: Your InfiniBand or network interfaces
   ```yaml
   fixed_env_vars:
     NCCL_SOCKET_IFNAME: "YOUR_INTERFACE"  # e.g., enp81s0f1np1
     NCCL_IB_HCA: "YOUR_IB_DEVICES"        # e.g., mlx5_0:1,mlx5_1:1
   ```

## Step 5: Test Your Setup

### Test 1: Simple Python Test
```bash
cd /home/dn/amd-dev/tools/scripts/tests/optimizer
python3 << 'EOF'
from optimizer import ParameterOptimizer
param_space = {
    'TEST_PARAM': {'type': 'categorical', 'values': [1, 2, 3]}
}
opt = ParameterOptimizer(param_space, method='random')
params = opt.suggest()
print(f"✓ Optimizer works! Suggested: {params}")
EOF
```

### Test 2: Parser Test
```bash
python3 parser.py
```

### Test 3: Database Test
```bash
python3 << 'EOF'
from results_db import ResultsDatabase
import os
db = ResultsDatabase('test.db')
print("✓ Database created successfully")
db.close()
os.remove('test.db')
print("✓ Database operations work!")
EOF
```

### Test 4: Manual RCCL Test
Before running optimization, verify a manual RCCL test works:

```bash
# Single node test
mpirun --np 8 -H localhost:8 \
  --allow-run-as-root \
  --bind-to numa \
  /home/dn/amd-dev/amd/rccl-tests/build/all_reduce_perf \
  -b 256M -e 256M -f 2 -g 1 -n 5

# Multi-node test (adjust IPs)
mpirun --np 16 -H 172.30.160.145:8,172.30.160.150:8 \
  --allow-run-as-root \
  --bind-to numa \
  --mca oob_tcp_if_include enp81s0f1np1 \
  --mca btl_tcp_if_include enp81s0f1np1 \
  /home/dn/amd-dev/amd/rccl-tests/build/all_reduce_perf \
  -b 256M -e 256M -f 2 -g 1 -n 5
```

If this test succeeds, you're ready to run the optimizer!

## Step 6: Run First Optimization

```bash
# Quick test run (5 iterations, ~10 minutes)
./optimize_rccl.py --config config.yaml --iterations 5

# If successful, run full optimization
./optimize_rccl.py --config config.yaml --iterations 50
```

## Troubleshooting Installation

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'skopt'`

**Solution**:
```bash
pip install --upgrade scikit-optimize
# or
pip install --user scikit-optimize
```

### Permission Errors

**Problem**: `Permission denied` when running scripts

**Solution**:
```bash
chmod +x optimize_rccl.py analyze.py
```

### Path Issues

**Problem**: `FileNotFoundError: config.yaml not found`

**Solution**:
```bash
# Always run from the optimizer directory
cd /home/dn/amd-dev/tools/scripts/tests/optimizer
./optimize_rccl.py --config config.yaml
```

### MPI Errors

**Problem**: `mpirun: command not found`

**Solution**:
```bash
# Install OpenMPI
sudo apt-get install openmpi-bin libopenmpi-dev

# or add to PATH
export PATH=/opt/openmpi/bin:$PATH
```

### RCCL Not Found

**Problem**: `error while loading shared libraries: librccl.so`

**Solution**:
```bash
# Add RCCL to library path
export LD_LIBRARY_PATH=/path/to/rccl/lib:$LD_LIBRARY_PATH

# Or update config.yaml with correct path
```

### Database Errors

**Problem**: `sqlite3.OperationalError: database is locked`

**Solution**:
- Only run one optimizer instance at a time
- Or use different output directories for multiple runs

### Plotting Errors

**Problem**: `No display name and no $DISPLAY environment variable`

**Solution**:
```bash
# Use non-interactive backend
export MPLBACKEND=Agg
# Then run analysis with --export-report to save plots
./analyze.py results.db --export-report report/
```

## Verification Checklist

Before running optimization, verify:

- [ ] Python 3.7+ installed
- [ ] All Python dependencies installed (`pip list | grep scikit-optimize`)
- [ ] RCCL library accessible
- [ ] rccl-tests built and accessible
- [ ] MPI working (test with `mpirun hostname`)
- [ ] Network interfaces configured
- [ ] config.yaml updated for your system
- [ ] Manual RCCL test succeeds
- [ ] Scripts are executable (`ls -la *.py`)
- [ ] No syntax errors (`python3 -m py_compile *.py`)

## Next Steps

Once installed:
1. Read `QUICKSTART.md` for a quick tutorial
2. Read `README.md` for detailed documentation
3. Run your first optimization!

## Getting Help

If you encounter issues:
1. Check this troubleshooting section
2. Verify all prerequisites
3. Test each component individually
4. Check system logs: `dmesg | tail`
5. Check MPI logs in output directories

## Uninstall

To remove:
```bash
# Remove Python packages
pip uninstall scikit-optimize pandas matplotlib seaborn pyyaml colorama tabulate tqdm

# Remove output directories
rm -rf optimization_runs/
rm -rf test_outputs/

# The optimizer itself is just scripts, no system install needed
```

Happy optimizing! 🚀


