# Troubleshooting Guide

## Issue: Optimizer Hangs on First Iteration

### Likely Causes

The optimizer isn't actually hung - **the RCCL test is running but taking a long time**. Here's why:

**Your default config runs:**
- 256M message size (large!)
- 20 iterations + 5 warmup = 25 iterations
- 16 processes across 2 nodes
- **Expected time: 3-5 minutes per test**

### Solution 1: Use Quick Test Config (Recommended First Step)

I've created `config_quick_test.yaml` with faster settings:

```bash
cd /home/dn/amd-dev/tools/scripts/tests/optimizer

# Run with quick test config (completes in ~2 minutes per test)
./optimize_rccl.py --config config_quick_test.yaml --iterations 5
```

**Quick test differences:**
- 1M message size (vs 256M)
- 5 iterations + 1 warmup (vs 20 + 5)
- 2 parameters only (vs 9)
- Random search (faster than Bayesian for small tests)

### Solution 2: Test Single RCCL Run

Run a single test manually to see timing:

```bash
cd /home/dn/amd-dev/tools/scripts/tests/optimizer

# This will show you exactly what the optimizer runs
./run_single_test.sh
```

This should complete in 30-60 seconds. If it hangs here, you have an RCCL/MPI issue.

### Solution 3: Modify Default Config for Faster Iterations

Edit `config.yaml`:

```yaml
test_config:
  minbytes: "64M"     # Reduced from 256M
  maxbytes: "64M"     # Reduced from 256M
  iterations: 10      # Reduced from 20
  warmup_iters: 2     # Reduced from 5
```

This will make each test run ~60-90 seconds instead of 3-5 minutes.

### Solution 4: Add Real-Time Progress

The optimizer shows output, but it may not be flushing. Run with:

```bash
# Force unbuffered output
python3 -u ./optimize_rccl.py --config config_quick_test.yaml --iterations 5 2>&1 | tee optimization.log
```

### Diagnostic Steps

1. **Verify setup works:**
   ```bash
   ./test_setup.py
   ```
   Should show all ✓ PASS

2. **Run single test manually:**
   ```bash
   ./run_single_test.sh
   ```
   Should complete in <60 seconds

3. **Check if optimizer created output:**
   ```bash
   ls -la optimization_runs/
   ```
   Look for run directories

4. **Monitor what's happening:**
   ```bash
   # In another terminal
   watch -n 1 'ps aux | grep -E "(mpirun|all_reduce)" | grep -v grep'
   ```

### Understanding Test Timing

| Config | Message Size | Iterations | Expected Time |
|--------|-------------|------------|---------------|
| Quick Test | 1M | 5+1 | ~30 sec |
| Fast | 64M | 10+2 | ~90 sec |
| Default | 256M | 20+5 | ~240 sec (4 min) |
| Production | 256M | 20+5 | ~300 sec (5 min) |

### Common Issues

#### Issue: Actually Hung (No Progress After 10 Minutes)

**Symptoms:**
- No output files created in `optimization_runs/`
- `ps aux | grep mpirun` shows nothing
- No CPU/GPU activity

**Fixes:**
1. Check Python syntax:
   ```bash
   python3 -m py_compile optimize_rccl.py
   ```

2. Run with full traceback:
   ```bash
   python3 -u optimize_rccl.py --config config_quick_test.yaml --iterations 5
   ```

3. Check for import errors:
   ```bash
   python3 -c "from optimizer import *; from executor import *; print('OK')"
   ```

#### Issue: MPI Hangs

**Symptoms:**
- `./run_single_test.sh` hangs
- Simple MPI test passes but RCCL test hangs

**Fixes:**
1. Check GPU availability:
   ```bash
   rocm-smi
   ```

2. Test without optimizer:
   ```bash
   mpirun --np 8 -H 172.30.160.145:8 --allow-run-as-root \
     /home/dn/amd-dev/amd/rccl-tests/build/all_reduce_perf \
     -b 1M -e 1M -f 2 -g 1 -n 1
   ```

3. Check library paths:
   ```bash
   ldd /home/dn/amd-dev/amd/rccl-tests/build/all_reduce_perf | grep "not found"
   ```

#### Issue: Permission Denied

**Fix:**
```bash
chmod +x optimize_rccl.py analyze.py test_setup.py run_single_test.sh
```

#### Issue: Can't Find Modules

**Symptoms:**
```
ModuleNotFoundError: No module named 'skopt'
```

**Fix:**
```bash
pip install -r requirements.txt
# or
pip install --user -r requirements.txt
```

### Recommended Workflow for First Run

1. **Test setup:**
   ```bash
   ./test_setup.py
   ```

2. **Quick test (5 minutes total):**
   ```bash
   ./optimize_rccl.py --config config_quick_test.yaml --iterations 5
   ```

3. **If successful, run production config:**
   ```bash
   ./optimize_rccl.py --config config.yaml --iterations 20
   ```

### Monitoring Progress

While optimizer runs:

**Terminal 1 - Run optimizer:**
```bash
./optimize_rccl.py --config config_quick_test.yaml --iterations 5
```

**Terminal 2 - Monitor processes:**
```bash
watch -n 2 'ps aux | grep -E "(optimize|mpirun|all_reduce)" | grep -v grep'
```

**Terminal 3 - Watch output files:**
```bash
watch -n 5 'find optimization_runs -name "*.log" -exec tail -5 {} \;'
```

### Still Having Issues?

1. **Check logs:**
   ```bash
   ls -lrt optimization_runs/run_*/test_outputs/run_*/
   cat optimization_runs/run_*/test_outputs/run_0001/output.log
   ```

2. **Increase verbosity in config:**
   ```yaml
   output:
     verbosity: 2  # 0=quiet, 1=normal, 2=verbose
   
   fixed_env_vars:
     NCCL_DEBUG: "INFO"  # More debug output
   ```

3. **Run in debug mode:**
   ```bash
   python3 -u -m pdb optimize_rccl.py --config config_quick_test.yaml --iterations 5
   ```

4. **Test each component separately:**
   ```bash
   # Test optimizer
   python3 optimizer.py
   
   # Test parser
   python3 parser.py
   
   # Test executor
   python3 << 'EOF'
   from executor import RCCLTestExecutor
   import yaml
   with open('config_quick_test.yaml') as f:
       config = yaml.safe_load(f)
   executor = RCCLTestExecutor(config['test_config'])
   print(executor.build_mpirun_command({'TEST': '1'}, 1))
   EOF
   ```

### Getting Help

If still stuck, gather this info:

```bash
# System info
uname -a
rocm-smi
mpirun --version

# Optimizer status
./test_setup.py > setup_test_results.txt 2>&1

# Try single test
./run_single_test.sh > single_test_results.txt 2>&1

# Check for output
find optimization_runs -type f -ls
```

Then share:
- `setup_test_results.txt`
- `single_test_results.txt`
- Output from optimizer (first 100 lines)
- Any error messages

## Quick Reference

```bash
# Fastest test (30 sec per iteration, 5 iterations = 3 min total)
./optimize_rccl.py --config config_quick_test.yaml --iterations 5

# Production run (4 min per iteration, 20 iterations = 80 min total)
./optimize_rccl.py --config config.yaml --iterations 20

# Debug mode
python3 -u ./optimize_rccl.py --config config_quick_test.yaml --iterations 5 2>&1 | tee debug.log

# Check if it's really running
ps aux | grep -E "(mpirun|all_reduce)"

# Kill if needed
pkill -9 -f optimize_rccl
pkill -9 mpirun
```


