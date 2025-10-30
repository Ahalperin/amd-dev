# Quick Start Guide

Get started with RCCL parameter optimization in 5 minutes!

## Prerequisites

- RCCL and rccl-tests installed
- Python 3.7+
- MPI environment configured
- Multiple nodes accessible (optional but recommended)

## Installation

```bash
# Navigate to optimizer directory
cd /home/dn/amd-dev/tools/scripts/tests/optimizer

# Install dependencies
pip install -r requirements.txt
```

## Step 1: Configure Your Environment

Edit `config.yaml` to match your setup:

```bash
nano config.yaml
```

**Key settings to update:**

1. **MPI Hosts** (line ~20):
   ```yaml
   mpi_hosts: "YOUR_HOST_IP:8,YOUR_HOST_IP2:8"  # Update with your IPs
   ```

2. **Network Interface** (line ~24):
   ```yaml
   oob_tcp_if: "YOUR_NETWORK_INTERFACE"  # e.g., enp81s0f1np1
   btl_tcp_if: "YOUR_NETWORK_INTERFACE"
   ```

3. **Paths** (line ~30):
   ```yaml
   LD_LIBRARY_PATH: "/path/to/your/rccl/build/release:..."
   LD_PRELOAD: "/path/to/your/librccl.so"
   ```

4. **IB HCA** (line ~38):
   ```yaml
   NCCL_IB_HCA: "YOUR_IB_DEVICES"  # e.g., "mlx5_0:1,mlx5_1:1,..."
   ```

## Step 2: Test Your Setup

Before running full optimization, verify your setup works:

```bash
# Run a single RCCL test manually to verify
mpirun --np 8 -H YOUR_HOST:8 \
  /home/dn/amd-dev/amd/rccl-tests/build/all_reduce_perf \
  -b 256M -e 256M -f 2 -g 1 -n 5
```

If this works, you're ready to optimize!

## Step 3: Run Your First Optimization

Start with a quick 20-iteration run:

```bash
# Run optimization (should take 30-60 minutes depending on test size)
./optimize_rccl.py --config config.yaml --iterations 20
```

You'll see:
- Parameter configurations being tested
- Real-time performance results
- Progress updates every 5 iterations
- Final best configuration

## Step 4: Analyze Results

```bash
# Find your results database
ls -lt optimization_runs/

# View summary
./analyze.py optimization_runs/run_TIMESTAMP/rccl_optimization_results.db --summary

# Generate full report with plots
./analyze.py optimization_runs/run_TIMESTAMP/rccl_optimization_results.db \
  --export-report my_analysis
```

## Step 5: Use Best Configuration

The best configuration is saved in `best_config.txt`:

```bash
# View best config
cat optimization_runs/run_TIMESTAMP/best_config.txt

# Use it in your tests
source optimization_runs/run_TIMESTAMP/best_config.txt
mpirun ... /path/to/rccl-tests/build/all_reduce_perf ...
```

## Example Output

### During Optimization:
```
================================================================================
RCCL Parameter Optimization
================================================================================

Configuration:
  Test: all_reduce_perf
  Message size: 256M - 256M
  Processes: 16
  Method: bayesian
  Objective: busbw_oop (maximize)
  Max iterations: 20

────────────────────────────────────────────────────────────────────────────────
Iteration 1/20
────────────────────────────────────────────────────────────────────────────────

Testing parameters:
  NCCL_IB_QPS_PER_CONNECTION = 1
  NCCL_IB_TC = 104
  RCCL_LL128_FORCE_ENABLE = 1

✓ Test completed successfully
  Bus BW (out-of-place): 343.57 GB/s
  Bus BW (in-place):     348.93 GB/s
  Objective (busbw_oop): 343.57
```

### Final Summary:
```
================================================================================
Optimization Complete
================================================================================

  Total time: 45.2 minutes
  Total runs: 20
  Successful runs: 19
  Success rate: 95.0%

Best Configuration Found:
  busbw_oop: 368.45 GB/s
  Bus BW (out-of-place): 368.45 GB/s
  Bus BW (in-place): 372.18 GB/s

Optimal Parameters:
  NCCL_IB_QPS_PER_CONNECTION = 4
  NCCL_IB_TC = 160
  NCCL_IB_FIFO_TC = 192
  RCCL_LL128_FORCE_ENABLE = 1
  NCCL_PXN_DISABLE = 0
  NCCL_IB_USE_INLINE = 1

  → Best configuration saved to: best_config.txt
```

## Common Issues

### "Connection refused" or MPI errors
- Check hosts are accessible: `ping YOUR_HOST_IP`
- Verify SSH keys are set up for passwordless access
- Test MPI manually: `mpirun -H HOST:8 hostname`

### "Library not found"
- Update paths in `config.yaml`
- Verify: `ls -la /path/to/librccl.so`
- Check: `echo $LD_LIBRARY_PATH`

### "No improvement found"
- Normal! It means you've reached near-optimal configuration
- Early stopping prevents wasting time
- Results are still valid

### Tests timeout
- Increase `timeout` in config (default: 300s)
- Reduce test iterations: `iterations: 10` instead of 20
- Check system isn't overloaded

## Next Steps

1. **Run Longer Optimization**: Use 50-100 iterations for production
   ```bash
   ./optimize_rccl.py --config config.yaml --iterations 100
   ```

2. **Optimize Different Tests**: Try `alltoall_perf`, `all_gather_perf`, etc.
   ```yaml
   test_name: "alltoall_perf"
   ```

3. **Optimize Different Message Sizes**: Create configs for small/large messages
   ```yaml
   minbytes: "1M"
   maxbytes: "1M"
   ```

4. **Add More Parameters**: Explore additional RCCL parameters in `config.yaml`

5. **Compare Methods**: Try random search vs Bayesian
   ```bash
   ./optimize_rccl.py --config config.yaml --method random --iterations 50
   ```

## Tips for Success

✅ **Start simple** - Optimize 3-5 parameters first
✅ **Use Bayesian** - Most efficient method (3-5x faster than random)
✅ **Validate** - Set `validation_runs: 5` in config
✅ **Monitor** - Watch for consistent improvement trends
✅ **Save results** - Keep databases for future reference
✅ **Document** - Note which config works best for your workload

## Getting Help

Check the full README.md for:
- Detailed parameter descriptions
- Advanced usage examples
- Troubleshooting guide
- Best practices

Happy optimizing! 🚀


