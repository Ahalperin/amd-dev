# RCCL Parameter Optimizer

An intelligent parameter optimization framework for RCCL (ROCm Communication Collectives Library) tests using Bayesian Optimization to efficiently find the best parameter combinations for maximum performance.

## Overview

This tool automates the process of finding optimal RCCL configuration parameters by:
- **Smart Search**: Uses Bayesian Optimization instead of exhaustive grid search
- **Automated Testing**: Executes RCCL tests with different parameter combinations
- **Performance Tracking**: Stores all results in a database for analysis
- **Result Analysis**: Provides visualization and statistical analysis
- **Efficiency**: Typically finds near-optimal configurations in 30-100 iterations

## Features

✅ **Multiple Optimization Methods**
- Bayesian Optimization (recommended) - intelligent search
- Random Search - baseline comparison
- Grid Search - exhaustive search

✅ **Flexible Configuration**
- YAML-based parameter space definition
- Support for categorical, integer, and continuous parameters
- Easy to add/remove parameters

✅ **Robust Execution**
- Automatic error handling and recovery
- Timeout protection
- Detailed logging of all runs

✅ **Comprehensive Analysis**
- Convergence plots
- Parameter importance analysis
- Performance distribution visualization
- Validation runs for best configuration

## Installation

### 1. Install Dependencies

```bash
cd /home/dn/amd-dev/tools/scripts/tests/optimizer
pip install -r requirements.txt
```

Dependencies include:
- `scikit-optimize` - Bayesian optimization
- `pyyaml` - Configuration parsing
- `pandas` - Data manipulation
- `matplotlib/seaborn` - Visualization
- `colorama` - Terminal colors
- `tabulate` - Pretty tables

### 2. Configure Parameters

Edit `config.yaml` to define:
- Test configuration (which test, message sizes, MPI settings)
- Fixed environment variables (paths, network config)
- Parameters to optimize (with their ranges/values)
- Optimization settings (method, iterations, objective)

## Quick Start

### Basic Usage

```bash
# 1. Edit config.yaml to match your setup
nano config.yaml

# 2. Run optimization
./optimize_rccl.py --config config.yaml

# 3. Analyze results
./analyze.py optimization_runs/run_TIMESTAMP/rccl_optimization_results.db --all
```

### Example: Optimize All-Reduce Performance

```bash
# Run 50 iterations of Bayesian optimization
./optimize_rccl.py --config config.yaml --iterations 50

# View summary
./analyze.py optimization_runs/run_*/rccl_optimization_results.db --summary

# Generate full report
./analyze.py optimization_runs/run_*/rccl_optimization_results.db --export-report analysis_report
```

## Configuration Guide

### Test Configuration

```yaml
test_config:
  test_name: "all_reduce_perf"      # RCCL test to run
  minbytes: "256M"                  # Message size range
  maxbytes: "256M"
  stepfactor: 2
  iterations: 20                    # Test iterations
  warmup_iters: 5
  
  # MPI settings
  mpi_hosts: "172.30.160.145:8,172.30.160.150:8"
  num_processes: 16
  bind_to: "numa"
```

### Fixed Environment Variables

These are set for all runs but not optimized:

```yaml
fixed_env_vars:
  LD_LIBRARY_PATH: "/home/dn/amd-dev/amd/rccl/build/release:..."
  NCCL_IB_HCA: "ionic_0:1,ionic_1:1,..."
  NCCL_DEBUG: "VERSION"
  # ... other fixed settings
```

### Optimization Parameters

Define parameters to optimize with their type and range:

```yaml
optimize_params:
  NCCL_IB_QPS_PER_CONNECTION:
    type: "categorical"
    values: [1, 2, 4, 8]
    description: "Number of QPs per IB connection"
  
  NCCL_IB_TC:
    type: "categorical"
    values: [104, 106, 160, 192]
    description: "InfiniBand Traffic Class"
  
  NCCL_BUFFSIZE:
    type: "integer"
    range: [524288, 4194304]  # 512KB to 4MB
    description: "Buffer size in bytes"
```

### Optimization Settings

```yaml
optimization:
  method: "bayesian"              # bayesian, random, or grid
  max_iterations: 50              # Maximum iterations to run
  n_initial_points: 10            # Random samples before Bayesian
  objective: "busbw_oop"          # Metric to optimize
  direction: "maximize"           # maximize or minimize
  validation_runs: 5              # Validate best config
  early_stopping_patience: 15     # Stop if no improvement
```

## Usage Examples

### Different Optimization Methods

```bash
# Bayesian optimization (recommended)
./optimize_rccl.py --config config.yaml --method bayesian --iterations 50

# Random search (baseline)
./optimize_rccl.py --config config.yaml --method random --iterations 100

# Grid search (exhaustive)
./optimize_rccl.py --config config.yaml --method grid
```

### Analysis Commands

```bash
# Print summary statistics
./analyze.py results.db --summary

# Show convergence plot
./analyze.py results.db --plot-convergence

# Show parameter importance
./analyze.py results.db --plot-importance

# Show parameter distributions
./analyze.py results.db --plot-distribution

# Generate comprehensive report
./analyze.py results.db --export-report ./report/

# Show everything
./analyze.py results.db --all
```

## Output Files

After optimization, you'll find:

```
optimization_runs/
└── run_YYYYMMDD_HHMMSS/
    ├── rccl_optimization_results.db    # SQLite database with all results
    ├── best_config.txt                 # Best configuration (shell export format)
    └── test_outputs/                   # Detailed test outputs
        ├── run_0001/
        │   ├── output.log
        │   ├── error.log
        │   └── parameters.txt
        ├── run_0002/
        └── ...
```

## Understanding Results

### Metrics

The optimizer tracks these metrics:
- **busbw_oop** - Bus bandwidth (out-of-place) [usually the main objective]
- **busbw_ip** - Bus bandwidth (in-place)
- **algbw_oop** - Algorithmic bandwidth (out-of-place)
- **algbw_ip** - Algorithmic bandwidth (in-place)
- **time_oop/ip** - Execution time in microseconds

### Best Configuration

The best configuration is saved in `best_config.txt` and can be used directly:

```bash
# Load best configuration
source best_config.txt

# Run test with best config
mpirun ... /path/to/rccl-tests/build/all_reduce_perf ...
```

## Advanced Usage

### Custom Objective Function

Edit `config.yaml` to optimize different metrics:

```yaml
optimization:
  objective: "algbw_oop"    # Optimize algorithmic bandwidth
  # or
  objective: "busbw_ip"     # Optimize in-place bus bandwidth
```

### Multi-Test Optimization

To optimize across multiple message sizes, create separate configs:

```bash
# Small messages
./optimize_rccl.py --config config_small.yaml

# Large messages  
./optimize_rccl.py --config config_large.yaml

# Compare results
./analyze.py run1/results.db --summary
./analyze.py run2/results.db --summary
```

### Parallel Execution (Future Enhancement)

Currently runs sequentially. For parallel execution across multiple nodes, you could:
1. Run multiple optimizer instances with different random seeds
2. Merge databases afterward for analysis

## Troubleshooting

### Tests Keep Failing

Check:
1. MPI hosts are accessible: `ping 172.30.160.145`
2. Network interfaces are correct in config
3. RCCL library paths are correct
4. Run a test manually first to verify setup

### Out of Memory

Reduce:
- `test_config.iterations` - fewer test iterations
- `optimization.max_iterations` - fewer optimization iterations
- Message sizes in test config

### Slow Performance

- Use `NCCL_DEBUG: "VERSION"` instead of "INFO" or "TRACE"
- Reduce `test_config.iterations`
- Increase `timeout` if tests are being killed early

### Database Locked

Only one process can write to the database at a time. Close other instances or use different output directories.

## Tips for Best Results

1. **Start Small**: Begin with 20-30 iterations to understand the parameter space
2. **Use Bayesian**: Bayesian optimization is typically 3-5x more efficient than random search
3. **Focus on Key Parameters**: Start with 3-5 most impactful parameters
4. **Validate Results**: Always run validation (set `validation_runs: 5`)
5. **Check Convergence**: If early stopping triggers, you've likely found near-optimal config
6. **Multiple Runs**: For critical workloads, run optimization 2-3 times with different random seeds

## Parameter Recommendations

### High-Impact Parameters (start here)
- `NCCL_IB_QPS_PER_CONNECTION` - [1, 2, 4, 8]
- `NCCL_IB_TC` - [104, 106, 160, 192]
- `RCCL_LL128_FORCE_ENABLE` - [0, 1]
- `NCCL_PXN_DISABLE` - [0, 1]

### Medium-Impact Parameters
- `NCCL_IB_FIFO_TC` - [104, 160, 192, 224]
- `NCCL_IB_USE_INLINE` - [0, 1]
- `NET_OPTIONAL_RECV_COMPLETION` - [0, 1]

### Advanced Parameters (use with caution)
- `NCCL_BUFFSIZE` - Buffer size tuning
- `NCCL_ALGO` - Force specific algorithm
- `NCCL_MIN_NCHANNELS` / `NCCL_MAX_NCHANNELS`

## Contributing

To add new parameters:
1. Add to `optimize_params` in `config.yaml`
2. Specify type (categorical, integer, real)
3. Define values or range
4. Run optimization!

## Support

For issues or questions:
1. Check the output logs in `optimization_runs/`
2. Review database with: `sqlite3 results.db "SELECT * FROM optimization_runs;"`
3. Verify test works manually before optimizing

## License

Part of the AMD ROCm development tools.

## References

- [RCCL Documentation](https://github.com/ROCmSoftwarePlatform/rccl)
- [Bayesian Optimization](https://scikit-optimize.github.io/)
- [NCCL Environment Variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)


