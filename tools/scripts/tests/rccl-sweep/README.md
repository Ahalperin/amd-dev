# RCCL Test Sweep Tool

A systematic test automation tool for running RCCL collective tests across multiple configurations including node scaling and channel sweeps.

## Features

- **Multiple Collectives**: Support for all standard RCCL collectives (all_reduce, reduce_scatter, all_gather, alltoall, broadcast, reduce)
- **Node Scaling**: Automatically scales from 1 to N nodes based on available servers
- **Channel Sweep**: Configurable channel range with step (e.g., 4-64 step 4)
- **Full Logging**: Stores command-line and complete test output for each run
- **SQLite Database**: All results stored in queryable database
- **Progress Tracking**: Real-time progress with ETA estimates

## Installation

```bash
cd /home/amir/amd-dev/tools/scripts/tests/rccl-sweep

# Create and activate conda environment
conda env create -f environment.yml
conda activate rccl-sweep

# Make scripts executable
chmod +x rccl_sweep.py
```

**Note**: Always activate the conda environment before running the tool:
```bash
conda activate rccl-sweep
```

## Quick Start

1. **Create servers file** with your node IPs:
   ```bash
   # Create servers.txt with one IP per line
   # Comments after IP or lines starting with # are ignored
   ```
   Example `servers.txt`:
   ```
   172.30.160.145
   172.30.160.150
   172.30.160.204  # node 3
   #172.30.160.193  # commented out - will be skipped
   ```

2. **Set MY_PATH** environment variable pointing to directory with RCCL libs and rccl-tests executables:
   ```bash
   export MY_PATH=/home/amir/rccl-bins/develop
   ```
   Or configure directly in `sweep_config.yaml`:
   ```yaml
   paths:
     rccl_path: "/home/amir/rccl-bins/develop"
   ```

3. **Run a sweep**:
   ```bash
   # Run all collectives across all nodes with channel sweep
   # Uses ./servers.txt by default
   ./rccl_sweep.py --channels 4:64:4

   # Run single collective on specific nodes
   ./rccl_sweep.py --collective all_reduce --nodes 2 --channels 4:64:4

   # Dry run to see what would be executed
   ./rccl_sweep.py --channels 4:64:4 --dry-run
   ```

## CLI Reference

```
usage: rccl_sweep.py [-h] [--servers SERVERS] --channels CHANNELS
                     [--collective {all,all_reduce,reduce_scatter,all_gather,alltoall,broadcast,reduce}]
                     [--nodes NODES] [--config CONFIG] [--min-bytes MIN_BYTES]
                     [--max-bytes MAX_BYTES] [--dry-run] [--verbose]

Options:
  --servers, -s       Path to servers.txt file with node IPs (default: ./servers.txt)
  --channels, -c      Channel range as MIN:MAX:STEP, e.g., "4:64:4" (required)
  --collective        Specific collective or "all" (default: all)
  --nodes, -n         Node count or range: N or MIN-MAX (e.g., 2 or 1-4)
  --config            Path to config file (default: sweep_config.yaml)
  --min-bytes         Override minimum message size (e.g., 1M, 256M)
  --max-bytes         Override maximum message size (e.g., 1G, 16G)
  --dry-run, -d       Show full commands without executing
  --verbose, -v       Verbose output
```

## Examples

### Run Full Sweep
```bash
# All collectives, all nodes (1-9), channels 4-64 step 4
# Uses ./servers.txt by default
./rccl_sweep.py -c 4:64:4
```

### Run Specific Collective
```bash
# Only all_reduce
./rccl_sweep.py --collective all_reduce -c 4:64:4
```

### Run on Specific Node Count
```bash
# Only 2-node configuration
./rccl_sweep.py --nodes 2 -c 4:64:4
```

### Run on Node Range
```bash
# Run on 1 and 2 nodes
./rccl_sweep.py --nodes 1-2 -c 4:64:4

# Run on 2, 3, and 4 nodes
./rccl_sweep.py --nodes 2-4 -c 4:64:4
```

### Custom Message Sizes
```bash
# Override to run 256M to 1G only
./rccl_sweep.py -c 4:64:4 --min-bytes 256M --max-bytes 1G
```

### Dry Run Mode
```bash
# See what commands would be executed
./rccl_sweep.py -c 4:64:4 --dry-run
```

### Custom Servers File
```bash
# Use a different servers file
./rccl_sweep.py -s /path/to/my_servers.txt -c 4:64:4
```

## Output Structure

```
sweep_results/run_YYYYMMDD_HHMMSS/
├── sweep_results.db              # SQLite database with all metrics
├── summary.csv                   # CSV export for easy analysis
└── outputs/
    ├── all_reduce_perf_1node_4ch/
    │   ├── command.txt           # Full mpirun command
    │   └── output.log            # Complete test output
    ├── all_reduce_perf_1node_8ch/
    │   ├── command.txt
    │   └── output.log
    └── ...
```

## Configuration (sweep_config.yaml)

Key configuration sections:

### Paths
```yaml
paths:
  # Single directory containing RCCL libs and rccl-tests executables
  rccl_path: "${MY_PATH}"  # Set MY_PATH env var or use absolute path
```

**Note**: Set `MY_PATH` before running:
```bash
export MY_PATH=/home/amir/rccl-bins/develop
```

### Test Defaults
```yaml
test_defaults:
  min_bytes: "1M"
  max_bytes: "16G"
  step_factor: 2
  iterations: 20
  warmup_iters: 5
  timeout: 600
```

### Environment Variables
All NCCL/RCCL environment variables are configured in the `env_vars` section.

## Database Queries

Access results using SQLite:

```bash
# Open database
sqlite3 sweep_results/run_*/sweep_results.db

# Get all successful runs
SELECT collective, num_nodes, num_channels, avg_busbw, max_busbw 
FROM sweep_runs WHERE status='success' ORDER BY avg_busbw DESC;

# Get best configuration per collective
SELECT collective, num_nodes, num_channels, MAX(avg_busbw) as best_busbw 
FROM sweep_runs WHERE status='success' GROUP BY collective;

# Export to CSV
.mode csv
.output results.csv
SELECT * FROM sweep_runs WHERE status='success';
```

## Plotting Results

Generate bus bandwidth vs message size graphs using `plot_busbw.py`:

```bash
# Plot specific collective and node count
python plot_busbw.py all_reduce_perf 1 -o all_reduce_1node.png
python plot_busbw.py all_reduce_perf 2 -o all_reduce_2node.png

# Use custom database path
python plot_busbw.py reduce_scatter_perf 1 --db /path/to/sweep_results.db -o output.png

# Display interactively (no -o flag)
python plot_busbw.py alltoall_perf 2
```

**Collective types**: `all_reduce_perf`, `reduce_scatter_perf`, `all_gather_perf`, `alltoall_perf`, `broadcast_perf`, `reduce_perf`

The script will:
- Query the database for matching runs
- Plot bus bandwidth (in-place) vs message size on a log₂ x-axis
- Show multiple sessions as separate curves if data spans multiple sweeps

## Test Matrix

For a full sweep with 9 servers and channels 4:64:4:
- 6 collectives × 9 node counts × 16 channel values = **864 tests**
- Estimated time: ~3 minutes per test = **~43 hours**

For focused testing:
- Single collective: 144 tests (~7 hours)
- Single node count: 96 tests (~5 hours)
- Both: 16 tests (~48 minutes)

## Environment Variables

The tool sets these NCCL environment variables (from sweep_config.yaml):

| Variable | Default | Description |
|----------|---------|-------------|
| NCCL_MIN_NCHANNELS | (swept) | Minimum channels |
| NCCL_MAX_NCHANNELS | (swept) | Maximum channels |
| NCCL_IB_HCA | ionic_0:1,... | InfiniBand HCA config |
| NCCL_IB_TC | 104 | Traffic class |
| NCCL_IB_QPS_PER_CONNECTION | 2 | QPs per connection |
| NCCL_IB_SPLIT_DATA_ON_QPS | 1 | Split data on QPs |
| ... | ... | See sweep_config.yaml |

## Troubleshooting

### Tests timing out
- Increase `timeout` in sweep_config.yaml
- Reduce message size range

### Connection errors
- Verify servers are accessible: `ping <server_ip>`
- Check SSH keys: `ssh <server_ip> hostname`
- Verify MPI interfaces in config

### Library not found
- Update paths in sweep_config.yaml
- Ensure LD_LIBRARY_PATH includes RCCL lib directory

## Files

| File | Description |
|------|-------------|
| rccl_sweep.py | Main CLI entry point |
| analyze_sweep.py | Results analysis tool |
| plot_busbw.py | Generate bus bandwidth graphs |
| sweep_config.yaml | Default configuration |
| sweep_executor.py | Test execution engine |
| sweep_parser.py | Output parsing |
| sweep_db.py | Database operations |
| servers.txt | Server IP list (user-created) |
| environment.yml | Conda environment specification |
| requirements.txt | Pip dependencies (fallback) |

## License

Part of AMD ROCm development tools.

