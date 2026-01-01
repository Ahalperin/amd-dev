# RCCL-Sweep User Workflow Guide

## Background

**RCCL-Sweep** is a systematic test automation tool for benchmarking and tuning AMD RCCL (ROCm Communication Collectives Library) performance across different configurations. The tool addresses the challenge of finding optimal RCCL parameters (channels, algorithms, protocols) for various collective operations across different cluster topologies and message sizes.

### Purpose

- **Performance Benchmarking**: Run systematic sweeps of RCCL collective operations across multiple nodes, GPUs, and configurations
- **Auto-Tuning**: Automatically find the best algorithm/protocol/channel configurations for each message size range
- **Hotspot Detection**: Identify performance degradation points ("hotspots") where tuning can improve throughput
- **Tuner Configuration Generation**: Generate RCCL tuner configuration files that can be loaded at runtime for optimized performance

### Key Concepts

| Term | Description |
|------|-------------|
| **Collective** | Communication patterns like `all_reduce`, `all_gather`, `reduce_scatter`, `alltoall`, `broadcast`, `reduce` |
| **Channels** | Number of parallel communication channels (affects bandwidth utilization) |
| **Algorithm** | Communication algorithm: `RING`, `TREE`, `DIRECT` |
| **Protocol** | Data transfer protocol: `SIMPLE`, `LL` (Low Latency), `LL128` |
| **Bus Bandwidth (busbw)** | Effective bandwidth achieved by the collective operation (GB/s) |
| **Hotspot** | Message size range where performance drops unexpectedly below expected levels |

---

## User Workflow

### Prerequisites

1. **Setup conda environment**:
   ```bash
   cd /home/amir/amd-dev/tools/scripts/tests/rccl-sweep
   conda env create -f environment.yml
   conda activate rccl-sweep
   ```

2. **Configure servers** - Create `servers.txt` with cluster node IPs:
   ```
   172.30.160.145
   172.30.160.150
   172.30.160.204  # comments allowed
   #172.30.160.193  # commented nodes are skipped
   ```

3. **Set RCCL path** - Point to RCCL libraries and rccl-tests executables:
   ```bash
   export MY_PATH=/path/to/rccl-bins
   ```

---

### Workflow Option A: Manual Step-by-Step Approach

For fine-grained control, run each step individually:

#### Step 1: Run Sweeps

Execute performance sweeps across configurations:

```bash
# Basic sweep with channel range
./rccl_sweep.py --channels 4:64:4 --nodes 1-2

# Sweep specific collectives with algorithm/protocol
./rccl_sweep.py --collective all_reduce,all_gather \
    --channels 32:256:32 \
    --algo RING,TREE --proto SIMPLE \
    --nodes 1-2

# Dry run to preview commands
./rccl_sweep.py --channels 4:64:4 --dry-run
```

#### Step 2: Filter Error Entries

Remove test results that had errors:

```bash
# Check for errors
python filter_metrics.py sweep_results/run_*/metrics.csv --count

# Remove error entries from metrics
python filter_metrics.py sweep_results/run_*/metrics.csv --prune-err
```

#### Step 3: Merge Results

Combine metrics from multiple sweep runs:

```bash
python merge_metrics.py --base-path ./sweep_results -o sweep_results/merged_metrics.csv
```

#### Step 4: Optimize Metrics

Select best configuration for each (collective, nodes, message size) combination:

```bash
python optimize_metrics.py sweep_results/merged_metrics.csv -o sweep_results/optimized_metrics.csv
```

#### Step 5: Detect Hotspots

Identify performance degradation areas:

```bash
python detect_hotspots.py sweep_results/optimized_metrics.csv \
    --threshold 0.10 \
    -o sweep_results/hotspots_report.csv \
    --verbose
```

#### Step 6: Run Targeted Sweeps (if hotspots found)

Based on hotspot report, run additional sweeps on problematic size ranges with different configurations.

#### Step 7: Generate Tuner Configuration

Create the final RCCL tuner config file:

```bash
python generate_tuner_config.py sweep_results/optimized_metrics.csv \
    -o generated_tuner.csv \
    --include-algo-proto
```

---

### Workflow Option B: Automated Pipeline (Recommended)

For most use cases, use the automated auto-tuner that orchestrates all steps:

```bash
# Basic auto-tune
python rccl_autotune.py -n 1-2 -c 32:256:32

# With specific collectives and size range
python rccl_autotune.py \
    --nodes 1-2 \
    --channels 32:64:8 \
    --collective all_reduce,all_gather \
    --algo RING,TREE \
    --proto SIMPLE \
    --min-size 1M \
    --max-size 512M \
    --output-dir ./sweep_results \
    --tuner-output ./my_tuner.conf

# Dry run to preview
python rccl_autotune.py -n 1-2 -c 32:64:8 --dry-run

# From YAML config file
python rccl_autotune.py --config autotune_config.yaml
```

The automated pipeline performs:
1. Initial broad sweeps across configurations
2. Error filtering and metrics merging
3. Metrics optimization
4. Hotspot detection
5. Targeted refinement sweeps (iterative)
6. Final tuner config generation

---

## Output Artifacts Explained

### Directory Structure

```
sweep_results/
├── run_YYYYMMDD_HHMMSS/           # Per-run results
│   ├── sweep_results.db           # SQLite database with all metrics
│   ├── summary.csv                # High-level summary per test
│   ├── metrics.csv                # Detailed per-message-size metrics
│   └── outputs/
│       └── <collective>_<nodes>/
│           ├── command.txt        # Full mpirun command executed
│           └── output.log         # Complete test output
├── merged_metrics.csv             # Combined metrics from all runs
├── optimized_metrics.csv          # Best config per (collective, nodes, size)
├── hotspots_report.csv            # Detected performance hotspots
├── generated_tuner.csv            # Final RCCL tuner configuration
└── plots/                         # Bus bandwidth graphs
    └── <collective>_<nodes>.png
```

### Key Files

| File | Description |
|------|-------------|
| **`summary.csv`** | Quick overview: collective, nodes, channels, avg/max busbw, status, duration |
| **`metrics.csv`** | Detailed per-size data: size_bytes, time_us, busbw, algo, proto, nchannels |
| **`merged_metrics.csv`** | All metrics combined from multiple runs for comprehensive analysis |
| **`optimized_metrics.csv`** | Best configuration selected for each unique (collective, nodes, size) |
| **`hotspots_report.csv`** | Lists size ranges with performance drops, expected vs actual bandwidth |
| **`generated_tuner.csv`** | RCCL tuner format: collective, min_bytes, max_bytes, algo, proto, channels, nNodes, nRanks |
| **`sweep_results.db`** | SQLite database for querying results programmatically |

### Metrics CSV Columns

| Column | Description |
|--------|-------------|
| `collective` | Operation type (e.g., `all_reduce_perf`) |
| `num_nodes` | Number of nodes used |
| `num_gpus` | Total GPU count |
| `size_bytes` | Message size in bytes |
| `time_ip_us` | In-place operation time (microseconds) |
| `busbw_ip` | In-place bus bandwidth (GB/s) |
| `algo` | Algorithm used (RING, TREE, Direct) |
| `proto` | Protocol used (SIMPLE, LL, LL128) |
| `nchannels` | Number of channels |
| `errors_ip` | Error count (should be 0) |

### Hotspots Report Columns

| Column | Description |
|--------|-------------|
| `collective`, `num_nodes`, `num_gpus` | Configuration identifier |
| `hotspot_start_bytes`, `hotspot_end_bytes` | Size range where drop occurs |
| `expected_busbw` | Running maximum bandwidth before drop |
| `actual_busbw_min`, `actual_busbw_max` | Bandwidth range in hotspot |
| `drop_percent_min`, `drop_percent_max` | Performance drop percentage |
| `current_algo`, `current_proto`, `current_nchannels` | Current configuration at hotspot |

### Generated Tuner Config Format

The tuner CSV follows the NCCL/RCCL tuner format:
```csv
collective_type,min_bytes,max_bytes,algorithm,protocol,channels,nNodes,nRanks,numPipeOps,regBuff
allreduce,1048576,2097152,0,2,32,2,16,-1,-1
```

Where algorithm values: `0`=RING, `1`=TREE, `2`=Direct, `-1`=default  
Protocol values: `0`=LL, `1`=LL128, `2`=SIMPLE, `-1`=default

---

## Visualization

Generate bus bandwidth plots:

```bash
# Plot specific collective and node count
python plot_busbw.py all_reduce_perf 1 -o all_reduce_1node.png
python plot_busbw.py all_reduce_perf 2 -o all_reduce_2node.png

# Interactive display
python plot_busbw.py reduce_scatter_perf 1
```

---

## Querying Results with SQLite

```bash
sqlite3 sweep_results/sweep_results.db

# Get all successful runs
SELECT collective, num_nodes, num_channels, avg_busbw, max_busbw 
FROM sweep_runs WHERE status='success' ORDER BY avg_busbw DESC;

# Get best configuration per collective
SELECT collective, num_nodes, num_channels, MAX(avg_busbw) as best_busbw 
FROM sweep_runs WHERE status='success' GROUP BY collective;
```

---

## Configuration Reference

### `autotune_config.yaml`

```yaml
sweep:
  nodes: "1-2"
  channels: "32:256:32"
  collectives: [all_reduce, all_gather]
  algos: [RING, TREE]
  protos: [SIMPLE]
  min_size: "1M"
  max_size: "512M"

hotspot:
  threshold: 0.10       # 10% drop triggers hotspot
  min_drop_gbps: 0.0    # Minimum absolute drop
  max_iterations: 3     # Refinement iterations

output:
  dir: "./sweep_results"
  tuner_conf: "generated_tuner.csv"
  report_csv: "tuning_report.csv"
```

### `unsupported_combos.yaml`

Defines algorithm/protocol combinations that are not supported by RCCL:
- `all_gather` + TREE algorithm
- `reduce_scatter` + TREE algorithm  
- `broadcast` + TREE algorithm
- `reduce` + TREE algorithm
- `alltoall` + TREE algorithm
- TREE + SIMPLE on single-node MI300X/MI355X

---

## Quick Reference Commands

| Task | Command |
|------|---------|
| Run basic sweep | `./rccl_sweep.py -c 4:64:4 -n 1-2` |
| Run auto-tune | `python rccl_autotune.py -n 1-2 -c 32:256:32` |
| Dry run | Add `--dry-run` to any command |
| Merge metrics | `python merge_metrics.py` |
| Detect hotspots | `python detect_hotspots.py merged_metrics.csv --threshold 0.1` |
| Generate tuner config | `python generate_tuner_config.py optimized_metrics.csv` |
| Plot results | `python plot_busbw.py all_reduce_perf 1 -o plot.png` |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Tests timing out | Increase `timeout` in `sweep_config.yaml` or reduce message size range |
| Connection errors | Verify SSH keys and server accessibility (`ssh <server_ip> hostname`) |
| Library not found | Check `MY_PATH` env var and `LD_LIBRARY_PATH` |
| No metrics generated | Check output logs in `sweep_results/run_*/outputs/` |
| Hotspot not improving | Try different algo/proto combinations or wider channel range |


