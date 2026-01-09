# rccl-tune-predict

ML-based tool for predicting optimal RCCL tuning parameters (`algo`, `proto`, `nchannels`) for given workload configurations.

## Overview

This tool consumes sweep data from [rccl-sweep](../rccl-sweep/) and trains a machine learning model to predict bus bandwidth (`busbw`) for any combination of:
- Workload parameters: `collective`, `num_nodes`, `num_gpus`, `size_bytes`
- Tuning parameters: `algo`, `proto`, `nchannels`

At inference time, it searches over the tuning parameter space to find the configuration that maximizes predicted bandwidth.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Train a Model

Train from rccl-sweep results:

```bash
python train.py --data ../rccl-sweep/sweep_results/merged_metrics.csv
```

This creates `models/busbw_model.pkl`.

### 2. Predict Optimal Config

Single prediction:

```bash
python predict.py --collective all_reduce --nodes 4 --gpus 32 --size 16M
```

Output:
```
Optimal config for all_reduce (4 nodes, 32 GPUs, 16.0 MB):
  algo:           RING
  proto:          LL128
  nchannels:      24
  predicted_busbw: 285.30 GB/s
```

Multiple sizes:

```bash
python predict.py --collective all_gather --nodes 4 --gpus 32 --sizes 1M,16M,64M,512M
```

Output:
```
Optimal configs for all_gather (4 nodes, 32 GPUs):
----------------------------------------------------------------------
        Size        Algo     Proto   nchannels  BusBW (GB/s)
----------------------------------------------------------------------
      1.0 MB        RING        LL          12         45.23
     16.0 MB        RING     LL128          24        285.30
     64.0 MB        TREE     LL128          28        312.45
    512.0 MB        RING    SIMPLE          32        298.67
----------------------------------------------------------------------
```

## CLI Reference

### train.py

```
usage: train.py [-h] --data DATA [--output OUTPUT] [--test-split TEST_SPLIT]
                [--n-estimators N_ESTIMATORS] [--max-depth MAX_DEPTH]
                [--learning-rate LEARNING_RATE] [--show-importance]

Options:
  --data, -d          Path to metrics CSV file (required)
  --output, -o        Path to save model (default: models/busbw_model.pkl)
  --test-split, -t    Fraction for testing (default: 0.2)
  --n-estimators      Number of boosting stages (default: 200)
  --max-depth         Maximum tree depth (default: 6)
  --learning-rate     Learning rate (default: 0.1)
  --show-importance   Display feature importances after training
```

### predict.py

```
usage: predict.py [-h] [--model MODEL] --collective COLLECTIVE --nodes NODES 
                  --gpus GPUS [--size SIZE] [--sizes SIZES] 
                  [--max-nchannels MAX_NCHANNELS] [--algos ALGOS] 
                  [--protos PROTOS] [--json]

Options:
  --model, -m         Path to trained model (default: models/busbw_model.pkl)
  --collective, -c    Collective type (required)
  --nodes, -n         Number of nodes (required)
  --gpus, -g          Total number of GPUs (required)
  --size, -s          Message size (e.g., 16M, 1048576)
  --sizes             Comma-separated sizes (e.g., 1M,16M,64M)
  --max-nchannels     Maximum channels to search (default: 32)
  --algos             Limit search to specific algorithms (e.g., RING,TREE)
  --protos            Limit search to specific protocols (e.g., LL128,SIMPLE)
  --json              Output as JSON
```

**Supported collectives:** `all_reduce`, `all_gather`, `reduce_scatter`, `alltoall`, `broadcast`, `reduce`

**Size suffixes:** `K` (KiB), `M` (MiB), `G` (GiB)

## Python API

```python
from core import BusbwPredictor, find_optimal_config, load_sweep_data

# Train a model
df = load_sweep_data('path/to/metrics.csv')
from core.utils import prepare_features
X, y = prepare_features(df)

model = BusbwPredictor()
model.fit(X, y, test_split=0.2)
model.save('models/my_model.pkl')

# Load and predict
model = BusbwPredictor.load('models/my_model.pkl')
config = find_optimal_config(
    model, 
    collective='all_reduce',
    num_nodes=4, 
    num_gpus=32, 
    size_bytes=16*1024*1024
)

print(f"Best: algo={config['algo']}, proto={config['proto']}, "
      f"nchannels={config['nchannels']}, busbw={config['predicted_busbw']:.2f}")
```

## Model Details

### Features

| Feature | Description |
|---------|-------------|
| `collective_encoded` | Integer-encoded collective type |
| `num_nodes` | Number of nodes |
| `num_gpus` | Total number of GPUs |
| `size_bytes` | Message size in bytes |
| `log_size` | log2(size_bytes) for better scaling |
| `gpus_per_node` | Derived: num_gpus / num_nodes |
| `bytes_per_gpu` | Derived: size_bytes / num_gpus |
| `algo_encoded` | Integer-encoded algorithm |
| `proto_encoded` | Integer-encoded protocol |
| `nchannels` | Number of channels |

### Collective Encoding

| Collective | Code |
|------------|------|
| all_reduce | 0 |
| all_gather | 1 |
| reduce_scatter | 2 |
| alltoall | 3 |
| broadcast | 4 |
| reduce | 5 |

### Algorithm Encoding

| Algorithm | Code |
|-----------|------|
| TREE | 0 |
| RING | 1 |
| COLLNETDIRECT | 2 |
| COLLNETCHAIN | 3 |
| NVLS | 4 |
| NVLSTREE | 5 |

### Protocol Encoding

| Protocol | Code |
|----------|------|
| LL | 0 |
| LL128 | 1 |
| SIMPLE | 2 |

### Search Strategy

The tool uses a coarse-to-fine search:
1. For each (algo, proto) combination, search nchannels in steps of 4
2. Fine-tune around the best value in steps of 1
3. Return the configuration with highest predicted bandwidth

## File Structure

```
rccl-tune-predict/
├── PLAN.md               # Design document
├── README.md             # This file
├── requirements.txt      # Python dependencies
├── train.py              # CLI: Train model
├── predict.py            # CLI: Predict optimal config
├── core/                 # Python package
│   ├── __init__.py
│   ├── busbw_predictor.py # BusbwPredictor class
│   ├── search.py         # Config search/optimization
│   └── utils.py          # Data loading, feature engineering
└── models/               # Saved trained models
    └── .gitkeep
```

## Requirements

- Python 3.8+
- scikit-learn >= 1.0
- pandas >= 1.3
- numpy >= 1.20
