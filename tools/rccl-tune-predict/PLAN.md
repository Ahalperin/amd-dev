# rccl-tune-predict: ML-Based RCCL Configuration Predictor

## Overview

A standalone tool that consumes `optimized_metrics.csv` from [rccl-sweep](../rccl-sweep/) and uses ML to predict optimal RCCL tuning parameters (algo, proto, nchannels) for given workload parameters.

**Approach**: Train a model to predict `busbw` given all parameters, then search over (algo, proto, nchannels) space to find the configuration that maximizes predicted bandwidth.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TRAINING PHASE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  optimized_metrics.csv ──► train.py ──► models/busbw_model.pkl             │
│                                                                             │
│  Features: num_nodes, num_gpus, size_bytes, log_size, algo, proto, nchannels│
│  Target: busbw_ip (in-place bus bandwidth)                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                             PREDICTION PHASE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  User query: (collective, num_nodes, num_gpus, size_bytes)                  │
│       │                                                                     │
│       ▼                                                                     │
│  predict.py ──► Load model ──► Search over (algo, proto, nchannels)        │
│       │                              │                                      │
│       │                              ▼                                      │
│       │                    For each combination:                            │
│       │                      - Predict busbw                                │
│       │                      - Track best config                            │
│       │                              │                                      │
│       ▼                              ▼                                      │
│  Output: algo=RING, proto=LL128, nchannels=24, predicted_busbw=285.3       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
rccl-tune-predict/
├── PLAN.md               # This file
├── README.md             # Usage documentation
├── requirements.txt      # Python dependencies
├── train.py              # CLI: Train model from sweep data
├── predict.py            # CLI: Predict optimal config
├── core/                 # Python package
│   ├── __init__.py
│   ├── busbw_predictor.py # BusbwPredictor class
│   ├── search.py         # Config search/optimization
│   └── utils.py          # Data loading, feature engineering
└── models/               # Saved trained models
    └── .gitkeep
```

## Core Components

### 1. Model Class (`core/busbw_predictor.py`)

**BusbwPredictor** class:
- Wraps `sklearn.ensemble.GradientBoostingRegressor`
- Features:
  - `num_nodes` - Number of nodes
  - `num_gpus` - Total GPUs
  - `size_bytes` - Message size in bytes
  - `log_size` - log2(size_bytes) for better scaling
  - `gpus_per_node` - Derived feature
  - `bytes_per_gpu` - Derived feature
  - `algo` - Algorithm (integer encoded: Direct=0, RING=1, Tree=2, etc.)
  - `proto` - Protocol (integer encoded: SIMPLE=0, LL128=1)
  - `nchannels` - Number of channels (1..N)
- Target: `busbw_ip` (in-place bus bandwidth)
- Methods:
  - `fit(X, y)` - Train the model
  - `predict(X)` - Predict busbw for given features
  - `save(path)` - Serialize model to pickle
  - `load(path)` - Load model from pickle

### 2. Config Search (`core/search.py`)

**find_optimal_config()** function:
- Input: (collective, num_nodes, num_gpus, size_bytes), trained model
- Algorithm:
  1. Enumerate all (algo, proto) combinations
  2. For each combo, coarse search over nchannels (step=4)
  3. Fine-tune nchannels around best coarse value (step=1)
  4. Return config with highest predicted busbw
- Output: `{algo, proto, nchannels, predicted_busbw}`

### 3. CLI Tools

**train.py**:
```bash
# Train from rccl-sweep results
python train.py \
    --data ../rccl-sweep/sweep_results/optimized_metrics.csv \
    --output models/busbw_model.pkl \
    --test-split 0.2

# Output:
# Training on 3687 samples, testing on 922 samples
# Model R² score: 0.94
# Model MAE: 12.3 GB/s
# Saved model to models/busbw_model.pkl
```

**predict.py**:
```bash
# Single prediction
python predict.py \
    --model models/busbw_model.pkl \
    --collective all_gather \
    --nodes 4 \
    --gpus 32 \
    --size 16777216

# Output:
# Optimal config for all_gather (4 nodes, 32 GPUs, 16.0 MB):
#   algo: RING
#   proto: LL128
#   nchannels: 24
#   predicted_busbw: 285.3 GB/s

# Batch prediction (multiple sizes)
python predict.py \
    --model models/busbw_model.pkl \
    --collective all_gather \
    --nodes 4 \
    --gpus 32 \
    --sizes 1048576,16777216,67108864,536870912

# Output table with optimal config for each size
```

## Implementation Tasks

- [ ] Create directory structure and `requirements.txt`
- [ ] Implement `BusbwPredictor` class with train/predict/save/load
- [ ] Implement config search with coarse-to-fine nchannels optimization
- [ ] Create `train.py` CLI for model training
- [ ] Create `predict.py` CLI for config prediction
- [ ] Write README with usage examples

## Dependencies

```
scikit-learn>=1.0
pandas>=1.3
numpy>=1.20
```

## Key Design Decisions

1. **Predict busbw, not config directly** - This allows the model to learn the relationship between ALL parameters and performance, then we search for optimal config at inference time.

2. **Continuous nchannels** - Model can predict performance for ANY nchannels value (1..N), not just the discrete values tested in the sweep.

3. **Integer encoding for algo/proto** - These are treated as categorical inputs; the model learns their effect on busbw through the training data.

4. **Coarse-to-fine search** - Efficient search strategy: first scan nchannels in steps of 4, then refine around the best value.

5. **Separate from rccl-sweep** - This tool only consumes sweep results; it doesn't run any RCCL tests itself.

