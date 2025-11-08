# InferenceMAX Benchmark Setup with Custom RCCL

This directory contains scripts for running InferenceMAX benchmarks with custom RCCL libraries.

## Setup Complete!

The following has been configured:
- ✓ Custom RCCL build location
- ✓ InferenceMAX repository
- ✓ bench_serving repository  
- ✓ Docker and GPU access
- ✓ HuggingFace cache directory
- ✓ Helper scripts

## Quick Start

### 1. Configure HuggingFace Token

Edit `benchmark.env` and set your HuggingFace token:
```bash
export HF_TOKEN="hf_your_token_here"
```

### 2. Download Model (first time only)

```bash
./download_model.sh
```

This downloads DeepSeek-R1 (~600GB, takes ~45 minutes on fast connection).

### 3. Run Quick Test (50 requests)

```bash
./quick_test.sh
```

## Files Overview

- `benchmark.env` - Environment configuration (edit this for your setup)
- `download_model.sh` - Downloads DeepSeek-R1 model
- `run_50req_test.sh` - Standard 50-request test (without custom RCCL)
- `quick_test.sh` - Quick test wrapper
- `InferenceMAX/` - InferenceMAX repository
- `bench_serving/` - Benchmark client repository

## Verifying Custom RCCL

After running a benchmark, check that your custom RCCL was used:

## Benchmark Parameters

Edit `benchmark.env` or set environment variables:

- `TP` - Tensor parallel size (number of GPUs)
- `ISL` - Input sequence length
- `OSL` - Output sequence length
- `CONC` - Concurrency (simultaneous requests)
- `NUM_PROMPTS` - Total number of requests
- `RESULT_FILENAME` - Output JSON filename

## References

- InferenceMAX: https://inferencemax.ai/
- SGLang: https://github.com/sgl-project/sglang
