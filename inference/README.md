# InferenceMAX Benchmark Setup with Custom RCCL

This directory contains scripts for running InferenceMAX benchmarks with custom RCCL libraries.

## Setup Complete!

The following has been configured:
- ✓ Custom RCCL build location
- ✓ Docker and GPU access
- ✓ HuggingFace cache directory
- ✓ Helper scripts
- ✓ SGLang benchmark tools (built-in, no separate download needed)

## Quick Start

### 1. Configure HuggingFace Token

Edit `test_inference.sh` and set your HuggingFace token, or export it as an environment variable:
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
./test_inference.sh
```

Or start the server manually:

```bash
# Start server in detached mode
./start_server_container.sh server --detached --tp 8 --ep 1 --port 8888

# Run benchmark (in another terminal or script)
./test_inference.sh

# After benchmark completes, dump NPKIT logs (if needed)
curl http://localhost:8888/destroy_nccl_comm
```

## Files Overview

- `start_server_container.sh` - Start SGLang server with custom RCCL (applies comm_shutdown patch automatically)
- `test_inference.sh` - Run inference benchmark test
- `download_model.sh` - Downloads DeepSeek-R1 model
- `sglang_patch/comm_shutdown.patch` - Patch to enable NPKIT log dumping

## NPKIT Logs

To collect NPKIT profiling logs:

1. **Patch is applied automatically**: The `start_server_container.sh` script automatically applies the `comm_shutdown.patch` to enable NPKIT log dumping.

2. **After test completion**: Once your benchmark/test is complete, trigger NPKIT log dump by calling:
   ```bash
   curl http://localhost:8888/destroy_nccl_comm
   ```
   (Replace `localhost:8888` with your server host and port if different)

3. **Logs location**: NPKIT logs will be saved to `outputs/npkit/` directory.

**Note**: The `comm_shutdown` patch must be applied for NPKIT logs to be generated. This is done automatically when using `start_server_container.sh`.

## Verifying Custom RCCL

After running a benchmark, check that your custom RCCL was used by examining the RCCL debug logs in `outputs/logs/`.

## Benchmark Parameters

### Server Parameters (start_server_container.sh)

- `--tp TP` - Tensor parallel size (default: 8)
- `--ep EP` - Expert parallel size (default: 1)
- `--port PORT` - Server port (default: 8888)
- `--num-prompts NUM` - Number of prompts for benchmark mode (default: 50)
- `--detached, -d` - Run server in detached mode

### Test Parameters (test_inference.sh)

Edit `test_inference.sh` or set environment variables:

- `TP` - Tensor parallel size (number of GPUs)
- `EP` - Expert parallel size
- `PORT` - Server port
- `ISL` - Input sequence length
- `OSL` - Output sequence length
- `CONC` - Concurrency (simultaneous requests)
- `NUM_PROMPTS` - Total number of requests

## References

- InferenceMAX: https://inferencemax.ai/
- SGLang: https://github.com/sgl-project/sglang
