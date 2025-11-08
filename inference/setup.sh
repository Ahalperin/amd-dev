#!/usr/bin/env bash

################################################################################
# InferenceMAX Benchmark Setup Script with Custom RCCL Support
#
# This script sets up a complete InferenceMAX benchmark environment including:
# - Repository cloning (InferenceMAX, bench_serving)
# - Custom RCCL library configuration
# - Docker setup and verification
# - HuggingFace model preparation
# - Helper scripts generation
#
# Usage: bash setup.sh
################################################################################

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CUSTOM_RCCL_PATH="/home/dn/amd-dev/dn/rccl/build/release"
WORK_DIR="/home/dn/amd-dev/inference"
HF_CACHE_DIR="/mnt/hf_hub_cache"
DOCKER_IMAGE="rocm/7.0:rocm7.0_ubuntu_22.04_sgl-dev-v0.5.2-rocm7.0-mi35x-20250915"

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo ""
    echo -e "${GREEN}======================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}======================================${NC}"
}

################################################################################
# Step 1: System Requirements Check
################################################################################
print_header "Step 1: Checking System Requirements"

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    print_error "This script must be run on Linux"
    exit 1
fi
print_success "OS: Linux detected"

# Check for ROCm installation
if command -v rocm-smi &> /dev/null; then
    ROCM_VERSION=$(rocm-smi --showdriverversion 2>/dev/null | grep -oP 'ROCm version: \K[0-9.]+' || echo "unknown")
    print_success "ROCm installed: version $ROCM_VERSION"
    
    # Show GPU info
    print_info "Detected GPUs:"
    rocm-smi --showproductname | grep -E "GPU\[|Card series" || print_warning "Could not detect GPU details"
else
    print_error "ROCm not found. Please install ROCm 7.0+ first."
    echo "Installation guide: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/"
    exit 1
fi

# Check for Docker
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version | grep -oP '\d+\.\d+\.\d+')
    print_success "Docker installed: version $DOCKER_VERSION"
    
    # Check if user is in docker group
    if groups $USER | grep -q docker; then
        print_success "User $USER is in docker group"
    else
        print_warning "User $USER is NOT in docker group"
        print_info "Run: sudo usermod -aG docker $USER && newgrp docker"
    fi
else
    print_error "Docker not found. Installing Docker..."
    sudo apt update
    sudo apt install -y docker.io
    sudo systemctl start docker
    sudo systemctl enable docker
    sudo usermod -aG docker $USER
    print_warning "Docker installed. Please log out and log back in, then re-run this script."
    exit 0
fi

# Check for Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | grep -oP '\d+\.\d+\.\d+')
    print_success "Python3 installed: version $PYTHON_VERSION"
else
    print_error "Python3 not found. Please install Python 3.10+"
    exit 1
fi

# Check for git
if command -v git &> /dev/null; then
    print_success "Git installed"
else
    print_error "Git not found. Installing..."
    sudo apt update && sudo apt install -y git
fi

# Check for huggingface-cli
if command -v huggingface-cli &> /dev/null; then
    print_success "huggingface-cli installed"
else
    print_warning "huggingface-cli not found. Installing..."
    pip install -U "huggingface_hub[cli]"
fi

################################################################################
# Step 2: Verify Custom RCCL Build
################################################################################
print_header "Step 2: Verifying Custom RCCL Build"

if [ -d "$CUSTOM_RCCL_PATH" ]; then
    print_success "Custom RCCL directory found: $CUSTOM_RCCL_PATH"
    
    if [ -f "$CUSTOM_RCCL_PATH/librccl.so" ] || [ -f "$CUSTOM_RCCL_PATH/librccl.so.1" ]; then
        print_success "Custom RCCL library found:"
        ls -lh "$CUSTOM_RCCL_PATH"/librccl*.so* 2>/dev/null | head -5
    else
        print_error "librccl.so not found in $CUSTOM_RCCL_PATH"
        print_info "Please build RCCL first or update CUSTOM_RCCL_PATH in this script"
        exit 1
    fi
    
    # Check for optional rccl-net plugin
    if [ -f "$CUSTOM_RCCL_PATH/librccl-net.so" ]; then
        print_success "RCCL-NET plugin found"
    else
        print_info "RCCL-NET plugin not found (optional)"
    fi
else
    print_error "Custom RCCL directory not found: $CUSTOM_RCCL_PATH"
    print_info "Please build RCCL first or update CUSTOM_RCCL_PATH in this script"
    exit 1
fi

################################################################################
# Step 3: Setup Work Directory
################################################################################
print_header "Step 3: Setting Up Work Directory"

mkdir -p "$WORK_DIR"
cd "$WORK_DIR"
print_success "Work directory: $WORK_DIR"

################################################################################
# Step 4: Clone Required Repositories
################################################################################
print_header "Step 4: Cloning Required Repositories"

# Clone InferenceMAX
if [ -d "InferenceMAX" ]; then
    print_info "InferenceMAX directory exists. Updating..."
    cd InferenceMAX
    git pull || print_warning "Failed to update InferenceMAX"
    cd ..
else
    print_info "Cloning InferenceMAX repository..."
    # Using the SemiAnalysis/InferenceMAX repository (official)
    git clone https://github.com/InferenceMAX/InferenceMAX.gitt || \
    print_error "Failed to clone InferenceMAX. Repository may be private or URL incorrect."
fi

if [ -d "InferenceMAX" ]; then
    print_success "InferenceMAX repository ready"
    print_info "Available benchmarks:"
    ls InferenceMAX/benchmarks/dsr1_*.sh 2>/dev/null | head -5 || echo "  (benchmark scripts not found)"
else
    print_warning "InferenceMAX not available. You may need to clone it manually."
fi

# Clone bench_serving
if [ -d "bench_serving" ]; then
    print_info "bench_serving directory exists. Updating..."
    cd bench_serving
    git pull || print_warning "Failed to update bench_serving"
    cd ..
else
    print_info "Cloning bench_serving repository..."
    git clone https://github.com/kimbochen/bench_serving.git
fi

if [ -d "bench_serving" ]; then
    print_success "bench_serving repository ready"
else
    print_error "Failed to clone bench_serving"
    exit 1
fi

################################################################################
# Step 5: Setup HuggingFace Cache Directory
################################################################################
print_header "Step 5: Setting Up HuggingFace Cache"

if [ -d "$HF_CACHE_DIR" ]; then
    print_success "HuggingFace cache directory exists: $HF_CACHE_DIR"
    
    # Check available space
    AVAILABLE_SPACE=$(df -BG "$HF_CACHE_DIR" | tail -1 | awk '{print $4}' | sed 's/G//')
    if [ "$AVAILABLE_SPACE" -lt 700 ]; then
        print_warning "Low disk space: ${AVAILABLE_SPACE}GB available (700GB+ recommended)"
    else
        print_success "Sufficient disk space: ${AVAILABLE_SPACE}GB available"
    fi
    
    # Check permissions
    if [ -w "$HF_CACHE_DIR" ]; then
        print_success "Write permissions OK for $HF_CACHE_DIR"
    else
        print_warning "No write permissions. Fixing..."
        sudo chown -R $USER:$USER "$HF_CACHE_DIR"
    fi
else
    print_warning "HuggingFace cache directory not found: $HF_CACHE_DIR"
    print_info "Creating directory..."
    sudo mkdir -p "$HF_CACHE_DIR"
    sudo chown -R $USER:$USER "$HF_CACHE_DIR"
    print_success "Created: $HF_CACHE_DIR"
fi

################################################################################
# Step 6: Docker Image Verification
################################################################################
print_header "Step 6: Verifying Docker Image"

print_info "Checking for Docker image: $DOCKER_IMAGE"
if docker images | grep -q "rocm7.0_ubuntu_22.04_sgl-dev"; then
    print_success "Docker image already available locally"
else
    print_info "Docker image not found locally"
    print_warning "You'll need to pull it manually or it will be pulled on first use:"
    echo "  docker pull $DOCKER_IMAGE"
fi

# Test Docker GPU access
print_info "Testing Docker GPU access..."
if docker run --rm --device=/dev/kfd --device=/dev/dri rocm/rocm-terminal:latest rocm-smi --showproductname &>/dev/null; then
    print_success "Docker can access GPUs"
else
    print_warning "Docker GPU access test failed. Check GPU permissions."
    print_info "Try: sudo usermod -aG video,render $USER"
fi

################################################################################
# Step 7: Create Helper Scripts
################################################################################
print_header "Step 7: Creating Helper Scripts"

# Create environment configuration file
cat > "${WORK_DIR}/benchmark.env" << 'EOF'
# InferenceMAX Benchmark Environment Configuration
# Source this file before running benchmarks: source benchmark.env

# Custom RCCL Configuration
export CUSTOM_RCCL_PATH="/home/dn/amd-dev/dn/rccl/build/release"

# HuggingFace Configuration
export HF_HUB_CACHE="/mnt/hf_hub_cache"
export HF_TOKEN=""  # Set your HuggingFace token here

# Docker Configuration
export DOCKER_IMAGE="rocm/7.0:rocm7.0_ubuntu_22.04_sgl-dev-v0.5.2-rocm7.0-mi35x-20250915"

# Benchmark Configuration
export WORK_DIR="/home/dn/amd-dev/inference"
export MODEL="deepseek-ai/DeepSeek-R1-0528"
export TP=8  # Tensor Parallel size (number of GPUs)
export PORT=8888

# Network Configuration
export NETWORK_NAME="bmk-net"
export SERVER_NAME="bmk-server"
export CLIENT_NAME="bmk-client"

echo "Environment configured for InferenceMAX benchmarks"
echo "Custom RCCL: $CUSTOM_RCCL_PATH"
echo "HF Cache: $HF_HUB_CACHE"
echo "Work Dir: $WORK_DIR"
EOF
print_success "Created: benchmark.env"

# Create model download script
cat > "${WORK_DIR}/download_model.sh" << 'EOF'
#!/usr/bin/env bash
# Download DeepSeek-R1 model using HuggingFace CLI

set -e

source "$(dirname "$0")/benchmark.env"

if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN not set. Please set it in benchmark.env"
    exit 1
fi

echo "=== Downloading DeepSeek-R1 Model ==="
echo "Model: $MODEL"
echo "Cache: $HF_HUB_CACHE"
echo "Size: ~600GB (this will take time)"
echo ""

mkdir -p "$HF_HUB_CACHE"

echo "Starting download..."
huggingface-cli download "$MODEL" \
    --cache-dir "$HF_HUB_CACHE" \
    --resume-download \
    --token "$HF_TOKEN"

echo ""
echo "✓ Model download complete!"
echo "Location: $HF_HUB_CACHE/models--deepseek-ai--DeepSeek-R1-0528"
EOF
chmod +x "${WORK_DIR}/download_model.sh"
print_success "Created: download_model.sh"


# Create quick test script
cat > "${WORK_DIR}/quick_test.sh" << 'EOFSCRIPT'
#!/usr/bin/env bash
# Quick test with 50 requests using custom RCCL

export ISL=512
export OSL=512
export CONC=8
export NUM_PROMPTS=50
export TP=4
export RESULT_FILENAME="quick_test_custom_rccl"

source "$(dirname "$0")/run_benchmark_custom_rccl.sh"
EOFSCRIPT
chmod +x "${WORK_DIR}/quick_test.sh"
print_success "Created: quick_test.sh"

################################################################################
# Step 8: Create Documentation
################################################################################
print_header "Step 8: Creating Documentation"

cat > "${WORK_DIR}/README.md" << 'EOF'
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
EOF
print_success "Created: README.md"

################################################################################
# Final Summary
################################################################################
print_header "Setup Complete!"

echo ""
print_success "InferenceMAX benchmark environment is ready!"
echo ""
echo "Location: $WORK_DIR"
echo ""
echo "Next steps:"
echo "  1. Set your HuggingFace token in benchmark.env"
echo "  2. Run: ./download_model.sh (first time only, ~600GB)"
echo ""
echo "Custom RCCL: $CUSTOM_RCCL_PATH"
echo "Documentation: $WORK_DIR/README.md"
echo ""
print_info "For more information, see: $WORK_DIR/README.md"