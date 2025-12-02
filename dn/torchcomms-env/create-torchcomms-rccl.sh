#!/usr/bin/env bash
#
# TorchComms RCCL Installation Script (Simplified)
# This script builds torchcomms with standard RCCL backend
#
# Prerequisites:
# - ROCm installed (tested with 7.0.1 and 7.1.1)
# - Conda environment 'torchcomms' already created
#
# Usage:
#   bash create-torchcomms-rccl.sh
#
# Note: This is the simpler alternative to RCCLX.
# For Meta's enhanced RCCLX, use create-torchcomms-dev-tools.sh
#

set -e  # Exit on error

echo "=========================================="
echo "TorchComms RCCL Installation (Standard)"
echo "=========================================="

# Detect ROCm version
if [ -d "/opt/rocm-7.0.1" ]; then
    ROCM_VERSION="7.0.1"
    export ROCM_HOME=/opt/rocm-7.0.1
elif [ -d "/opt/rocm-7.1.1" ]; then
    ROCM_VERSION="7.1.1"
    export ROCM_HOME=/opt/rocm-7.1.1
elif [ -d "/opt/rocm" ]; then
    ROCM_VERSION=$(readlink -f /opt/rocm | grep -oP 'rocm-\K[0-9.]+' || echo "unknown")
    if [ "$ROCM_VERSION" = "unknown" ]; then
        echo "Warning: Could not detect ROCm version, using /opt/rocm"
        export ROCM_HOME=/opt/rocm
    else
        export ROCM_HOME=/opt/rocm-${ROCM_VERSION}
    fi
else
    echo "Error: ROCm not found. Please install ROCm first."
    exit 1
fi

echo "✓ Detected ROCm ${ROCM_VERSION} at ${ROCM_HOME}"

# Activate environment
eval "$(conda shell.bash hook)"
conda activate torchcomms || { echo "Error: torchcomms environment not found. Run create-torchcomms-dev-tools.sh first."; exit 1; }

# Navigate to script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}/torchcomms" || { echo "Error: torchcomms directory not found"; exit 1; }

# Set ROCm environment variables for CMake
echo "Setting ROCm environment variables..."
export ROCM_PATH=${ROCM_HOME}
export HIP_PATH=${ROCM_HOME}
export CMAKE_PREFIX_PATH="${ROCM_HOME}:$CMAKE_PREFIX_PATH"
export RCCL_INCLUDE=${ROCM_HOME}/include/rccl

# Set backend configuration for standard RCCL
export USE_NCCL=OFF
export USE_NCCLX=OFF
export USE_GLOO=OFF
export USE_RCCL=ON      # Use standard RCCL
export USE_RCCLX=OFF
export USE_SYSTEM_LIBS=0  # Build own dependencies to avoid ABI issues

echo ""
echo "Build Configuration:"
echo "  ROCM_HOME: ${ROCM_HOME}"
echo "  USE_RCCL: ${USE_RCCL}"
echo "  USE_SYSTEM_LIBS: ${USE_SYSTEM_LIBS}"
echo ""

# Install PyTorch if needed
if [ -f "requirements.txt" ]; then
    echo "Installing PyTorch from requirements.txt..."
    pip install -r requirements.txt
fi

# Clean and rebuild
echo "Cleaning previous build artifacts..."
rm -rf build
pip uninstall -y torchcomms 2>/dev/null || true

# Build and install
echo ""
echo "Building and installing torchcomms with RCCL..."
echo "This may take several minutes..."
pip install --no-build-isolation -v .

# Verify installation
echo ""
echo "Verifying installation..."
if python -c "from torchcomms import new_comm, ReduceOp; import torch; print('✓ TorchComms:', 'OK'); print('✓ PyTorch:', torch.__version__)" 2>&1; then
    echo ""
    echo "=========================================="
    echo "✓ RCCL Installation completed!"
    echo "=========================================="
    echo ""
    echo "Backend: RCCL (Standard)"
    echo ""
    echo "Use 'rccl' backend in your code:"
    echo "  comm = new_comm('rccl', torch.device('cuda'))"
    echo ""
else
    echo ""
    echo "✗ Installation verification failed"
    exit 1
fi
