#!/usr/bin/env bash
#
# TorchComms Installation Script for AMD GPUs
# This script sets up torchcomms with RCCLX backend (Meta's enhanced RCCL)
#
# Prerequisites:
# - ROCm installed (tested with 7.0.1 and 7.1.1)
# - Conda or Miniconda installed
#
# Usage:
#   bash create-torchcomms-dev-tools.sh
#

set -e  # Exit on error

echo "=========================================="
echo "TorchComms + RCCLX Installation"
echo "=========================================="

# Detect ROCm version
if [ -d "/opt/rocm-7.0.1" ]; then
    ROCM_VERSION="7.0.1"
elif [ -d "/opt/rocm-7.1.1" ]; then
    ROCM_VERSION="7.1.1"
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

if [ -z "$ROCM_HOME" ]; then
    export ROCM_HOME=/opt/rocm-${ROCM_VERSION}
fi

echo "✓ Detected ROCm ${ROCM_VERSION} at ${ROCM_HOME}"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
    bash ~/miniconda.sh -b -p $HOME/miniconda
    echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
    export PATH="$HOME/miniconda/bin:$PATH"
    conda init bash
    echo "✓ Miniconda installed"
else
    echo "✓ Conda found: $(conda --version)"
fi

# Create conda environment if it doesn't exist
if conda env list | grep -q "^torchcomms "; then
    echo "✓ torchcomms environment already exists"
else
    echo "Creating torchcomms conda environment..."
    conda create -n torchcomms python=3.10 -y
    echo "✓ Environment created"
fi

# Activate environment
eval "$(conda shell.bash hook)"
conda activate torchcomms

# Install prerequisites
echo "Installing prerequisites..."
conda install conda-forge::glog=0.4.0 conda-forge::gflags conda-forge::fmt -y

# Set ROCm environment variables for CMake
echo "Setting ROCm environment variables..."
export ROCM_PATH=${ROCM_HOME}
export HIP_PATH=${ROCM_HOME}
export CMAKE_PREFIX_PATH="${ROCM_HOME}:$CMAKE_PREFIX_PATH"

# Navigate to torchcomms directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}/torchcomms" || cd "${SCRIPT_DIR}" || exit 1

# Install PyTorch if needed
if [ -f "requirements.txt" ]; then
    echo "Installing PyTorch from requirements.txt..."
    pip install -r requirements.txt
fi

# Step 1: Build RCCLX first
echo ""
echo "=========================================="
echo "Step 1: Building RCCLX from source"
echo "=========================================="

if [ ! -d "comms/rcclx/develop" ]; then
    echo "Error: RCCLX source directory not found at comms/rcclx/develop"
    exit 1
fi

cd comms/rcclx/develop

if [ ! -f "install.sh" ]; then
    echo "Error: RCCLX install.sh not found"
    exit 1
fi

echo "Building RCCLX for gfx950 architecture..."
./install.sh --prefix build --amdgpu_targets gfx950 --disable-colltrace -j20

if [ ! -d "build/release/build" ]; then
    echo "Error: RCCLX build failed"
    exit 1
fi

export BUILD_DIR=${PWD}/build/release/build
export RCCLX_INCLUDE=${BUILD_DIR}/include/rccl
export RCCLX_LIB=${BUILD_DIR}/lib

echo "✓ RCCLX built successfully"
echo "  RCCLX_INCLUDE: ${RCCLX_INCLUDE}"
echo "  RCCLX_LIB: ${RCCLX_LIB}"

# Return to torchcomms root
cd "${SCRIPT_DIR}/torchcomms"

# Step 2: Build torchcomms with RCCLX
echo ""
echo "=========================================="
echo "Step 2: Building torchcomms with RCCLX"
echo "=========================================="

# Set backend configuration for RCCLX
export USE_NCCL=OFF
export USE_NCCLX=OFF
export USE_GLOO=OFF
export USE_RCCL=OFF
export USE_RCCLX=ON       # Use RCCLX backend
export USE_SYSTEM_LIBS=0  # Build own dependencies (avoids fmt ABI issues)

echo ""
echo "Build Configuration:"
echo "  ROCM_HOME: ${ROCM_HOME}"
echo "  USE_RCCLX: ${USE_RCCLX}"
echo "  RCCLX_INCLUDE: ${RCCLX_INCLUDE}"
echo "  RCCLX_LIB: ${RCCLX_LIB}"
echo "  USE_SYSTEM_LIBS: ${USE_SYSTEM_LIBS}"
echo ""

# Clean previous build artifacts
if [ -d "build" ]; then
    echo "Cleaning previous torchcomms build artifacts..."
    rm -rf build
fi

# Uninstall any existing torchcomms
pip uninstall -y torchcomms 2>/dev/null || true

# Build and install torchcomms
echo ""
echo "Building and installing torchcomms with RCCLX..."
echo "This may take several minutes..."
pip install --no-build-isolation -v .

# Verify installation
echo ""
echo "Verifying installation..."
if python -c "from torchcomms import new_comm, ReduceOp; import torch; print('✓ TorchComms:', 'OK'); print('✓ PyTorch:', torch.__version__); print('✓ ROCm/HIP:', torch.version.hip if hasattr(torch.version, 'hip') else 'N/A')" 2>&1; then
    echo ""
    echo "=========================================="
    echo "✓ Installation completed successfully!"
    echo "=========================================="
    echo ""
    echo "Backend: RCCLX (Meta's enhanced RCCL)"
    echo ""
    echo "To use torchcomms, activate the environment:"
    echo "  conda activate torchcomms"
    echo ""
    echo "Example usage:"
    echo "  from torchcomms import new_comm, ReduceOp"
    echo "  comm = new_comm('rcclx', torch.device('cuda'))"
    echo ""
    echo "Note: Use backend name 'rcclx' (not 'rccl') in your code"
    echo ""
else
    echo ""
    echo "✗ Installation verification failed"
    echo "Please check the error messages above"
    exit 1
fi
