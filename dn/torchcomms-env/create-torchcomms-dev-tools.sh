#!/usr/bin/env bash
#
# TorchComms Installation Script for AMD GPUs
# This script sets up torchcomms with both RCCL and RCCLX backends
# - RCCL: Standard AMD ROCm Collective Communications Library
# - RCCLX: Meta's enhanced RCCL with additional optimizations
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
echo "TorchComms + RCCL + RCCLX Installation"
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
    # Check if Miniconda is already installed but not in PATH
    if [ -d "$HOME/miniconda" ]; then
        echo "Miniconda found at $HOME/miniconda, initializing..."
        source $HOME/miniconda/etc/profile.d/conda.sh
        conda init bash
        echo "✓ Miniconda initialized"
    else
        echo "Conda not found. Installing Miniconda..."
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
        bash ~/miniconda.sh -b -p $HOME/miniconda
        source $HOME/miniconda/etc/profile.d/conda.sh
        conda init bash
        echo "✓ Miniconda installed"
    fi
else
    echo "✓ Conda found: $(conda --version)"
fi

# Ensure conda is available (source it if needed)
if ! command -v conda &> /dev/null; then
    if [ -f "$HOME/miniconda/etc/profile.d/conda.sh" ]; then
        source $HOME/miniconda/etc/profile.d/conda.sh
    else
        echo "Error: Conda not available. Please restart your shell and try again."
        exit 1
    fi
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
conda install conda-forge::glog=0.4.0 conda-forge::gflags conda-forge::fmt conda-forge::libunwind -y

# Set ROCm environment variables for CMake
echo "Setting ROCm environment variables..."
export ROCM_PATH=${ROCM_HOME}
export HIP_PATH=${ROCM_HOME}
export CMAKE_PREFIX_PATH="${ROCM_HOME}:$CMAKE_PREFIX_PATH"

# Navigate to torchcomms directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TORCHCOMMS_DIR="${SCRIPT_DIR}/../torchcomms"

# Clone torchcomms if it doesn't exist
if [ ! -d "${TORCHCOMMS_DIR}" ]; then
    echo "Cloning torchcomms repository..."
    cd "${SCRIPT_DIR}/.."
    git clone --recursive https://github.com/meta-pytorch/torchcomms.git
    if [ $? -ne 0 ]; then
        echo "Error: Failed to clone torchcomms repository"
        exit 1
    fi
    echo "✓ Torchcomms repository cloned"
fi

cd "${TORCHCOMMS_DIR}" || exit 1
echo "Working directory: ${PWD}"

# Install PyTorch with ROCm support
echo ""
echo "Checking for PyTorch with ROCm ${ROCM_VERSION} support..."
if python -c "import torch" &> /dev/null; then
    PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    PYTORCH_ROCM=$(python -c "import torch; print(torch.version.hip if hasattr(torch.version, 'hip') else 'N/A')")
    echo "✓ PyTorch already installed: ${PYTORCH_VERSION}"
    echo "  PyTorch ROCm version: ${PYTORCH_ROCM}"
    
    # Verify it's the ROCm version, not CUDA
    if [[ "$PYTORCH_ROCM" == "N/A" ]]; then
        echo ""
        echo "✗ ERROR: PyTorch is installed but it's the CUDA version, not ROCm!"
        echo "  You have ROCm ${ROCM_VERSION} installed, but PyTorch is built for CUDA."
        echo ""
        echo "Please uninstall the current PyTorch and install the ROCm version:"
        echo "  pip uninstall torch torchvision torchaudio"
        echo ""
        echo "Then run this script again to install PyTorch nightly with ROCm ${ROCM_VERSION} support."
        exit 1
    fi
else
    echo "Installing PyTorch nightly builds with ROCm ${ROCM_VERSION} support..."
    
    # Install PyTorch nightly builds (tested working version: 2.10.0.dev)
    if [[ "$ROCM_VERSION" == "7.0"* ]] || [[ "$ROCM_VERSION" == "7.1"* ]]; then
        echo "Installing PyTorch nightly (2.10.0.dev+) from PyTorch repository..."
        pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.2
    else
        echo ""
        echo "✗ ERROR: Unsupported ROCm version: ${ROCM_VERSION}"
        echo "  This script only supports ROCm 7.0.x and 7.1.x"
        echo "  Please install ROCm 7.0.x or 7.1.x and try again."
        exit 1
    fi
    
    if [ $? -ne 0 ]; then
        echo ""
        echo "✗ ERROR: Failed to install PyTorch nightly builds"
        echo ""
        echo "Alternative options:"
        echo "1. Build PyTorch from source for ROCm ${ROCM_VERSION}"
        echo "   See: https://github.com/pytorch/pytorch#from-source"
        echo ""
        echo "2. Use stable PyTorch from AMD: https://repo.radeon.com/rocm/manylinux/"
        exit 1
    fi
    
    echo "✓ PyTorch nightly installed successfully"
fi

# Verify PyTorch installation works with ROCm
echo ""
echo "Verifying PyTorch with ROCm..."
if python -c "import torch; print('  PyTorch version:', torch.__version__); print('  ROCm/HIP version:', torch.version.hip if hasattr(torch.version, 'hip') else 'N/A'); print('  CUDA available (ROCm):', torch.cuda.is_available()); print('  Device count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)" 2>&1; then
    echo "✓ PyTorch with ROCm verified successfully"
else
    echo "✗ PyTorch verification failed"
    exit 1
fi

# Step 1: Build RCCLX using the torchcomms build script
echo ""
echo "=========================================="
echo "Step 1: Building RCCLX from source"
echo "=========================================="

if [ ! -f "build_rcclx.sh" ]; then
    echo "Error: build_rcclx.sh not found in torchcomms directory"
    echo "Current directory: ${PWD}"
    exit 1
fi

echo "Building RCCLX for gfx950 architecture using build_rcclx.sh..."
# Unset USE_SYSTEM_LIBS so build_rcclx.sh will build glog and other dependencies from source
unset USE_SYSTEM_LIBS
./build_rcclx.sh --amdgpu_targets gfx950

if [ $? -ne 0 ]; then
    echo "Error: RCCLX build failed"
    exit 1
fi

echo "✓ RCCLX built successfully"

# Step 2: Build torchcomms with RCCL and RCCLX
echo ""
echo "=========================================="
echo "Step 2: Building torchcomms with RCCL + RCCLX"
echo "=========================================="

# Set backend configuration for both RCCL and RCCLX
export USE_NCCL=OFF
export USE_NCCLX=OFF
export USE_GLOO=OFF
export USE_RCCL=ON        # Use standard RCCL backend
export USE_RCCLX=ON       # Use RCCLX backend (Meta's enhanced RCCL)
export USE_SYSTEM_LIBS=0  # Build own dependencies (avoids fmt ABI issues)

echo ""
echo "Build Configuration:"
echo "  ROCM_HOME: ${ROCM_HOME}"
echo "  USE_RCCL: ${USE_RCCL}"
echo "  USE_RCCLX: ${USE_RCCLX}"
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
echo "Building and installing torchcomms with RCCL + RCCLX backends..."
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
    echo "Backends installed:"
    echo "  - RCCL: Standard AMD ROCm collective communications"
    echo "  - RCCLX: Meta's enhanced RCCL with optimizations"
    echo ""
    echo "To use torchcomms, activate the environment:"
    echo "  conda activate torchcomms"
    echo ""
    echo "Example usage:"
    echo "  from torchcomms import new_comm, ReduceOp"
    echo "  import torch"
    echo ""
    echo "  # Use standard RCCL backend:"
    echo "  comm = new_comm('rccl', torch.device('cuda'))"
    echo ""
    echo "  # OR use RCCLX backend (Meta's enhanced version):"
    echo "  comm = new_comm('rcclx', torch.device('cuda'))"
    echo ""
else
    echo ""
    echo "✗ Installation verification failed"
    echo "Please check the error messages above"
    exit 1
fi
