#!/bin/bash
#
# Helper script to use ROCm 7.1.1
# Source this file to set environment for ROCm 7.1.1
#
# Usage:
#   source use-rocm-7.1.sh
#   OR
#   . use-rocm-7.1.sh
#

ROCM_VERSION="7.1.1"
ROCM_BASE="/opt/rocm-${ROCM_VERSION}"

if [ ! -d "$ROCM_BASE" ]; then
    echo "Error: ROCm ${ROCM_VERSION} not found at ${ROCM_BASE}"
    echo "Please install it first using install_rocm_7.1.1.sh"
    return 1 2>/dev/null || exit 1
fi

# Remove old ROCm paths if they exist
export PATH=$(echo $PATH | tr ':' '\n' | grep -v '/opt/rocm' | tr '\n' ':' | sed 's/:$//')
export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -v '/opt/rocm' | tr '\n' ':' | sed 's/:$//')

# Set ROCm 7.1.1 environment
export ROCM_PATH="${ROCM_BASE}"
export ROCM_VERSION="${ROCM_VERSION}"
export PATH="${ROCM_BASE}/bin:${ROCM_BASE}/llvm/bin:${PATH}"
export LD_LIBRARY_PATH="${ROCM_BASE}/lib:${ROCM_BASE}/lib64:${LD_LIBRARY_PATH}"
export CMAKE_PREFIX_PATH="${ROCM_BASE}:${CMAKE_PREFIX_PATH}"

# HIP settings
export HIP_PATH="${ROCM_BASE}"
export HIP_PLATFORM="amd"
export HSA_PATH="${ROCM_BASE}"

echo "✓ ROCm ${ROCM_VERSION} environment loaded"
echo "  ROCM_PATH: ${ROCM_PATH}"
echo "  HIP_PATH: ${HIP_PATH}"
echo ""
echo "Verify with:"
echo "  rocm-smi"
echo "  rocminfo | grep -i 'Name:'"
echo "  hipcc --version"

