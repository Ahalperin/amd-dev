#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e
# Exit if an undefined variable is used
set -u
# Fail on pipe errors
set -o pipefail

# Parse command line arguments
NPKIT_FLAG=""
RCCL_BRANCH="drop/2025-08"
AMD_ANP_BRANCH="tags/v1.1.0-5"

while [[ $# -gt 0 ]]; do
    case $1 in
        --npkit)
            NPKIT_FLAG="--npkit-enable"
            echo "NPKit profiling enabled"
            shift
            ;;
        --rccl-branch)
            RCCL_BRANCH="$2"
            echo "RCCL branch set to: $RCCL_BRANCH"
            shift 2
            ;;
        --amd-anp-branch)
            AMD_ANP_BRANCH="$2"
            echo "AMD-ANP branch set to: $AMD_ANP_BRANCH"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --npkit                Enable NPKit profiling support in RCCL"
            echo "  --rccl-branch BRANCH   Specify RCCL branch to checkout (default: drop/2025-08)"
            echo "  --amd-anp-branch BRANCH Specify AMD-ANP branch/tag to checkout (default: tags/v1.1.0-5)"
            echo "  -h, --help             Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "============================================"
echo "Starting RCCL build process"
echo "============================================"
echo "RCCL Branch: ${RCCL_BRANCH}"
echo "AMD-ANP Branch: ${AMD_ANP_BRANCH}"
echo "NPKit: ${NPKIT_FLAG:-disabled}"
echo "============================================"
echo ""

# set environment variables
export OMPI_HOME=/opt/ompi-4.1.6/
export OMPI_LIB_PATH=/opt/ompi-4.1.6/build/ompi/.libs/
export RCCL_HOME=/home/dn/amd-dev/dn/rccl/
export RCCL_INSTALL_DIR=${RCCL_HOME}/build/release/
export ROCM_HOME=/opt/rocm-7.0.1/

# checkout git rccl to specified branch/tag
cd ~/amd-dev/dn/rccl/
git fetch -p
echo "Checking out RCCL: origin/${RCCL_BRANCH}"
git checkout -B ${RCCL_BRANCH}
git pull --rebase origin

# build rccl based on specified branch
echo "Building RCCL..."
cd /home/dn/amd-dev/dn/rccl
sudo rm -rf build
./install.sh -l --prefix build/ --disable-mscclpp --disable-msccl-kernel --amdgpu_targets gfx950 ${NPKIT_FLAG}

# build rccl-tests
echo "Building RCCL tests..."
cd /home/dn/amd-dev/dn/rccl-tests/
sudo rm -rf build
make MPI=1 MPI_HOME=${OMPI_HOME} NCCL_HOME=${RCCL_INSTALL_DIR} -j

# checkout amd-anp to specified branch/tag
cd ~/amd-dev/dn/amd-anp
git fetch -p
echo "Checking out AMD-ANP: origin/${AMD_ANP_BRANCH}"
git checkout -B ${AMD_ANP_BRANCH}
git pull --rebase origin

# build and install rccl-network plugin (depends on AINIC driver that is installed on bare-metal)
echo "Building AMD-ANP network plugin..."
cd /home/dn/amd-dev/dn/amd-anp
sudo rm -rf build
sudo make RCCL_HOME=${RCCL_HOME} MPI_INCLUDE=${OMPI_HOME}/include/ MPI_LIB_PATH=${OMPI_HOME}/lib ROCM_PATH=${ROCM_HOME}

echo "Installing AMD-ANP network plugin..."
sudo make RCCL_HOME=${RCCL_HOME} ROCM_PATH=${ROCM_HOME} install

echo "Build completed successfully!"
