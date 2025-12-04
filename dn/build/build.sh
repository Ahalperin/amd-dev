#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e
# Exit if an undefined variable is used
set -u
# Fail on pipe errors
set -o pipefail

WORKSPACE_ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
DN_DIR=${WORKSPACE_ROOT}/dn

# Parse command line arguments
NPKIT_FLAG=""
RCCL_BRANCH="drop/2025-08"
AMD_ANP_BRANCH="tags/v1.1.0-5"
LOG_DIR="${DN_DIR}/build"
ENABLE_LOGGING=true
RCCL_DISABLE_MSCCL_FLAGS=""

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
        --log-dir)
            LOG_DIR="$2"
            echo "Log directory set to: $LOG_DIR"
            shift 2
            ;;
        --no-log)
            ENABLE_LOGGING=false
            echo "Logging disabled"
            shift
            ;;
        --rccl-disable-msccl)
            RCCL_DISABLE_MSCCL_FLAGS="--disable-mscclpp --disable-msccl-kernel"
            echo "RCCL MSCCL disabled"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --npkit                Enable NPKit profiling support in RCCL"
            echo "  --rccl-branch BRANCH   Specify RCCL branch to checkout (default: drop/2025-08)"
            echo "  --amd-anp-branch BRANCH Specify AMD-ANP branch/tag to checkout (default: tags/v1.1.0-5)"
            echo "  --log-dir DIR          Directory for log file (default: /home/dn/amd-dev/dn/build)"
            echo "  --no-log               Disable logging to file"
            echo "  --rccl-disable-msccl   Disable MSCCL++ and MSCCL kernel in RCCL build"
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

# Setup logging to file if enabled
if [ "$ENABLE_LOGGING" = true ]; then
    
    # Use fixed log filename
    LOG_FILE="${LOG_DIR}/build.log"
    
    # Remove previous log file if it exists
    if [ -f "${LOG_FILE}" ]; then
        rm -f "${LOG_FILE}"
    fi
    
    # Create a named pipe for tee
    PIPE_NAME="/tmp/build_pipe_$$"
    mkfifo "${PIPE_NAME}"
    
    # Start tee in background, redirecting both stdout and stderr
    tee "${LOG_FILE}" < "${PIPE_NAME}" &
    TEE_PID=$!
    
    # Redirect stdout and stderr to the named pipe
    exec > "${PIPE_NAME}" 2>&1
    
    # Remove the pipe file (will remain available until all file descriptors are closed)
    rm "${PIPE_NAME}"
    
    # Trap to ensure tee process is cleaned up on exit
    cleanup() {
        # Close file descriptors
        exec 1>&- 2>&-
        # Wait for tee to finish
        wait ${TEE_PID} 2>/dev/null || true
    }
    trap cleanup EXIT
    
    echo "============================================"
    echo "Logging enabled"
    echo "Log file: ${LOG_FILE}"
    echo "You can monitor progress in real-time with:"
    echo "  tail -F ${LOG_FILE}"
    echo "============================================"
fi

echo "============================================"
echo "Starting RCCL build process"
echo "============================================"
echo "RCCL Branch: ${RCCL_BRANCH}"
echo "AMD-ANP Branch: ${AMD_ANP_BRANCH}"
echo "NPKit: ${NPKIT_FLAG:-disabled}"
echo "MSCCL: $([ -n \"${RCCL_DISABLE_MSCCL_FLAGS}\" ] && echo 'disabled' || echo 'enabled')"
echo "============================================"
echo ""

# set environment variables
export OMPI_HOME=/opt/ompi-4.1.6/
export OMPI_LIB_PATH=/opt/ompi-4.1.6/lib/
export RCCL_HOME=${DN_DIR}/rccl/
export RCCL_INSTALL_DIR=${RCCL_HOME}/build/release/
export ROCM_HOME=/opt/rocm-7.0.1/

# checkout git rccl to specified branch/tag
cd ${DN_DIR}/rccl/
git fetch -p
echo "Checking out RCCL: ${RCCL_BRANCH}"
git checkout ${RCCL_BRANCH}
# Only pull if we're on a branch (not a tag/detached HEAD)
if git symbolic-ref -q HEAD > /dev/null; then
    git pull --rebase
fi

# build rccl based on specified branch
echo "Building RCCL..."
cd ${DN_DIR}/rccl
# No need to remove build directory if it already exists, this is a significant time saver
# sudo rm -rf build
./install.sh -l --prefix build/ --amdgpu_targets gfx950 ${RCCL_DISABLE_MSCCL_FLAGS} ${NPKIT_FLAG}

# build rccl-tests
echo "Building RCCL tests..."
cd ${DN_DIR}/rccl-tests/
sudo rm -rf build
make MPI=1 MPI_HOME=${OMPI_HOME} NCCL_HOME=${RCCL_INSTALL_DIR} -j

# checkout amd-anp to specified branch/tag
cd ${DN_DIR}/amd-anp
git fetch -p
echo "Checking out AMD-ANP: ${AMD_ANP_BRANCH}"
git checkout ${AMD_ANP_BRANCH}
# Only pull if we're on a branch (not a tag/detached HEAD)
if git symbolic-ref -q HEAD > /dev/null; then
    git pull --rebase
fi

# build and install rccl-network plugin (depends on AINIC driver that is installed on bare-metal)
echo "Building AMD-ANP network plugin..."
cd ${DN_DIR}/amd-anp
sudo rm -rf build
sudo make RCCL_HOME=${RCCL_HOME} MPI_INCLUDE=${OMPI_HOME}/include/ MPI_LIB_PATH=${OMPI_HOME}/lib ROCM_PATH=${ROCM_HOME}

# We do not install the network plugin to the shared library path to allow multiple versions.
# echo "Installing AMD-ANP network plugin..."
# sudo make RCCL_HOME=${RCCL_HOME} ROCM_PATH=${ROCM_HOME} install

echo "Build completed successfully!"
