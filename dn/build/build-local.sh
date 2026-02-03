#!/bin/bash

# Get workspace root using git, then get dn/ directory
WORKSPACE_ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
DN_DIR=${WORKSPACE_ROOT}/dn

# Parse command line arguments
NPKIT_FLAG=""
BUILD_TARGET=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --npkit)
            NPKIT_FLAG="--npkit-enable"
            echo "NPKit profiling enabled"
            shift
            ;;
        --target)
            if [[ -z "$2" ]] || [[ "$2" =~ ^- ]]; then
                echo "Error: --target requires a value (rccl, rccl-tests, or amd-anp)"
                exit 1
            fi
            BUILD_TARGET="$2"
            if [[ "$BUILD_TARGET" != "rccl" ]] && [[ "$BUILD_TARGET" != "rccl-tests" ]] && [[ "$BUILD_TARGET" != "amd-anp" ]]; then
                echo "Error: Invalid target '$BUILD_TARGET'. Must be one of: rccl, rccl-tests, amd-anp"
                exit 1
            fi
            echo "Building target: $BUILD_TARGET"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --npkit              Enable NPKit profiling support in RCCL"
            echo "  --target TARGET      Build only the specified target (rccl, rccl-tests, or amd-anp)"
            echo "                       If not specified, all targets will be built"
            echo "  -h, --help           Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# set environment variables
export OMPI_HOME=/opt/ompi-4.1.6/
export OMPI_LIB_PATH=/opt/ompi-4.1.6/build/ompi/.libs/
export RCCL_HOME=${DN_DIR}/rccl/
export RCCL_INSTALL_DIR=${RCCL_HOME}/build/release/
export ROCM_HOME=/opt/rocm/

# build rccl
if [[ -z "$BUILD_TARGET" ]] || [[ "$BUILD_TARGET" == "rccl" ]]; then
    cd ${DN_DIR}/rccl && sudo rm -rf build && CMAKE_EXPORT_COMPILE_COMMANDS=ON ./install.sh -l --prefix build/ --disable-msccl-kernel ${NPKIT_FLAG} --log-trace
fi

# build rccl-tests
if [[ -z "$BUILD_TARGET" ]] || [[ "$BUILD_TARGET" == "rccl-tests" ]]; then
    cd ${DN_DIR}/rccl-tests/ && sudo rm -rf build && make ROCM_PATH=${ROCM_HOME} MPI=1 MPI_HOME=${OMPI_HOME} NCCL_HOME=${RCCL_INSTALL_DIR} -j
fi

# build and install rccl-network plugin (depends on AINIC driver that is installed on bare-metal)
if [[ -z "$BUILD_TARGET" ]] || [[ "$BUILD_TARGET" == "amd-anp" ]]; then
    cd ${DN_DIR}/amd-anp && sudo rm -rf build && sudo make RCCL_HOME=${RCCL_HOME} MPI_INCLUDE=${OMPI_HOME}/include/ MPI_LIB_PATH=${OMPI_HOME}/lib ROCM_PATH=${ROCM_HOME}
fi

