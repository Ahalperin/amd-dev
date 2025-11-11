#!/bin/bash

# Parse command line arguments
NPKIT_FLAG=""
AMD_ANP_NPKIT=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --npkit)
            NPKIT_FLAG="--npkit-enable"
            AMD_ANP_NPKIT="ENABLE_NPKIT=1"
            echo "NPKit profiling enabled"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --npkit    Enable NPKit profiling support in RCCL and AMD-ANP"
            echo "  -h, --help Show this help message"
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
export RCCL_HOME=/home/dn/amd-dev/dn/rccl/
export RCCL_INSTALL_DIR=${RCCL_HOME}/build/release/
export ROCM_HOME=/opt/rocm-7.0.1/
# build rccl & rccl-tests
cd /home/dn/amd-dev/dn/rccl && sudo rm -rf build && ./install.sh -l --prefix build/ --disable-mscclpp --disable-msccl-kernel ${NPKIT_FLAG}
# cd /home/dn/amd-dev/dn/rccl-tests/ && sudo rm -rf build && mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=Release -DUSE_MPI=ON -DCMAKE_PREFIX_PATH="/home/dn/amd-dev/dn/rccl/;/opt/ompi-4.1.6/" -DGPU_TARGETS=gfx950 .. && make -j6
cd /home/dn/amd-dev/dn/rccl-tests/ && sudo rm -rf build && make MPI=1 MPI_HOME=${OMPI_HOME} NCCL_HOME=${RCCL_INSTALL_DIR} -j
# build and install rccl-network plugin (depends on AINIC driver that is installed on bare-metal)
cd /home/dn/amd-dev/dn/amd-anp && sudo rm -rf build && sudo make ${AMD_ANP_NPKIT} RCCL_HOME=${RCCL_HOME} MPI_INCLUDE=${OMPI_HOME}/include/ MPI_LIB_PATH=${OMPI_HOME}/lib ROCM_PATH=${ROCM_HOME}
cd /home/dn/amd-dev/dn/amd-anp && sudo make RCCL_HOME=${RCCL_HOME} ROCM_PATH=${ROCM_HOME} install

