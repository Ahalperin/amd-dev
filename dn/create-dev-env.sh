#!/usr/bin/env bash

########################################################
# Create DEV environment
########################################################

DN_DIR=$(dirname "$0")

# clone amd repos & checkout to specific branches
pushd ${DN_DIR}
git clone --recurse-submodules -b develop git@github.com:Ahalperin/rccl.git
cd rccl
git remote add upstream https://github.com/ROCm/rccl.git
popd

pushd ${DN_DIR}
git clone --recurse-submodules -b develop git@github.com:Ahalperin/rccl-tests.git
cd rccl-tests
git remote add upstream https://github.com/ROCm/rccl-tests.git
popd

pushd ${DN_DIR}
git clone git@github.com:Ahalperin/amd-anp.git
cd amd-anp
git remote add upstream https://github.com/rocm/amd-anp.git
popd