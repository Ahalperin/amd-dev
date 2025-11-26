#!/usr/bin/env bash

########################################################
# Create DEV environment
########################################################

DN_DIR=$(dirname "$0")

# clone amd repos & checkout to specific branches
pushd ${DN_DIR}
if [ ! -d "rccl" ]; then
  git clone --recurse-submodules -b develop git@github.com:Ahalperin/rccl.git
  cd rccl
  git remote add upstream https://github.com/ROCm/rccl.git
else
  echo "Skipping rccl - repository already exists"
fi
popd

pushd ${DN_DIR}
if [ ! -d "rccl-tests" ]; then
  git clone --recurse-submodules -b develop git@github.com:Ahalperin/rccl-tests.git
  cd rccl-tests
  git remote add upstream https://github.com/ROCm/rccl-tests.git
else
  echo "Skipping rccl-tests - repository already exists"
fi
popd

pushd ${DN_DIR}
if [ ! -d "amd-anp" ]; then
  git clone git@github.com:Ahalperin/amd-anp.git
  cd amd-anp
  git remote add upstream https://github.com/rocm/amd-anp.git
else
  echo "Skipping amd-anp - repository already exists"
fi
popd

# clone meta repos & checkout to specific branches
pushd ${DN_DIR}
if [ ! -d "torchcomms" ]; then
  git clone git@github.com:Ahalperin/torchcomms.git
  cd torchcomms
  git remote add upstream https://github.com/meta-pytorch/torchcomms.git
else
  echo "Skipping torchcomms - repository already exists"
fi
popd