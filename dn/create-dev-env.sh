#!/usr/bin/env bash

########################################################
# Create DEV environment
########################################################

# clone amd repos & checkout to specific branches
cd ~/amd-dev/dn
git clone --recurse-submodules -b develop git@github.com:Ahalperin/rccl.git
cd rccl
git remote add upstream https://github.com/ROCm/rccl.git

cd ~/amd-dev/dn/
git clone --recurse-submodules -b develop git@github.com:Ahalperin/rccl-tests.git
cd rccl-tests
git remote add upstream https://github.com/ROCm/rccl-tests.git

cd ~/amd-dev/dn
git clone git@github.com:Ahalperin/amd-anp.git
cd amd-anp
git remote add upstream https://github.com/rocm/amd-anp.git