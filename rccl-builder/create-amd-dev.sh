########################################################
# Create AMD DEV environment
########################################################

# Exit immediately if any command fails
set -e
set -o pipefail

# needed for ssh-copy-id to remote servers
# ssh-keygen -t ed25519 -C "dn@$(hostname)" -f ~/.ssh/id_ed25519
[ ! -f ~/.ssh/id_ed25519 ] && ssh-keygen -t ed25519 -C "dn@$(hostname)" -f ~/.ssh/id_ed25519 -N "" -q

# allow user to access gpus
sudo usermod -aG render $USER
sudo usermod -aG video $USER

# add aliases to bashrc if they don't already exist
grep -q "alias c='clear'" ~/.bashrc || echo "alias c='clear'" >> ~/.bashrc
grep -q "alias gs='git status'" ~/.bashrc || echo "alias gs='git status'" >> ~/.bashrc
source ~/.bashrc

# install ompi-4.1.6 only if not already installed
if [ ! -d /opt/ompi-4.1.6 ]; then
    cd ~/
    wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.6.tar.gz
    tar -zxf openmpi-4.1.6.tar.gz
    cd openmpi-4.1.6
    sudo ./configure --prefix=/opt/ompi-4.1.6
    make -j16 install
    cd ..
    sudo rm -rf openmpi-4.1.6 openmpi-4.1.6.tar.gz
fi

# clone amd repos & checkout to specific branches
cd ~
git clone https://github.com/Ahalperin/amd-dev.git
cd ~/amd-dev
mkdir -p amd
cd ~/amd-dev/amd
git clone --recurse-submodules -b "develop" "https://github.com/ROCm/rccl"
git clone --recurse-submodules -b "develop" "https://github.com/ROCm/rccl-tests"
git clone https://github.com/rocm/amd-anp

# checkout git rccl to drop/2025-08
cd ~/amd/rccl/
git checkout drop/2025-08
git switch -c drop/2025-08

# checkout amd-anp to v1.1.0-5
cd ~/amd-dev/amd/amd-anp
git checkout tags/v1.1.0-5
git switch -c v1.1.0-5

# install openmpi-4.1.6
sudo rm -rf /opt/ompi-4.1.6/
cd ~/
wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.6.tar.gz
tar -zxf openmpi-4.1.6.tar.gz
cd openmpi-4.1.6
sudo ./configure --prefix=/opt/ompi-4.1.6
sudo make -j16 install
cd ..
sudo rm -rf openmpi-4.1.6 openmpi-4.1.6.tar.gz

# set environment variables
export OMPI_HOME=/opt/ompi-4.1.6/
export OMPI_LIB_PATH=/opt/ompi-4.1.6/build/ompi/.libs/
export RCCL_HOME=/home/dn/amd-dev/amd/rccl/
export RCCL_INSTALL_DIR=${RCCL_HOME}/build/release/
export ROCM_HOME=/opt/rocm-7.0.1/
# build rccl & rccl-tests
cd /home/dn/amd-dev/amd/rccl && sudo rm -rf build && ./install.sh -l --prefix build/ --disable-mscclpp --disable-msccl-kernel
# cd /home/dn/amd-dev/amd/rccl-tests/ && sudo rm -rf build && mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=Release -DUSE_MPI=ON -DCMAKE_PREFIX_PATH="/home/dn/amd-dev/amd/rccl/;/opt/ompi-4.1.6/" -DGPU_TARGETS=gfx950 .. && make -j6
cd /home/dn/amd-dev/amd/rccl-tests/ && sudo rm -rf build && make MPI=1 MPI_HOME=${OMPI_HOME} NCCL_HOME=${RCCL_INSTALL_DIR} -j
# build and install rccl-network plugin (depends on AINIC driver that is installed on bare-metal)
cd /home/dn/amd-dev/amd/amd-anp && sudo rm -rf build && sudo make RCCL_HOME=${RCCL_HOME} MPI_INCLUDE=${OMPI_HOME}/include/ MPI_LIB_PATH=${OMPI_HOME}/lib ROCM_PATH=${ROCM_HOME}
cd /home/dn/amd-dev/amd/amd-anp && sudo make RCCL_HOME=${RCCL_HOME} ROCM_PATH=${ROCM_HOME} install