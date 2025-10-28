########################################################
# Create AMD DEV environment
########################################################

# needed for ssh-copy-id to remote servers
# ssh-keygen -t ed25519 -C "dn@$(hostname)" -f ~/.ssh/id_ed25519
[ ! -f ~/.ssh/id_ed25519 ] && ssh-keygen -t ed25519 -C "dn@$(hostname)" -f ~/.ssh/id_ed25519 -N "" -q

# allow user to access gpus
sudo usermod -aG render $USER
sudo usermod -aG video $USER

# add aliases to bashrc if they don't already exist
grep -q "alias c='clear'" ~/.bashrc || echo "alias c='clear'" >> ~/.bashrc
grep -q "alias gits='git status'" ~/.bashrc || echo "alias gits='git status'" >> ~/.bashrc
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

# create rccl-builder docker container
# cd ~/amd-dev/rccl-builder
# ./build_rccl_builder

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

# checkout amd-anp to drop/2025-08
cd ~/amd-dev/amd/amd-anp
git checkout tags/v1.1.0-5
git switch -c v1.1.0-5

# cd ~/amd-dev
# docker run -d \
#   --name rccl-builder \
#   --workdir /workspace \
#   -v /opt/rocm-7.0.1:/opt/rocm-7.0.1 \
#   -v $(pwd)/amd:/workspace \
#   -v $(pwd)/tools:/tools \
#   -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
#   -e DISPLAY=$DISPLAY \
#   -e USER_ID=$(id -u) \
#   -e GROUP_ID=$(id -g) \
#   --device /dev/kfd \
#   --device /dev/dri \
#   --security-opt seccomp=unconfined \
#   --security-opt apparmor=unconfined \
#   --shm-size=512m \
#   --privileged \
#   --network=host \
#   --pid=host \
#   --ipc=host \
#   -v /dev:/dev \
#   -v /sys:/sys:ro \
#   -v /proc:/proc \
#   rccl-builder:latest \
#   tail -f /dev/null


# build rccl & rccl-tests
# docker exec -it rccl-builder bash -c "cd /workspace/rccl && ROCM_PATH=/opt/rocm-7.0.1 ./install.sh -l --prefix build/ --disable-mscclpp --disable-msccl-kernel"
# docker exec -it rccl-builder bash -c 'cd /workspace/rccl-tests/ && mkdir -p build && cd build && ROCM_PATH=/opt/rocm-7.0.1 cmake -DCMAKE_BUILD_TYPE=Release -DUSE_MPI=ON -DCMAKE_PREFIX_PATH="/home/dn/amd-dev/amd/rccl/install;${MPI_INSTALL_PREFIX}" -DGPU_TARGETS=gfx950 .. && make -j6'

cd /home/dn/amd-dev/amd/rccl && sudo rm -rf build && ./install.sh -l --prefix build/ --disable-mscclpp --disable-msccl-kernel
cd /home/dn/amd-dev/amd/rccl-tests/ && sudo rm -rf build && mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=Release -DUSE_MPI=ON -DCMAKE_PREFIX_PATH="/home/dn/amd-dev/amd/rccl/;/opt/ompi-4.1.6/" -DGPU_TARGETS=gfx950 .. && make -j6

# build and install rccl-network plugin (depends on AINIC driver that is installed on bare-metal)
cd /home/dn/amd-dev/amd/amd-anp && sudo make RCCL_HOME=/home/dn/amd-dev/amd/rccl/ MPI_INCLUDE=/opt/ompi-4.1.6/include/ MPI_LIB_PATH=/opt/ompi-4.1.6/build/ompi/.libs/ ROCM_PATH=/opt/rocm-7.0.1/
cd /home/dn/amd-dev/amd/amd-anp && sudo make RCCL_HOME=/home/dn/amd-dev/amd/rccl/ ROCM_PATH=/opt/rocm-7.0.1/ install