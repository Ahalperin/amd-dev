########################################################
# Create AMD DEV environment
########################################################

# needed for ssh-copy-id to remote servers
ssh-keygen -t ed25519 -C "dn@$(hostname)" -f ~/.ssh/id_ed25519

# allow user to access gpus
sudo usermod -aG render $USER
sudo usermod -aG video $USER

# add aliases to bashrc if they don't already exist
grep -q "alias c='clear'" ~/.bashrc || echo "alias c='clear'" >> ~/.bashrc
grep -q "alias gits='git status'" ~/.bashrc || echo "alias gits='git status'" >> ~/.bashrc
source ~/.bashrc

# create rccl-builder docker container
cd ~/amd-dev/rccl-builder
./build_rccl_builder

cd ~
git clone https://github.com/Ahalperin/amd-dev.git
cd ~/amd-dev
mkdir -p amd
cd ~/amd-dev/amd
git clone --recurse-submodules -b "develop" "https://github.com/ROCm/rccl"
git clone --recurse-submodules -b "develop" "https://github.com/ROCm/rccl-tests"

# checkout git rccl to drop/2025-08
cd ~/amd/rccl/
git checkout drop/2025-08
git switch -c drop/2025-08

cd ~/amd-dev
docker run -d \
  --name rccl-builder \
  --workdir /workspace \
  -v $(pwd)/amd:/workspace \
  -v $(pwd)/tools:/tools \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -e DISPLAY=$DISPLAY \
  -e USER_ID=$(id -u) \
  -e GROUP_ID=$(id -g) \
  --device /dev/kfd \
  --device /dev/dri \
  --security-opt seccomp=unconfined \
  --security-opt apparmor=unconfined \
  --shm-size=512m \
  --privileged \
  --network=host \
  --pid=host \
  --ipc=host \
  -v /dev:/dev \
  -v /sys:/sys:ro \
  -v /proc:/proc \
  rccl-builder:latest \
  tail -f /dev/null

# build rccl & rccl-tests
docker exec -it rccl-builder bash -c "cd /workspace/rccl && ./install.sh -l --prefix build/ --disable-mscclpp --disable-msccl-kernel"
docker exec -it rccl-builder bash -c 'cd /workspace/rccl-tests/ && mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=Release -DUSE_MPI=ON -DCMAKE_PREFIX_PATH="/home/dn/amd-dev/amd/rccl/install;${MPI_INSTALL_PREFIX}" -DGPU_TARGETS=gfx950 .. && make -j6'