## RCCL Builder

### Build RCCL Builder Docker Image

```bash
cd cd ~/amd-dev/rccl-builder
./build_rccl_builder
```

### fetch source tree

```bash
cd ~
git clone https://github.com/Ahalperin/amd-dev.git
cd ~/amd-dev
mkdir -p amd
cd ~/amd-dev/amd
git clone --recurse-submodules -b "develop" "https://github.com/ROCm/rccl"
git clone --recurse-submodules -b "develop" "https://github.com/ROCm/rccl-tests"
```

### Create RCCL Builder Container

Run it from amd-dev directory
```shell
cd ~/amd-dev
```

### when running from a container one can grant the container accessing the GPUs
- --device /dev/kfd - Grants access to the Kernel Fusion Driver, which is the main compute interface for AMD GPUs
- --device /dev/dri - Provides access to all GPU render nodes through the Direct Rendering Interface
- ---security-opt seccomp=unconfined - Enables memory mapping operations that may be required for GPU communication

```bash
# run container with root permissions and grant access to the GPU drivers
docker run --rm -it \
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
  rccl-builder:latest
```


### Build RCCL (gfx942 ==> MI300X)
### Build RCCL (gfx950 ==> MI325X)
### Build RCCL (-l ==> local installed GPU type)

```bash
# cd /workspace/rccl && ./install.sh --amdgpu_targets=gfx942 --prefix=/workspace/rccl/install/ --tests_build
# cd /workspace/rccl && ./install.sh --amdgpu_targets=gfx950 --prefix=/workspace/rccl/install/ --tests_build
# use -l to build RCCL for the local installed GPU's type

# build RCCL specific version
git checkout drop/2025-08
git switch -c drop/2025-08
cd /workspace/rccl && ./install.sh -l --prefix build/ --disable-mscclpp --disable-msccl-kernel
```

### Build RCCL-Tests

```bash
cd /workspace/rccl-tests/ && mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=Release -DUSE_MPI=ON -DCMAKE_PREFIX_PATH="/workspace/rccl/install;${MPI_INSTALL_PREFIX}" -DGPU_TARGETS=gfx942 .. && make -j6
cd /workspace/rccl-tests/ && mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=Release -DUSE_MPI=ON -DCMAKE_PREFIX_PATH="/workspace/rccl/install;${MPI_INSTALL_PREFIX}" -DGPU_TARGETS=gfx950 .. && make -j6

cd /workspace/rccl-tests/ && make MPI=1 MPI_HOME=/opt/ompi/ NCCL_HOME=/workspace/rccl -j 20
```
