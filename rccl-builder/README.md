## RCCL Builder

### Build RCCL Builder Docker Image

```bash
./rccl-builder/build_rccl_builder
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

```bash
cd /workspace/rccl && ./install.sh --amdgpu_targets=gfx942 --prefix=/workspace/rccl/install/ --tests_build
cd /workspace/rccl && ./install.sh --amdgpu_targets=gfx950 --prefix=/workspace/rccl/install/ --tests_build
```

### Build RCCL-Tests

```bash
cd /workspace/rccl-tests/ && mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=Release -DUSE_MPI=ON -DCMAKE_PREFIX_PATH="/workspace/rccl/install;${MPI_INSTALL_PREFIX}" -DGPU_TARGETS=gfx942 .. && make -j6
```
