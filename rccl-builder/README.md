## RCCL Builder

### Build RCCL Builder Docker Image

```bash
utils/containers/rccl_builder/build_rccl_builder
```

### Create RCCL Builder Container

Run it from the directory that contains rccl and rccl_tests:

```bash
# run container with root 
docker run --rm -it --name rccl-builder --workdir /workspace -v $(pwd):/workspace -v /tmp/.X11-unix:/tmp/.X11-unix:rw -e DISPLAY=$DISPLAY -e USER_ID=$(id -u) -e GROUP_ID=$(id -g) rccl-builder:latest
# run container with user
docker run --rm -it --name rccl-builder --workdir /workspace -v $(pwd):/workspace -v /tmp/.X11-unix:/tmp/.X11-unix:rw -e DISPLAY=$DISPLAY -e USER_ID=$(id -u) -e GROUP_ID=$(id -g) rccl-builder:latest su user -c 'cd /workspace && exec /bin/bash'
```

### Build RCCL (gfx942 ==> MI300X)

```bash
cd /worspace/rccl && ./install.sh --amdgpu_targets=${GPU_TARGETS} --prefix=/workspace/rccl/install/ -tests_build
```

### Build RCCL-Tests

```bash
cd /workspace/rccl-tests/ && mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=Release -DUSE_MPI=ON -DCMAKE_PREFIX_PATH="/workspace/rccl/install;${MPI_INSTALL_PREFIX}" -DGPU_TARGETS=gfx942 .. && make -j6
```