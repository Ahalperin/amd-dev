cd ~/amd-dev/amd
git clone https://github.com/rocm/amd-anp
cd amd-anp
git checkout tags/v1.1.0-5
git switch -c v1.1.0-5

# build and install rccl-network plugin (depends on AINIC driver that is installed on bare-metal)
sudo make RCCL_HOME=/home/dn/amd-dev/amd/rccl/ MPI_INCLUDE=/opt/ompi-4.1.6/include/ MPI_LIB_PATH=/opt/ompi-4.1.6/lib/ ROCM_PATH=/opt/rocm-7.0.1/
sudo make RCCL_HOME=/home/dn/amd-dev/amd/rccl/ ROCM_PATH=/opt/rocm-7.0.1/ install