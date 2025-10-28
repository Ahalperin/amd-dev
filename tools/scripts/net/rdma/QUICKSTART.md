# Quick Start: AMD Pensando RDMA Setup

## For New Servers

Run this single command to install and configure RDMA:

```bash
cd /home/dn/amd-dev
sudo ./tools/scripts/net/rdma/install-rdma-devices.sh
```

That's it! The script will:
- ✅ Install required packages
- ✅ Load RDMA modules
- ✅ Configure network interfaces  
- ✅ **Configure automatic loading on boot** (new!)
- ✅ Verify everything works

## After Reboot

RDMA should work automatically. To verify:

```bash
rdma link show | grep ACTIVE
```

## If RDMA Is Missing After Reboot

Just re-run the script:

```bash
cd /home/dn/amd-dev
sudo ./tools/scripts/net/rdma/install-rdma-devices.sh
```

## For RCCL Tests

Add to your environment:

```bash
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME="enp9s0,enp25s0,enp105s0,enp121s0,enp137s0,enp153s0,enp233s0,enp249s0"
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=106
export NCCL_DEBUG=INFO
```

## More Help

- Full documentation: [README.md](./README.md)
- Having issues? [TROUBLESHOOTING.md](./TROUBLESHOOTING.md)
- Verify setup: `../net-discovery.sh`
