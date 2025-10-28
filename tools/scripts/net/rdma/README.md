# AMD Pensando RDMA Installation Scripts

This directory contains tools for installing and configuring RDMA support on AMD Pensando DSC (Distributed Services Card) NICs.

## Scripts

### `install-rdma-devices.sh`

Automated installation script that configures RDMA support for Pensando DSC NICs.

#### Features

- ✅ Detects Pensando DSC NICs automatically
- ✅ Checks current RDMA status
- ✅ Installs required packages (`ionic-common`, `ionic-dkms`)
- ✅ Loads RDMA kernel modules (`ionic_rdma`, `ib_peer_mem`)
- ✅ Updates netplan configuration for new interface names
- ✅ Brings up network interfaces
- ✅ Verifies RDMA functionality
- ✅ Provides RCCL configuration examples

#### Prerequisites

- Root/sudo access
- AMD Pensando DSC NICs installed
- AMD AINIC repository available at `/opt/amd/ainic/deb-repo`
- Ubuntu 22.04 with netplan (or compatible system)

#### Usage

**Basic usage (recommended):**
```bash
sudo ./install-rdma-devices.sh
```

**Dry-run mode (see what would be done without making changes):**
```bash
sudo ./install-rdma-devices.sh --dry-run
```

**Skip netplan configuration updates:**
```bash
sudo ./install-rdma-devices.sh --skip-netplan
```

**Help:**
```bash
./install-rdma-devices.sh --help
```

#### What the Script Does

1. **Pre-flight Checks:**
   - Verifies root privileges
   - Detects Pensando DSC NICs
   - Checks if RDMA is already configured

2. **Package Installation:**
   - Updates apt cache
   - Installs `ionic-common` and `ionic-dkms` packages
   - These provide the RDMA-enabled drivers

3. **Module Loading:**
   - Unloads existing `ionic` module
   - Loads new `ionic` module (with RDMA support)
   - Loads `ionic_rdma` module
   - Loads `ib_peer_mem` module (if available)

4. **Interface Configuration:**
   - Detects Pensando network interfaces
   - Brings up all interfaces
   - Updates netplan configuration (changes `enp*np0` to `enp*`)
   - Applies netplan configuration

5. **Verification:**
   - Checks RDMA devices in `/sys/class/infiniband/`
   - Displays RDMA link status
   - Verifies active RDMA links
   - Shows InfiniBand device information

6. **Configuration Output:**
   - Provides RCCL environment variable examples
   - Suggests verification commands

#### Example Output

```
═══════════════════════════════════════════════════════════════════════
  AMD Pensando RDMA Installation
═══════════════════════════════════════════════════════════════════════

[SUCCESS] Found 8 Pensando DSC NIC(s)
[INFO] ionic_rdma module not loaded - RDMA support needs to be installed
[SUCCESS] Required packages are available
[SUCCESS] Packages installed successfully
[SUCCESS] RDMA modules loaded successfully
[SUCCESS] Found 8 RDMA device(s)
[SUCCESS] 8 RDMA link(s) are ACTIVE

═══════════════════════════════════════════════════════════════════════
  Installation Complete!
═══════════════════════════════════════════════════════════════════════
```

#### Troubleshooting

**Package not found:**
```bash
# Ensure the AMD AINIC repository is configured
ls /opt/amd/ainic/deb-repo/
cat /etc/apt/sources.list.d/ainic.list
```

**Module load failure:**
```bash
# Check kernel logs
dmesg | tail -50

# Check module dependencies
modinfo ionic_rdma
```

**RDMA devices not created:**
```bash
# Verify modules are loaded
lsmod | grep ionic

# Check kernel logs
dmesg | grep ionic

# Manually reload modules
sudo modprobe -r ionic
sudo modprobe ionic
sudo modprobe ionic_rdma
```

**Network interfaces not coming up:**
```bash
# Check interface status
ip link show | grep enp

# Manually bring up interface
sudo ip link set enp9s0 up

# Check netplan configuration
sudo netplan --debug apply
```

#### Post-Installation

After successful installation:

1. **Verify RDMA is working:**
   ```bash
   # Check RDMA devices
   ls /sys/class/infiniband/
   
   # Check RDMA link status
   rdma link show
   
   # List InfiniBand devices
   ibv_devinfo -l
   ```

2. **Test RDMA performance between two nodes:**
   ```bash
   # On node 1
   ib_send_bw -d ionic_0 -i 1 -F --report_gbits
   
   # On node 2 (connect to node 1)
   ib_send_bw -d ionic_0 -i 1 -F --report_gbits <node1_ip>
   ```

3. **Configure RCCL to use RDMA:**
   ```bash
   export NCCL_IB_DISABLE=0
   export NCCL_SOCKET_IFNAME="enp9s0,enp25s0,enp105s0,enp121s0,enp137s0,enp153s0,enp233s0,enp249s0"
   export NCCL_IB_GID_INDEX=3
   export NCCL_IB_TC=106
   export NCCL_IB_QPS_PER_CONNECTION=1
   export NCCL_IB_TIMEOUT=22
   export NCCL_IB_RETRY_CNT=7
   export NCCL_IB_ADAPTIVE_ROUTING=1
   export NCCL_DEBUG=INFO
   ```

4. **Make configuration persistent:**
   
   The netplan changes are automatically made persistent. For module loading on boot:
   
   ```bash
   # Ensure modules load on boot
   echo "ionic" | sudo tee -a /etc/modules
   echo "ionic_rdma" | sudo tee -a /etc/modules
   echo "ib_peer_mem" | sudo tee -a /etc/modules
   ```

## Related Scripts

- `../net-discovery.sh` - Discover and display network configuration including RDMA devices
- `../connectivity/test-connectivity.sh` - Test network connectivity between nodes

## Support

For issues or questions:
- Check kernel logs: `dmesg | grep -i "ionic\|rdma"`
- Check module status: `lsmod | grep ionic`
- Verify packages: `dpkg -l | grep ionic`
- Contact AMD support for Pensando-specific issues

## Version History

- **v1.0** (2024-10-28): Initial release
  - Automated RDMA installation for Pensando DSC NICs
  - Netplan configuration updates
  - Comprehensive verification and troubleshooting

## References

- AMD Pensando Documentation
- RDMA Programming Guide
- NCCL Documentation: https://docs.nvidia.com/deeplearning/nccl/


