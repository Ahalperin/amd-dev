# RDMA Troubleshooting Guide for AMD Pensando NICs

## Common Issue: RDMA Not Available After Reboot

### Symptom
After rebooting the server:
- ❌ All network interfaces show "DOWN"
- ❌ RDMA devices show "N/A" in net-discovery.sh
- ❌ Interface names revert to `enp*np0` format
- ❌ No IP addresses configured
- ❌ `ionic_rdma` module not loaded

### Root Cause
The RDMA kernel modules (`ionic_rdma`, `ib_peer_mem`) were not configured to load automatically at boot. The system loaded the default kernel `ionic` driver instead of the DKMS version with RDMA support.

### Solution 1: Run the Installation Script (Quick Fix)

```bash
cd /home/dn/amd-dev
sudo ./tools/scripts/net/rdma/install-rdma-devices.sh
```

The script will:
1. Reload the RDMA-enabled drivers
2. Configure persistent module loading (as of latest version)
3. Bring up interfaces and restore RDMA

### Solution 2: Manual Configuration (If Script Not Available)

#### Step 1: Reload RDMA Modules
```bash
# Unload old driver
sudo modprobe -r ionic

# Load RDMA-enabled drivers
sudo modprobe ionic
sudo modprobe ionic_rdma
sudo modprobe ib_peer_mem
```

#### Step 2: Configure Persistent Loading
```bash
# Create module loading configuration
sudo tee /etc/modules-load.d/ionic-rdma.conf > /dev/null <<EOF
ionic
ionic_rdma
ib_peer_mem
EOF

# Create modprobe configuration
sudo tee /etc/modprobe.d/ionic-rdma.conf > /dev/null <<EOF
softdep ionic post: ionic_rdma
install ionic /sbin/modprobe --ignore-install ionic && /sbin/modprobe ionic_rdma || /sbin/modprobe --ignore-install ionic
EOF

# Update initramfs
sudo update-initramfs -u
```

#### Step 3: Apply Network Configuration
```bash
sudo netplan apply
```

### Verification

After applying the fix, verify RDMA is working:

```bash
# Check loaded modules
lsmod | grep ionic

# Should show:
# ionic_rdma            237568  0
# ib_peer_mem            20480  1 ionic_rdma
# ionic                 258048  1 ionic_rdma

# Check RDMA devices
ls /sys/class/infiniband/

# Should show: ionic_0 through ionic_7 (plus any other RDMA devices)

# Check RDMA link status
rdma link show

# Should show all ionic links as ACTIVE/LINK_UP

# Run net-discovery
cd /home/dn/amd-dev
./tools/scripts/net/net-discovery.sh
```

Expected output should show:
- ✅ All interfaces UP
- ✅ IP addresses configured
- ✅ RDMA devices listed (ionic_0, ionic_1, etc.)
- ✅ 400Gbps link speed

### Testing RDMA After Reboot

After configuring persistent loading, test with a reboot:

```bash
# Reboot the server
sudo reboot

# After reboot, check RDMA immediately
lsmod | grep ionic_rdma
rdma link show
ls /sys/class/infiniband/ | grep ionic
```

All commands should show RDMA is active without manual intervention.

## Other Common Issues

### Issue: Wrong ionic Module Loaded

**Symptom:**
```bash
$ lsmod | grep ionic
ionic                 200704  0    # ← Wrong! Should be ~258KB
```

**Solution:**
```bash
# The kernel driver is loading instead of DKMS
# Create/update module blacklist
sudo tee -a /etc/modprobe.d/ionic-rdma.conf > /dev/null <<EOF
# Prefer DKMS version
install ionic /sbin/modprobe --ignore-install ionic && /sbin/modprobe ionic_rdma
EOF

sudo update-initramfs -u
sudo reboot
```

### Issue: RDMA Links Show DOWN

**Symptom:**
```bash
$ rdma link show
link ionic_0/1 state DOWN physical_state DISABLED netdev enp9s0
```

**Solution:**
```bash
# Bring up network interfaces
sudo ip link set enp9s0 up
sudo ip link set enp25s0 up
# ... repeat for all interfaces

# Or apply netplan
sudo netplan apply
```

### Issue: No IP Addresses After RDMA Configuration

**Symptom:**
Interfaces are UP but have no IP addresses.

**Solution:**
```bash
# Check if netplan was updated correctly
cat /etc/netplan/*.yaml | grep -A5 "enp"

# Ensure interface names are correct (without np0)
# If they still have np0, update manually:
sudo sed -i 's/enp\([0-9]*\)s0np0/enp\1s0/g' /etc/netplan/*.yaml

# Apply netplan
sudo netplan apply
```

### Issue: Modules Load But No RDMA Devices

**Symptom:**
```bash
$ lsmod | grep ionic_rdma
ionic_rdma            237568  0

$ ls /sys/class/infiniband/ | grep ionic
# No output
```

**Solution:**
```bash
# Check dmesg for errors
sudo dmesg | grep -i "ionic\|rdma" | tail -20

# Try reloading modules in correct order
sudo modprobe -r ionic_rdma
sudo modprobe -r ionic
sudo modprobe ionic
sleep 2
sudo modprobe ionic_rdma

# Check again
ls /sys/class/infiniband/
```

### Issue: DKMS Modules Not Building

**Symptom:**
```bash
$ dpkg -l | grep ionic-dkms
ii  ionic-dkms   25.08.2.001   all   AMD Pensando IONIC Driver(s) as DKMS

$ ls /lib/modules/$(uname -r)/updates/dkms/ | grep ionic
# No output
```

**Solution:**
```bash
# Rebuild DKMS modules
sudo dkms status
sudo dkms remove ionic/25.08.2.001 --all
sudo dkms install ionic/25.08.2.001

# Verify modules are built
ls -lh /lib/modules/$(uname -r)/updates/dkms/*ionic*
```

## Prevention

To prevent RDMA issues after system updates or reboots:

1. **Always use the latest installation script** which configures persistence automatically
2. **Verify after kernel updates** that DKMS modules rebuild correctly
3. **Test after each reboot** to ensure modules load automatically
4. **Document your configuration** in `/etc/netplan/` with comments
5. **Keep backups** of working netplan configurations

## Quick Verification Checklist

Use this checklist to verify RDMA is properly configured:

- [ ] `ionic-dkms` package installed (`dpkg -l | grep ionic-dkms`)
- [ ] DKMS modules built (`ls /lib/modules/$(uname -r)/updates/dkms/*ionic*`)
- [ ] Module loading configured (`cat /etc/modules-load.d/ionic-rdma.conf`)
- [ ] Modprobe configuration correct (`cat /etc/modprobe.d/ionic-rdma.conf`)
- [ ] ionic_rdma module loaded (`lsmod | grep ionic_rdma`)
- [ ] RDMA devices exist (`ls /sys/class/infiniband/ | grep ionic`)
- [ ] RDMA links active (`rdma link show | grep ACTIVE`)
- [ ] Network interfaces UP (`ip link show | grep enp.*UP`)
- [ ] IP addresses configured (`ip addr show | grep 172`)
- [ ] Netplan updated (`grep -v np0 /etc/netplan/*.yaml | grep enp`)

## Getting Help

If issues persist:

1. **Collect diagnostic information:**
   ```bash
   # Save this output
   {
     echo "=== System Info ==="
     uname -a
     echo ""
     echo "=== Installed Packages ==="
     dpkg -l | grep -E "ionic|rdma"
     echo ""
     echo "=== Loaded Modules ==="
     lsmod | grep -E "ionic|ib_"
     echo ""
     echo "=== RDMA Devices ==="
     ls -la /sys/class/infiniband/
     echo ""
     echo "=== RDMA Links ==="
     rdma link show
     echo ""
     echo "=== Network Interfaces ==="
     ip link show | grep -A2 "enp"
     echo ""
     echo "=== Kernel Messages ==="
     dmesg | grep -i "ionic\|rdma" | tail -50
   } > ~/rdma-diagnostics.txt
   ```

2. **Check recent logs:**
   ```bash
   journalctl -xe | grep -i "ionic\|rdma" | tail -50
   ```

3. **Review AMD Pensando documentation** for your specific NIC model

4. **Contact AMD Support** with diagnostic information

## Related Documentation

- [install-rdma-devices.sh](./install-rdma-devices.sh) - Main installation script
- [README.md](./README.md) - General RDMA installation guide
- [../net-discovery.sh](../net-discovery.sh) - Network discovery tool


