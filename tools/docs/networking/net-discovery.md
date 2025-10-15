# AMD GPU Network Discovery

## Overview
AMD GFX950 GPUs have integrated 400 Gb/s network interfaces via Pensando DSC (Distributed Services Card) Ethernet Controllers.

## Quick Start

Run the automated discovery script:
```bash
/home/dn/amd-dev/tools/scripts/net/net-discovery.sh
```

## Script Location
- **Path**: `/home/dn/amd-dev/tools/scripts/net/net-discovery.sh`
- **Purpose**: Automatically discovers and displays GPU-to-network interface mappings
- **Features**:
  - Detects all AMD GPU Processing Accelerators
  - Maps each GPU to its dedicated network interface
  - Shows IP addresses, link speeds, and status
  - Displays NUMA node topology
  - Provides network configuration details

## Expected Output
The script displays:
- GPU PCIe addresses mapped to network interfaces
- 400 Gb/s link speeds per GPU
- IP addresses and network status
- NUMA node information
- Network controller details (Pensando DSC with ionic driver)
- MTU settings (9144 bytes - jumbo frames)
- Useful commands for monitoring and troubleshooting

## Network Configuration
- **Network Controller**: Pensando DSC Ethernet Controller
- **Driver**: ionic
- **Speed**: 400 Gb/s per GPU
- **MTU**: 9144 bytes (jumbo frames for high throughput)
- **Topology**: Separate subnets per NUMA node
  - NUMA Node 0: 172.65.1.0/24
  - NUMA Node 1: 172.33.1.0/24

## Manual Commands

### List GPU PCIe Addresses
```bash
lspci | grep "Processing accelerators.*AMD" | awk '{print $1}'
```

### Check Network Interface Status
```bash
ip link show | grep -E "enp(9|25|105|121|137|153|233|249)"
```

### Verify Link Speed
```bash
ethtool enp9s0 | grep Speed
```

### Monitor Network Traffic
```bash
ip -s link show enp9s0
```

### Test GPU-to-GPU Connectivity
```bash
ping -I enp9s0 <target_gpu_ip>
```
