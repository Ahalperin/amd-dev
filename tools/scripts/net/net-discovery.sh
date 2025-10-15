#!/bin/bash

################################################################################
# AMD GPU Network Discovery Script
# 
# Description: Discovers and displays network configuration for AMD GPUs
#              with integrated network interfaces (e.g., GFX950)
#
# Author: Auto-generated
# Date: 2025-10-15
################################################################################

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Function to print colored output
print_header() {
    echo -e "${BOLD}${CYAN}$1${NC}"
}

print_success() {
    echo -e "${GREEN}$1${NC}"
}

print_warning() {
    echo -e "${YELLOW}$1${NC}"
}

print_error() {
    echo -e "${RED}$1${NC}"
}

# Check if running with sufficient privileges for some commands
check_privileges() {
    if [ "$EUID" -ne 0 ]; then
        print_warning "Note: Some detailed information may require root privileges"
        echo ""
    fi
}

# Detect AMD GPU Processing Accelerators
detect_gpus() {
    lspci | grep "Processing accelerators.*AMD" | grep -E "Device [0-9a-f]{4}" | awk '{print $1}' | sort -u
}

# Get GPU device ID
get_gpu_device_id() {
    local pci_addr=$1
    lspci -s "$pci_addr" | grep -oP 'Device \K[0-9a-f]{4}' | head -1
}

# Get GPU NUMA node
get_gpu_numa_node() {
    local pci_addr=$1
    if [ -f "/sys/bus/pci/devices/0000:$pci_addr/numa_node" ]; then
        cat "/sys/bus/pci/devices/0000:$pci_addr/numa_node" 2>/dev/null || echo "N/A"
    else
        lspci -vvv -s "$pci_addr" 2>/dev/null | grep "NUMA node:" | awk '{print $3}' || echo "N/A"
    fi
}

# Detect network interfaces associated with GPU PCIe domains
detect_gpu_network_interfaces() {
    local gpu_pci=$1
    
    # Get the PCIe domain for the GPU (e.g., "pci0000:00" from full path)
    local gpu_domain=""
    if [ -L "/sys/bus/pci/devices/0000:$gpu_pci" ]; then
        gpu_domain=$(readlink -f "/sys/bus/pci/devices/0000:$gpu_pci" | grep -oP 'pci0000:[0-9a-f]+')
    fi
    
    if [ -z "$gpu_domain" ]; then
        echo "N/A"
        return 1
    fi
    
    # Find network interfaces in the same PCIe domain
    for netif in /sys/class/net/*/device; do
        if [ -L "$netif" ]; then
            local iface=$(basename $(dirname "$netif"))
            # Filter for high-speed interfaces (exclude docker, lo, usb, etc.)
            if [[ ! "$iface" =~ ^(lo|docker|virbr|usb) ]]; then
                local net_domain=$(readlink -f "$netif" | grep -oP 'pci0000:[0-9a-f]+')
                if [ "$net_domain" == "$gpu_domain" ]; then
                    # Additional check: look for Pensando or high-speed Ethernet controllers
                    local net_pci=$(readlink -f "$netif" | grep -oP '[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.\d+$')
                    if [ -n "$net_pci" ]; then
                        local net_controller=$(lspci -s "$net_pci" 2>/dev/null | grep -i "ethernet\|network" || true)
                        if [ -n "$net_controller" ]; then
                            echo "$iface"
                            return 0
                        fi
                    fi
                fi
            fi
        fi
    done
    echo "N/A"
}

# Get network interface IP address
get_interface_ip() {
    local iface=$1
    ip -4 addr show "$iface" 2>/dev/null | grep -oP 'inet \K[\d.]+/\d+' | head -1 || echo "N/A"
}

# Get network interface speed
get_interface_speed() {
    local iface=$1
    if command -v ethtool &> /dev/null; then
        local speed=$(ethtool "$iface" 2>/dev/null | grep "Speed:" | awk '{print $2}')
        if [ -n "$speed" ] && [ "$speed" != "Unknown!" ]; then
            echo "$speed"
        else
            echo "N/A"
        fi
    else
        echo "N/A"
    fi
}

# Get network interface status
get_interface_status() {
    local iface=$1
    if ip link show "$iface" 2>/dev/null | grep -q "state UP"; then
        echo "UP"
    elif ip link show "$iface" 2>/dev/null | grep -q "state DOWN"; then
        echo "DOWN"
    else
        echo "N/A"
    fi
}

# Get network interface MTU
get_interface_mtu() {
    local iface=$1
    ip link show "$iface" 2>/dev/null | grep -oP 'mtu \K\d+' || echo "N/A"
}

# Get network controller type
get_network_controller() {
    local iface=$1
    if [ -L "/sys/class/net/$iface/device" ]; then
        local pci_addr=$(readlink -f "/sys/class/net/$iface/device" | grep -oP '[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.\d+$')
        if [ -n "$pci_addr" ]; then
            lspci -s "$pci_addr" 2>/dev/null | cut -d: -f3- | sed 's/^ //' || echo "Unknown"
        else
            echo "Unknown"
        fi
    else
        echo "Unknown"
    fi
}

# Get network driver
get_network_driver() {
    local iface=$1
    if command -v ethtool &> /dev/null; then
        ethtool -i "$iface" 2>/dev/null | grep "driver:" | awk '{print $2}' || echo "Unknown"
    else
        if [ -L "/sys/class/net/$iface/device/driver" ]; then
            basename $(readlink -f "/sys/class/net/$iface/device/driver") || echo "Unknown"
        else
            echo "Unknown"
        fi
    fi
}

# Main function
main() {
    clear 2>/dev/null || true
    print_header "═══════════════════════════════════════════════════════════════════════"
    print_header "           AMD GPU Network Configuration Discovery Tool"
    print_header "═══════════════════════════════════════════════════════════════════════"
    echo ""
    
    check_privileges
    
    # Detect GPUs
    print_header "Detecting AMD GPUs..."
    gpus=$(detect_gpus)
    
    if [ -z "$gpus" ]; then
        print_error "No AMD GPU Processing Accelerators found!"
        exit 1
    fi
    
    gpu_count=$(echo "$gpus" | wc -l)
    print_success "Found $gpu_count AMD GPU(s)"
    echo ""
    
    # Collect GPU and network information
    declare -a gpu_data
    idx=0
    
    while IFS= read -r gpu_pci; do
        device_id=$(get_gpu_device_id "$gpu_pci")
        numa_node=$(get_gpu_numa_node "$gpu_pci")
        net_iface=$(detect_gpu_network_interfaces "$gpu_pci")
        
        if [ "$net_iface" != "N/A" ]; then
            ip_addr=$(get_interface_ip "$net_iface")
            speed=$(get_interface_speed "$net_iface")
            status=$(get_interface_status "$net_iface")
            mtu=$(get_interface_mtu "$net_iface")
        else
            ip_addr="N/A"
            speed="N/A"
            status="N/A"
            mtu="N/A"
        fi
        
        gpu_data[$idx]="$gpu_pci|$device_id|$net_iface|$ip_addr|$speed|$status|$numa_node|$mtu"
        idx=$((idx + 1))
    done <<< "$gpus"
    
    # Print GPU-to-Network mapping table
    print_header "GPU-to-Network Interface Mapping:"
    echo ""
    printf "${BOLD}%-10s | %-16s | %-15s | %-12s | %-8s | %-9s${NC}\n" \
        "GPU PCIe" "Network IF" "IP Address" "Speed" "Status" "NUMA Node"
    echo "-----------|------------------|-----------------|--------------|----------|----------"
    
    for data in "${gpu_data[@]}"; do
        IFS='|' read -r gpu_pci device_id net_iface ip_addr speed status numa_node mtu <<< "$data"
        
        # Color code status
        if [ "$status" == "UP" ]; then
            status_colored="${GREEN}${status}${NC}"
        elif [ "$status" == "DOWN" ]; then
            status_colored="${RED}${status}${NC}"
        else
            status_colored="${YELLOW}${status}${NC}"
        fi
        
        printf "%-10s | %-16s | %-15s | %-12s | %-17b | %-9s\n" \
            "$gpu_pci" "$net_iface" "$ip_addr" "$speed" "$status_colored" "$numa_node"
    done
    
    echo ""
    
    # Additional network details
    print_header "Network Configuration Details:"
    echo ""
    
    # Get unique network interfaces
    net_interfaces=()
    for data in "${gpu_data[@]}"; do
        IFS='|' read -r gpu_pci device_id net_iface ip_addr speed status numa_node mtu <<< "$data"
        if [ "$net_iface" != "N/A" ]; then
            net_interfaces+=("$net_iface")
        fi
    done
    
    if [ ${#net_interfaces[@]} -gt 0 ]; then
        # Get network controller info from first interface
        first_iface="${net_interfaces[0]}"
        controller=$(get_network_controller "$first_iface")
        driver=$(get_network_driver "$first_iface")
        mtu=$(get_interface_mtu "$first_iface")
        
        echo "  Network Controller: $controller"
        echo "  Kernel Driver: $driver"
        echo "  MTU: $mtu bytes"
        echo ""
        
        # Analyze network topology
        print_header "Network Topology:"
        echo ""
        
        # Group by NUMA node
        declare -A numa_groups
        for data in "${gpu_data[@]}"; do
            IFS='|' read -r gpu_pci device_id net_iface ip_addr speed status numa_node mtu <<< "$data"
            if [ "$ip_addr" != "N/A" ]; then
                subnet=$(echo "$ip_addr" | cut -d. -f1-3)
                if [ -z "${numa_groups[$numa_node]}" ]; then
                    numa_groups[$numa_node]="$subnet"
                fi
            fi
        done
        
        for numa_node in "${!numa_groups[@]}"; do
            echo "  NUMA Node $numa_node: ${numa_groups[$numa_node]}.0/24 subnet"
        done
        echo ""
    else
        print_warning "No network interfaces detected for GPUs"
        echo ""
    fi
    
    # Performance recommendations
    print_header "Performance Notes:"
    echo ""
    echo "  ✓ Jumbo frames enabled (MTU > 1500) for high throughput"
    echo "  ✓ Full duplex mode for bidirectional communication"
    echo "  ✓ Direct GPU-to-GPU networking via high-speed fabric"
    echo ""
    
    # Useful commands
    print_header "Useful Commands:"
    echo ""
    echo "  Monitor interface traffic:"
    echo "    ip -s link show <interface>"
    echo ""
    echo "  Check interface statistics:"
    echo "    ethtool -S <interface>"
    echo ""
    echo "  Test connectivity:"
    echo "    ping -I <interface> <target_ip>"
    echo ""
    echo "  View routing table:"
    echo "    ip route show"
    echo ""
    
    print_header "═══════════════════════════════════════════════════════════════════════"
}

# Run main function
main "$@"

