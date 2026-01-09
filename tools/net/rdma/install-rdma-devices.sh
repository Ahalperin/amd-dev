#!/bin/bash
#
# AMD Pensando NIC RDMA Installation Script
# This script automatically installs and configures RDMA support for Pensando DSC NICs
#
# Usage: sudo ./install-rdma-devices.sh [--skip-netplan] [--dry-run]
#
# Author: AMD Development Team
# Version: 1.0
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;36m'
NC='\033[0m' # No Color

# Script options
DRY_RUN=false
SKIP_NETPLAN=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --skip-netplan)
            SKIP_NETPLAN=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dry-run       Show what would be done without making changes"
            echo "  --skip-netplan  Skip netplan configuration updates"
            echo "  -h, --help      Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_section() {
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"
    echo ""
}

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]] && [[ $DRY_RUN == false ]]; then
        log_error "This script must be run as root (use sudo)"
        exit 1
    fi
}

# Check if Pensando NICs are present
check_pensando_nics() {
    log_section "Checking for Pensando NICs"
    
    local nic_count=$(lspci | grep -c "Pensando Systems DSC Ethernet Controller" || true)
    
    if [[ $nic_count -eq 0 ]]; then
        log_error "No Pensando DSC Ethernet Controllers found on this system"
        exit 1
    fi
    
    log_success "Found $nic_count Pensando DSC NIC(s)"
    return 0
}

# Check if RDMA modules are already loaded
check_rdma_status() {
    log_section "Checking Current RDMA Status"
    
    if lsmod | grep -q "ionic_rdma"; then
        log_success "ionic_rdma module is already loaded"
        
        local ionic_count=$(ls /sys/class/infiniband/ 2>/dev/null | grep -c "ionic_" || true)
        log_info "Found $ionic_count ionic RDMA device(s)"
        
        if [[ $ionic_count -gt 0 ]]; then
            log_warning "RDMA appears to be already configured. Use --force to reinstall."
            echo ""
            rdma link show | grep ionic || true
            return 1
        fi
    else
        log_info "ionic_rdma module not loaded - RDMA support needs to be installed"
    fi
    
    return 0
}

# Check if required packages are available
check_packages() {
    log_section "Checking Package Availability"
    
    # Check if ainic repository is configured
    if [[ ! -d "/opt/amd/ainic/deb-repo" ]]; then
        log_error "AMD AINIC repository not found at /opt/amd/ainic/deb-repo"
        log_error "Please ensure the Pensando driver packages are available"
        exit 1
    fi
    
    log_success "AMD AINIC repository found"
    
    # Check if packages are available
    apt-cache policy ionic-dkms &>/dev/null
    if [[ $? -ne 0 ]]; then
        log_error "ionic-dkms package not available in repositories"
        log_info "Running: apt update"
        apt update
    fi
    
    local ionic_common_available=$(apt-cache policy ionic-common 2>/dev/null | grep -c "Candidate" || true)
    local ionic_dkms_available=$(apt-cache policy ionic-dkms 2>/dev/null | grep -c "Candidate" || true)
    
    if [[ $ionic_common_available -eq 0 ]] || [[ $ionic_dkms_available -eq 0 ]]; then
        log_error "Required packages not available in repositories"
        exit 1
    fi
    
    log_success "Required packages are available"
}

# Install RDMA packages
install_rdma_packages() {
    log_section "Installing RDMA Packages"
    
    # Check if already installed
    local ionic_common_installed=$(dpkg -l | grep -c "^ii.*ionic-common" || true)
    local ionic_dkms_installed=$(dpkg -l | grep -c "^ii.*ionic-dkms" || true)
    
    if [[ $ionic_common_installed -gt 0 ]] && [[ $ionic_dkms_installed -gt 0 ]]; then
        log_info "Packages already installed:"
        dpkg -l | grep "ionic-common\|ionic-dkms"
        return 0
    fi
    
    if [[ $DRY_RUN == true ]]; then
        log_info "[DRY-RUN] Would install: ionic-common ionic-dkms"
        return 0
    fi
    
    log_info "Installing ionic-common and ionic-dkms..."
    
    export DEBIAN_FRONTEND=noninteractive
    apt update -qq
    apt install -y ionic-common ionic-dkms
    
    if [[ $? -eq 0 ]]; then
        log_success "Packages installed successfully"
    else
        log_error "Package installation failed"
        exit 1
    fi
}

# Load RDMA kernel modules
load_rdma_modules() {
    log_section "Loading RDMA Kernel Modules"
    
    if [[ $DRY_RUN == true ]]; then
        log_info "[DRY-RUN] Would unload: ionic"
        log_info "[DRY-RUN] Would load: ionic, ionic_rdma, ib_peer_mem"
        return 0
    fi
    
    # Unload existing ionic module (if loaded)
    if lsmod | grep -q "^ionic "; then
        log_info "Unloading existing ionic module..."
        modprobe -r ionic 2>/dev/null || log_warning "Could not unload ionic (may be in use)"
    fi
    
    # Load new modules
    log_info "Loading ionic module..."
    modprobe ionic
    
    log_info "Loading ionic_rdma module..."
    modprobe ionic_rdma
    
    log_info "Loading ib_peer_mem module..."
    modprobe ib_peer_mem 2>/dev/null || log_warning "ib_peer_mem not available (optional)"
    
    # Verify modules are loaded
    if lsmod | grep -q "ionic_rdma"; then
        log_success "RDMA modules loaded successfully"
        lsmod | grep "ionic\|ib_peer"
    else
        log_error "Failed to load RDMA modules"
        exit 1
    fi
}

# Wait for RDMA devices to be created
wait_for_rdma_devices() {
    log_info "Waiting for RDMA devices to be created..."
    
    local max_wait=10
    local count=0
    
    while [[ $count -lt $max_wait ]]; do
        local ionic_devices=$(ls /sys/class/infiniband/ 2>/dev/null | grep -c "ionic_" || true)
        
        if [[ $ionic_devices -gt 0 ]]; then
            log_success "Found $ionic_devices RDMA device(s)"
            return 0
        fi
        
        sleep 1
        ((count++))
    done
    
    log_warning "RDMA devices not detected after ${max_wait}s"
    return 1
}

# Bring up network interfaces
bring_up_interfaces() {
    log_section "Configuring Network Interfaces"
    
    # Detect Pensando interfaces
    local pensando_interfaces=$(ip link show | grep -oP "enp\d+s0(?!np0)" | sort -u)
    
    if [[ -z "$pensando_interfaces" ]]; then
        log_error "No Pensando network interfaces found"
        return 1
    fi
    
    log_info "Detected interfaces:"
    echo "$pensando_interfaces" | while read iface; do
        echo "  - $iface"
    done
    
    if [[ $DRY_RUN == true ]]; then
        log_info "[DRY-RUN] Would bring up interfaces and configure RDMA"
        return 0
    fi
    
    # Bring up each interface
    echo "$pensando_interfaces" | while read iface; do
        log_info "Bringing up interface: $iface"
        ip link set "$iface" up 2>&1 || log_warning "Could not bring up $iface"
    done
    
    # Wait for links to come up
    sleep 3
}

# Update netplan configuration
update_netplan() {
    log_section "Updating Netplan Configuration"
    
    if [[ $SKIP_NETPLAN == true ]]; then
        log_info "Skipping netplan configuration (--skip-netplan)"
        return 0
    fi
    
    # Find netplan configuration files
    local netplan_files=$(find /etc/netplan -name "*.yaml" 2>/dev/null)
    
    if [[ -z "$netplan_files" ]]; then
        log_warning "No netplan configuration files found"
        return 0
    fi
    
    log_info "Found netplan configuration files:"
    echo "$netplan_files"
    
    # Check if any files need updating
    local needs_update=false
    for file in $netplan_files; do
        if grep -q "enp[0-9]*s0np0" "$file" 2>/dev/null; then
            needs_update=true
            break
        fi
    done
    
    if [[ $needs_update == false ]]; then
        log_info "Netplan configuration already up to date"
        return 0
    fi
    
    if [[ $DRY_RUN == true ]]; then
        log_info "[DRY-RUN] Would update netplan files to use new interface names"
        return 0
    fi
    
    # Backup and update each file
    for file in $netplan_files; do
        if grep -q "enp[0-9]*s0np0" "$file" 2>/dev/null; then
            log_info "Updating: $file"
            
            # Create backup
            cp "$file" "${file}.backup-$(date +%Y%m%d-%H%M%S)"
            
            # Update interface names (remove np0 suffix)
            sed -i 's/enp\([0-9]*\)s0np0/enp\1s0/g' "$file"
            
            # Fix permissions
            chmod 600 "$file"
            
            log_success "Updated: $file (backup created)"
        fi
    done
    
    # Apply netplan configuration
    log_info "Applying netplan configuration..."
    netplan apply 2>&1 | grep -v "WARNING" || true
    
    sleep 3
    log_success "Netplan configuration applied"
}

# Configure persistent module loading
configure_persistent_modules() {
    log_section "Configuring Persistent Module Loading"
    
    if [[ $DRY_RUN == true ]]; then
        log_info "[DRY-RUN] Would configure modules to load at boot"
        return 0
    fi
    
    # Create modules-load.d configuration
    log_info "Creating /etc/modules-load.d/ionic-rdma.conf..."
    cat > /etc/modules-load.d/ionic-rdma.conf << 'EOF'
# AMD Pensando RDMA modules
# Load these modules at boot to enable RDMA support
ionic
ionic_rdma
ib_peer_mem
EOF
    
    # Create modprobe configuration
    log_info "Creating /etc/modprobe.d/ionic-rdma.conf..."
    cat > /etc/modprobe.d/ionic-rdma.conf << 'EOF'
# AMD Pensando RDMA module configuration
# Ensure ionic_rdma loads after ionic
softdep ionic post: ionic_rdma

# Prefer DKMS version over in-kernel driver
install ionic /sbin/modprobe --ignore-install ionic && /sbin/modprobe ionic_rdma || /sbin/modprobe --ignore-install ionic
EOF
    
    # Update initramfs
    log_info "Updating initramfs..."
    update-initramfs -u -k $(uname -r) 2>&1 | grep -v "^W:" || true
    
    log_success "Persistent module loading configured"
    log_info "RDMA modules will load automatically on next boot"
}

# Verify RDMA functionality
verify_rdma() {
    log_section "Verifying RDMA Configuration"
    
    # Check RDMA devices
    local rdma_devices=$(ls /sys/class/infiniband/ 2>/dev/null | grep "ionic_" || true)
    local rdma_count=$(echo "$rdma_devices" | grep -c "ionic_" || true)
    
    if [[ $rdma_count -eq 0 ]]; then
        log_error "No ionic RDMA devices found"
        return 1
    fi
    
    log_success "Found $rdma_count RDMA device(s):"
    echo "$rdma_devices" | while read dev; do
        echo "  - $dev"
    done
    
    echo ""
    log_info "RDMA Link Status:"
    rdma link show | grep "ionic_" || true
    
    echo ""
    log_info "InfiniBand Devices:"
    ibv_devinfo -l 2>/dev/null || log_warning "ibv_devinfo not available"
    
    # Count active links
    local active_links=$(rdma link show | grep -c "state ACTIVE" || true)
    
    if [[ $active_links -gt 0 ]]; then
        log_success "$active_links RDMA link(s) are ACTIVE"
    else
        log_warning "No RDMA links are in ACTIVE state"
        log_info "You may need to configure IP addresses and bring up interfaces"
    fi
    
    return 0
}

# Generate RCCL configuration example
generate_rccl_config() {
    log_section "RCCL Configuration Example"
    
    cat << 'EOF'
To use RDMA with RCCL, add these environment variables:

# Enable RDMA
export NCCL_IB_DISABLE=0

# Specify Pensando network interfaces (adjust based on your configuration)
export NCCL_SOCKET_IFNAME="enp9s0,enp25s0,enp105s0,enp121s0,enp137s0,enp153s0,enp233s0,enp249s0"

# Pensando RDMA optimizations
export NCCL_IB_GID_INDEX=3          # RoCEv2
export NCCL_IB_TC=106               # Traffic class for RDMA
export NCCL_IB_QPS_PER_CONNECTION=1
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=7
export NCCL_IB_ADAPTIVE_ROUTING=1

# Debug output (optional)
export NCCL_DEBUG=INFO

# Test RDMA bandwidth (on two nodes):
# Node 1: ib_send_bw -d ionic_0 -i 1 -F --report_gbits
# Node 2: ib_send_bw -d ionic_0 -i 1 -F --report_gbits <node1_ip>
EOF
}

# Main installation flow
main() {
    log_section "AMD Pensando RDMA Installation"
    
    if [[ $DRY_RUN == true ]]; then
        log_warning "Running in DRY-RUN mode - no changes will be made"
    fi
    
    # Pre-flight checks
    check_root
    check_pensando_nics
    
    # Check current status
    if check_rdma_status; then
        check_packages
        install_rdma_packages
        load_rdma_modules
        wait_for_rdma_devices
        bring_up_interfaces
        update_netplan
    else
        log_info "RDMA appears to be already configured"
    fi
    
    # Configure persistence and verification
    if [[ $DRY_RUN == false ]]; then
        configure_persistent_modules
        verify_rdma
        echo ""
        generate_rccl_config
    fi
    
    echo ""
    log_section "Installation Complete!"
    log_success "AMD Pensando RDMA support is now configured"
    
    if [[ $DRY_RUN == false ]]; then
        log_info "Run './net-discovery.sh' to verify the configuration"
    fi
}

# Run main function
main "$@"


