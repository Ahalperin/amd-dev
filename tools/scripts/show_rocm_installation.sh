#!/bin/bash

################################################################################
# ROCm Installation Information Script
# Description: Displays comprehensive ROCm installation details including
#              version, packages, GPU info, and firmware versions
################################################################################

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Function to print section headers
print_header() {
    echo -e "\n${BOLD}${BLUE}===================================================================${NC}"
    echo -e "${BOLD}${CYAN}$1${NC}"
    echo -e "${BOLD}${BLUE}===================================================================${NC}\n"
}

# Function to print subsection headers
print_subheader() {
    echo -e "${BOLD}${YELLOW}--- $1 ---${NC}"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to print status
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $2"
    else
        echo -e "${RED}✗${NC} $2"
    fi
}

################################################################################
# Main Script
################################################################################

echo -e "${BOLD}${GREEN}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════════╗
║         ROCm Installation Information Display Tool           ║
╚═══════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# System Information
print_header "System Information"
echo -e "${BOLD}Hostname:${NC} $(hostname)"
echo -e "${BOLD}Kernel:${NC} $(uname -r)"
echo -e "${BOLD}OS:${NC} $(cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2)"
echo -e "${BOLD}Date:${NC} $(date)"

# ROCm Version Information
print_header "ROCm Version Information"

# Check ROCm installation directory
if [ -d "/opt/rocm" ]; then
    print_status 0 "ROCm installation found at /opt/rocm"
    
    if [ -f "/opt/rocm/.info/version" ]; then
        ROCM_VERSION=$(cat /opt/rocm/.info/version)
        echo -e "${BOLD}ROCm Version:${NC} ${GREEN}${ROCM_VERSION}${NC}"
    fi
    
    if [ -L "/opt/rocm" ]; then
        ROCM_LINK=$(readlink -f /opt/rocm)
        echo -e "${BOLD}ROCm Symlink Points To:${NC} ${ROCM_LINK}"
    fi
else
    print_status 1 "ROCm installation not found at /opt/rocm"
fi

# ROCm SMI Version
if command_exists rocm-smi; then
    print_status 0 "rocm-smi command available"
    rocm-smi --version 2>/dev/null | grep -E "version|VERSION" || echo "  $(rocm-smi --version 2>&1 | head -5)"
else
    print_status 1 "rocm-smi command not found"
fi

# ROCminfo
if command_exists rocminfo; then
    print_status 0 "rocminfo command available"
    ROCMINFO_VERSION=$(rocminfo --version 2>/dev/null | head -1)
    [ -n "$ROCMINFO_VERSION" ] && echo "  $ROCMINFO_VERSION"
else
    print_status 1 "rocminfo command not found"
fi

# HIP Version
if command_exists hipcc; then
    print_status 0 "HIP compiler (hipcc) available"
    HIP_VERSION=$(hipcc --version 2>/dev/null | head -1)
    [ -n "$HIP_VERSION" ] && echo "  $HIP_VERSION"
else
    print_status 1 "hipcc command not found"
fi

# ROCm Installed Packages
print_header "ROCm Installed Packages"

if command_exists dpkg; then
    print_subheader "Core ROCm Packages"
    dpkg -l | grep -E "^ii.*rocm|^ii.*hip|^ii.*hsa|^ii.*rccl" | awk '{printf "  %-35s %s\n", $2, $3}' | head -30
    
    TOTAL_ROCM_PACKAGES=$(dpkg -l | grep -E "^ii.*rocm|^ii.*hip|^ii.*hsa|^ii.*rccl|^ii.*roc" | wc -l)
    echo -e "\n${BOLD}Total ROCm-related packages installed:${NC} ${GREEN}${TOTAL_ROCM_PACKAGES}${NC}"
elif command_exists rpm; then
    print_subheader "Core ROCm Packages (RPM)"
    rpm -qa | grep -E "rocm|hip|hsa|rccl" | head -30
else
    print_status 1 "Package manager not found (dpkg/rpm)"
fi

# GPU Information
print_header "AMD GPU Information"

if command_exists rocm-smi; then
    print_subheader "GPU Detection"
    GPU_COUNT=$(rocm-smi --showid 2>/dev/null | grep -c "GPU\[" || echo "0")
    echo -e "${BOLD}Number of GPUs Detected:${NC} ${GREEN}${GPU_COUNT}${NC}"
    
    if [ "$GPU_COUNT" -gt 0 ]; then
        echo ""
        print_subheader "GPU Product Information"
        rocm-smi --showproductname 2>/dev/null || echo "  Unable to retrieve GPU product info"
    fi
else
    print_status 1 "rocm-smi not available for GPU detection"
fi

# List GPUs using lspci
if command_exists lspci; then
    echo ""
    print_subheader "PCI GPU Devices"
    lspci | grep -i "vga\|3d\|display" | grep -i "amd\|ati" || echo "  No AMD GPUs found via lspci"
fi

# GPU Firmware Information
print_header "GPU Firmware Information"

if command_exists rocm-smi; then
    print_subheader "VBIOS Versions"
    rocm-smi --showvbios 2>/dev/null | grep -E "GPU\[|VBIOS" || echo "  Unable to retrieve VBIOS info"
    
    echo ""
    print_subheader "Firmware Versions (First GPU)"
    rocm-smi --showfwinfo 2>/dev/null | grep "GPU\[0\]" | head -15 || echo "  Unable to retrieve firmware info"
else
    print_status 1 "rocm-smi not available for firmware information"
fi

# Kernel Driver Information
print_header "AMD GPU Kernel Driver Information"

if command_exists modinfo; then
    if modinfo amdgpu >/dev/null 2>&1; then
        print_status 0 "amdgpu kernel module loaded"
        AMDGPU_VERSION=$(modinfo amdgpu 2>/dev/null | grep "^version:" | awk '{print $2}')
        echo -e "${BOLD}AMDGPU Driver Version:${NC} ${GREEN}${AMDGPU_VERSION}${NC}"
        
        echo ""
        print_subheader "Firmware Files"
        modinfo amdgpu 2>/dev/null | grep "^firmware:" | head -20 | sed 's/^firmware:/  /'
        TOTAL_FIRMWARE=$(modinfo amdgpu 2>/dev/null | grep -c "^firmware:")
        echo -e "  ${BOLD}...and ${TOTAL_FIRMWARE} total firmware files${NC}"
    else
        print_status 1 "amdgpu kernel module not found"
    fi
else
    print_status 1 "modinfo command not available"
fi

# Check if amdgpu is loaded
if lsmod | grep -q amdgpu; then
    print_status 0 "amdgpu module currently loaded"
else
    print_status 1 "amdgpu module not loaded"
fi

# ROCm Libraries
print_header "ROCm Library Paths"

if [ -d "/opt/rocm/lib" ]; then
    print_status 0 "ROCm lib directory exists"
    LIB_COUNT=$(find /opt/rocm/lib -maxdepth 1 -name "*.so*" 2>/dev/null | wc -l)
    echo -e "  ${BOLD}Shared libraries found:${NC} ${LIB_COUNT}"
fi

if [ -d "/opt/rocm/lib64" ]; then
    print_status 0 "ROCm lib64 directory exists"
    LIB64_COUNT=$(find /opt/rocm/lib64 -maxdepth 1 -name "*.so*" 2>/dev/null | wc -l)
    echo -e "  ${BOLD}Shared libraries found:${NC} ${LIB64_COUNT}"
fi

# Environment Variables
print_header "ROCm Environment Variables"

ROCM_ENV_VARS=$(env | grep -i rocm | sort)
if [ -n "$ROCM_ENV_VARS" ]; then
    echo "$ROCM_ENV_VARS" | while IFS= read -r line; do
        VAR_NAME=$(echo "$line" | cut -d'=' -f1)
        VAR_VALUE=$(echo "$line" | cut -d'=' -f2-)
        echo -e "  ${BOLD}${VAR_NAME}${NC}=${VAR_VALUE}"
    done
else
    echo "  No ROCm-specific environment variables set"
fi

HIP_ENV_VARS=$(env | grep -i hip | sort)
if [ -n "$HIP_ENV_VARS" ]; then
    echo ""
    print_subheader "HIP Environment Variables"
    echo "$HIP_ENV_VARS" | while IFS= read -r line; do
        VAR_NAME=$(echo "$line" | cut -d'=' -f1)
        VAR_VALUE=$(echo "$line" | cut -d'=' -f2-)
        echo -e "  ${BOLD}${VAR_NAME}${NC}=${VAR_VALUE}"
    done
fi

# ROCm Path Check
print_header "ROCm Path Configuration"

if echo "$PATH" | grep -q "rocm"; then
    print_status 0 "ROCm found in PATH"
    echo "$PATH" | tr ':' '\n' | grep rocm | while read -r path; do
        echo "  ${path}"
    done
else
    print_status 1 "ROCm not found in PATH"
fi

if echo "$LD_LIBRARY_PATH" | grep -q "rocm"; then
    print_status 0 "ROCm found in LD_LIBRARY_PATH"
    echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep rocm | while read -r path; do
        echo "  ${path}"
    done
else
    echo -e "${YELLOW}⚠${NC} ROCm not found in LD_LIBRARY_PATH (may be normal)"
fi

# ROCm Tools Check
print_header "ROCm Development Tools"

TOOLS=("hipcc" "rocm-smi" "rocminfo" "rocprof" "rccl-prim-test" "rocm-clang" "amdclang" "amdclang++")
for tool in "${TOOLS[@]}"; do
    if command_exists "$tool"; then
        TOOL_PATH=$(which "$tool" 2>/dev/null)
        print_status 0 "$tool available at: $TOOL_PATH"
    else
        print_status 1 "$tool not found"
    fi
done

# Summary
print_header "Installation Summary"

echo -e "${BOLD}ROCm Installation:${NC}"
if [ -n "$ROCM_VERSION" ]; then
    echo -e "  Version: ${GREEN}${ROCM_VERSION}${NC}"
else
    echo -e "  Version: ${RED}Not detected${NC}"
fi

if [ "$GPU_COUNT" -gt 0 ]; then
    echo -e "  GPUs Detected: ${GREEN}${GPU_COUNT}${NC}"
else
    echo -e "  GPUs Detected: ${RED}0${NC}"
fi

if [ -n "$AMDGPU_VERSION" ]; then
    echo -e "  Kernel Driver: ${GREEN}${AMDGPU_VERSION}${NC}"
else
    echo -e "  Kernel Driver: ${RED}Not detected${NC}"
fi

echo ""
echo -e "${BOLD}${GREEN}Script completed successfully!${NC}"
echo -e "${BOLD}${BLUE}===================================================================${NC}\n"


