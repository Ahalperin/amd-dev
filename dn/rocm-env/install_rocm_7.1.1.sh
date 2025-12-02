#!/bin/bash
#
# Install ROCm 7.1.1 side-by-side with existing ROCm 7.0.1
# For Ubuntu 22.04 (Jammy)
#
# This script will NOT remove your existing ROCm 7.0.1 installation
#

set -e  # Exit on error

echo "=================================="
echo "ROCm 7.1.1 Side-by-Side Installation"
echo "=================================="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "This script must be run with sudo"
    echo "Usage: sudo bash $0"
    exit 1
fi

# Verify Ubuntu version
UBUNTU_VERSION=$(lsb_release -rs)
if [ "$UBUNTU_VERSION" != "22.04" ]; then
    echo "Warning: This script is designed for Ubuntu 22.04, but you have $UBUNTU_VERSION"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "Step 1: Adding ROCm repository..."
# Add ROCm repository key
mkdir -p --mode=0755 /etc/apt/keyrings
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
    gpg --dearmor | tee /etc/apt/keyrings/rocm.gpg > /dev/null

# Add ROCm repository for Ubuntu 22.04 (jammy)
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/7.1.1 jammy main" | \
    tee /etc/apt/sources.list.d/rocm.list

# Add priority to prefer ROCm packages
echo -e 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600' | \
    tee /etc/apt/preferences.d/rocm-pin-600

echo ""
echo "Step 2: Updating package lists..."
apt-get update

echo ""
echo "Step 3: Installing ROCm 7.1.1..."
echo "This will install to /opt/rocm-7.1.1 (7.0.1 will remain in /opt/rocm-7.0.1)"
echo ""

# Install ROCm 7.1.1 with version-specific package
# Using the versioned package ensures side-by-side installation
apt-get install -y rocm-hip-sdk7.1.1 rocm-libs7.1.1 rocm-llvm7.1.1

echo ""
echo "Step 4: Installing additional ROCm 7.1.1 components..."
# Install additional components with specific version
apt-get install -y \
    rocm-smi-lib7.1.1 \
    hip-runtime-amd7.1.1 \
    rocm-device-libs7.1.1 \
    rocm-core7.1.1

echo ""
echo "Step 5: Installing RCCL 7.1.1..."
apt-get install -y rccl7.1.1

echo ""
echo "=================================="
echo "Installation Complete!"
echo "=================================="
echo ""
echo "Installed ROCm versions:"
ls -d /opt/rocm* | grep -v "^/opt/rocm$"
echo ""
echo "Current /opt/rocm symlink points to:"
ls -la /opt/rocm | grep -- "->"
echo ""
echo "=================================="
echo "How to Use ROCm 7.1.1:"
echo "=================================="
echo ""
echo "Option 1: Set environment variables (temporary, per-session)"
echo "  export ROCM_PATH=/opt/rocm-7.1.1"
echo "  export PATH=\$ROCM_PATH/bin:\$PATH"
echo "  export LD_LIBRARY_PATH=\$ROCM_PATH/lib:\$LD_LIBRARY_PATH"
echo ""
echo "Option 2: Change system default (affects all users)"
echo "  sudo update-alternatives --config rocm"
echo ""
echo "Option 3: Create a wrapper script (recommended for development)"
echo "  See: /home/amir/amd-dev/dn/use-rocm-7.1.sh"
echo ""
echo "=================================="
echo "Verify installation:"
echo "=================================="
echo "  /opt/rocm-7.1.1/bin/rocminfo"
echo "  /opt/rocm-7.1.1/bin/rocm-smi"
echo ""

