#!/bin/bash

set -e  # Exit immediately if a command fails

# Default values
DEFAULT_USER="dn"
DEFAULT_VERSION="develop"

usage() {
    echo "Usage: $0 [-u user] [-v version]"
    echo ""
    echo "Options:"
    echo "  -u USER      Username for paths (default: $DEFAULT_USER)"
    echo "  -v VERSION   RCCL version/tag name (default: $DEFAULT_VERSION)"
    echo "  -h           Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 -u myuser -v release-1.0"
    exit 1
}

# Parse command-line arguments
while getopts "u:v:h" opt; do
    case $opt in
        u) MY_USER="$OPTARG" ;;
        v) RCCL_VERSION="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done

# Use defaults if not provided
MY_USER="${MY_USER:-$DEFAULT_USER}"
RCCL_VERSION="${RCCL_VERSION:-$DEFAULT_VERSION}"

# Paths
RCCL_BUILD_DIR="/home/${MY_USER}/amd-dev/dn/rccl/build/release"
AMD_ANP_BUILD_DIR="/home/${MY_USER}/amd-dev/dn/amd-anp/build"
RCCL_TESTS_BUILD_DIR="/home/${MY_USER}/amd-dev/dn/rccl-tests/build"
RCCL_PACKAGES_DIR="/home/${MY_USER}/rccl-packages"
TMP_PACKAGE_DIR="${RCCL_PACKAGES_DIR}/tmp"
VERSION_PACKAGE_DIR="${RCCL_PACKAGES_DIR}/${RCCL_VERSION}"

echo "Creating RCCL package with:"
echo "  User: $MY_USER"
echo "  Version: $RCCL_VERSION"
echo "  Output: $VERSION_PACKAGE_DIR/rccl-package.tar"
echo ""

# make sure the temporary package directory exists
mkdir -p "$TMP_PACKAGE_DIR"

# copy rccl bins
cp "$RCCL_BUILD_DIR/librccl.so.1.0" "$TMP_PACKAGE_DIR/."
ln -s librccl.so.1.0 "$TMP_PACKAGE_DIR/librccl.so.1"
ln -s librccl.so.1.0 "$TMP_PACKAGE_DIR/librccl.so"

# copy amd-anp bins
cp "$AMD_ANP_BUILD_DIR/librccl-net.so" "$TMP_PACKAGE_DIR/."

# copy rccl-tests executables using find from build directory
find "$RCCL_TESTS_BUILD_DIR" -type f -executable | xargs -I {} cp {} "$TMP_PACKAGE_DIR/."

# make sure the version package directory exists
mkdir -p "$VERSION_PACKAGE_DIR"

# tar the package and save it in the version directory
tar -cvf "$VERSION_PACKAGE_DIR/rccl-package.tar" -C "$TMP_PACKAGE_DIR" .

# remove the temporary package directory
rm -rf "$TMP_PACKAGE_DIR"

echo ""
echo "Package created successfully: $VERSION_PACKAGE_DIR/rccl-package.tar"
