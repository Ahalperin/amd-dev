#!/bin/bash

set -e  # Exit immediately if a command fails

# Default values
DEFAULT_USER=$USER
DEFAULT_VERSION="develop"
# DEFAULT_VERSION="2025-06-J13A-1"
INCLUDE_DN_TUNER=false

usage() {
    echo "Usage: $0 [-u user] [-v version] [-d docker_image] [--dn-tuner]"
    echo ""
    echo "Options:"
    echo "  -u USER          Username for paths (default: $DEFAULT_USER)"
    echo "  -v VERSION       RCCL version/tag name (default: $DEFAULT_VERSION)"
    echo "  -d DOCKER_IMAGE  Docker image to extract files from (optional)"
    echo "  --dn-tuner       Include dn-tuner library and config files"
    echo "  -h               Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 -u myuser -v release-1.0"
    echo "  $0 -u myuser -v release-1.0 -d rccl-build:7.1.1-develop-main"
    echo "  $0 -u myuser -v release-1.0 --dn-tuner"
    exit 1
}

# Parse command-line arguments (handle long options first, rebuild args without them)
ARGS=()
for arg in "$@"; do
    case $arg in
        --dn-tuner)
            INCLUDE_DN_TUNER=true
            ;;
        *)
            ARGS+=("$arg")
            ;;
    esac
done

# Reset positional parameters to filtered args
set -- "${ARGS[@]}"

# Parse short options
while getopts "u:v:d:h" opt; do
    case $opt in
        u) MY_USER="$OPTARG" ;;
        v) RCCL_VERSION="$OPTARG" ;;
        d) DOCKER_IMAGE="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done

# Use defaults if not provided
MY_USER="${MY_USER:-$DEFAULT_USER}"
RCCL_VERSION="${RCCL_VERSION:-$DEFAULT_VERSION}"


# package paths
RCCL_PACKAGES_DIR="/home/${MY_USER}/rccl-packages"
TMP_PACKAGE_DIR="${RCCL_PACKAGES_DIR}/tmp"
VERSION_PACKAGE_DIR="${RCCL_PACKAGES_DIR}/${RCCL_VERSION}"

echo "Creating RCCL package with:"
echo "  User: $MY_USER"
echo "  Version: $RCCL_VERSION"
if [ -n "$DOCKER_IMAGE" ]; then
    echo "  Docker Image: $DOCKER_IMAGE"
fi
if [ "$INCLUDE_DN_TUNER" = true ]; then
    echo "  DN-Tuner: included"
fi
echo "  Output: $VERSION_PACKAGE_DIR/rccl-package.tar"
echo ""

# make sure the temporary package directory exists
mkdir -p "$TMP_PACKAGE_DIR"

# Function to copy dn-tuner files
copy_dn_tuner() {
    DN_TUNER_DIR="/home/${MY_USER}/amd-dev/dn/dn-tuner"
    
    echo "Including dn-tuner library and config files..."
    
    # copy dn-tuner library
    cp "$DN_TUNER_DIR/librccl-tunerv4-dn.so" "$TMP_PACKAGE_DIR/."
    
    # copy dn-tuner config files
    mkdir -p "$TMP_PACKAGE_DIR/dn-tuner-conf"
    cp -r "$DN_TUNER_DIR/conf/"* "$TMP_PACKAGE_DIR/dn-tuner-conf/."
}

# Function to copy files from host
copy_from_host() {
    # build paths
    RCCL_BUILD_DIR="/home/${MY_USER}/amd-dev/dn/rccl/build/release"
    AMD_ANP_BUILD_DIR="/home/${MY_USER}/amd-dev/dn/amd-anp/build"
    RCCL_TESTS_BUILD_DIR="/home/${MY_USER}/amd-dev/dn/rccl-tests/build"

    # copy rccl bins
    cp "$RCCL_BUILD_DIR/librccl.so.1.0" "$TMP_PACKAGE_DIR/."
    ln -sf librccl.so.1.0 "$TMP_PACKAGE_DIR/librccl.so.1"
    ln -sf librccl.so.1.0 "$TMP_PACKAGE_DIR/librccl.so"

    # copy amd-anp bins
    cp "$AMD_ANP_BUILD_DIR/librccl-anp.so" "$TMP_PACKAGE_DIR/librccl-net.so"

    # copy rccl-tests executables using find from build directory
    find "$RCCL_TESTS_BUILD_DIR" -type f -executable | xargs -I {} cp {} "$TMP_PACKAGE_DIR/."
}

# Function to copy files from docker container
copy_from_docker() {
    # build paths
    RCCL_BUILD_DIR="/workspace/rccl/build/release"
    AMD_ANP_BUILD_DIR="/workspace/amd-anp/build"
    RCCL_TESTS_BUILD_DIR="/workspace/rccl-tests/build"

    local container_id="$1"
    
    # copy rccl bins
    docker cp "$container_id:$RCCL_BUILD_DIR/librccl.so.1.0" "$TMP_PACKAGE_DIR/."
    ln -sf librccl.so.1.0 "$TMP_PACKAGE_DIR/librccl.so.1"
    ln -sf librccl.so.1.0 "$TMP_PACKAGE_DIR/librccl.so"

    # copy amd-anp bins
    docker cp "$container_id:$AMD_ANP_BUILD_DIR/librccl-anp.so" "$TMP_PACKAGE_DIR/."
    ln -sf librccl-anp.so "$TMP_PACKAGE_DIR/librccl-net.so"

    # copy rccl-tests executables using find from build directory
    # First, get the list of executables from the container
    docker exec "$container_id" find "$RCCL_TESTS_BUILD_DIR" -type f -executable -name "*_perf" | while read -r file; do
        docker cp "$container_id:$file" "$TMP_PACKAGE_DIR/."
    done
}

# Copy files from docker image or host
if [ -n "$DOCKER_IMAGE" ]; then
    # Create and start a temporary container from the docker image
    # Use 'sleep infinity' to keep the container running
    echo "Creating temporary container from image: $DOCKER_IMAGE"
    CONTAINER_ID=$(docker run -d "$DOCKER_IMAGE" sleep infinity)
    
    # Ensure container is removed even if script fails
    trap "docker rm -f $CONTAINER_ID 2>/dev/null || true" EXIT
    
    echo "Copying files from container..."
    copy_from_docker "$CONTAINER_ID"
    
    # Stop and remove the temporary container
    docker stop "$CONTAINER_ID" >/dev/null
    docker rm -f "$CONTAINER_ID" >/dev/null
    trap - EXIT
else
    # Copy from host filesystem
    copy_from_host
fi

# Copy dn-tuner files if requested
if [ "$INCLUDE_DN_TUNER" = true ]; then
    copy_dn_tuner
fi

# make sure the version package directory exists
mkdir -p "$VERSION_PACKAGE_DIR"

# tar the package and save it in the version directory
tar -cvf "$VERSION_PACKAGE_DIR/rccl-package.tar" -C "$TMP_PACKAGE_DIR" .

# remove the temporary package directory
rm -rf "$TMP_PACKAGE_DIR"

echo ""
echo "Package created successfully: $VERSION_PACKAGE_DIR/rccl-package.tar"
