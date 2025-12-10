#!/bin/bash

# Usage function
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Deploy RCCL binaries to remote cluster nodes."
    echo ""
    echo "Options:"
    echo "  -u, --user USER          Remote username (default: dn)"
    echo "  -v, --version VERSION    Package version (default: develop)"
    echo "  -h, --help               Show this help message"
    echo ""
    echo "Environment variables (used if options not provided):"
    echo "  REMOTE_USER, PACKAGE_VERSION"
    exit 0
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -u|--user)
            ARG_REMOTE_USER="$2"
            shift 2
            ;;
        -v|--version)
            ARG_PACKAGE_VERSION="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Configurable variables (CLI args > environment variables > defaults)
REMOTE_USER="${ARG_REMOTE_USER:-${REMOTE_USER:-dn}}"
PACKAGE_VERSION="${ARG_PACKAGE_VERSION:-${PACKAGE_VERSION:-develop}}"

# Fixed configuration
REMOTE_PASS="drive1234!"
PACKAGE_FILE="rccl-package.tar"
REMOTE_DIR="/home/${REMOTE_USER}/rccl-bins/${PACKAGE_VERSION}"
SOURCE_FILE="/home/amir/rccl-packages/${PACKAGE_VERSION}/${PACKAGE_FILE}"

# Get script directory and servers file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVERS_FILE="$SCRIPT_DIR/servers.txt"

# Display configuration
echo "Configuration:"
echo "  REMOTE_USER:     $REMOTE_USER"
echo "  PACKAGE_VERSION: $PACKAGE_VERSION"
echo ""

# Check if servers file exists
if [ ! -f "$SERVERS_FILE" ]; then
    echo "Error: Servers file $SERVERS_FILE not found."
    exit 1
fi

# Check if source file exists
if [ ! -f "$SOURCE_FILE" ]; then
    echo "Error: Source file $SOURCE_FILE not found."
    exit 1
fi

# Check if sshpass is installed
if ! command -v sshpass &> /dev/null; then
    echo "Error: sshpass is not installed. Please install it to use this script."
    exit 1
fi

# Loop through each server in the file
while IFS= read -r REMOTE_HOST || [ -n "$REMOTE_HOST" ]; do
    # Skip empty lines or comments
    [[ -z "$REMOTE_HOST" || "$REMOTE_HOST" =~ ^# ]] && continue

    echo "--------------------------------------------------"
    echo "Deploying to $REMOTE_HOST..."

    echo "Creating remote directory $REMOTE_DIR on $REMOTE_HOST..."
    sshpass -p "$REMOTE_PASS" ssh -n -o StrictHostKeyChecking=no "$REMOTE_USER@$REMOTE_HOST" "mkdir -p $REMOTE_DIR"

    echo "Copying $SOURCE_FILE to $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR..."
    sshpass -p "$REMOTE_PASS" scp -o StrictHostKeyChecking=no "$SOURCE_FILE" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/"

    echo "Extracting tar file on remote host $REMOTE_HOST..."
    sshpass -p "$REMOTE_PASS" ssh -n -o StrictHostKeyChecking=no "$REMOTE_USER@$REMOTE_HOST" "cd $REMOTE_DIR && tar -xvf $PACKAGE_FILE"

done < "$SERVERS_FILE"

echo "--------------------------------------------------"
echo "Done."
