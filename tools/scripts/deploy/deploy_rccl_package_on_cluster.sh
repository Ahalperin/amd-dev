#!/bin/bash

# Usage function
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Deploy RCCL binaries to remote cluster nodes."
    echo ""
    echo "Options:"
    echo "  -tu, --to-user USER      User to deploy (default: dn)"
    echo "  -fu, --from-user USER    User to take the package from (default: dn)"
    echo "  -v, --version VERSION    Package version (default: develop)"
    echo "  -sv, --show-versions     Show existing package versions in given --from_user"
    echo "  -h, --help               Show this help message"
    echo ""
    echo "Environment variables (used if options not provided):"
    echo "  REMOTE_USER, FROM_USER, PACKAGE_VERSION"
    exit 0
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -tu|--to-user)
            ARG_REMOTE_USER="$2"
            shift 2
            ;;
        -fu|--from-user)
            ARG_FROM_USER="$2"
            shift 2
            ;;
        -v|--version)
            ARG_PACKAGE_VERSION="$2"
            shift 2
            ;;
        -sv|--show-versions)
            SHOW_VERSIONS=true
            shift
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
FROM_USER="${ARG_FROM_USER:-${FROM_USER:-dn}}"
PACKAGE_VERSION="${ARG_PACKAGE_VERSION:-${PACKAGE_VERSION:-develop}}"

# Handle --show-versions option
if [ "$SHOW_VERSIONS" = true ]; then
    PACKAGES_DIR="/home/${FROM_USER}/rccl-packages"
    echo "Available package versions for user '${FROM_USER}':"
    if [ -d "$PACKAGES_DIR" ]; then
        versions=$(find "$PACKAGES_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | sort)
        if [ -n "$versions" ]; then
            echo "$versions"
        else
            echo "(none)"
        fi
    else
        echo "(none)"
    fi
    exit 0
fi

# Fixed configuration
REMOTE_PASS="drive1234!"
PACKAGE_FILE_NAME="rccl-package.tar"
SOURCE_FILE="/home/${FROM_USER}/rccl-packages/${PACKAGE_VERSION}/${PACKAGE_FILE_NAME}"
REMOTE_DIR="/home/${REMOTE_USER}/rccl-bins/${PACKAGE_VERSION}"

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
    sshpass -p "$REMOTE_PASS" ssh -n -o StrictHostKeyChecking=no "$REMOTE_USER@$REMOTE_HOST" "cd $REMOTE_DIR && tar -xvf $PACKAGE_FILE_NAME"

done < "$SERVERS_FILE"

echo "--------------------------------------------------"
echo "Done."
