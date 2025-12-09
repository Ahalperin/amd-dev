#!/bin/bash

# Variables
REMOTE_USER="dn"
PACKAGE_VERSION="develop"

# Configuration
REMOTE_PASS="drive1234!"
PACKAGE_FILE="rccl-package.tar"
REMOTE_DIR="/home/${REMOTE_USER}/rccl-bins/${PACKAGE_VERSION}"
SOURCE_FILE="/home/amir/rccl-packages/${PACKAGE_VERSION}/${PACKAGE_FILE}"

# Get script directory and servers file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVERS_FILE="$SCRIPT_DIR/servers.txt"

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
