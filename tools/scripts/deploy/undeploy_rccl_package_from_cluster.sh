#!/bin/bash

# Usage function
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Remove deployed RCCL binaries from remote cluster nodes."
    echo ""
    echo "Options:"
    echo "  -tu, --to-user USER      User to undeploy from (default: dn)"
    echo "  -v, --version VERSION    Package version to remove (default: develop)"
    echo "  -sv, --show-versions     Show existing package versions on each server (no removal)"
    echo "  -h, --help               Show this help message"
    echo ""
    echo "Environment variables (used if options not provided):"
    echo "  REMOTE_USER, PACKAGE_VERSION"
    echo ""
    echo "Examples:"
    echo "  # Undeploy using defaults (version=develop, user=dn)"
    echo "  $0"
    echo ""
    echo "  # Undeploy a specific version"
    echo "  $0 -v my-feature-branch"
    echo ""
    echo "  # Undeploy from user 'testuser'"
    echo "  $0 -tu testuser -v develop"
    echo ""
    echo "  # Show available package versions on all servers"
    echo "  $0 -sv"
    echo ""
    echo "  # Show available package versions for a specific user"
    echo "  $0 -tu testuser -sv"
    exit 0
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -tu|--to-user)
            ARG_REMOTE_USER="$2"
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
PACKAGE_VERSION="${ARG_PACKAGE_VERSION:-${PACKAGE_VERSION:-develop}}"

# Fixed configuration
REMOTE_PASS="drive1234!"
REMOTE_BASE_DIR="/home/${REMOTE_USER}/rccl-bins"
REMOTE_DIR="${REMOTE_BASE_DIR}/${PACKAGE_VERSION}"

# Get script directory and servers file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVERS_FILE="$SCRIPT_DIR/servers.txt"

# Check if servers file exists
if [ ! -f "$SERVERS_FILE" ]; then
    echo "Error: Servers file $SERVERS_FILE not found."
    exit 1
fi

# Check if sshpass is installed
if ! command -v sshpass &> /dev/null; then
    echo "Error: sshpass is not installed. Please install it to use this script."
    exit 1
fi

# Handle --show-versions option
if [ "$SHOW_VERSIONS" = true ]; then
    echo "Showing existing package versions for user '${REMOTE_USER}' on each server:"
    echo ""

    while IFS= read -r REMOTE_HOST || [ -n "$REMOTE_HOST" ]; do
        # Skip empty lines or comments
        [[ -z "$REMOTE_HOST" || "$REMOTE_HOST" =~ ^# ]] && continue

        echo "--------------------------------------------------"
        echo "Server: $REMOTE_HOST"

        versions=$(sshpass -p "$REMOTE_PASS" ssh -n -o StrictHostKeyChecking=no "$REMOTE_USER@$REMOTE_HOST" \
            "if [ -d '$REMOTE_BASE_DIR' ]; then find '$REMOTE_BASE_DIR' -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | sort; fi" 2>/dev/null)

        if [ -n "$versions" ]; then
            echo "$versions"
        else
            echo "(none)"
        fi

    done < "$SERVERS_FILE"

    echo "--------------------------------------------------"
    exit 0
fi

# Display configuration
echo "Configuration:"
echo "  REMOTE_USER:     $REMOTE_USER"
echo "  PACKAGE_VERSION: $PACKAGE_VERSION"
echo "  REMOTE_DIR:      $REMOTE_DIR"
echo ""

# Confirmation prompt
read -p "Are you sure you want to remove '$PACKAGE_VERSION' from all cluster nodes? [y/N] " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Loop through each server in the file
while IFS= read -r REMOTE_HOST || [ -n "$REMOTE_HOST" ]; do
    # Skip empty lines or comments
    [[ -z "$REMOTE_HOST" || "$REMOTE_HOST" =~ ^# ]] && continue

    echo "--------------------------------------------------"
    echo "Undeploying from $REMOTE_HOST..."

    # Check if directory exists before attempting removal
    exists=$(sshpass -p "$REMOTE_PASS" ssh -n -o StrictHostKeyChecking=no "$REMOTE_USER@$REMOTE_HOST" \
        "[ -d '$REMOTE_DIR' ] && echo 'yes' || echo 'no'" 2>/dev/null)

    if [ "$exists" = "yes" ]; then
        echo "Removing $REMOTE_DIR on $REMOTE_HOST..."
        sshpass -p "$REMOTE_PASS" ssh -n -o StrictHostKeyChecking=no "$REMOTE_USER@$REMOTE_HOST" "rm -rf '$REMOTE_DIR'"
        echo "Removed successfully."
    else
        echo "Directory $REMOTE_DIR does not exist on $REMOTE_HOST. Skipping."
    fi

done < "$SERVERS_FILE"

echo "--------------------------------------------------"
echo "Done."

