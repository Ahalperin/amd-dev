#!/bin/bash

################################################################################
# GPU Full Mesh Connection Test - Package Creator
#
# Creates a deployable tar.gz package from source files
################################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_NAME="gpu-mesh-test"
OUTPUT_FILE="gpu-mesh-test.tar.gz"

# Source files
MAIN_SCRIPT="$SCRIPT_DIR/gpu-full-mesh-conn-test.py"
NET_DISCOVERY="$SCRIPT_DIR/../net-discovery.sh"
INSTALL_SCRIPT="$SCRIPT_DIR/package/$PACKAGE_NAME/install.sh"
README="$SCRIPT_DIR/package/$PACKAGE_NAME/README.md"
SERVERS_SAMPLE="$SCRIPT_DIR/package/$PACKAGE_NAME/servers.list.sample"

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Creates a deployable gpu-mesh-test.tar.gz package"
    echo ""
    echo "Options:"
    echo "  -o, --output DIR    Output directory for the tar.gz (default: current directory)"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Example:"
    echo "  $0                      # Creates package in current directory"
    echo "  $0 -o /tmp              # Creates package in /tmp"
}

OUTPUT_DIR="$SCRIPT_DIR"

while [[ $# -gt 0 ]]; do
    case $1 in
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            print_usage
            exit 1
            ;;
    esac
done

echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║       GPU Full Mesh Connection Test - Package Creator       ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check required source files exist
echo -e "${YELLOW}Checking source files...${NC}"

missing_files=0

if [ ! -f "$MAIN_SCRIPT" ]; then
    echo -e "  ${RED}✗${NC} Missing: $MAIN_SCRIPT"
    missing_files=1
else
    echo -e "  ${GREEN}✓${NC} Found: gpu-full-mesh-conn-test.py"
fi

if [ ! -f "$NET_DISCOVERY" ]; then
    echo -e "  ${RED}✗${NC} Missing: $NET_DISCOVERY"
    missing_files=1
else
    echo -e "  ${GREEN}✓${NC} Found: net-discovery.sh"
fi

if [ ! -f "$INSTALL_SCRIPT" ]; then
    echo -e "  ${RED}✗${NC} Missing: $INSTALL_SCRIPT"
    missing_files=1
else
    echo -e "  ${GREEN}✓${NC} Found: install.sh"
fi

if [ ! -f "$README" ]; then
    echo -e "  ${RED}✗${NC} Missing: $README"
    missing_files=1
else
    echo -e "  ${GREEN}✓${NC} Found: README.md"
fi

if [ ! -f "$SERVERS_SAMPLE" ]; then
    echo -e "  ${RED}✗${NC} Missing: $SERVERS_SAMPLE"
    missing_files=1
else
    echo -e "  ${GREEN}✓${NC} Found: servers.list.sample"
fi

if [ $missing_files -eq 1 ]; then
    echo ""
    echo -e "${RED}Error: Missing required source files${NC}"
    exit 1
fi

# Create temporary build directory
BUILD_DIR=$(mktemp -d)
PACKAGE_DIR="$BUILD_DIR/$PACKAGE_NAME"
mkdir -p "$PACKAGE_DIR"

echo ""
echo -e "${YELLOW}Building package...${NC}"

# Copy files to package directory
cp "$MAIN_SCRIPT" "$PACKAGE_DIR/"
cp "$NET_DISCOVERY" "$PACKAGE_DIR/"
cp "$INSTALL_SCRIPT" "$PACKAGE_DIR/"
cp "$README" "$PACKAGE_DIR/"
cp "$SERVERS_SAMPLE" "$PACKAGE_DIR/"

# Make scripts executable
chmod +x "$PACKAGE_DIR/gpu-full-mesh-conn-test.py"
chmod +x "$PACKAGE_DIR/net-discovery.sh"
chmod +x "$PACKAGE_DIR/install.sh"

echo -e "  ${GREEN}✓${NC} Copied all files to build directory"

# Create output directory if needed
mkdir -p "$OUTPUT_DIR"

# Create tar.gz
echo -e "  ${GREEN}✓${NC} Creating tar.gz archive..."
cd "$BUILD_DIR"
tar -czf "$OUTPUT_DIR/$OUTPUT_FILE" "$PACKAGE_NAME"

# Cleanup
rm -rf "$BUILD_DIR"

# Get package info
PACKAGE_PATH="$OUTPUT_DIR/$OUTPUT_FILE"
PACKAGE_SIZE=$(ls -lh "$PACKAGE_PATH" | awk '{print $5}')
FILE_COUNT=$(tar -tzf "$PACKAGE_PATH" | grep -v '/$' | wc -l)

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                   Package Created Successfully              ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "Output:     ${CYAN}$PACKAGE_PATH${NC}"
echo -e "Size:       $PACKAGE_SIZE"
echo -e "Files:      $FILE_COUNT"
echo ""
echo -e "${YELLOW}Package contents:${NC}"
tar -tzf "$PACKAGE_PATH" | sed 's/^/  /'
echo ""
echo -e "${CYAN}To deploy:${NC}"
echo "  scp $PACKAGE_PATH user@server:/tmp/"
echo "  ssh user@server 'cd /tmp && tar -xzf $OUTPUT_FILE && cd $PACKAGE_NAME && sudo ./install.sh'"
echo ""

