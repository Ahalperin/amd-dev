#!/bin/bash

# Script to check ping and SSH connectivity for servers in servers.list
# Usage: ./check_connectivity.sh

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# SSH user and password
SSH_USER="dn"
SSH_PASSWORD="drive1234!"

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVERS_FILE="${SCRIPT_DIR}/servers.list"

# Check if servers.list exists
if [[ ! -f "$SERVERS_FILE" ]]; then
    echo -e "${RED}Error: servers.list not found in ${SCRIPT_DIR}${NC}"
    exit 1
fi

# Initialize counters
total_servers=0
ping_reachable=0
ping_unreachable=0
ssh_reachable=0
ssh_unreachable=0

echo "=================================="
echo "Server Connectivity Check"
echo "=================================="
echo ""
echo "Reading servers from: $SERVERS_FILE"
echo "SSH User: $SSH_USER"
echo ""

# Read the file and process each line
while IFS= read -r line || [[ -n "$line" ]]; do
    # Skip empty lines and header line
    if [[ -z "$line" ]] || [[ "$line" =~ ^Vendor/Hostname ]] || [[ "$line" =~ ^[[:space:]]*$ ]]; then
        continue
    fi
    
    # Extract IP address (handles lines with just IP or tab-separated fields)
    ip=$(echo "$line" | awk '{print $1}')
    
    # Skip if not a valid IP format
    if [[ ! "$ip" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        continue
    fi
    
    total_servers=$((total_servers + 1))
    
    # Ping the server (send 2 packets, timeout 2 seconds)
    printf "Testing %-15s ... " "$ip"
    if ping -c 2 -W 2 "$ip" > /dev/null 2>&1; then
        printf "${GREEN}PING: ✓${NC} | "
        ping_reachable=$((ping_reachable + 1))
        
        # Test SSH connectivity (timeout 5 seconds)
        # Redirect stdin to /dev/null to prevent ssh from consuming the while loop's stdin
        if ssh -o ConnectTimeout=5 -o BatchMode=yes -o StrictHostKeyChecking=no "${SSH_USER}@${ip}" "exit" < /dev/null > /dev/null 2>&1; then
            printf "${GREEN}SSH: ✓${NC}\n"
            ssh_reachable=$((ssh_reachable + 1))
        else
            printf "${RED}SSH: ✗${NC} - ${BLUE}Setting up SSH key...${NC}\n"
            ssh_unreachable=$((ssh_unreachable + 1))
            
            # Try to copy SSH key to enable passwordless authentication
            echo "  Running ssh-copy-id for ${SSH_USER}@${ip}"
            
            # Check if sshpass is available
            if command -v sshpass > /dev/null 2>&1; then
                # Use sshpass for automatic password authentication
                if sshpass -p "${SSH_PASSWORD}" ssh-copy-id -o ConnectTimeout=5 -o StrictHostKeyChecking=no -f "${SSH_USER}@${ip}" > /dev/null 2>&1; then
                    echo -e "  ${GREEN}✓ SSH key copied successfully${NC}"
                    # Test SSH again
                    if ssh -o ConnectTimeout=5 -o BatchMode=yes -o StrictHostKeyChecking=no "${SSH_USER}@${ip}" "exit" < /dev/null > /dev/null 2>&1; then
                        echo -e "  ${GREEN}✓ SSH now working!${NC}"
                        ssh_unreachable=$((ssh_unreachable - 1))
                        ssh_reachable=$((ssh_reachable + 1))
                    fi
                else
                    echo -e "  ${RED}✗ Failed to copy SSH key${NC}"
                fi
            else
                # Fallback: use expect script
                expect << EOF > /dev/null 2>&1
                    set timeout 10
                    spawn ssh-copy-id -o ConnectTimeout=5 -o StrictHostKeyChecking=no -f ${SSH_USER}@${ip}
                    expect {
                        "*password:" { send "${SSH_PASSWORD}\r"; exp_continue }
                        eof
                    }
EOF
                if [ $? -eq 0 ]; then
                    echo -e "  ${GREEN}✓ SSH key copied successfully${NC}"
                    # Test SSH again
                    if ssh -o ConnectTimeout=5 -o BatchMode=yes -o StrictHostKeyChecking=no "${SSH_USER}@${ip}" "exit" < /dev/null > /dev/null 2>&1; then
                        echo -e "  ${GREEN}✓ SSH now working!${NC}"
                        ssh_unreachable=$((ssh_unreachable - 1))
                        ssh_reachable=$((ssh_reachable + 1))
                    fi
                else
                    echo -e "  ${RED}✗ Failed to copy SSH key${NC}"
                    echo -e "  ${YELLOW}Note: Install 'sshpass' or 'expect' for automatic authentication${NC}"
                fi
            fi
            echo ""
        fi
    else
        # Ping failed - skip SSH test and count as unreachable for both
        printf "${RED}PING: ✗${NC} | ${RED}SSH: ✗${NC} ${YELLOW}(not tested - ping failed)${NC}\n"
        ping_unreachable=$((ping_unreachable + 1))
        ssh_unreachable=$((ssh_unreachable + 1))
    fi
done < "$SERVERS_FILE"

# Print summary
echo ""
echo "=================================="
echo "Summary"
echo "=================================="
echo "Total servers: $total_servers"
echo ""
echo "PING Results:"
echo -e "  ${GREEN}Reachable: $ping_reachable${NC}"
echo -e "  ${RED}Unreachable: $ping_unreachable${NC}"
echo ""
echo "SSH Results (user: $SSH_USER):"
echo -e "  ${GREEN}Reachable: $ssh_reachable${NC}"
echo -e "  ${RED}Unreachable: $ssh_unreachable${NC}"

# Exit with appropriate code
if [[ $ping_unreachable -gt 0 ]] || [[ $ssh_unreachable -gt 0 ]]; then
    echo ""
    echo -e "${YELLOW}Warning: Some connectivity tests failed${NC}"
    exit 1
else
    echo ""
    echo -e "${GREEN}All connectivity tests passed!${NC}"
    exit 0
fi

