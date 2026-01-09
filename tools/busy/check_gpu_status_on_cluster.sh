#!/bin/bash

# Script to check GPU usage status across multiple servers
# Reads server IPs from servers.txt and reports if GPUs are in use or free

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVERS_FILE="${SCRIPT_DIR}/servers.txt"

if [[ ! -f "$SERVERS_FILE" ]]; then
    echo "Error: servers.txt not found at $SERVERS_FILE"
    exit 1
fi

echo "Checking GPU status on servers..."
echo "=================================================================="

while IFS= read -r line || [[ -n "$line" ]]; do
    # Skip empty lines and comment lines (starting with #)
    [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue

    # Strip inline comments (everything after #) and trim whitespace
    server="${line%%#*}"
    server="${server%% }"
    server="${server## }"
    [[ -z "$server" ]] && continue

    # Check GPU status via SSH (-n prevents SSH from consuming stdin)
    # Get the first GPU process name (excluding gpuagent) if GPUs are in use
    result=$(sshpass -p 'drive1234!' ssh -n -o ConnectTimeout=5 -o StrictHostKeyChecking=no "dn@$server" \
        "pids=\$(rocm-smi --showpids 2>/dev/null | grep -E '^[0-9]+' | grep -v 'gpuagent' | awk '{print \$1}')
         if [[ -n \"\$pids\" ]]; then
             first_pid=\$(echo \"\$pids\" | head -1)
             proc_name=\$(ps -p \$first_pid -o user=,comm= 2>/dev/null || echo 'unknown')
             echo \"GPUs-in-use|\$proc_name\"
         else
             echo 'GPUs-free|'
         fi" 2>/dev/null)

    if [[ -z "$result" ]]; then
        status="unreachable"
        proc_name=""
    else
        status=$(echo "$result" | cut -d'|' -f1)
        proc_name=$(echo "$result" | cut -d'|' -f2)
    fi

    printf "%-20s %-15s %s\n" "$server" "$status" "$proc_name"

done < "$SERVERS_FILE"

echo "=================================================================="
echo "Done."

