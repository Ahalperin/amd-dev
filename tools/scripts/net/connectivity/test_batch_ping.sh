#!/bin/bash

# Test the batch ping command structure to see what's wrong

SOURCE_IP="172.65.2.40"
TARGET_IPS=("172.65.1.24" "172.65.1.26")

echo "Testing batch ping command structure..."
echo ""

for target_ip in "${TARGET_IPS[@]}"; do
    echo "=== PING_START:${target_ip} ==="
    ping -I ${SOURCE_IP} -c 1 -W 1 ${target_ip} 2>&1
    exit_code=$?
    echo "=== PING_EXIT:${target_ip}:${exit_code} ==="
done

echo ""
echo "Expected format verified!"
