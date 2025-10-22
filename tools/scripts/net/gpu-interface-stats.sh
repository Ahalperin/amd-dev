#!/bin/bash

for iface in $(ip link | grep -oP '^\d+: \K[^:]+(?=:)' | grep -v lo); do
    echo "=== $iface ==="
    ethtool -S $iface | grep -E "(packets|bytes|errors|dropped)" | head -10
    echo ""
done