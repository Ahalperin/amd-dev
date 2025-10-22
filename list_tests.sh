#!/bin/bash

# Script to list all available test names from tests.txt
# Usage: ./list_tests.sh

TESTS_FILE="./tools/docs/tests.txt"

if [ ! -f "$TESTS_FILE" ]; then
    echo "Error: Tests file not found: $TESTS_FILE"
    exit 1
fi

echo "Available tests:"
echo "================"
echo ""

# Extract all test names and organize by category
awk '
    /"category":/ {
        gsub(/^[[:space:]]*"category":[[:space:]]*"|"[[:space:]]*,?[[:space:]]*$/, "")
        category=$0
        if (category != prev_category) {
            if (prev_category != "") print ""
            print "Category: " category
            print "---"
            prev_category=category
        }
    }
    /"name":/ {
        gsub(/^[[:space:]]*"name":[[:space:]]*"|"[[:space:]]*,?[[:space:]]*$/, "")
        print "  " $0
    }
' "$TESTS_FILE"

echo ""
echo "================"
echo "Usage: ./run_test.sh <test_name>"

