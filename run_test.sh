#!/bin/bash

# Script to run a test from tests.txt and save results
# Usage: ./run_test.sh <test_name>

if [ $# -ne 1 ]; then
    echo "Usage: $0 <test_name>"
    echo "Example: $0 proto_simple_all_reduce_with_ld"
    exit 1
fi

TEST_NAME="$1"
TESTS_FILE="./tools/docs/tests.txt"
RESULTS_BASE_DIR="./results"

# Check if tests file exists
if [ ! -f "$TESTS_FILE" ]; then
    echo "Error: Tests file not found: $TESTS_FILE"
    exit 1
fi

# Extract the command for the given test name
# Look for the test name and extract the multi-line command
echo "Searching for test: $TEST_NAME"

# Use awk to extract the command between "name": "$TEST_NAME" and the next closing brace
COMMAND=$(awk -v test="$TEST_NAME" '
    /"name": "'"$TEST_NAME"'"/ { found=1; next }
    found && /"command":/ {
        # Start capturing from mpirun
        in_command=1
        line=$0
        # Remove leading whitespace and "command": 
        sub(/^[[:space:]]*"command":[[:space:]]*/, "", line)
        command=line
        # Check if command ends on same line (has closing JSON quote after the bash -c closing quote)
        if (line ~ /"[[:space:]]*$/) {
            # This is a one-line command, but we need to keep the bash -c quotes
            # The line ends with ..." so we keep everything
            print command
            exit
        }
        next
    }
    in_command {
        # Check if this line ends the command (ends with closing JSON quote after content)
        if ($0 ~ /"[[:space:]]*$/) {
            # This is the last line - it has the content ending with a quote
            # We need to keep the last quote (part of bash -c) but not process further
            command=command "\n" $0
            print command
            exit
        } else if ($0 ~ /^[[:space:]]*}/) {
            # Hit closing brace without finding end quote
            print command
            exit
        } else {
            # Continue building the command
            command=command "\n" $0
        }
    }
' "$TESTS_FILE")

if [ -z "$COMMAND" ]; then
    echo "Error: Test '$TEST_NAME' not found in $TESTS_FILE"
    exit 1
fi

echo "Found test command"
echo "---"
echo "$COMMAND"
echo "---"

# Create results directory for this test
TEST_RESULTS_DIR="${RESULTS_BASE_DIR}/${TEST_NAME}"
mkdir -p "$TEST_RESULTS_DIR"

echo ""
echo "Running test and saving results to: $TEST_RESULTS_DIR"
echo ""

# Clean up any existing output files in the current directory
rm -f nccl_debug.log topo.xml graph.xml

# Run the command and capture output
OUTPUT_FILE="${TEST_RESULTS_DIR}/command_output.log"
echo "Running command..."
echo "Command output will be saved to: $OUTPUT_FILE"
echo ""

# Clean up the command - remove line continuation backslashes and join lines
# The backslashes are for readability in the file but cause issues with eval
COMMAND_CLEAN=""
while IFS= read -r line; do
    # Remove trailing backslash and whitespace
    line=$(echo "$line" | sed 's/[[:space:]]*\\[[:space:]]*$//')
    COMMAND_CLEAN="${COMMAND_CLEAN} ${line}"
done <<< "$COMMAND"

# Trim leading space
COMMAND_CLEAN="${COMMAND_CLEAN# }"

# Execute the command and save output
eval "$COMMAND_CLEAN" 2>&1 | tee "$OUTPUT_FILE"
EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "Command finished with exit code: $EXIT_CODE"

# Move generated files to results directory
echo ""
echo "Collecting generated files..."

FILES_MOVED=0

if [ -f "nccl_debug.log" ]; then
    mv nccl_debug.log "$TEST_RESULTS_DIR/"
    echo "✓ Saved nccl_debug.log"
    FILES_MOVED=$((FILES_MOVED+1))
else
    echo "✗ nccl_debug.log not found"
fi

if [ -f "topo.xml" ]; then
    mv topo.xml "$TEST_RESULTS_DIR/"
    echo "✓ Saved topo.xml"
    FILES_MOVED=$((FILES_MOVED+1))
else
    echo "✗ topo.xml not found"
fi

if [ -f "graph.xml" ]; then
    mv graph.xml "$TEST_RESULTS_DIR/"
    echo "✓ Saved graph.xml"
    FILES_MOVED=$((FILES_MOVED+1))
else
    echo "✗ graph.xml not found"
fi

echo "✓ Saved command_output.log"

echo ""
echo "================================================"
echo "Test completed: $TEST_NAME"
echo "Results saved to: $TEST_RESULTS_DIR"
echo "Files collected: $((FILES_MOVED+1))"
echo "Exit code: $EXIT_CODE"
echo "================================================"

exit $EXIT_CODE

