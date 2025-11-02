#!/bin/bash
# Verification script to check TorchComms indexing setup

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TC_DIR="${1:-$(cd "$SCRIPT_DIR/../../../../meta/torchcomms" && pwd)}"

echo "🔍 Verifying TorchComms indexing setup..."
echo ""
echo "TorchComms directory: $TC_DIR"
echo ""

# Initialize counters
CHECKS_PASSED=0
CHECKS_TOTAL=0

check() {
    CHECKS_TOTAL=$((CHECKS_TOTAL + 1))
    if eval "$2"; then
        echo "✅ $1"
        CHECKS_PASSED=$((CHECKS_PASSED + 1))
        return 0
    else
        echo "❌ $1"
        return 1
    fi
}

# Check clangd installation
echo "🔧 Checking Prerequisites:"
check "clangd is installed" "command -v clangd &>/dev/null" || echo "   Install: brew install llvm (macOS) or apt install clangd (Ubuntu)"
if command -v clangd &>/dev/null; then
    CLANGD_VERSION=$(clangd --version 2>&1 | head -1)
    echo "   Version: $CLANGD_VERSION"
fi
echo ""

# Check deployed files
echo "📁 Checking Deployed Files:"
check "compile_commands.json exists" "test -f '$TC_DIR/compile_commands.json'"
check ".clangd config exists" "test -f '$TC_DIR/.clangd'"
check "VSCode settings exist" "test -f '$TC_DIR/.vscode/settings.json'"
check "README exists" "test -f '$TC_DIR/README-INDEXING.md'"
echo ""

# Check compile_commands.json content
if [ -f "$TC_DIR/compile_commands.json" ]; then
    echo "📊 Compilation Database Stats:"
    ENTRY_COUNT=$(grep -c '"file":' "$TC_DIR/compile_commands.json" 2>/dev/null || echo "0")
    FILE_SIZE=$(ls -lh "$TC_DIR/compile_commands.json" | awk '{print $5}')
    
    check "Has source file entries (found $ENTRY_COUNT)" "test $ENTRY_COUNT -gt 0"
    echo "   File size: $FILE_SIZE"
    echo ""
fi

# Check if key source files are indexed
echo "🔎 Checking Sample Files:"
for file in "comms/ctran/Ctran.cc" "comms/ctran/algos/AllReduce/AllReduceDirect.cc" "comms/torchcomms/TorchComm.cpp"; do
    check "$file is indexed" "grep -q '$TC_DIR/$file' '$TC_DIR/compile_commands.json' 2>/dev/null"
done
echo ""

# Check directory structure
echo "📂 Checking Code Structure:"
check "comms/ctran exists" "test -d '$TC_DIR/comms/ctran'"
check "comms/torchcomms exists" "test -d '$TC_DIR/comms/torchcomms'"
check "comms/utils exists" "test -d '$TC_DIR/comms/utils'"
echo ""

# Summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📋 Summary: $CHECKS_PASSED/$CHECKS_TOTAL checks passed"
echo ""

if [ $CHECKS_PASSED -eq $CHECKS_TOTAL ]; then
    echo "✅ All checks passed! Setup is complete."
    echo ""
    echo "🚀 Next steps:"
    echo "   1. Open VSCode: code '$TC_DIR'"
    echo "   2. Install 'clangd' extension"
    echo "   3. Disable 'C/C++' extension"
    echo "   4. Reload window and start navigating!"
    echo ""
    echo "📖 See: $TC_DIR/README-INDEXING.md"
    exit 0
elif [ $CHECKS_PASSED -gt $((CHECKS_TOTAL / 2)) ]; then
    echo "⚠️  Some checks failed, but setup is mostly complete."
    echo "   Review the failures above and fix if needed."
    exit 0
else
    echo "❌ Multiple checks failed. Setup may be incomplete."
    echo ""
    echo "💡 Try running setup again:"
    echo "   cd $SCRIPT_DIR"
    echo "   ./setup.sh"
    exit 1
fi






