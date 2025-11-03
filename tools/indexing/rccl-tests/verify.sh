#!/bin/bash
# Verification script to check RCCL-tests indexing setup

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RT_DIR="${1:-$(cd "$SCRIPT_DIR/../../../amd/rccl-tests" && pwd)}"

echo "🔍 Verifying RCCL-tests indexing setup..."
echo ""
echo "RCCL-tests directory: $RT_DIR"
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
check "compile_commands.json exists" "test -f '$RT_DIR/compile_commands.json'"
check ".clangd config exists" "test -f '$RT_DIR/.clangd'"
check "VSCode settings exist" "test -f '$RT_DIR/.vscode/settings.json'"
check "README exists" "test -f '$RT_DIR/README-INDEXING.md'"
echo ""

# Check compile_commands.json content
if [ -f "$RT_DIR/compile_commands.json" ]; then
    echo "📊 Compilation Database Stats:"
    ENTRY_COUNT=$(grep -c '"file":' "$RT_DIR/compile_commands.json" 2>/dev/null || echo "0")
    FILE_SIZE=$(ls -lh "$RT_DIR/compile_commands.json" | awk '{print $5}')
    
    check "Has source file entries (found $ENTRY_COUNT)" "test $ENTRY_COUNT -gt 0"
    echo "   File size: $FILE_SIZE"
    echo ""
fi

# Check if key source files are indexed
echo "🔎 Checking Sample Files:"
for file in "src/all_reduce.cu" "src/common.cu" "src/common.h"; do
    check "$file is indexed" "grep -q '$RT_DIR/$file' '$RT_DIR/compile_commands.json' 2>/dev/null"
done
echo ""

# Check directory structure
echo "📂 Checking Code Structure:"
check "src/ exists" "test -d '$RT_DIR/src'"
check "src/all_reduce.cu exists" "test -f '$RT_DIR/src/all_reduce.cu'"
check "src/common.h exists" "test -f '$RT_DIR/src/common.h'"
echo ""

# Summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📋 Summary: $CHECKS_PASSED/$CHECKS_TOTAL checks passed"
echo ""

if [ $CHECKS_PASSED -eq $CHECKS_TOTAL ]; then
    echo "✅ All checks passed! Setup is complete."
    echo ""
    echo "🚀 Next steps:"
    echo "   1. Open VSCode: code '$RT_DIR'"
    echo "   2. Install 'clangd' extension"
    echo "   3. Disable 'C/C++' extension"
    echo "   4. Reload window and start navigating!"
    echo ""
    echo "📖 See: $RT_DIR/README-INDEXING.md"
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








