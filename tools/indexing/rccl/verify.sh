#!/bin/bash
# Verification script to check RCCL indexing setup

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RCCL_DIR="${1:-$(cd "$SCRIPT_DIR/../../../amd/rccl" && pwd)}"

echo "🔍 Verifying RCCL indexing setup..."
echo ""
echo "RCCL directory: $RCCL_DIR"
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
check "compile_commands.json exists" "test -f '$RCCL_DIR/compile_commands.json'"
check ".clangd config exists" "test -f '$RCCL_DIR/.clangd'"
check "VSCode settings exist" "test -f '$RCCL_DIR/.vscode/settings.json'"
check "README exists" "test -f '$RCCL_DIR/README-INDEXING.md'"
echo ""

# Check compile_commands.json content
if [ -f "$RCCL_DIR/compile_commands.json" ]; then
    echo "📊 Compilation Database Stats:"
    ENTRY_COUNT=$(grep -c '"file":' "$RCCL_DIR/compile_commands.json" 2>/dev/null || echo "0")
    FILE_SIZE=$(ls -lh "$RCCL_DIR/compile_commands.json" | awk '{print $5}')
    
    check "Has source file entries (found $ENTRY_COUNT)" "test $ENTRY_COUNT -gt 0"
    echo "   File size: $FILE_SIZE"
    echo "   Sample entry:"
    head -20 "$RCCL_DIR/compile_commands.json" | tail -15 | sed 's/^/     /'
    echo ""
fi

# Check if key source files are indexed
echo "🔎 Checking Sample Files:"
for file in "src/rccl_wrap.cc" "src/proxy.cc" "src/init.cc"; do
    check "$file is indexed" "grep -q '$RCCL_DIR/$file' '$RCCL_DIR/compile_commands.json' 2>/dev/null"
done
echo ""

# Test clangd on a sample file
if command -v clangd &>/dev/null && [ -f "$RCCL_DIR/src/rccl_wrap.cc" ]; then
    echo "🧪 Testing clangd parsing:"
    cd "$RCCL_DIR"
    
    # Try to check the file (with timeout)
    if timeout 5 clangd --check="src/rccl_wrap.cc" >/dev/null 2>&1; then
        echo "✅ clangd can parse source files"
        CHECKS_PASSED=$((CHECKS_PASSED + 1))
    else
        echo "⚠️  clangd check timed out or failed (this is OK, indexing may still work)"
    fi
    CHECKS_TOTAL=$((CHECKS_TOTAL + 1))
    echo ""
fi

# Summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📋 Summary: $CHECKS_PASSED/$CHECKS_TOTAL checks passed"
echo ""

if [ $CHECKS_PASSED -eq $CHECKS_TOTAL ]; then
    echo "✅ All checks passed! Setup is complete."
    echo ""
    echo "🚀 Next steps:"
    echo "   1. Open VSCode: code '$RCCL_DIR'"
    echo "   2. Install 'clangd' extension"
    echo "   3. Disable 'C/C++' extension"
    echo "   4. Reload window and start navigating!"
    echo ""
    echo "📖 See: $RCCL_DIR/README-INDEXING.md"
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



