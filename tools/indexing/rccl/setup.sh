#!/bin/bash
# Setup script to deploy RCCL indexing files
# This script copies configuration files to the RCCL directory and generates compile_commands.json

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_RCCL_DIR="$(cd "$SCRIPT_DIR/../../../amd/rccl" && pwd)"
RCCL_DIR="${1:-$DEFAULT_RCCL_DIR}"

echo "🔧 Setting up RCCL code indexing..."
echo ""

# Verify RCCL directory
if [ ! -d "$RCCL_DIR" ]; then
    echo "❌ Error: RCCL directory not found: $RCCL_DIR"
    echo "   Usage: $0 [RCCL_ROOT_DIR]"
    exit 1
fi

if [ ! -d "$RCCL_DIR/src" ]; then
    echo "❌ Error: Not a valid RCCL directory (src/ not found): $RCCL_DIR"
    exit 1
fi

echo "📁 RCCL directory: $RCCL_DIR"
echo ""

# Deploy .clangd configuration
echo "📝 Deploying .clangd configuration..."
cp "$SCRIPT_DIR/clangd-config" "$RCCL_DIR/.clangd"
echo "   ✅ Created $RCCL_DIR/.clangd"

# Deploy VSCode settings
echo "📝 Deploying VSCode settings..."
mkdir -p "$RCCL_DIR/.vscode"
if [ -f "$RCCL_DIR/.vscode/settings.json" ]; then
    echo "   ⚠️  $RCCL_DIR/.vscode/settings.json already exists"
    echo "   Creating backup: .vscode/settings.json.bak"
    cp "$RCCL_DIR/.vscode/settings.json" "$RCCL_DIR/.vscode/settings.json.bak"
fi
cp "$SCRIPT_DIR/vscode-settings.json" "$RCCL_DIR/.vscode/settings.json"
echo "   ✅ Created $RCCL_DIR/.vscode/settings.json"

# Deploy documentation
echo "📝 Deploying README..."
cp "$SCRIPT_DIR/README.md" "$RCCL_DIR/README-INDEXING.md"
echo "   ✅ Created $RCCL_DIR/README-INDEXING.md"

# Generate compile_commands.json
echo ""
echo "🔨 Generating compile_commands.json..."
python3 "$SCRIPT_DIR/generate_compile_commands.py" "$RCCL_DIR"

echo ""
echo "✅ Setup complete!"
echo ""
echo "📚 Next steps:"
echo "   1. Open VSCode in: $RCCL_DIR"
echo "   2. Install 'clangd' extension by LLVM (if not already installed)"
echo "   3. Disable/uninstall 'C/C++' extension by Microsoft (or disable for this workspace)"
echo "   4. Reload VSCode window (Cmd+Shift+P → 'Reload Window')"
echo "   5. Open any .cc/.h file and enjoy code navigation!"
echo ""
echo "🔍 Available features:"
echo "   - F12: Go to Definition"
echo "   - Shift+F12: Find All References"
echo "   - Cmd+Click: Jump to symbol"
echo "   - Hover: See type info and docs"
echo ""
echo "📖 For more info: cat $RCCL_DIR/README-INDEXING.md"





