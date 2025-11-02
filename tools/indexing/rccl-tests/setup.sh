#!/bin/bash
# Setup script to deploy RCCL-tests indexing files
# This script copies configuration files to the RCCL-tests directory and generates compile_commands.json

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_RT_DIR="$(cd "$SCRIPT_DIR/../../../amd/rccl-tests" && pwd)"
RT_DIR="${1:-$DEFAULT_RT_DIR}"

echo "🔧 Setting up RCCL-tests code indexing..."
echo ""

# Verify RCCL-tests directory
if [ ! -d "$RT_DIR" ]; then
    echo "❌ Error: RCCL-tests directory not found: $RT_DIR"
    echo "   Usage: $0 [RCCL_TESTS_ROOT_DIR]"
    exit 1
fi

if [ ! -d "$RT_DIR/src" ]; then
    echo "❌ Error: Not a valid RCCL-tests directory (src/ not found): $RT_DIR"
    exit 1
fi

echo "📁 RCCL-tests directory: $RT_DIR"
echo ""

# Deploy .clangd configuration
echo "📝 Deploying .clangd configuration..."
cp "$SCRIPT_DIR/clangd-config" "$RT_DIR/.clangd"
echo "   ✅ Created $RT_DIR/.clangd"

# Deploy VSCode settings
echo "📝 Deploying VSCode settings..."
mkdir -p "$RT_DIR/.vscode"
if [ -f "$RT_DIR/.vscode/settings.json" ]; then
    echo "   ⚠️  $RT_DIR/.vscode/settings.json already exists"
    echo "   Creating backup: .vscode/settings.json.bak"
    cp "$RT_DIR/.vscode/settings.json" "$RT_DIR/.vscode/settings.json.bak"
fi
cp "$SCRIPT_DIR/vscode-settings.json" "$RT_DIR/.vscode/settings.json"
echo "   ✅ Created $RT_DIR/.vscode/settings.json"

# Deploy documentation
echo "📝 Deploying README..."
cp "$SCRIPT_DIR/README.md" "$RT_DIR/README-INDEXING.md"
echo "   ✅ Created $RT_DIR/README-INDEXING.md"

# Generate compile_commands.json
echo ""
echo "🔨 Generating compile_commands.json..."
python3 "$SCRIPT_DIR/generate_compile_commands.py" "$RT_DIR"

echo ""
echo "✅ Setup complete!"
echo ""
echo "📚 Next steps:"
echo "   1. Open VSCode in: $RT_DIR"
echo "   2. Install 'clangd' extension by LLVM (if not already installed)"
echo "   3. Disable/uninstall 'C/C++' extension by Microsoft (or disable for this workspace)"
echo "   4. Reload VSCode window (Cmd+Shift+P → 'Reload Window')"
echo "   5. Open any .cu/.cc file and enjoy code navigation!"
echo ""
echo "🔍 Available features:"
echo "   - F12: Go to Definition"
echo "   - Shift+F12: Find All References"
echo "   - Cmd+Click: Jump to symbol"
echo "   - Hover: See type info and docs"
echo ""
echo "📖 For more info: cat $RT_DIR/README-INDEXING.md"



