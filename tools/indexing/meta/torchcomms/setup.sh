#!/bin/bash
# Setup script to deploy Meta TorchComms indexing files
# This script copies configuration files to the TorchComms directory and generates compile_commands.json

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_TC_DIR="$(cd "$SCRIPT_DIR/../../../../meta/torchcomms" && pwd)"
TC_DIR="${1:-$DEFAULT_TC_DIR}"

echo "🔧 Setting up TorchComms code indexing..."
echo ""

# Verify TorchComms directory
if [ ! -d "$TC_DIR" ]; then
    echo "❌ Error: TorchComms directory not found: $TC_DIR"
    echo "   Usage: $0 [TORCHCOMMS_ROOT_DIR]"
    exit 1
fi

if [ ! -d "$TC_DIR/comms" ]; then
    echo "❌ Error: Not a valid TorchComms directory (comms/ not found): $TC_DIR"
    exit 1
fi

echo "📁 TorchComms directory: $TC_DIR"
echo ""

# Deploy .clangd configuration
echo "📝 Deploying .clangd configuration..."
cp "$SCRIPT_DIR/clangd-config" "$TC_DIR/.clangd"
echo "   ✅ Created $TC_DIR/.clangd"

# Deploy VSCode settings
echo "📝 Deploying VSCode settings..."
mkdir -p "$TC_DIR/.vscode"
if [ -f "$TC_DIR/.vscode/settings.json" ]; then
    echo "   ⚠️  $TC_DIR/.vscode/settings.json already exists"
    echo "   Creating backup: .vscode/settings.json.bak"
    cp "$TC_DIR/.vscode/settings.json" "$TC_DIR/.vscode/settings.json.bak"
fi
cp "$SCRIPT_DIR/vscode-settings.json" "$TC_DIR/.vscode/settings.json"
echo "   ✅ Created $TC_DIR/.vscode/settings.json"

# Deploy documentation
echo "📝 Deploying README..."
cp "$SCRIPT_DIR/README.md" "$TC_DIR/README-INDEXING.md"
echo "   ✅ Created $TC_DIR/README-INDEXING.md"

# Generate compile_commands.json
echo ""
echo "🔨 Generating compile_commands.json..."
python3 "$SCRIPT_DIR/generate_compile_commands.py" "$TC_DIR"

echo ""
echo "✅ Setup complete!"
echo ""
echo "📚 Next steps:"
echo "   1. Open VSCode in: $TC_DIR"
echo "   2. Install 'clangd' extension by LLVM (if not already installed)"
echo "   3. Disable/uninstall 'C/C++' extension by Microsoft (or disable for this workspace)"
echo "   4. Reload VSCode window (Cmd+Shift+P → 'Reload Window')"
echo "   5. Open any .cc/.cu file and enjoy code navigation!"
echo ""
echo "🔍 Available features:"
echo "   - F12: Go to Definition"
echo "   - Shift+F12: Find All References"
echo "   - Cmd+Click: Jump to symbol"
echo "   - Hover: See type info and docs"
echo ""
echo "📖 For more info: cat $TC_DIR/README-INDEXING.md"










