#!/bin/bash
# Helper script to regenerate compile_commands.json for all AMD development projects
# Usage: ./regenerate-compile-commands.sh [project]
# Where project can be: rccl, rccl-tests, amd-anp, or all (default)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PROJECT="${1:-all}"

# Function to print colored status messages
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to regenerate rccl compile_commands.json
regenerate_rccl() {
    print_status "Regenerating compile_commands.json for RCCL..."
    
    if [ ! -d "dn/rccl" ]; then
        print_error "dn/rccl directory not found!"
        return 1
    fi
    
    cd dn/rccl/build/release
    cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DBUILD_TESTS=ON ../..
    
    if [ -f "compile_commands.json" ]; then
        ln -sf "$PWD/compile_commands.json" ../../compile_commands.json
        print_status "RCCL compile_commands.json created successfully ($(wc -l < compile_commands.json) lines)"
    else
        print_error "Failed to generate RCCL compile_commands.json"
        return 1
    fi
    
    cd "$SCRIPT_DIR"
}

# Function to regenerate rccl-tests compile_commands.json
regenerate_rccl_tests() {
    print_status "Regenerating compile_commands.json for RCCL Tests..."
    
    if [ ! -d "dn/rccl-tests" ]; then
        print_error "dn/rccl-tests directory not found!"
        return 1
    fi
    
    cd dn/rccl-tests/build
    cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..
    
    if [ -f "compile_commands.json" ]; then
        ln -sf "$PWD/compile_commands.json" ../compile_commands.json
        print_status "RCCL Tests compile_commands.json created successfully ($(wc -l < compile_commands.json) lines)"
    else
        print_error "Failed to generate RCCL Tests compile_commands.json"
        return 1
    fi
    
    cd "$SCRIPT_DIR"
}

# Function to regenerate amd-anp compile_commands.json
regenerate_amd_anp() {
    print_status "Regenerating compile_commands.json for AMD ANP..."
    
    if [ ! -d "dn/amd-anp" ]; then
        print_error "dn/amd-anp directory not found!"
        return 1
    fi
    
    # Check if bear is installed
    if ! command -v bear &> /dev/null; then
        print_error "bear is not installed. Please install it with: sudo apt-get install bear"
        return 1
    fi
    
    # Check if RCCL_HOME is set or use default
    RCCL_HOME="${RCCL_HOME:-$SCRIPT_DIR/dn/rccl}"
    RCCL_BUILD="${RCCL_BUILD:-$RCCL_HOME/build/release}"
    
    if [ ! -d "$RCCL_HOME" ]; then
        print_error "RCCL_HOME not found at: $RCCL_HOME"
        print_error "Please set RCCL_HOME environment variable or ensure dn/rccl exists"
        return 1
    fi
    
    cd dn/amd-anp
    
    # Touch source files to trigger rebuild
    print_status "Touching source files to trigger rebuild..."
    find src -name "*.cc" -exec touch {} \;
    
    # Use bear to capture compilation commands
    print_status "Capturing compilation commands with bear..."
    bear -- make RCCL_HOME="$RCCL_HOME" RCCL_BUILD="$RCCL_BUILD" 2>&1 | tee /tmp/amd-anp-build.log || true
    
    if [ -f "compile_commands.json" ]; then
        print_status "AMD ANP compile_commands.json created successfully ($(wc -l < compile_commands.json) lines)"
    else
        print_warning "AMD ANP compile_commands.json may not have been created"
        print_warning "Check /tmp/amd-anp-build.log for build errors"
    fi
    
    cd "$SCRIPT_DIR"
}

# Main script logic
case "$PROJECT" in
    rccl)
        regenerate_rccl
        ;;
    rccl-tests)
        regenerate_rccl_tests
        ;;
    amd-anp)
        regenerate_amd_anp
        ;;
    all)
        print_status "Regenerating compile_commands.json for all projects..."
        regenerate_rccl
        regenerate_rccl_tests
        regenerate_amd_anp
        print_status "All done!"
        ;;
    *)
        print_error "Unknown project: $PROJECT"
        echo "Usage: $0 [rccl|rccl-tests|amd-anp|all]"
        exit 1
        ;;
esac

print_status "=================================="
print_status "Compile database regeneration complete!"
print_status "You may need to restart clangd or reload your IDE for changes to take effect."
print_status "In VS Code/Cursor, you can run: 'Clangd: Restart language server'"



