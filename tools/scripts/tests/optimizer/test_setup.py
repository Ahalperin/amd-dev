#!/usr/bin/env python3
"""
Test script to verify RCCL optimizer setup before running full optimization.
This helps diagnose issues with configuration, connectivity, and test execution.
"""

import sys
import subprocess
import yaml
from pathlib import Path
import time


def print_header(text):
    """Print a section header."""
    print(f"\n{'='*80}")
    print(f"  {text}")
    print(f"{'='*80}\n")


def print_result(test_name, passed, message=""):
    """Print test result."""
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"{status} - {test_name}")
    if message:
        print(f"       {message}")


def test_config_exists():
    """Test if config.yaml exists."""
    print_header("1. Checking Configuration File")
    
    config_path = Path("config.yaml")
    exists = config_path.exists()
    
    print_result("config.yaml exists", exists)
    return exists


def test_config_valid():
    """Test if config.yaml is valid YAML."""
    print_header("2. Validating Configuration")
    
    try:
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required sections
        required_sections = ['test_config', 'fixed_env_vars', 'optimize_params', 'optimization']
        missing = [s for s in required_sections if s not in config]
        
        if missing:
            print_result("Configuration structure", False, f"Missing sections: {missing}")
            return False, None
        
        print_result("YAML syntax", True)
        print_result("Required sections", True)
        return True, config
    
    except Exception as e:
        print_result("Configuration parsing", False, str(e))
        return False, None


def test_rccl_test_exists(config):
    """Test if RCCL test executable exists."""
    print_header("3. Checking RCCL Test Executable")
    
    test_name = config['test_config']['test_name']
    test_path = f"/home/dn/amd-dev/amd/rccl-tests/build/{test_name}"
    
    exists = Path(test_path).exists()
    print_result(f"Test executable: {test_path}", exists)
    
    if exists:
        # Check if executable
        is_exec = Path(test_path).is_file() and Path(test_path).stat().st_mode & 0o111
        print_result("File is executable", is_exec)
        return is_exec
    
    return False


def test_mpi():
    """Test if MPI is available."""
    print_header("4. Checking MPI Installation")
    
    try:
        result = subprocess.run(['which', 'mpirun'], 
                               capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            mpi_path = result.stdout.strip()
            print_result("mpirun found", True, mpi_path)
            
            # Get version
            result = subprocess.run(['mpirun', '--version'], 
                                   capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                version = result.stdout.strip().split('\n')[0]
                print_result("MPI version", True, version)
            
            return True
        else:
            print_result("mpirun found", False)
            return False
    
    except Exception as e:
        print_result("MPI check", False, str(e))
        return False


def test_network_connectivity(config):
    """Test network connectivity to MPI hosts."""
    print_header("5. Testing Network Connectivity")
    
    mpi_hosts = config['test_config']['mpi_hosts']
    # Parse hosts (format: "ip:slots,ip:slots")
    host_ips = []
    for host_spec in mpi_hosts.split(','):
        ip = host_spec.split(':')[0].strip()
        host_ips.append(ip)
    
    all_ok = True
    for ip in host_ips:
        try:
            result = subprocess.run(['ping', '-c', '1', '-W', '2', ip], 
                                   capture_output=True, timeout=5)
            success = result.returncode == 0
            print_result(f"Ping {ip}", success)
            all_ok = all_ok and success
        except Exception as e:
            print_result(f"Ping {ip}", False, str(e))
            all_ok = False
    
    return all_ok


def test_simple_mpi(config):
    """Test simple MPI command."""
    print_header("6. Testing Simple MPI Command")
    
    mpi_hosts = config['test_config']['mpi_hosts']
    
    cmd = [
        'mpirun',
        '--np', '2',
        '-H', mpi_hosts.split(',')[0],  # Just first host
        '--allow-run-as-root',
        'hostname'
    ]
    
    print(f"Running: {' '.join(cmd)}")
    print("(This should complete quickly...)")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print_result("Simple MPI test", True)
            print(f"       Output: {result.stdout.strip()}")
            return True
        else:
            print_result("Simple MPI test", False, f"Exit code: {result.returncode}")
            if result.stderr:
                print(f"       Error: {result.stderr[:200]}")
            return False
    
    except subprocess.TimeoutExpired:
        print_result("Simple MPI test", False, "TIMEOUT - MPI command hung!")
        print("       This is likely the same issue affecting the optimizer.")
        print("       Check:")
        print("       - SSH passwordless access between nodes")
        print("       - Network connectivity")
        print("       - MPI configuration")
        return False
    except Exception as e:
        print_result("Simple MPI test", False, str(e))
        return False


def test_rccl_simple(config):
    """Test simple RCCL test execution."""
    print_header("7. Testing RCCL Test Execution (Quick)")
    
    test_name = config['test_config']['test_name']
    test_path = f"/home/dn/amd-dev/amd/rccl-tests/build/{test_name}"
    mpi_hosts = config['test_config']['mpi_hosts'].split(',')[0]  # Just first host
    
    cmd = [
        'mpirun',
        '--np', '2',
        '-H', mpi_hosts,
        '--allow-run-as-root',
        '--bind-to', 'numa',
        test_path,
        '-b', '1M',
        '-e', '1M',
        '-f', '2',
        '-g', '1',
        '-n', '1',
        '-w', '0'
    ]
    
    print(f"Running quick RCCL test...")
    print(f"Command: {' '.join(cmd[:4])} ... {test_name} -b 1M -e 1M ...")
    print("(This may take 30-60 seconds...)")
    
    try:
        start = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        elapsed = time.time() - start
        
        if result.returncode == 0:
            print_result("RCCL test execution", True, f"Completed in {elapsed:.1f}s")
            
            # Check for bandwidth in output
            if 'GB/s' in result.stdout:
                print("       Test produced valid output with bandwidth metrics")
            return True
        else:
            print_result("RCCL test execution", False, f"Exit code: {result.returncode}")
            if result.stderr:
                print(f"       Error: {result.stderr[:200]}")
            return False
    
    except subprocess.TimeoutExpired:
        print_result("RCCL test execution", False, "TIMEOUT (>120s)")
        print("       The test is taking too long or hanging.")
        print("       Possible issues:")
        print("       - GPU initialization problems")
        print("       - Network configuration issues")
        print("       - Library path problems")
        return False
    except Exception as e:
        print_result("RCCL test execution", False, str(e))
        return False


def test_python_deps():
    """Test if Python dependencies are installed."""
    print_header("8. Checking Python Dependencies")
    
    deps = {
        'skopt': 'scikit-optimize',
        'yaml': 'pyyaml',
        'pandas': 'pandas',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'colorama': 'colorama',
        'tabulate': 'tabulate'
    }
    
    all_ok = True
    for module, package in deps.items():
        try:
            __import__(module)
            print_result(f"{package}", True)
        except ImportError:
            print_result(f"{package}", False, f"Run: pip install {package}")
            all_ok = False
    
    return all_ok


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("  RCCL OPTIMIZER SETUP TEST")
    print("  This will verify your configuration and environment")
    print("="*80)
    
    results = []
    
    # Test 1: Config exists
    if not test_config_exists():
        print("\n❌ Cannot proceed without config.yaml")
        sys.exit(1)
    
    # Test 2: Config valid
    valid, config = test_config_valid()
    results.append(("Configuration", valid))
    if not valid:
        print("\n❌ Cannot proceed with invalid configuration")
        sys.exit(1)
    
    # Test 3: RCCL test exists
    results.append(("RCCL Test Executable", test_rccl_test_exists(config)))
    
    # Test 4: MPI
    results.append(("MPI Installation", test_mpi()))
    
    # Test 5: Network
    results.append(("Network Connectivity", test_network_connectivity(config)))
    
    # Test 6: Simple MPI
    mpi_ok = test_simple_mpi(config)
    results.append(("Simple MPI Test", mpi_ok))
    
    # Test 7: RCCL execution (only if MPI works)
    if mpi_ok:
        results.append(("RCCL Test Execution", test_rccl_simple(config)))
    else:
        print_header("7. Testing RCCL Test Execution")
        print("⊘ SKIPPED - Fix MPI issues first")
        results.append(("RCCL Test Execution", False))
    
    # Test 8: Python deps
    results.append(("Python Dependencies", test_python_deps()))
    
    # Summary
    print_header("TEST SUMMARY")
    
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    
    for name, ok in results:
        status = "✓" if ok else "✗"
        print(f"  {status} {name}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed! You're ready to run the optimizer.")
        print("\nRun: ./optimize_rccl.py --config config.yaml --iterations 20")
    else:
        print("\n❌ Some tests failed. Please fix the issues above before running optimization.")
        print("\nCommon fixes:")
        print("  - Install missing dependencies: pip install -r requirements.txt")
        print("  - Check network connectivity: ping <your_host_ip>")
        print("  - Verify SSH keys: ssh <your_host_ip> hostname")
        print("  - Check RCCL paths in config.yaml")
        print("  - Ensure GPUs are accessible")
    
    sys.exit(0 if passed == total else 1)


if __name__ == '__main__':
    main()


