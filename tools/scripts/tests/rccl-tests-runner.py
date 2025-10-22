#!/usr/bin/env python3
"""
Script to run NCCL/RCCL tests with mpirun across multiple nodes.
"""

import argparse
import subprocess
import sys
import os
from datetime import datetime


def run_single_test(ip, np, test_name, minbytes, maxbytes, stepfactor, output_dir):
    """Run a single NCCL test with the given parameters.
    
    Args:
        ip: Single IP address or list of IP addresses
        np: Total number of processes
        test_name: Name of the test executable
        minbytes: Minimum byte size
        maxbytes: Maximum byte size
        stepfactor: Step factor
        output_dir: Output directory path
    """
    # Handle IP address(es)
    if isinstance(ip, list):
        ip_list = ip
    else:
        ip_list = [ip]
    
    # Calculate processes per host - distribute evenly
    num_hosts = len(ip_list)
    procs_per_host = np // num_hosts
    remainder = np % num_hosts
    
    # Build host string for mpirun: host1:slots,host2:slots,...
    host_specs = []
    for i, host_ip in enumerate(ip_list):
        # Distribute remainder processes to first hosts
        slots = procs_per_host + (1 if i < remainder else 0)
        host_specs.append(f"{host_ip}:{slots}")
    
    host_string = ','.join(host_specs)
    
    # Create prefix from test parameters
    prefix = f"{test_name}_{minbytes}_{maxbytes}_{stepfactor}"
    
    # Define output file names with prefix inside output directory
    test_results_log = os.path.join(output_dir, f"{prefix}.test.results.log")
    nccl_debug_log = os.path.join(output_dir, f"{prefix}.nccl_debug.log")
    topo_xml = os.path.join(output_dir, f"{prefix}.topo.xml")
    graph_xml = os.path.join(output_dir, f"{prefix}.graph.xml")
    
    # Build the rccl-test command content
    bash_command = (
        f'NCCL_DEBUG=INFO NCCL_DEBUG_FILE={nccl_debug_log} '
        f'NCCL_TOPO_DUMP_FILE={topo_xml} '
        f'NCCL_GRAPH_DUMP_FILE={graph_xml} '
        f'NCCL_DEBUG_SUBSYS=GRAPH,ALL '
        f'NCCL_IB_GID_INDEX=1 '
        f'NCCL_SOCKET_IFNAME=enp81s0f1np1 '
        f'LD_LIBRARY_PATH=/home/dn/amd-dev/amd/rccl/build/release:\$LD_LIBRARY_PATH '
        f'/home/dn/amd-dev/amd/rccl-tests/build/{test_name} '
        f'-b {minbytes} -e {maxbytes} -f {stepfactor} -g 1'
    )
    
    # Build the mpirun command
    cmd = [
        'mpirun',
        '-H', host_string,
        '-np', str(np),
        '--mca', 'oob_tcp_if_include', 'enp81s0f1np1',
        '--mca', 'btl_tcp_if_include', 'enp81s0f1np1',
        'bash', '-c', f'{bash_command}'
    ]
    
    # Print host distribution info
    if num_hosts > 1:
        print(f"Running on {num_hosts} host(s):")
        for i, host_ip in enumerate(ip_list):
            slots = procs_per_host + (1 if i < remainder else 0)
            print(f"  {host_ip}: {slots} process(es)")
        print()
    
    print(f"Executing command:")
    print(' '.join(cmd))
    print()
    print(f"Note: Output files will be created in: {output_dir}/")
    print(f"  - {prefix}.test.results.log (test output)")
    print(f"  - {prefix}.nccl_debug.log")
    print(f"  - {prefix}.topo.xml")
    print(f"  - {prefix}.graph.xml")
    print()
    sys.stdout.flush()  # Ensure output is displayed before subprocess starts
    
    # Execute the command and capture output
    try:
        with open(test_results_log, 'w') as log_file:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Read and display output in real-time while writing to file
            for line in process.stdout:
                print(line, end='')
                log_file.write(line)
                sys.stdout.flush()
            
            process.wait()
            returncode = process.returncode
        
        print(f"\nTest output saved to: {os.path.basename(test_results_log)}")
        return returncode
    except KeyboardInterrupt:
        print("\nCommand interrupted by user")
        return 130
    except Exception as e:
        print(f"Error executing command: {e}", file=sys.stderr)
        return 1


def parse_ip_list(ip_string):
    """Parse IP address or list of IP addresses.
    
    Supports formats:
    - Single IP: 172.30.160.200
    - Comma-separated: 172.30.160.200,172.30.160.201
    - Bracketed list: [172.30.160.200, 172.30.160.201]
    
    Returns:
        str or list: Single IP string or list of IP strings
    """
    ip_string = ip_string.strip()
    
    # Handle bracketed list format: [ip1, ip2, ...]
    if ip_string.startswith('[') and ip_string.endswith(']'):
        ip_string = ip_string[1:-1]  # Remove brackets
    
    # Check if comma-separated
    if ',' in ip_string:
        ip_list = [ip.strip() for ip in ip_string.split(',')]
        return ip_list
    else:
        return ip_string


def read_params_from_file(filepath):
    """Read test parameters from a space-separated CSV file.
    
    Expected format (one test per line):
    IP NP TEST_NAME MINBYTES MAXBYTES STEPFACTOR
    
    IP can be:
    - Single IP: 172.30.160.200
    - List: [172.30.160.200, 172.30.160.201]
    
    Examples:
    172.30.160.200 8 all_reduce_perf 4 10000000 2
    [172.30.160.200, 172.30.160.201] 8 all_reduce_perf 4 10000000 2
    """
    tests = []
    try:
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Handle bracketed IP list - need special parsing
                if line.startswith('['):
                    # Find the closing bracket
                    close_bracket = line.find(']')
                    if close_bracket == -1:
                        print(f"Warning: Line {line_num} has unclosed bracket. Skipping.", file=sys.stderr)
                        print(f"  Line: {line}", file=sys.stderr)
                        continue
                    
                    ip_part = line[:close_bracket+1]
                    rest = line[close_bracket+1:].strip()
                    parts = [ip_part] + rest.split()
                else:
                    parts = line.split()
                
                if len(parts) != 6:
                    print(f"Warning: Line {line_num} has {len(parts)} fields, expected 6. Skipping.", file=sys.stderr)
                    print(f"  Line: {line}", file=sys.stderr)
                    continue
                
                try:
                    test_params = {
                        'ip': parse_ip_list(parts[0]),
                        'np': int(parts[1]),
                        'test_name': parts[2],
                        'minbytes': int(parts[3]),
                        'maxbytes': int(parts[4]),
                        'stepfactor': int(parts[5])
                    }
                    tests.append(test_params)
                except ValueError as e:
                    print(f"Warning: Line {line_num} has invalid values: {e}. Skipping.", file=sys.stderr)
                    print(f"  Line: {line}", file=sys.stderr)
                    continue
        
        return tests
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file '{filepath}': {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Run NCCL/RCCL tests with mpirun',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with command-line arguments (single host)
  %(prog)s --ip 172.30.160.200 --np 8 --test-name all_reduce_perf \\
           -b 4 -e 10000000 -f 2
  
  # Run with multiple hosts (comma-separated)
  %(prog)s --ip 172.30.160.200,172.30.160.201 --np 8 --test-name all_reduce_perf \\
           -b 4 -e 10000000 -f 2
  
  # Run from parameter file
  %(prog)s --params-file tests.csv
  
Parameter file format (space-separated):
  IP NP TEST_NAME MINBYTES MAXBYTES STEPFACTOR
  
  Single host:
  172.30.160.200 8 all_reduce_perf 4 10000000 2
  
  Multiple hosts (bracketed list):
  [172.30.160.200, 172.30.160.201] 8 all_reduce_perf 4 10000000 2
"""
    )
    
    parser.add_argument(
        '--params-file',
        help='Path to CSV file with test parameters (space-separated)'
    )
    parser.add_argument(
        '--ip', 
        help='IP address(es) of target node(s). Single IP or comma-separated list (e.g., "172.30.160.200,172.30.160.201")'
    )
    parser.add_argument(
        '--np', 
        type=int,
        help='Number of processes to spawn'
    )
    parser.add_argument(
        '--test-name',
        help='Name of the test executable (e.g., all_reduce_perf)'
    )
    parser.add_argument(
        '-b', '--minbytes',
        type=int,
        help='Min size in bytes (starting byte size for the test)'
    )
    parser.add_argument(
        '-e', '--maxbytes',
        type=int,
        help='Max size in bytes (ending byte size for the test)'
    )
    parser.add_argument(
        '-f', '--stepfactor',
        type=int,
        help='Increment factor (multiplicative step factor, e.g., 2 for doubling, 4 for 4x growth)'
    )
    
    args = parser.parse_args()
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"nccl_test_run_{timestamp}"
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}/")
        print()
    except Exception as e:
        print(f"Error creating output directory '{output_dir}': {e}", file=sys.stderr)
        sys.exit(1)
    
    # Determine mode: file-based or argument-based
    if args.params_file:
        # File-based mode
        if any([args.ip, args.np, args.test_name, args.minbytes, args.maxbytes, args.stepfactor]):
            print("Warning: --params-file specified, ignoring other command-line arguments", file=sys.stderr)
        
        tests = read_params_from_file(args.params_file)
        if not tests:
            print("Error: No valid tests found in parameter file", file=sys.stderr)
            sys.exit(1)
        
        print(f"Found {len(tests)} test(s) in parameter file\n")
        
        # Run all tests
        failed_tests = []
        for i, test in enumerate(tests, 1):
            print(f"{'='*80}")
            print(f"Running test {i}/{len(tests)}")
            print(f"{'='*80}")
            returncode = run_single_test(**test, output_dir=output_dir)
            if returncode != 0:
                failed_tests.append((i, test, returncode))
            print()
        
        # Summary
        if failed_tests:
            print(f"\n{'='*80}")
            print(f"Summary: {len(failed_tests)}/{len(tests)} test(s) failed:")
            for test_num, test, code in failed_tests:
                print(f"  Test {test_num}: {test['ip']} {test['test_name']} (exit code: {code})")
            print(f"\nAll output files are in: {output_dir}/")
            sys.exit(1)
        else:
            print(f"\n{'='*80}")
            print(f"All {len(tests)} test(s) completed successfully!")
            print(f"All output files are in: {output_dir}/")
            sys.exit(0)
    else:
        # Argument-based mode (original behavior)
        required_args = ['ip', 'np', 'test_name', 'minbytes', 'maxbytes', 'stepfactor']
        missing = [arg for arg in required_args if getattr(args, arg.replace('-', '_')) is None]
        
        if missing:
            parser.error(f"the following arguments are required when not using --params-file: {', '.join('--' + arg for arg in missing)}")
        
        # Parse IP list from command-line argument
        ip_parsed = parse_ip_list(args.ip)
        
        returncode = run_single_test(
            ip_parsed, args.np, args.test_name,
            args.minbytes, args.maxbytes, args.stepfactor,
            output_dir
        )
        sys.exit(returncode)


if __name__ == '__main__':
    main()
