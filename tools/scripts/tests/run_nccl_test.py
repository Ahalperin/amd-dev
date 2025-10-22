#!/usr/bin/env python3
"""
Script to run NCCL/RCCL tests with mpirun across multiple nodes.
"""

import argparse
import subprocess
import sys
import os
from datetime import datetime


def run_single_test(ip, np, test_name, from_bytes, to_bytes, step_factor, output_dir):
    """Run a single NCCL test with the given parameters."""
    # Create prefix from test parameters
    prefix = f"{test_name}_{from_bytes}_{to_bytes}_{step_factor}"
    
    # Define output file names with prefix inside output directory
    test_results_log = os.path.join(output_dir, f"{prefix}.test.results.log")
    nccl_debug_log = os.path.join(output_dir, f"{prefix}.nccl_debug.log")
    topo_xml = os.path.join(output_dir, f"{prefix}.topo.xml")
    graph_xml = os.path.join(output_dir, f"{prefix}.graph.xml")
    
    # Build the mpirun command
    cmd = [
        'mpirun',
        '-H', f'{ip}:{np}',
        '-np', str(np),
        '--mca', 'oob_tcp_if_include', 'enp81s0f1np1',
        '--mca', 'btl_tcp_if_include', 'enp81s0f1np1',
        'bash', '-c',
        f'NCCL_DEBUG=INFO NCCL_DEBUG_FILE={nccl_debug_log} NCCL_TOPO_DUMP_FILE={topo_xml} '
        f'NCCL_GRAPH_DUMP_FILE={graph_xml} NCCL_DEBUG_SUBSYS=GRAPH,ALL '
        f'/home/dn/amd-dev/amd/rccl-tests/build/{test_name} '
        f'-b {from_bytes} -e {to_bytes} -f {step_factor} -g 1'
    ]
    
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


def read_params_from_file(filepath):
    """Read test parameters from a space-separated CSV file.
    
    Expected format (one test per line):
    IP NP TEST_NAME FROM_BYTES TO_BYTES STEP_FACTOR
    
    Example:
    172.30.160.200 8 all_reduce_perf 4 10000000 2
    """
    tests = []
    try:
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) != 6:
                    print(f"Warning: Line {line_num} has {len(parts)} fields, expected 6. Skipping.", file=sys.stderr)
                    print(f"  Line: {line}", file=sys.stderr)
                    continue
                
                try:
                    test_params = {
                        'ip': parts[0],
                        'np': int(parts[1]),
                        'test_name': parts[2],
                        'from_bytes': int(parts[3]),
                        'to_bytes': int(parts[4]),
                        'step_factor': int(parts[5])
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
  # Run with command-line arguments
  %(prog)s --ip 172.30.160.200 --np 8 --test-name all_reduce_perf \\
           --from-bytes 4 --to-bytes 10000000 --step-factor 2
  
  # Run from parameter file
  %(prog)s --params-file tests.csv
  
Parameter file format (space-separated):
  IP NP TEST_NAME FROM_BYTES TO_BYTES STEP_FACTOR
  172.30.160.200 8 all_reduce_perf 4 10000000 2
  172.30.160.201 2 all_gather_perf 1000 100000000 4
"""
    )
    
    parser.add_argument(
        '--params-file',
        help='Path to CSV file with test parameters (space-separated)'
    )
    parser.add_argument(
        '--ip', 
        help='IP address of the target node'
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
        '--from-bytes',
        type=int,
        help='Starting byte size for the test'
    )
    parser.add_argument(
        '--to-bytes',
        type=int,
        help='Ending byte size for the test'
    )
    parser.add_argument(
        '--step-factor',
        type=int,
        help='Multiplicative step factor for the test (e.g., 2 for doubling, 4 for 4x growth)'
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
        if any([args.ip, args.np, args.test_name, args.from_bytes, args.to_bytes, args.step_factor]):
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
        required_args = ['ip', 'np', 'test_name', 'from_bytes', 'to_bytes', 'step_factor']
        missing = [arg for arg in required_args if getattr(args, arg.replace('-', '_')) is None]
        
        if missing:
            parser.error(f"the following arguments are required when not using --params-file: {', '.join('--' + arg for arg in missing)}")
        
        returncode = run_single_test(
            args.ip, args.np, args.test_name,
            args.from_bytes, args.to_bytes, args.step_factor,
            output_dir
        )
        sys.exit(returncode)


if __name__ == '__main__':
    main()
