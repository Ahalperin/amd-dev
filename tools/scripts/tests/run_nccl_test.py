#!/usr/bin/env python3
"""
Script to run NCCL/RCCL tests with mpirun across multiple nodes.
"""

import argparse
import subprocess
import sys
<<<<<<< HEAD


def main():
    parser = argparse.ArgumentParser(
        description='Run NCCL/RCCL tests with mpirun',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--ip', 
        required=True,
        help='IP address of the target node'
    )
    parser.add_argument(
        '--np', 
        type=int,
        required=True,
        help='Number of processes to spawn'
    )
    parser.add_argument(
        '--test-name',
        required=True,
        help='Name of the test executable (e.g., all_reduce_perf)'
    )
    parser.add_argument(
        '--from-bytes',
        type=int,
        required=True,
        help='Starting byte size for the test'
    )
    parser.add_argument(
        '--to-bytes',
        type=int,
        required=True,
        help='Ending byte size for the test'
    )
    parser.add_argument(
        '--step-bytes',
        type=int,
        required=True,
        help='Byte step size for the test'
    )
    
    args = parser.parse_args()
    
    # Build the mpirun command
    cmd = [
        'mpirun',
        '-H', f'{args.ip}:{args.np}',
        '-np', str(args.np),
=======
import os


def run_single_test(ip, np, test_name, from_bytes, to_bytes, step_factor):
    """Run a single NCCL test with the given parameters."""
    # Build the mpirun command
    cmd = [
        'mpirun',
        '-H', f'{ip}:{np}',
        '-np', str(np),
>>>>>>> 6b42e4b (test auto initial)
        '--mca', 'oob_tcp_if_include', 'enp81s0f1np1',
        '--mca', 'btl_tcp_if_include', 'enp81s0f1np1',
        'bash', '-c',
        f'NCCL_DEBUG=INFO NCCL_DEBUG_FILE=nccl_debug.log NCCL_TOPO_DUMP_FILE=topo.xml '
        f'NCCL_GRAPH_DUMP_FILE=graph.xml NCCL_DEBUG_SUBSYS=GRAPH,ALL '
<<<<<<< HEAD
        f'/home/dn/amd-dev/amd/rccl-tests/build/{args.test_name} '
        f'-b {args.from_bytes} -e {args.to_bytes} -f {args.step_bytes} -g 1'
=======
        f'/home/dn/amd-dev/amd/rccl-tests/build/{test_name} '
        f'-b {from_bytes} -e {to_bytes} -f {step_factor} -g 1'
>>>>>>> 6b42e4b (test auto initial)
    ]
    
    print(f"Executing command:")
    print(' '.join(cmd))
    print()
<<<<<<< HEAD
    
    # Execute the command
    try:
        result = subprocess.run(cmd, check=False)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\nCommand interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Error executing command: {e}", file=sys.stderr)
        sys.exit(1)
=======
    print(f"Note: Output files will be created in the current directory:")
    print(f"  - test.results.log (test output)")
    print(f"  - nccl_debug.log")
    print(f"  - topo.xml")
    print(f"  - graph.xml")
    print()
    sys.stdout.flush()  # Ensure output is displayed before subprocess starts
    
    # Execute the command and capture output
    try:
        with open('test.results.log', 'w') as log_file:
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
        
        print(f"\nTest output saved to: test.results.log")
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
            returncode = run_single_test(**test)
            if returncode != 0:
                failed_tests.append((i, test, returncode))
            print()
        
        # Summary
        if failed_tests:
            print(f"\n{'='*80}")
            print(f"Summary: {len(failed_tests)}/{len(tests)} test(s) failed:")
            for test_num, test, code in failed_tests:
                print(f"  Test {test_num}: {test['ip']} {test['test_name']} (exit code: {code})")
            sys.exit(1)
        else:
            print(f"\n{'='*80}")
            print(f"All {len(tests)} test(s) completed successfully!")
            sys.exit(0)
    else:
        # Argument-based mode (original behavior)
        required_args = ['ip', 'np', 'test_name', 'from_bytes', 'to_bytes', 'step_factor']
        missing = [arg for arg in required_args if getattr(args, arg.replace('-', '_')) is None]
        
        if missing:
            parser.error(f"the following arguments are required when not using --params-file: {', '.join('--' + arg for arg in missing)}")
        
        returncode = run_single_test(
            args.ip, args.np, args.test_name,
            args.from_bytes, args.to_bytes, args.step_factor
        )
        sys.exit(returncode)
>>>>>>> 6b42e4b (test auto initial)


if __name__ == '__main__':
    main()

