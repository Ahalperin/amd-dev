#!/usr/bin/env python3
"""
Script to run NCCL/RCCL tests with mpirun across multiple nodes.
"""

import argparse
import subprocess
import sys


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
        '--mca', 'oob_tcp_if_include', 'enp81s0f1np1',
        '--mca', 'btl_tcp_if_include', 'enp81s0f1np1',
        'bash', '-c',
        f'NCCL_DEBUG=INFO NCCL_DEBUG_FILE=nccl_debug.log NCCL_TOPO_DUMP_FILE=topo.xml '
        f'NCCL_GRAPH_DUMP_FILE=graph.xml NCCL_DEBUG_SUBSYS=GRAPH,ALL '
        f'/home/dn/amd-dev/amd/rccl-tests/build/{args.test_name} '
        f'-b {args.from_bytes} -e {args.to_bytes} -f {args.step_bytes} -g 1'
    ]
    
    print(f"Executing command:")
    print(' '.join(cmd))
    print()
    
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


if __name__ == '__main__':
    main()

