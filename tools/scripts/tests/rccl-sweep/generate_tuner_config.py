#!/usr/bin/env python3
"""
Generate RCCL Tuner Configuration

Converts optimized_metrics.csv to RCCL tuner.conf format.

Usage:
    python generate_tuner_config.py sweep_results/optimized_metrics.csv
    python generate_tuner_config.py sweep_results/optimized_metrics.csv -o my_tuner.conf
    python generate_tuner_config.py sweep_results/optimized_metrics.csv --include-algo-proto
"""

import argparse
import sys
from pathlib import Path

# Add script directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from autotune.config_generator import TunerConfigGenerator


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate RCCL tuner configuration from optimized metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - generates tuner.conf in same directory as input
  %(prog)s sweep_results/optimized_metrics.csv

  # Specify output path
  %(prog)s sweep_results/optimized_metrics.csv -o my_tuner.conf

  # Include algorithm and protocol (instead of -1 for defaults)
  %(prog)s sweep_results/optimized_metrics.csv --include-algo-proto

  # Don't merge adjacent size ranges
  %(prog)s sweep_results/optimized_metrics.csv --no-merge

  # Also generate a CSV report
  %(prog)s sweep_results/optimized_metrics.csv --csv-report tuning_report.csv

Output Format:
  The generated tuner.conf file follows the NCCL tuner CSV format:
  collective_type,min_bytes,max_bytes,algorithm,protocol,channels,nNodes,nRanks,numPipeOps,regBuff

  Example entries:
  allreduce,1,4096,-1,-1,32,2,16,-1,-1
  allgather,1048576,2097152,-1,-1,64,1,8,-1,-1
"""
    )
    
    parser.add_argument(
        'input_file',
        type=Path,
        help='Path to optimized_metrics.csv'
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=None,
        metavar='FILE',
        help='Output path for tuner.conf (default: <input_dir>/generated_tuner.conf)'
    )
    parser.add_argument(
        '--csv-report',
        type=Path,
        default=None,
        metavar='FILE',
        help='Also generate a CSV report at this path'
    )
    parser.add_argument(
        '--no-merge',
        action='store_true',
        help='Do not merge adjacent size ranges with identical configs'
    )
    parser.add_argument(
        '--include-algo-proto',
        action='store_true',
        help='Include algorithm and protocol in output (default: use -1 for both)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not args.input_file.exists():
        print(f"Error: Input file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)
    
    # Set default output path
    if args.output is None:
        args.output = args.input_file.parent / 'generated_tuner.conf'
    
    # Generate config
    generator = TunerConfigGenerator(merge_ranges=not args.no_merge)
    
    try:
        entries = generator.generate(
            args.input_file,
            args.output,
            include_algo_proto=args.include_algo_proto,
        )
        
        print(f"Generated tuner config: {args.output}")
        print(f"Total entries: {len(entries)}")
        
        if args.verbose:
            # Show summary by collective
            by_collective = {}
            for entry in entries:
                coll = entry.collective
                if coll not in by_collective:
                    by_collective[coll] = 0
                by_collective[coll] += 1
            
            print("\nEntries by collective:")
            for coll in sorted(by_collective.keys()):
                print(f"  {coll}: {by_collective[coll]}")
        
        # Generate CSV report if requested
        if args.csv_report:
            generator.generate_csv_report(entries, args.csv_report)
            print(f"Generated CSV report: {args.csv_report}")
        
        print("\n✓ Configuration generated successfully!")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

