#!/usr/bin/env python3
"""
RCCL Auto-Tuner

Automated RCCL performance tuning pipeline that:
1. Runs systematic sweeps across configurations
2. Merges and optimizes metrics
3. Detects performance hotspots
4. Runs targeted sweeps to refine hotspot areas
5. Generates tuner configuration file

Usage:
    # Basic usage with defaults
    python rccl_autotune.py -n 1-2 -c 32:256:32 --output-dir ./tuning_results

    # With hotspot refinement
    python rccl_autotune.py -n 1-4 -c 4:64:4 \\
        --max-iterations 3 \\
        --hotspot-threshold 0.10 \\
        --tuner-output ./my_tuner.conf

    # From config file
    python rccl_autotune.py --config autotune_config.yaml

    # Dry run - show commands without executing
    python rccl_autotune.py -n 1-2 -c 32:64:8 --dry-run
"""

import argparse
import sys
from pathlib import Path

# Add script directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from autotune import AutoTunePipeline
from autotune.pipeline import PipelineConfig, load_config_yaml


def parse_step_ranges(ranges_str: str) -> list:
    """
    Parse step ranges string into list of (min_size, max_size, step_size) tuples.
    
    Format: "START:END:STEP,START:END:STEP,..."
    Example: "4K:1M:4K,1M:64M:256K,64M:512M:4M"
    
    Returns:
        List of tuples: [(min_size, max_size, step_size), ...]
    """
    ranges = []
    for part in ranges_str.split(','):
        part = part.strip()
        if not part:
            continue
        components = part.split(':')
        if len(components) != 3:
            raise ValueError(
                f"Invalid step range format '{part}'. "
                f"Expected START:END:STEP (e.g., '4K:1M:4K')"
            )
        min_size, max_size, step_size = [c.strip() for c in components]
        ranges.append((min_size, max_size, step_size))
    
    if not ranges:
        raise ValueError("No valid step ranges found")
    
    return ranges


def main():
    """Main entry point for the RCCL auto-tuner."""
    parser = argparse.ArgumentParser(
        description='RCCL Auto-Tuner - Automated performance tuning for RCCL collectives',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with NCCL defaults (logarithmic size progression)
  %(prog)s -n 1-2 -c 32:256:32

  # Sweep specific collectives
  %(prog)s -n 1-2 -c 32:64:8 --collective all_reduce,all_gather

  # Sweep explicit channel values
  %(prog)s -n 1-2 -c 1,5,16,32,64,128 --collective all_reduce

  # With fixed step size for fine-grained small message sweeps
  %(prog)s -n 1-2 -c 32:64:8 --min-size 4K --max-size 256M --step-size 4K

  # With adaptive step ranges for efficient wide-range sweeps
  %(prog)s -n 1-2 -c 32:64:8 --step-ranges "4K:1M:4K,1M:64M:256K,64M:512M:4M"

  # With hotspot refinement
  %(prog)s -n 1-4 -c 4:64:4 --max-iterations 3 --hotspot-threshold 0.10

  # Specify output paths
  %(prog)s -n 1-2 -c 32:64:8 \\
      --output-dir ./tuning_results \\
      --tuner-output ./my_tuner.conf

  # From YAML config file
  %(prog)s --config autotune_config.yaml

  # Dry run - show plan without executing
  %(prog)s -n 1-2 -c 32:64:8 --dry-run
"""
    )
    
    # Config file (takes precedence)
    parser.add_argument(
        '--config',
        type=Path,
        metavar='FILE',
        help='YAML configuration file (overrides command-line args)'
    )
    
    # Sweep parameters
    sweep_group = parser.add_argument_group('Sweep Parameters')
    sweep_group.add_argument(
        '--nodes', '-n',
        default='1-2',
        metavar='RANGE',
        help='Node count or range: N or MIN-MAX (default: 1-2)'
    )
    sweep_group.add_argument(
        '--channels', '-c',
        default='32:256:32',
        metavar='RANGE|LIST',
        help='Channel sweep: range START:END:STEP (e.g., 32:256:32) or list V1,V2,... (e.g., 1,5,16,32,64,128). Default: 32:256:32'
    )
    sweep_group.add_argument(
        '--collective',
        metavar='NAMES',
        help='Comma-separated collectives (default: all_reduce,all_gather). '
             'Options: all_reduce, reduce_scatter, all_gather, alltoall, broadcast, reduce'
    )
    sweep_group.add_argument(
        '--algo',
        default='RING,TREE',
        metavar='ALGOS',
        help='Comma-separated algorithms (default: RING,TREE)'
    )
    sweep_group.add_argument(
        '--proto',
        default='SIMPLE',
        metavar='PROTOS',
        help='Comma-separated protocols (default: SIMPLE)'
    )
    sweep_group.add_argument(
        '--min-size',
        default='1M',
        metavar='SIZE',
        help='Minimum message size (default: 1M)'
    )
    sweep_group.add_argument(
        '--max-size',
        default='512M',
        metavar='SIZE',
        help='Maximum message size (default: 512M)'
    )
    sweep_group.add_argument(
        '--step-size',
        metavar='SIZE',
        help='Fixed step size for message sizes (e.g., 1M, 4K). '
             'If not set, uses doubling factor (logarithmic progression)'
    )
    sweep_group.add_argument(
        '--step-ranges',
        metavar='RANGES',
        help='Size-based step ranges: START:END:STEP,... (e.g., "4K:1M:4K,1M:64M:256K,64M:512M:4M"). '
             'When specified, --min-size, --max-size, and --step-size are ignored.'
    )
    sweep_group.add_argument(
        '--servers', '-s',
        default='./servers.txt',
        metavar='FILE',
        help='Path to servers.txt file (default: ./servers.txt)'
    )
    
    # Hotspot parameters
    hotspot_group = parser.add_argument_group('Hotspot Detection')
    hotspot_group.add_argument(
        '--max-iterations',
        type=int,
        default=3,
        metavar='N',
        help='Maximum hotspot refinement iterations (default: 3)'
    )
    hotspot_group.add_argument(
        '--hotspot-threshold',
        type=float,
        default=0.10,
        metavar='PCT',
        help='Drop threshold for hotspot detection (default: 0.10 = 10%%)'
    )
    hotspot_group.add_argument(
        '--min-drop-gbps',
        type=float,
        default=0.0,
        metavar='GBPS',
        help='Minimum absolute busbw drop in GB/s (default: 0)'
    )
    
    # Output parameters
    output_group = parser.add_argument_group('Output')
    output_group.add_argument(
        '--output-dir', '-o',
        type=Path,
        default='./sweep_results',
        metavar='DIR',
        help='Output directory for results (default: ./sweep_results)'
    )
    output_group.add_argument(
        '--tuner-output',
        type=Path,
        metavar='FILE',
        help='Path for generated tuner CSV (default: <output-dir>/generated_tuner.csv)'
    )
    output_group.add_argument(
        '--report-output',
        type=Path,
        metavar='FILE',
        help='Path for tuning report CSV (default: <output-dir>/tuning_report.csv)'
    )
    
    # Execution options
    exec_group = parser.add_argument_group('Execution Options')
    exec_group.add_argument(
        '--dry-run', '-d',
        action='store_true',
        help='Show commands without executing sweeps'
    )
    exec_group.add_argument(
        '--combos-config',
        type=Path,
        metavar='FILE',
        help='Path to unsupported_combos.yaml (default: auto-detect in script directory)'
    )
    exec_group.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output (default)'
    )
    exec_group.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Reduce output verbosity'
    )
    
    args = parser.parse_args()
    
    # Build configuration
    if args.config:
        if not args.config.exists():
            print(f"Error: Config file not found: {args.config}", file=sys.stderr)
            sys.exit(1)
        config = load_config_yaml(args.config)
        print(f"Loaded configuration from: {args.config}")
    else:
        # Parse collectives
        if args.collective:
            collectives = [c.strip() for c in args.collective.split(',')]
            # Normalize names
            collectives = [
                c if c.endswith('_perf') else f"{c.replace('-', '_')}_perf"
                for c in collectives
            ]
        else:
            collectives = ['all_reduce_perf', 'all_gather_perf']
        
        # Parse algorithms
        algos = [a.strip().upper() for a in args.algo.split(',')]
        
        # Parse protocols
        protos = [p.strip().upper() for p in args.proto.split(',')]
        
        # Parse step ranges if provided
        step_ranges = None
        if args.step_ranges:
            try:
                step_ranges = parse_step_ranges(args.step_ranges)
            except ValueError as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)
            
            # Warn if min-size/max-size/step-size are also provided
            if args.min_size != '1M' or args.max_size != '512M' or args.step_size:
                print("Warning: --min-size, --max-size, and --step-size are ignored when --step-ranges is specified",
                      file=sys.stderr)
        
        config = PipelineConfig(
            output_dir=Path(args.output_dir),
            nodes=args.nodes,
            channels=args.channels,
            collectives=collectives,
            algos=algos,
            protos=protos,
            min_size=args.min_size,
            max_size=args.max_size,
            step_size=args.step_size,
            step_ranges=step_ranges,
            hotspot_threshold=args.hotspot_threshold,
            hotspot_min_drop_gbps=args.min_drop_gbps,
            max_iterations=args.max_iterations,
            tuner_output=args.tuner_output,
            report_output=args.report_output,
            servers_file=args.servers,
            combos_config=args.combos_config,
        )
    
    # Create and run pipeline
    pipeline = AutoTunePipeline(
        config=config,
        verbose=not args.quiet,
        dry_run=args.dry_run,
    )
    
    result = pipeline.run()
    
    # Exit with appropriate code
    if not result.success:
        print(f"\nPipeline failed: {result.error_message}", file=sys.stderr)
        sys.exit(1)
    
    print("\n✓ Auto-tuning completed successfully!")
    sys.exit(0)


if __name__ == '__main__':
    main()

