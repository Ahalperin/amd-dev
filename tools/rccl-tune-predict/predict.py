#!/usr/bin/env python3
"""Predict optimal RCCL configuration using trained model."""

import argparse
import sys
from pathlib import Path

# Add parent directory for imports when running as script
sys.path.insert(0, str(Path(__file__).parent))

from core.busbw_predictor import BusbwPredictor
from core.search import find_optimal_config, find_optimal_configs_batch
from core.utils import format_size


def parse_sizes(sizes_str: str) -> list:
    """Parse comma-separated size values."""
    sizes = []
    for s in sizes_str.split(','):
        s = s.strip()
        # Handle suffixes like K, M, G
        multiplier = 1
        if s.upper().endswith('G'):
            multiplier = 1024**3
            s = s[:-1]
        elif s.upper().endswith('M'):
            multiplier = 1024**2
            s = s[:-1]
        elif s.upper().endswith('K'):
            multiplier = 1024
            s = s[:-1]
        sizes.append(int(float(s) * multiplier))
    return sizes


def main():
    parser = argparse.ArgumentParser(
        description='Predict optimal RCCL configuration using trained ML model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single prediction
  python predict.py --collective all_reduce --nodes 4 --gpus 32 --size 16M

  # Multiple sizes
  python predict.py --collective all_gather --nodes 4 --gpus 32 \\
      --sizes 1M,16M,64M,512M

  # Limit search to specific algorithms
  python predict.py --collective all_reduce --nodes 2 --gpus 16 --size 32M \\
      --algos RING,TREE

Supported collectives: all_reduce, all_gather, reduce_scatter, alltoall, broadcast, reduce
"""
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='models/busbw_model.pkl',
        help='Path to trained model (default: models/busbw_model.pkl)'
    )
    parser.add_argument(
        '--collective', '-c',
        type=str,
        required=True,
        help='Collective type (all_reduce, all_gather, reduce_scatter, alltoall, broadcast, reduce)'
    )
    parser.add_argument(
        '--nodes', '-n',
        type=int,
        required=True,
        help='Number of nodes'
    )
    parser.add_argument(
        '--gpus', '-g',
        type=int,
        required=True,
        help='Total number of GPUs'
    )
    parser.add_argument(
        '--size', '-s',
        type=str,
        help='Message size (e.g., 16M, 1048576). Use --sizes for multiple.'
    )
    parser.add_argument(
        '--sizes',
        type=str,
        help='Comma-separated message sizes (e.g., 1M,16M,64M,512M)'
    )
    parser.add_argument(
        '--max-nchannels',
        type=int,
        default=32,
        help='Maximum number of channels to search (default: 32)'
    )
    parser.add_argument(
        '--algos',
        type=str,
        help='Comma-separated list of algorithms to search (default: all)'
    )
    parser.add_argument(
        '--protos',
        type=str,
        help='Comma-separated list of protocols to search (default: all)'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON'
    )
    
    args = parser.parse_args()
    
    # Validate size arguments
    if not args.size and not args.sizes:
        parser.error("Either --size or --sizes is required")
    
    # Load model
    try:
        model = BusbwPredictor.load(args.model)
    except FileNotFoundError:
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Parse search constraints
    algos = args.algos.split(',') if args.algos else None
    protos = args.protos.split(',') if args.protos else None
    
    # Parse sizes
    if args.sizes:
        sizes = parse_sizes(args.sizes)
    else:
        sizes = parse_sizes(args.size)
    
    # Run predictions
    results = find_optimal_configs_batch(
        model,
        args.collective,
        args.nodes,
        args.gpus,
        sizes,
        max_nchannels=args.max_nchannels,
        algos=algos,
        protos=protos,
    )
    
    # Output results
    if args.json:
        import json
        print(json.dumps(results, indent=2))
    elif len(results) == 1:
        # Single result - detailed output
        r = results[0]
        size = sizes[0]
        print(f"\nOptimal config for {args.collective} ({args.nodes} nodes, {args.gpus} GPUs, {format_size(size)}):")
        print(f"  algo:           {r['algo']}")
        print(f"  proto:          {r['proto']}")
        print(f"  nchannels:      {r['nchannels']}")
        print(f"  predicted_busbw: {r['predicted_busbw']:.2f} GB/s")
    else:
        # Multiple results - table output
        print(f"\nOptimal configs for {args.collective} ({args.nodes} nodes, {args.gpus} GPUs):")
        print("-" * 70)
        print(f"{'Size':>12s}  {'Algo':>10s}  {'Proto':>8s}  {'nchannels':>10s}  {'BusBW (GB/s)':>12s}")
        print("-" * 70)
        for r, size in zip(results, sizes):
            print(f"{format_size(size):>12s}  {r['algo']:>10s}  {r['proto']:>8s}  {r['nchannels']:>10d}  {r['predicted_busbw']:>12.2f}")
        print("-" * 70)


if __name__ == '__main__':
    main()


