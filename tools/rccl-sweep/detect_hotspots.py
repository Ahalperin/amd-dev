#!/usr/bin/env python3
"""
RCCL Hot-spot Detection Tool

Analyzes RCCL sweep baseline data to identify message size ranges where busbw
performance drops unexpectedly below expected levels, enabling targeted tuning
of algo/proto/nchannels for those specific ranges.

Usage:
    python detect_hotspots.py merged_metrics.csv -o hotspots_report.csv --threshold 0.1 --verbose
"""

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def format_bytes(size_bytes: int) -> str:
    """Convert bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.0f}{unit}" if size_bytes == int(size_bytes) else f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}TB"


@dataclass
class HotSpot:
    """Represents a detected hot-spot range."""
    collective: str
    num_nodes: int
    num_gpus: int
    start_bytes: int
    end_bytes: int
    expected_busbw: float  # Running max before drop
    actual_busbw_min: float  # Minimum busbw in hot-spot
    actual_busbw_max: float  # Maximum busbw in hot-spot
    drop_percent_max: float  # Maximum drop percentage
    drop_percent_min: float  # Minimum drop percentage
    algo: str
    proto: str
    nchannels: int
    data_points: int = 1  # Number of data points in this hot-spot


@dataclass
class MetricRow:
    """Represents a single row from the metrics CSV."""
    collective: str
    num_nodes: int
    num_gpus: int
    size_bytes: int
    busbw_ip: float
    algo: str
    proto: str
    nchannels: int


def parse_metrics_csv(file_path: Path) -> list[MetricRow]:
    """Parse the merged_metrics.csv file."""
    rows = []
    
    with open(file_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            try:
                # Parse required fields
                metric = MetricRow(
                    collective=row['collective'],
                    num_nodes=int(row['num_nodes']),
                    num_gpus=int(row['num_gpus']),
                    size_bytes=int(row['size_bytes']),
                    busbw_ip=float(row['busbw_ip']),
                    algo=row.get('algo', ''),
                    proto=row.get('proto', ''),
                    nchannels=int(row.get('nchannels', 0) or 0),
                )
                rows.append(metric)
            except (ValueError, KeyError) as e:
                # Skip malformed rows
                continue
    
    return rows


def detect_hotspots(
    rows: list[MetricRow],
    threshold: float = 0.10,
    min_drop_gbps: float = 0.0,
) -> list[HotSpot]:
    """
    Detect hot-spots where busbw drops below expected levels.
    
    Algorithm:
    1. Group rows by (collective, num_nodes, num_gpus)
    2. Sort each group by size_bytes
    3. Track max_busbw_seen as running maximum
    4. If busbw_ip < max_busbw_seen * (1 - threshold), mark as hot-spot
    5. Merge contiguous hot-spot ranges
    
    Args:
        rows: Parsed metric rows
        threshold: Drop percentage threshold (0.10 = 10%)
        min_drop_gbps: Minimum absolute busbw drop in GB/s to consider
        
    Returns:
        List of detected hot-spots
    """
    # Group by (collective, num_nodes, num_gpus)
    groups = defaultdict(list)
    for row in rows:
        key = (row.collective, row.num_nodes, row.num_gpus)
        groups[key].append(row)
    
    hotspots = []
    
    for (collective, num_nodes, num_gpus), group_rows in groups.items():
        # Sort by size_bytes
        group_rows.sort(key=lambda r: r.size_bytes)
        
        max_busbw_seen = 0.0
        current_hotspot: Optional[HotSpot] = None
        
        for row in group_rows:
            # Update running maximum
            if row.busbw_ip > max_busbw_seen:
                # Performance recovered - close any current hot-spot
                if current_hotspot is not None:
                    hotspots.append(current_hotspot)
                    current_hotspot = None
                max_busbw_seen = row.busbw_ip
            else:
                # Check if this is a hot-spot
                drop_threshold = max_busbw_seen * (1 - threshold)
                absolute_drop = max_busbw_seen - row.busbw_ip
                
                if row.busbw_ip < drop_threshold and absolute_drop >= min_drop_gbps:
                    drop_percent = (max_busbw_seen - row.busbw_ip) / max_busbw_seen * 100
                    
                    if current_hotspot is None:
                        # Start new hot-spot
                        current_hotspot = HotSpot(
                            collective=collective,
                            num_nodes=num_nodes,
                            num_gpus=num_gpus,
                            start_bytes=row.size_bytes,
                            end_bytes=row.size_bytes,
                            expected_busbw=max_busbw_seen,
                            actual_busbw_min=row.busbw_ip,
                            actual_busbw_max=row.busbw_ip,
                            drop_percent_max=drop_percent,
                            drop_percent_min=drop_percent,
                            algo=row.algo,
                            proto=row.proto,
                            nchannels=row.nchannels,
                            data_points=1,
                        )
                    else:
                        # Extend current hot-spot
                        current_hotspot.end_bytes = row.size_bytes
                        current_hotspot.actual_busbw_min = min(
                            current_hotspot.actual_busbw_min, row.busbw_ip
                        )
                        current_hotspot.actual_busbw_max = max(
                            current_hotspot.actual_busbw_max, row.busbw_ip
                        )
                        current_hotspot.drop_percent_max = max(
                            current_hotspot.drop_percent_max, drop_percent
                        )
                        current_hotspot.drop_percent_min = min(
                            current_hotspot.drop_percent_min, drop_percent
                        )
                        current_hotspot.data_points += 1
                else:
                    # Not a hot-spot (within threshold), but didn't exceed max
                    # Close any current hot-spot
                    if current_hotspot is not None:
                        hotspots.append(current_hotspot)
                        current_hotspot = None
        
        # Don't forget to close any remaining hot-spot
        if current_hotspot is not None:
            hotspots.append(current_hotspot)
    
    # Sort hotspots by collective, num_nodes, start_bytes
    hotspots.sort(key=lambda h: (h.collective, h.num_nodes, h.start_bytes))
    
    return hotspots


def write_csv_report(hotspots: list[HotSpot], output_path: Path) -> None:
    """Write hot-spots to CSV file."""
    fieldnames = [
        'collective',
        'num_nodes',
        'num_gpus',
        'hotspot_start_bytes',
        'hotspot_end_bytes',
        'hotspot_start_human',
        'hotspot_end_human',
        'expected_busbw',
        'actual_busbw_min',
        'actual_busbw_max',
        'drop_percent_min',
        'drop_percent_max',
        'current_algo',
        'current_proto',
        'current_nchannels',
        'data_points',
    ]
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for hs in hotspots:
            writer.writerow({
                'collective': hs.collective,
                'num_nodes': hs.num_nodes,
                'num_gpus': hs.num_gpus,
                'hotspot_start_bytes': hs.start_bytes,
                'hotspot_end_bytes': hs.end_bytes,
                'hotspot_start_human': format_bytes(hs.start_bytes),
                'hotspot_end_human': format_bytes(hs.end_bytes),
                'expected_busbw': f"{hs.expected_busbw:.2f}",
                'actual_busbw_min': f"{hs.actual_busbw_min:.2f}",
                'actual_busbw_max': f"{hs.actual_busbw_max:.2f}",
                'drop_percent_min': f"{hs.drop_percent_min:.1f}",
                'drop_percent_max': f"{hs.drop_percent_max:.1f}",
                'current_algo': hs.algo,
                'current_proto': hs.proto,
                'current_nchannels': hs.nchannels,
                'data_points': hs.data_points,
            })


def print_summary(hotspots: list[HotSpot]) -> None:
    """Print human-readable summary of hot-spots."""
    if not hotspots:
        print("No hot-spots detected.")
        return
    
    print(f"\n{'='*70}")
    print(f"  RCCL Hot-spot Detection Report")
    print(f"{'='*70}\n")
    print(f"Total hot-spots detected: {len(hotspots)}\n")
    
    # Group by collective for organized output
    by_collective = defaultdict(list)
    for hs in hotspots:
        by_collective[hs.collective].append(hs)
    
    for collective in sorted(by_collective.keys()):
        coll_hotspots = by_collective[collective]
        coll_name = collective.replace('_perf', '').replace('_', ' ').title()
        
        print(f"{coll_name}:")
        print(f"{'-'*50}")
        
        for hs in coll_hotspots:
            print(f"  {hs.num_nodes} node(s), {hs.num_gpus} GPUs:")
            print(f"    Range: {format_bytes(hs.start_bytes)} - {format_bytes(hs.end_bytes)}")
            print(f"    Expected busbw: {hs.expected_busbw:.2f} GB/s")
            print(f"    Actual busbw: {hs.actual_busbw_min:.2f} - {hs.actual_busbw_max:.2f} GB/s")
            print(f"    Drop: {hs.drop_percent_min:.1f}% - {hs.drop_percent_max:.1f}%")
            print(f"    Current config: algo={hs.algo}, proto={hs.proto}, nchannels={hs.nchannels}")
            print(f"    Data points: {hs.data_points}")
            print()
        
        print()


def generate_sweep_suggestions(hotspots: list[HotSpot]) -> list[dict]:
    """
    Generate sweep suggestions for each hotspot.
    
    Returns list of suggested sweep configurations to run for refinement.
    """
    suggestions = []
    
    # Alternative algorithms to try
    algo_alternatives = {
        'RING': ['TREE'],
        'TREE': ['RING'],
        'Direct': ['RING', 'TREE'],
    }
    
    for hs in hotspots:
        # Calculate channel range expansion
        current_channels = hs.nchannels or 32
        min_ch = max(4, current_channels // 2)
        max_ch = min(256, current_channels * 2)
        step = max(4, (max_ch - min_ch) // 8)
        
        suggestion = {
            'collective': hs.collective.replace('_perf', ''),
            'nodes': hs.num_nodes,
            'min_size': format_bytes(hs.start_bytes),
            'max_size': format_bytes(hs.end_bytes),
            'channels': f"{min_ch}:{max_ch}:{step}",
            'suggested_algos': algo_alternatives.get(hs.algo, ['RING', 'TREE']),
            'current_algo': hs.algo,
            'current_channels': hs.nchannels,
            'drop_percent': hs.drop_percent_max,
        }
        suggestions.append(suggestion)
    
    return suggestions


def hotspot_to_dict(hs: HotSpot) -> dict:
    """Convert a HotSpot to a dictionary for JSON serialization."""
    return {
        'collective': hs.collective,
        'num_nodes': hs.num_nodes,
        'num_gpus': hs.num_gpus,
        'start_bytes': hs.start_bytes,
        'end_bytes': hs.end_bytes,
        'start_bytes_human': format_bytes(hs.start_bytes),
        'end_bytes_human': format_bytes(hs.end_bytes),
        'expected_busbw': round(hs.expected_busbw, 2),
        'actual_busbw_min': round(hs.actual_busbw_min, 2),
        'actual_busbw_max': round(hs.actual_busbw_max, 2),
        'drop_percent_min': round(hs.drop_percent_min, 1),
        'drop_percent_max': round(hs.drop_percent_max, 1),
        'algo': hs.algo,
        'proto': hs.proto,
        'nchannels': hs.nchannels,
        'data_points': hs.data_points,
    }


def write_json_report(hotspots: list[HotSpot], output_path: Path, include_suggestions: bool = True) -> None:
    """Write hot-spots to JSON file."""
    import json
    
    data = {
        'hotspots': [hotspot_to_dict(hs) for hs in hotspots],
        'count': len(hotspots),
    }
    
    if include_suggestions:
        data['sweep_suggestions'] = generate_sweep_suggestions(hotspots)
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description='Detect RCCL performance hot-spots from sweep metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default 10% threshold
  python detect_hotspots.py sweep_results/merged_metrics.csv

  # Custom threshold and output file
  python detect_hotspots.py merged_metrics.csv -o hotspots.csv --threshold 0.15

  # Verbose output with minimum drop filter
  python detect_hotspots.py merged_metrics.csv --threshold 0.10 --min-drop 1.0 --verbose

  # Output as JSON for programmatic use
  python detect_hotspots.py merged_metrics.csv --json -o hotspots.json

  # JSON output with sweep suggestions
  python detect_hotspots.py merged_metrics.csv --json --suggest-sweeps
        """
    )
    
    parser.add_argument(
        'input_file',
        type=Path,
        help='Path to merged_metrics.csv file'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=None,
        help='Output path for report (default: <input_dir>/hotspots_report.csv or .json)'
    )
    
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=0.10,
        help='Drop percentage threshold (default: 0.10 = 10 percent)'
    )
    
    parser.add_argument(
        '--min-drop',
        type=float,
        default=0.0,
        help='Minimum absolute busbw drop in GB/s to consider (filter noise)'
    )
    
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output as JSON instead of CSV'
    )
    
    parser.add_argument(
        '--suggest-sweeps',
        action='store_true',
        help='Include sweep suggestions in JSON output (requires --json)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print detailed summary to stdout'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress output (useful with --json for piping)'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not args.input_file.exists():
        print(f"Error: Input file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)
    
    # Set default output path based on format
    if args.output is None:
        if args.json:
            args.output = args.input_file.parent / 'hotspots_report.json'
        else:
            args.output = args.input_file.parent / 'hotspots_report.csv'
    
    # Parse metrics
    if not args.quiet:
        print(f"Reading metrics from: {args.input_file}")
    rows = parse_metrics_csv(args.input_file)
    if not args.quiet:
        print(f"Loaded {len(rows)} metric rows")
    
    # Detect hot-spots
    if not args.quiet:
        print(f"Detecting hot-spots with threshold={args.threshold*100:.0f}%, min_drop={args.min_drop} GB/s")
    hotspots = detect_hotspots(rows, args.threshold, args.min_drop)
    if not args.quiet:
        print(f"Detected {len(hotspots)} hot-spots")
    
    # Write report in requested format
    if args.json:
        write_json_report(hotspots, args.output, include_suggestions=args.suggest_sweeps)
        if not args.quiet:
            print(f"JSON report written to: {args.output}")
    else:
        write_csv_report(hotspots, args.output)
        if not args.quiet:
            print(f"CSV report written to: {args.output}")
    
    # Print summary if verbose (and not in quiet mode)
    if not args.quiet and (args.verbose or len(hotspots) > 0):
        print_summary(hotspots)


if __name__ == '__main__':
    main()

