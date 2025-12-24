#!/usr/bin/env python3
"""
Analyze RCCL sweep results to find optimal configurations for each combination of:
- Number of nodes
- Collective type  
- Message size

Supports new database format with per-message-size metrics including algo/proto/nchannels.

Usage:
    python analyze_sweep.py /path/to/sweep_results/run_YYYYMMDD_HHMMSS
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict

# Try to import sweep_db module
try:
    from sweep_db import SweepDatabase
    HAS_SWEEP_DB = True
except ImportError:
    HAS_SWEEP_DB = False


def format_size(size_bytes):
    """Format byte size to human readable format."""
    if size_bytes >= 1024**3:
        return f"{size_bytes / (1024**3):.0f}G"
    elif size_bytes >= 1024**2:
        return f"{size_bytes / (1024**2):.0f}M"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.0f}K"
    else:
        return f"{size_bytes}B"


def load_from_database(db_path):
    """Load metrics from SQLite database."""
    db = SweepDatabase(str(db_path))
    
    # Get all metrics
    metrics = db.get_metrics()
    db.close()
    
    # Convert to our format
    data = []
    for m in metrics:
        data.append({
            'collective': m['collective'],
            'num_nodes': m['num_nodes'],
            'num_gpus': m['num_gpus'],
            'size': m['size_bytes'],
            'algo': m.get('algo'),
            'proto': m.get('proto'),
            'nchannels': m.get('nchannels'),
            'busbw_oop': m.get('busbw_oop'),
            'busbw_ip': m.get('busbw_ip'),
        })
    
    return data


def load_from_metrics_csv(csv_path):
    """Load metrics from metrics.csv file."""
    import csv
    
    data = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'collective': row['collective'],
                'num_nodes': int(row['num_nodes']),
                'num_gpus': int(row['num_gpus']),
                'size': int(row['size_bytes']),
                'algo': row.get('algo'),
                'proto': row.get('proto'),
                'nchannels': int(row['nchannels']) if row.get('nchannels') else None,
                'busbw_oop': float(row['busbw_oop']) if row.get('busbw_oop') else None,
                'busbw_ip': float(row['busbw_ip']) if row.get('busbw_ip') else None,
            })
    
    return data


def main():
    parser = argparse.ArgumentParser(
        description='Analyze RCCL sweep results to find optimal configurations.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
    # Analyze sweep results (auto-detects database or CSV)
    python analyze_sweep.py /path/to/sweep_results/run_20251224_111535
    
    # Show detailed per-size breakdown
    python analyze_sweep.py /path/to/results --detailed
        '''
    )
    parser.add_argument(
        'results_dir',
        type=Path,
        help='Path to the sweep results directory'
    )
    parser.add_argument(
        '--detailed', '-d',
        action='store_true',
        help='Show detailed per-size breakdown including algo/proto/nchannels'
    )
    parser.add_argument(
        '--sort-by',
        choices=['busbw_ip', 'busbw_oop', 'size'],
        default='busbw_ip',
        help='Sort results by (default: busbw_ip)'
    )
    args = parser.parse_args()
    
    results_dir = args.results_dir
    if not results_dir.exists():
        print(f"Error: Directory not found: {results_dir}")
        return 1
    
    # Try to load data from database first, then CSV
    db_path = results_dir / 'sweep_results.db'
    metrics_csv = results_dir / 'metrics.csv'
    
    data = []
    source = None
    
    if db_path.exists() and HAS_SWEEP_DB:
        try:
            data = load_from_database(db_path)
            source = f"database: {db_path.name}"
        except Exception as e:
            print(f"Warning: Could not read database: {e}")
    
    if not data and metrics_csv.exists():
        try:
            data = load_from_metrics_csv(metrics_csv)
            source = f"CSV: {metrics_csv.name}"
        except Exception as e:
            print(f"Warning: Could not read metrics CSV: {e}")
    
    if not data:
        print("Error: No data found! Need either sweep_results.db or metrics.csv")
        return 1
    
    print(f"Loaded {len(data)} metrics from {source}")
    print()
    
    # Organize data
    collectives = sorted(set(d['collective'] for d in data))
    nodes_list = sorted(set(d['num_nodes'] for d in data))
    sizes = sorted(set(d['size'] for d in data))
    
    # Group data by (collective, num_nodes, size)
    grouped = defaultdict(list)
    for d in data:
        key = (d['collective'], d['num_nodes'], d['size'])
        grouped[key].append(d)
    
    if args.detailed:
        # Detailed view: show algo/proto/nchannels for each size
        print("=" * 120)
        print("DETAILED PER-MESSAGE-SIZE METRICS")
        print("=" * 120)
        
        for collective in collectives:
            print(f"\n{'='*100}")
            print(f"COLLECTIVE: {collective}")
            print(f"{'='*100}")
            
            for num_nodes in nodes_list:
                # Check if any data exists
                has_data = any((collective, num_nodes, s) in grouped for s in sizes)
                if not has_data:
                    continue
                
                print(f"\n  Nodes: {num_nodes}")
                print(f"  {'-'*90}")
                print(f"  {'Size':>10} | {'Algo':>8} | {'Proto':>8} | {'nCh':>5} | {'BusBW-OOP':>12} | {'BusBW-IP':>12}")
                print(f"  {'-'*90}")
                
                for size in sizes:
                    key = (collective, num_nodes, size)
                    if key not in grouped:
                        continue
                    
                    # Should be one entry per size (unless multiple runs)
                    for entry in grouped[key]:
                        algo = entry.get('algo') or 'auto'
                        proto = entry.get('proto') or 'auto'
                        nch = entry.get('nchannels') or 'auto'
                        bw_oop = entry.get('busbw_oop') or 0
                        bw_ip = entry.get('busbw_ip') or 0
                        print(f"  {format_size(size):>10} | {algo:>8} | {proto:>8} | {nch:>5} | {bw_oop:>12.2f} | {bw_ip:>12.2f}")
    
    else:
        # Summary view: find best configuration per (collective, nodes, size)
        print("=" * 120)
        print("SUMMARY: BEST CONFIGURATIONS BY COLLECTIVE, NODES, AND MESSAGE SIZE")
        print("(Selected based on in-place busbw)")
        print("=" * 120)
        
        best_results = {}
        for key, entries in grouped.items():
            # Find entry with best in-place busbw
            best = max(entries, key=lambda x: x.get('busbw_ip') or 0)
            best_results[key] = best
        
        for collective in collectives:
            print(f"\n{'='*100}")
            print(f"COLLECTIVE: {collective}")
            print(f"{'='*100}")
            
            for num_nodes in nodes_list:
                has_data = any((collective, num_nodes, s) in best_results for s in sizes)
                if not has_data:
                    continue
                
                print(f"\n  Nodes: {num_nodes}")
                print(f"  {'-'*90}")
                print(f"  {'Size':>10} | {'Algo':>8} | {'Proto':>8} | {'nCh':>5} | {'BusBW-IP (GB/s)':>16}")
                print(f"  {'-'*90}")
                
                for size in sizes:
                    key = (collective, num_nodes, size)
                    if key in best_results:
                        entry = best_results[key]
                        algo = entry.get('algo') or 'auto'
                        proto = entry.get('proto') or 'auto'
                        nch = entry.get('nchannels') or 'auto'
                        bw_ip = entry.get('busbw_ip') or 0
                        print(f"  {format_size(size):>10} | {algo:>8} | {proto:>8} | {nch:>5} | {bw_ip:>16.2f}")
        
        # Compact summary table
        print("\n\n")
        print("=" * 140)
        print("COMPACT SUMMARY: nChannels @ BusBW-IP (GB/s)")
        print("=" * 140)
        
        # Build header with sizes
        size_strs = [format_size(s) for s in sizes[:10]]  # Limit to 10 sizes for readability
        header = f"{'Collective':<22} | {'N':>3} |"
        for s in size_strs:
            header += f" {s:>10} |"
        print(header)
        print("-" * len(header))
        
        for collective in collectives:
            for num_nodes in nodes_list:
                has_data = any((collective, num_nodes, s) in best_results for s in sizes)
                if not has_data:
                    continue
                
                row = f"{collective:<22} | {num_nodes:>3} |"
                for size in sizes[:10]:
                    key = (collective, num_nodes, size)
                    if key in best_results:
                        entry = best_results[key]
                        nch = entry.get('nchannels') or 0
                        bw = entry.get('busbw_ip') or 0
                        row += f" {nch:>3}@{bw:>5.0f} |"
                    else:
                        row += f" {'N/A':>10} |"
                print(row)
    
    # Export enhanced analysis
    import csv
    output_csv = results_dir / 'analysis.csv'
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['collective', 'num_nodes', 'num_gpus', 'size_bytes', 'size_human', 
                        'algo', 'proto', 'nchannels', 'busbw_oop', 'busbw_ip'])
        for d in sorted(data, key=lambda x: (x['collective'], x['num_nodes'], x['size'])):
            writer.writerow([
                d['collective'], d['num_nodes'], d['num_gpus'],
                d['size'], format_size(d['size']),
                d.get('algo'), d.get('proto'), d.get('nchannels'),
                d.get('busbw_oop'), d.get('busbw_ip')
            ])
    
    print(f"\n\nAnalysis saved to: {output_csv}")


if __name__ == '__main__':
    sys.exit(main() or 0)
