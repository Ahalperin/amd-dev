#!/usr/bin/env python3
"""
Analyze RCCL sweep results to find optimal number of channels for each combination of:
- Number of nodes
- Collective type  
- Message size

Best results selected based on in-place busbw.

Usage:
    python analyze_sweep.py /path/to/sweep_results/run_YYYYMMDD_HHMMSS
"""

import argparse
import re
import csv
from pathlib import Path
from collections import defaultdict

def parse_output_log(filepath):
    """Parse output.log to extract message size and in-place busbw data."""
    results = []
    with open(filepath, 'r') as f:
        for line in f:
            # Match data lines: size count type redop root oop_time oop_algbw oop_busbw #wrong ip_time ip_algbw ip_busbw #wrong
            match = re.match(r'\s*(\d+)\s+\d+\s+\w+\s+\w+\s+\S+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+\d+\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s+\d+', line)
            if match:
                size = int(match.group(1))
                inplace_busbw = float(match.group(2))
                results.append({'size': size, 'inplace_busbw': inplace_busbw})
    return results

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

def main():
    parser = argparse.ArgumentParser(
        description='Analyze RCCL sweep results to find optimal number of channels.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Example:
    python analyze_sweep.py /path/to/sweep_results/run_20251221_063252
        '''
    )
    parser.add_argument(
        'results_dir',
        type=Path,
        help='Path to the sweep results directory (contains outputs/ folder and summary.csv)'
    )
    args = parser.parse_args()
    
    results_dir = args.results_dir
    if not results_dir.exists():
        print(f"Error: Directory not found: {results_dir}")
        return 1
    
    outputs_dir = results_dir / 'outputs'
    if not outputs_dir.exists():
        print(f"Error: outputs/ directory not found in: {results_dir}")
        return 1
    
    # Data structure: {(collective, num_nodes, message_size): [(num_channels, inplace_busbw), ...]}
    all_data = defaultdict(list)
    
    for subdir in outputs_dir.iterdir():
        if not subdir.is_dir():
            continue
        
        output_log = subdir / 'output.log'
        if not output_log.exists():
            continue
        
        # Parse directory name: {collective}_{num_nodes}node_{num_channels}ch
        name = subdir.name
        match = re.match(r'(.+)_(\d+)node_(\d+)ch', name)
        if not match:
            continue
        
        collective = match.group(1)
        num_nodes = int(match.group(2))
        num_channels = int(match.group(3))
        
        # Parse output log
        results = parse_output_log(output_log)
        
        for r in results:
            key = (collective, num_nodes, r['size'])
            all_data[key].append((num_channels, r['inplace_busbw']))
    
    if not all_data:
        print("No data found!")
        return
    
    # Find best for each combination (collective, nodes, message size)
    best_results = {}
    for key, values in all_data.items():
        # Find the entry (num_channels, inplace_busbw) with highest inplace_busbw
        best = max(values, key=lambda x: x[1])
        best_results[key] = best
    
    # Organize data for display
    collectives = sorted(set(k[0] for k in best_results.keys()))
    nodes_list = sorted(set(k[1] for k in best_results.keys()))
    sizes = sorted(set(k[2] for k in best_results.keys()))
    
    print("=" * 120)
    print("OPTIMAL NUMBER OF CHANNELS BY COLLECTIVE, NODES, AND MESSAGE SIZE")
    print("(Selected based on best in-place busbw)")
    print("=" * 120)
    
    for collective in collectives:
        print(f"\n{'='*100}")
        print(f"COLLECTIVE: {collective}")
        print(f"{'='*100}")
        
        for num_nodes in nodes_list:
            # Check if any data exists for this combo
            has_data = any((collective, num_nodes, s) in best_results for s in sizes)
            if not has_data:
                continue
            
            print(f"\n  Nodes: {num_nodes}")
            print(f"  {'-'*80}")
            print(f"  {'Message Size':>15} | {'Best Channels':>14} | {'In-Place BusBW (GB/s)':>22}")
            print(f"  {'-'*80}")
            
            for size in sizes:
                key = (collective, num_nodes, size)
                if key in best_results:
                    num_channels, busbw = best_results[key]
                    print(f"  {format_size(size):>15} | {num_channels:>14} | {busbw:>22.2f}")
    
    # Create compact summary table
    print("\n\n")
    print("=" * 140)
    print("SUMMARY TABLE: Best Channels (format: ch@busbw)")
    print("=" * 140)
    
    # Build header
    size_strs = [format_size(s) for s in sizes]
    header = f"{'Collective':<22} | {'Nodes':>5} |"
    for s in size_strs:
        header += f" {s:>9} |"
    print(header)
    print("-" * len(header))
    
    for collective in collectives:
        for num_nodes in nodes_list:
            # Check if data exists
            has_data = any((collective, num_nodes, s) in best_results for s in sizes)
            if not has_data:
                continue
            
            row = f"{collective:<22} | {num_nodes:>5} |"
            for size in sizes:
                key = (collective, num_nodes, size)
                if key in best_results:
                    ch, bw = best_results[key]
                    row += f" {ch:>2}@{bw:>6.1f} |"
                else:
                    row += f" {'N/A':>9} |"
            print(row)
    
    # Export to CSV
    output_csv = results_dir / 'best_channels_analysis.csv'
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['collective', 'num_nodes', 'message_size', 'message_size_human', 'best_num_channels', 'inplace_busbw'])
        for key in sorted(best_results.keys()):
            collective, num_nodes, size = key
            num_channels, busbw = best_results[key]
            writer.writerow([collective, num_nodes, size, format_size(size), num_channels, busbw])
    
    print(f"\n\nDetailed results saved to: {output_csv}")

if __name__ == '__main__':
    main()
