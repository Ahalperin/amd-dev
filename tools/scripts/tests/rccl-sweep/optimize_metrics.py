#!/usr/bin/env python3
"""
Optimize metrics CSV by keeping only the best performing configuration
for each unique combination of collective-type, num_nodes, num_gpus, and size_bytes.

Best is defined as the configuration with the minimum nchannels among those
that are within a tolerance percentage of the best (minimum) time_ip_us.
"""

import argparse
import pandas as pd
from pathlib import Path


def optimize_metrics(input_file: str, output_file: str = None, tolerance_pct: float = 5.0) -> pd.DataFrame:
    """
    Read metrics CSV and keep only the best configuration for each unique combination.
    
    Strategy: For each group, find configs within tolerance_pct of the best time_ip_us,
    then pick the one with the minimum nchannels.
    
    Args:
        input_file: Path to the input metrics.csv file
        output_file: Path to the output file. If None, uses 'metrics_optimize.csv' 
                     in the same directory as input_file
        tolerance_pct: Percentage tolerance from the best time_ip_us (default: 5%)
    
    Returns:
        DataFrame with optimized metrics
    """
    # Read the input CSV
    df = pd.read_csv(input_file)
    
    # Define the grouping columns
    group_cols = ['collective', 'num_nodes', 'num_gpus', 'size_bytes']
    
    # Verify required columns exist
    required_cols = group_cols + ['time_ip_us', 'nchannels']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Collect best indices for each group
    best_indices = []
    
    for _, group in df.groupby(group_cols):
        # Find the minimum time_ip_us in this group
        min_time = group['time_ip_us'].min()
        threshold = min_time * (1 + tolerance_pct / 100)
        
        # Filter to configs within tolerance
        within_tolerance = group[group['time_ip_us'] <= threshold]
        
        # Among those, pick the one with minimum nchannels
        best_idx = within_tolerance['nchannels'].idxmin()
        best_indices.append(best_idx)
    
    # Select the best rows
    df_optimized = df.loc[best_indices].reset_index(drop=True)
    
    # Sort by the grouping columns for consistent output
    df_optimized = df_optimized.sort_values(group_cols).reset_index(drop=True)
    
    # Determine output file path
    if output_file is None:
        input_path = Path(input_file)
        output_file = input_path.parent / 'metrics_optimize.csv'
    
    # Write to output CSV
    df_optimized.to_csv(output_file, index=False)
    
    # Print summary
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Tolerance: {tolerance_pct}% from best time_ip_us")
    print(f"Original rows: {len(df)}")
    print(f"Optimized rows: {len(df_optimized)}")
    print(f"Unique combinations: {len(df_optimized)}")
    print(f"Rows removed: {len(df) - len(df_optimized)}")
    
    return df_optimized


def main():
    parser = argparse.ArgumentParser(
        description='Optimize metrics CSV by selecting min nchannels within tolerance of best time_ip_us'
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to the input metrics.csv file'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Path to the output file (default: metrics_optimize.csv in same directory as input)'
    )
    parser.add_argument(
        '-t', '--tolerance',
        type=float,
        default=5.0,
        help='Percentage tolerance from best time_ip_us (default: 5%%)'
    )
    
    args = parser.parse_args()
    
    optimize_metrics(args.input_file, args.output, args.tolerance)


if __name__ == '__main__':
    main()
