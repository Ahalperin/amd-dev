#!/usr/bin/env python3
"""
Filter metrics CSV by error status.

Identifies entries with non-zero errors_oop or errors_ip values.
"""

import argparse
import pandas as pd
import signal
import sys

# Handle broken pipe gracefully (e.g., when piping to head)
signal.signal(signal.SIGPIPE, signal.SIG_DFL)


def filter_metrics(input_file: str, count_only: bool = False, 
                   show_err: bool = False, show_good: bool = False,
                   prune_err: bool = False) -> None:
    """
    Filter metrics CSV based on error status.
    
    Args:
        input_file: Path to the input metrics CSV file
        count_only: If True, only show count of entries with errors
        show_err: If True, output entries with errors to stdout
        show_good: If True, output entries without errors to stdout
        prune_err: If True, remove entries with errors from the input file
    """
    # Read the input CSV
    df = pd.read_csv(input_file)
    
    # Check for required columns
    if 'errors_oop' not in df.columns or 'errors_ip' not in df.columns:
        print("Error: CSV must contain 'errors_oop' and 'errors_ip' columns", file=sys.stderr)
        sys.exit(1)
    
    # Identify entries with errors (positive errors_oop or errors_ip)
    # Note: -1 represents N/A (e.g., alltoall doesn't report in-place errors)
    # so we only consider positive values as actual errors
    has_errors = (df['errors_oop'] > 0) | (df['errors_ip'] > 0)
    
    df_errors = df[has_errors]
    df_good = df[~has_errors]
    
    if count_only:
        print(f"Total entries: {len(df)}")
        print(f"Entries with errors: {len(df_errors)}")
        print(f"Entries without errors: {len(df_good)}")
        return
    
    if show_err:
        df_errors.to_csv(sys.stdout, index=False)
        return
    
    if show_good:
        df_good.to_csv(sys.stdout, index=False)
        return
    
    if prune_err:
        # Remove entries with errors from the input file
        df_good.to_csv(input_file, index=False)
        print(f"Pruned {len(df_errors)} entries with errors from {input_file}")
        print(f"Remaining entries: {len(df_good)}")
        return
    
    # Default: show summary
    print(f"Input file: {input_file}")
    print(f"Total entries: {len(df)}")
    print(f"Entries with errors: {len(df_errors)}")
    print(f"Entries without errors: {len(df_good)}")
    
    if len(df_errors) > 0:
        print("\nError entries breakdown by algo:")
        print(df_errors.groupby('algo').size().to_string())
        
        print("\nError entries breakdown by proto:")
        print(df_errors.groupby('proto').size().to_string())
        
        print("\nError entries breakdown by (algo, proto):")
        print(df_errors.groupby(['algo', 'proto']).size().to_string())


def main():
    parser = argparse.ArgumentParser(
        description='Filter metrics CSV by error status'
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to the input metrics CSV file'
    )
    parser.add_argument(
        '--count',
        action='store_true',
        help='Show only the count of entries with/without errors'
    )
    parser.add_argument(
        '--show-err',
        action='store_true',
        help='Output entries with errors to stdout (CSV format)'
    )
    parser.add_argument(
        '--show-good',
        action='store_true',
        help='Output entries without errors to stdout (CSV format)'
    )
    parser.add_argument(
        '--prune-err',
        action='store_true',
        help='Remove entries with errors from the input file (modifies file in place)'
    )
    
    args = parser.parse_args()
    
    # Validate mutually exclusive options
    options = [args.count, args.show_err, args.show_good, args.prune_err]
    if sum(options) > 1:
        parser.error("--count, --show-err, --show-good, and --prune-err are mutually exclusive")
    
    filter_metrics(args.input_file, args.count, args.show_err, args.show_good, args.prune_err)


if __name__ == '__main__':
    main()

