#!/usr/bin/env python3
"""
Merge all metrics.csv files from run directories in sweep_results.

This script finds all run_* directories containing metrics.csv files
and merges them into a single combined CSV file.
"""

import argparse
import csv
import sys
from pathlib import Path


def find_metrics_files(
    base_path: Path,
    metrics_filename: str = "metrics.csv",
    dir_suffix: str | None = None,
) -> list[tuple[str, Path]]:
    """Find all metrics files in run_* directories.
    
    Args:
        base_path: Path to sweep_results directory
        metrics_filename: Name of the metrics file to look for
        dir_suffix: Optional suffix to filter directories (e.g., '_base')
    """
    metrics_files = []
    
    for item in sorted(base_path.iterdir()):
        if item.is_dir() and item.name.startswith("run_"):
            # Filter by directory suffix if specified
            if dir_suffix and not item.name.endswith(dir_suffix):
                continue
            metrics_path = item / metrics_filename
            if metrics_path.exists():
                metrics_files.append((item.name, metrics_path))
    
    return metrics_files


def merge_metrics(
    base_path: Path,
    output_file: Path,
    add_run_column: bool = False,
    metrics_filename: str = "metrics.csv",
    dir_suffix: str | None = None,
) -> int:
    """
    Merge all metrics files from run directories.
    
    Args:
        base_path: Path to sweep_results directory
        output_file: Path to save the merged CSV
        add_run_column: Whether to add a column identifying the source run
        metrics_filename: Name of the metrics file to look for
        dir_suffix: Optional suffix to filter directories (e.g., '_base')
        
    Returns:
        Total number of data rows merged
    """
    metrics_files = find_metrics_files(base_path, metrics_filename, dir_suffix)
    
    if not metrics_files:
        print(f"No metrics.csv files found in run_* directories under {base_path}")
        sys.exit(1)
    
    print(f"Found {len(metrics_files)} metrics.csv files:")
    for run_name, path in metrics_files:
        print(f"  - {run_name}: {path}")
    
    total_rows = 0
    header_written = False
    header = None
    
    with open(output_file, 'w', newline='') as outfile:
        writer = None
        
        for run_name, metrics_path in metrics_files:
            try:
                with open(metrics_path, 'r', newline='') as infile:
                    reader = csv.reader(infile)
                    file_header = next(reader)
                    
                    # Initialize writer with first file's header
                    if not header_written:
                        if add_run_column:
                            header = ['run_id'] + file_header
                        else:
                            header = file_header
                        writer = csv.writer(outfile)
                        writer.writerow(header)
                        header_written = True
                    else:
                        # Verify headers match
                        expected_header = header[1:] if add_run_column else header
                        if file_header != expected_header:
                            print(f"  Warning: Header mismatch in {metrics_path}")
                            print(f"    Expected: {expected_header}")
                            print(f"    Got: {file_header}")
                    
                    # Write data rows
                    row_count = 0
                    for row in reader:
                        if add_run_column:
                            writer.writerow([run_name] + row)
                        else:
                            writer.writerow(row)
                        row_count += 1
                    
                    total_rows += row_count
                    print(f"  Loaded {row_count} rows from {run_name}")
                    
            except Exception as e:
                print(f"  Warning: Failed to read {metrics_path}: {e}")
    
    if total_rows == 0:
        print("No data could be loaded from any metrics.csv files")
        sys.exit(1)
    
    print(f"\nMerged {total_rows} total rows into {output_file}")
    return total_rows


def main():
    parser = argparse.ArgumentParser(
        description="Merge all metrics files from run directories"
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        default=Path("./sweep_results"),
        help="Path to sweep_results directory (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output file path (default: <base_path>/merged_metrics.csv)",
    )
    parser.add_argument(
        "--add-run-column",
        action="store_true",
        help="Add a column to identify the source run",
    )
    parser.add_argument(
        "--metrics-filename",
        type=str,
        default="metrics.csv",
        help="Name of the metrics file to merge (default: %(default)s)",
    )
    parser.add_argument(
        "--dir-suffix",
        type=str,
        default=None,
        help="Only include run directories ending with this suffix (e.g., '_base')",
    )
    
    args = parser.parse_args()
    
    if not args.base_path.exists():
        print(f"Error: Base path does not exist: {args.base_path}")
        sys.exit(1)
    
    output_file = args.output or (args.base_path / "merged_metrics.csv")
    
    merge_metrics(
        base_path=args.base_path,
        output_file=output_file,
        add_run_column=args.add_run_column,
        metrics_filename=args.metrics_filename,
        dir_suffix=args.dir_suffix,
    )


if __name__ == "__main__":
    main()
