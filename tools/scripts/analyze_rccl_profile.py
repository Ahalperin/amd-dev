#!/usr/bin/env python3

"""
RCCL Profile Analysis Script
Analyzes profiling results from ROCm profiling tools
"""

import pandas as pd
import argparse
import sys
import os
import json
from pathlib import Path

def analyze_csv_profile(csv_file):
    """Analyze CSV profiling results"""
    try:
        df = pd.read_csv(csv_file)
        print(f"\n=== Analysis of {csv_file} ===")
        
        # Basic statistics
        print(f"Total records: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        
        # If it's a kernel trace file
        if 'KernelName' in df.columns and 'DurationNs' in df.columns:
            print("\n--- Kernel Analysis ---")
            df['Duration_ms'] = df['DurationNs'] / 1e6
            
            # Top kernels by duration
            print("\nTop 10 kernels by total duration:")
            kernel_stats = df.groupby('KernelName')['Duration_ms'].agg(['sum', 'count', 'mean']).round(3)
            kernel_stats = kernel_stats.sort_values('sum', ascending=False)
            print(kernel_stats.head(10))
            
            # Overall statistics
            print(f"\nOverall Statistics:")
            print(f"Total execution time: {df['Duration_ms'].sum():.3f} ms")
            print(f"Average kernel duration: {df['Duration_ms'].mean():.3f} ms")
            print(f"Median kernel duration: {df['Duration_ms'].median():.3f} ms")
            print(f"Longest kernel: {df['Duration_ms'].max():.3f} ms")
            print(f"Shortest kernel: {df['Duration_ms'].min():.3f} ms")
            
        # If it has HIP API traces
        if 'Name' in df.columns and 'dur' in df.columns:
            print("\n--- HIP API Analysis ---")
            df['Duration_ms'] = df['dur'] / 1000.0  # Convert microseconds to milliseconds
            
            # Top API calls
            print("\nTop 10 API calls by total duration:")
            api_stats = df.groupby('Name')['Duration_ms'].agg(['sum', 'count', 'mean']).round(3)
            api_stats = api_stats.sort_values('sum', ascending=False)
            print(api_stats.head(10))
            
        return True
        
    except Exception as e:
        print(f"Error analyzing CSV file {csv_file}: {e}")
        return False

def analyze_json_profile(json_file):
    """Analyze JSON profiling results"""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        print(f"\n=== Analysis of {json_file} ===")
        print(f"JSON structure keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
        
        # Chrome tracing format
        if 'traceEvents' in data:
            events = data['traceEvents']
            print(f"Total trace events: {len(events)}")
            
            # Analyze different event types
            event_types = {}
            durations = []
            
            for event in events:
                ph = event.get('ph', 'unknown')
                event_types[ph] = event_types.get(ph, 0) + 1
                
                if 'dur' in event:
                    durations.append(event['dur'])
            
            print(f"Event types: {event_types}")
            
            if durations:
                print(f"Duration statistics (microseconds):")
                print(f"  Total events with duration: {len(durations)}")
                print(f"  Average duration: {sum(durations)/len(durations):.3f}")
                print(f"  Max duration: {max(durations):.3f}")
                print(f"  Min duration: {min(durations):.3f}")
        
        return True
        
    except Exception as e:
        print(f"Error analyzing JSON file {json_file}: {e}")
        return False

def find_profile_files(directory):
    """Find all profile files in a directory"""
    profile_files = {
        'csv': [],
        'json': [],
        'db': [],
        'pftrace': []
    }
    
    for file_path in Path(directory).glob('**/*'):
        if file_path.is_file():
            suffix = file_path.suffix.lower()
            if suffix == '.csv':
                profile_files['csv'].append(str(file_path))
            elif suffix == '.json':
                profile_files['json'].append(str(file_path))
            elif suffix == '.db':
                profile_files['db'].append(str(file_path))
            elif suffix == '.pftrace':
                profile_files['pftrace'].append(str(file_path))
    
    return profile_files

def main():
    parser = argparse.ArgumentParser(description='Analyze RCCL profiling results')
    parser.add_argument('path', help='Path to profile file or directory')
    parser.add_argument('--type', choices=['csv', 'json', 'auto'], default='auto',
                       help='File type to analyze')
    parser.add_argument('--summary', action='store_true',
                       help='Show only summary information')
    
    args = parser.parse_args()
    
    if os.path.isfile(args.path):
        # Single file analysis
        if args.type == 'auto':
            if args.path.endswith('.csv'):
                analyze_csv_profile(args.path)
            elif args.path.endswith('.json'):
                analyze_json_profile(args.path)
            else:
                print(f"Unknown file type: {args.path}")
        elif args.type == 'csv':
            analyze_csv_profile(args.path)
        elif args.type == 'json':
            analyze_json_profile(args.path)
            
    elif os.path.isdir(args.path):
        # Directory analysis
        print(f"Analyzing profile files in: {args.path}")
        profile_files = find_profile_files(args.path)
        
        print(f"\nFound files:")
        for file_type, files in profile_files.items():
            print(f"  {file_type.upper()}: {len(files)} files")
        
        # Analyze CSV files
        for csv_file in profile_files['csv']:
            if not args.summary:
                analyze_csv_profile(csv_file)
            else:
                print(f"CSV: {csv_file}")
        
        # Analyze JSON files
        for json_file in profile_files['json']:
            if not args.summary:
                analyze_json_profile(json_file)
            else:
                print(f"JSON: {json_file}")
        
        # Report other file types
        if profile_files['pftrace']:
            print(f"\nPerfetto trace files (view at https://ui.perfetto.dev/):")
            for pf in profile_files['pftrace']:
                print(f"  {pf}")
                
    else:
        print(f"Path not found: {args.path}")
        sys.exit(1)

if __name__ == "__main__":
    main()