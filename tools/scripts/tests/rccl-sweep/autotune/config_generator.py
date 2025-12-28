#!/usr/bin/env python3
"""
Tuner Configuration Generator

Converts optimized metrics CSV to RCCL tuner configuration file format.
Handles name mappings, size range consolidation, and proper formatting.
"""

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


@dataclass
class TunerEntry:
    """Represents a single tuner configuration entry."""
    collective: str
    min_bytes: int
    max_bytes: int
    algorithm: str
    protocol: str
    channels: int
    num_nodes: int
    num_ranks: int
    num_pipe_ops: int = -1
    reg_buff: int = -1
    
    def to_line(self) -> str:
        """Convert to tuner.conf line format."""
        return (
            f"{self.collective},{self.min_bytes},{self.max_bytes},"
            f"{self.algorithm},{self.protocol},{self.channels},"
            f"{self.num_nodes},{self.num_ranks},{self.num_pipe_ops},{self.reg_buff}"
        )


class TunerConfigGenerator:
    """Generates RCCL tuner configuration files from optimized metrics."""
    
    # Collective name mappings (perf test names -> tuner config names)
    COLLECTIVE_MAP = {
        'all_reduce_perf': 'allreduce',
        'all_gather_perf': 'allgather',
        'reduce_scatter_perf': 'reducescatter',
        'alltoall_perf': 'alltoall',
        'broadcast_perf': 'broadcast',
        'reduce_perf': 'reduce',
    }
    
    # Algorithm mappings (sweep output -> tuner config)
    # -1 means keep RCCL default
    ALGO_MAP = {
        'TREE': 'tree',
        'RING': 'ring',
        'Direct': '-1',  # CollNet Direct - use default
        'COLLNET_DIRECT': 'collnet_direct',
        'COLLNET_CHAIN': 'collnet_chain',
        'NVLS': 'nvls',
        'NVLS_TREE': 'nvls_tree',
        'PAT': 'pat',
    }
    
    # Protocol mappings
    PROTO_MAP = {
        'SIMPLE': 'simple',
        'LL': 'll',
        'LL128': 'll128',
    }
    
    def __init__(self, merge_ranges: bool = True):
        """
        Initialize the generator.
        
        Args:
            merge_ranges: If True, merge adjacent size ranges with identical configs
        """
        self.merge_ranges = merge_ranges
    
    def generate(
        self,
        metrics_csv: Path,
        output_path: Path,
        include_algo_proto: bool = False,
    ) -> List[TunerEntry]:
        """
        Generate tuner configuration from optimized metrics.
        
        Args:
            metrics_csv: Path to optimized_metrics.csv
            output_path: Path for output tuner.conf file
            include_algo_proto: If True, include algorithm and protocol in output.
                               If False, use -1 (keep default) for both.
        
        Returns:
            List of TunerEntry objects written to the config
        """
        # Read and parse metrics
        entries = self._read_metrics(metrics_csv, include_algo_proto)
        
        # Optionally merge adjacent ranges
        if self.merge_ranges:
            entries = self._merge_adjacent_ranges(entries)
        
        # Sort entries for consistent output
        entries = self._sort_entries(entries)
        
        # Write config file
        self._write_config(entries, output_path, metrics_csv)
        
        return entries
    
    def _read_metrics(
        self,
        metrics_csv: Path,
        include_algo_proto: bool,
    ) -> List[TunerEntry]:
        """Read metrics CSV and convert to TunerEntry objects."""
        entries = []
        
        with open(metrics_csv, 'r', newline='') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                try:
                    collective = row['collective']
                    
                    # Map collective name
                    tuner_collective = self.COLLECTIVE_MAP.get(collective)
                    if tuner_collective is None:
                        # Try without _perf suffix
                        tuner_collective = collective.replace('_perf', '')
                    
                    # Get size (use as both min and max for now)
                    size_bytes = int(row['size_bytes'])
                    
                    # Map algorithm and protocol
                    algo = row.get('algo', '')
                    proto = row.get('proto', '')
                    
                    if include_algo_proto:
                        tuner_algo = self.ALGO_MAP.get(algo, '-1')
                        tuner_proto = self.PROTO_MAP.get(proto, '-1')
                    else:
                        tuner_algo = '-1'
                        tuner_proto = '-1'
                    
                    # Get channels
                    channels = int(row.get('nchannels', 0) or 0)
                    
                    # Get node/rank info
                    num_nodes = int(row['num_nodes'])
                    num_ranks = int(row['num_gpus'])  # num_gpus is total ranks
                    
                    entry = TunerEntry(
                        collective=tuner_collective,
                        min_bytes=size_bytes,
                        max_bytes=size_bytes,
                        algorithm=tuner_algo,
                        protocol=tuner_proto,
                        channels=channels,
                        num_nodes=num_nodes,
                        num_ranks=num_ranks,
                    )
                    entries.append(entry)
                    
                except (ValueError, KeyError) as e:
                    # Skip malformed rows
                    continue
        
        return entries
    
    def _merge_adjacent_ranges(self, entries: List[TunerEntry]) -> List[TunerEntry]:
        """
        Merge adjacent size ranges that have identical configurations.
        
        Adjacent means max_bytes of one entry + 1 == min_bytes of the next entry.
        """
        if not entries:
            return entries
        
        # Group by (collective, num_nodes, num_ranks)
        groups: Dict[Tuple, List[TunerEntry]] = defaultdict(list)
        for entry in entries:
            key = (entry.collective, entry.num_nodes, entry.num_ranks)
            groups[key].append(entry)
        
        merged = []
        
        for key, group_entries in groups.items():
            # Sort by min_bytes
            group_entries.sort(key=lambda e: e.min_bytes)
            
            # Merge adjacent ranges with same config
            current = None
            for entry in group_entries:
                if current is None:
                    current = TunerEntry(
                        collective=entry.collective,
                        min_bytes=entry.min_bytes,
                        max_bytes=entry.max_bytes,
                        algorithm=entry.algorithm,
                        protocol=entry.protocol,
                        channels=entry.channels,
                        num_nodes=entry.num_nodes,
                        num_ranks=entry.num_ranks,
                    )
                elif self._can_merge(current, entry):
                    # Extend current range
                    current.max_bytes = entry.max_bytes
                else:
                    # Save current and start new
                    merged.append(current)
                    current = TunerEntry(
                        collective=entry.collective,
                        min_bytes=entry.min_bytes,
                        max_bytes=entry.max_bytes,
                        algorithm=entry.algorithm,
                        protocol=entry.protocol,
                        channels=entry.channels,
                        num_nodes=entry.num_nodes,
                        num_ranks=entry.num_ranks,
                    )
            
            if current is not None:
                merged.append(current)
        
        return merged
    
    def _can_merge(self, current: TunerEntry, next_entry: TunerEntry) -> bool:
        """Check if two entries can be merged (adjacent and same config)."""
        # Check if adjacent (allowing for small gaps in size granularity)
        if next_entry.min_bytes > current.max_bytes + 1:
            return False
        
        # Check if same configuration
        return (
            current.algorithm == next_entry.algorithm and
            current.protocol == next_entry.protocol and
            current.channels == next_entry.channels
        )
    
    def _sort_entries(self, entries: List[TunerEntry]) -> List[TunerEntry]:
        """Sort entries by collective, num_nodes, num_ranks, min_bytes."""
        # Define collective order
        collective_order = [
            'allgather', 'allreduce', 'broadcast', 
            'reduce', 'reducescatter', 'alltoall'
        ]
        
        def sort_key(entry: TunerEntry):
            try:
                coll_idx = collective_order.index(entry.collective)
            except ValueError:
                coll_idx = 999
            return (coll_idx, entry.num_nodes, entry.num_ranks, entry.min_bytes)
        
        return sorted(entries, key=sort_key)
    
    def _write_config(
        self,
        entries: List[TunerEntry],
        output_path: Path,
        source_csv: Path,
    ) -> None:
        """Write tuner configuration file in CSV format."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            # Write CSV header row
            f.write("collective_type,min_bytes,max_bytes,algorithm,protocol,"
                    "channels,nNodes,nRanks,numPipeOps,regBuff\n")
            
            # Write entries
            for entry in entries:
                f.write(entry.to_line() + "\n")
    
    def generate_csv_report(
        self,
        entries: List[TunerEntry],
        output_path: Path,
    ) -> None:
        """Generate a CSV report of the tuning results."""
        output_path = Path(output_path)
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'collective', 'min_bytes', 'max_bytes', 'min_human', 'max_human',
                'algorithm', 'protocol', 'channels', 'num_nodes', 'num_ranks'
            ])
            
            for entry in entries:
                writer.writerow([
                    entry.collective,
                    entry.min_bytes,
                    entry.max_bytes,
                    self._format_bytes(entry.min_bytes),
                    self._format_bytes(entry.max_bytes),
                    entry.algorithm,
                    entry.protocol,
                    entry.channels,
                    entry.num_nodes,
                    entry.num_ranks,
                ])
    
    @staticmethod
    def _format_bytes(size_bytes: int) -> str:
        """Convert bytes to human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                if size_bytes == int(size_bytes):
                    return f"{int(size_bytes)}{unit}"
                return f"{size_bytes:.1f}{unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f}TB"


def main():
    """Command-line interface for the config generator."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate RCCL tuner configuration from optimized metrics'
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
        help='Output path for tuner CSV (default: <input_dir>/generated_tuner.csv)'
    )
    parser.add_argument(
        '--csv-report',
        type=Path,
        default=None,
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
    
    args = parser.parse_args()
    
    # Set default output path
    if args.output is None:
        args.output = args.input_file.parent / 'generated_tuner.csv'
    
    # Generate config
    generator = TunerConfigGenerator(merge_ranges=not args.no_merge)
    entries = generator.generate(
        args.input_file,
        args.output,
        include_algo_proto=args.include_algo_proto,
    )
    
    print(f"Generated tuner config: {args.output}")
    print(f"Total entries: {len(entries)}")
    
    # Generate CSV report if requested
    if args.csv_report:
        generator.generate_csv_report(entries, args.csv_report)
        print(f"Generated CSV report: {args.csv_report}")


if __name__ == '__main__':
    main()

