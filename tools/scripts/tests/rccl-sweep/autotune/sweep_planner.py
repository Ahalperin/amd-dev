#!/usr/bin/env python3
"""
Sweep Planner Module

Generates targeted sweep configurations based on hotspot analysis.
Produces sweep commands for rccl_sweep.py to run.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import math

from .hotspot_analyzer import Hotspot


@dataclass
class SweepConfig:
    """Configuration for a single sweep run."""
    collective: str
    nodes: int
    min_size: str  # Human-readable (e.g., "8MB")
    max_size: str  # Human-readable (e.g., "16MB")
    channels: str  # Range format (e.g., "16:64:8")
    algos: List[str] = field(default_factory=list)
    protos: List[str] = field(default_factory=lambda: ['SIMPLE'])
    step_size: Optional[str] = None  # If set, use fixed step instead of factor
    
    def to_command_args(self) -> List[str]:
        """Convert to rccl_sweep.py command-line arguments."""
        args = [
            '--nodes', str(self.nodes),
            '--collective', self.collective.replace('_perf', ''),
            '--min-size', self.min_size,
            '--max-size', self.max_size,
            '--channels', self.channels,
        ]
        
        if self.algos:
            args.extend(['--algo', ','.join(self.algos)])
        
        if self.protos:
            args.extend(['--proto', ','.join(self.protos)])
        
        # Only include step_size if there's actually a size range to step through
        if self.step_size and self.min_size != self.max_size:
            args.extend(['--step-size', self.step_size])
        
        return args
    
    def to_command(self, sweep_script: str = './rccl_sweep.py') -> str:
        """Generate full command string."""
        args = self.to_command_args()
        return f"{sweep_script} {' '.join(args)}"
    
    @property
    def description(self) -> str:
        """Human-readable description of this sweep."""
        algos_str = ','.join(self.algos) if self.algos else 'auto'
        return (f"{self.collective} @ {self.nodes} node(s), "
                f"{self.min_size}-{self.max_size}, channels={self.channels}, algo={algos_str}")


class SweepPlanner:
    """
    Plans targeted sweeps based on hotspot analysis results.
    """
    
    # Size step options for different granularities
    SIZE_STEPS = ['1K', '4K', '16K', '64K', '256K', '1M', '4M']
    
    def __init__(
        self,
        channel_expansion: float = 2.0,
        min_channels: int = 4,
        max_channels: int = 256,
        channel_step: int = 4,
    ):
        """
        Initialize the planner.
        
        Args:
            channel_expansion: Factor to expand channel search range
            min_channels: Minimum channels to sweep
            max_channels: Maximum channels to sweep
            channel_step: Step size for channel sweep
        """
        self.channel_expansion = channel_expansion
        self.min_channels = min_channels
        self.max_channels = max_channels
        self.channel_step = channel_step
    
    def plan_from_hotspots(
        self,
        hotspots: List[Hotspot],
        include_alternatives: bool = True,
        allowed_algos: Optional[set] = None,
    ) -> List[SweepConfig]:
        """
        Generate sweep configurations from hotspots.
        
        Args:
            hotspots: List of detected hotspots
            include_alternatives: If True, include alternative algorithm sweeps
            allowed_algos: If provided, only use algorithms from this set
            
        Returns:
            List of sweep configurations
        """
        configs = []
        
        # Group hotspots by (collective, nodes) to consolidate ranges
        groups: Dict[Tuple[str, int], List[Hotspot]] = {}
        for hs in hotspots:
            key = (hs.collective, hs.num_nodes)
            if key not in groups:
                groups[key] = []
            groups[key].append(hs)
        
        for (collective, nodes), group_hotspots in groups.items():
            # Sort by start_bytes
            group_hotspots.sort(key=lambda h: h.start_bytes)
            
            # Merge overlapping/adjacent ranges
            merged_ranges = self._merge_ranges(group_hotspots)
            
            for start_bytes, end_bytes, current_algo, current_channels in merged_ranges:
                # Calculate channel range
                channel_range = self._calculate_channel_range(current_channels)
                
                # Calculate size step based on range span (None if single size point)
                if start_bytes == end_bytes:
                    step_size = None  # Single size point, no stepping needed
                else:
                    step_size = self._calculate_size_step(start_bytes, end_bytes)
                
                # Check if current algo is in allowed set
                algo_to_use = current_algo
                if allowed_algos and current_algo:
                    algo_upper = current_algo.upper()
                    if algo_upper not in allowed_algos:
                        # Current algo not allowed, skip or use first allowed
                        if allowed_algos:
                            algo_to_use = list(allowed_algos)[0]
                        else:
                            algo_to_use = None
                
                # Create sweep config with current algorithm variations
                config = SweepConfig(
                    collective=collective,
                    nodes=nodes,
                    min_size=self._format_bytes(start_bytes),
                    max_size=self._format_bytes(end_bytes),
                    channels=channel_range,
                    algos=[algo_to_use] if algo_to_use else [],
                    protos=['SIMPLE'],
                    step_size=step_size,
                )
                configs.append(config)
                
                # Add alternative algorithm sweeps (filtered by allowed_algos)
                if include_alternatives and current_algo:
                    alt_algos = self._get_alternative_algos(current_algo)
                    # Filter alternatives by allowed set
                    if allowed_algos:
                        alt_algos = [a for a in alt_algos if a.upper() in allowed_algos]
                    for alt_algo in alt_algos:
                        alt_config = SweepConfig(
                            collective=collective,
                            nodes=nodes,
                            min_size=self._format_bytes(start_bytes),
                            max_size=self._format_bytes(end_bytes),
                            channels=channel_range,
                            algos=[alt_algo],
                            protos=['SIMPLE'],
                            step_size=step_size,
                        )
                        configs.append(alt_config)
        
        return configs
    
    def plan_initial_sweep(
        self,
        collectives: List[str] = None,
        nodes: str = "1-2",
        channels: str = "32:256:32",
        min_size: str = "1M",
        max_size: str = "512M",
        algos: List[str] = None,
        protos: List[str] = None,
    ) -> List[SweepConfig]:
        """
        Plan initial broad sweep configuration.
        
        Args:
            collectives: List of collectives to sweep
            nodes: Node range (e.g., "1-2")
            channels: Channel sweep range (e.g., "32:256:32")
            min_size: Minimum message size
            max_size: Maximum message size
            algos: Algorithms to sweep (default: ['RING', 'TREE'])
            protos: Protocols to sweep (default: ['SIMPLE'])
            
        Returns:
            List of sweep configurations
        """
        if collectives is None:
            collectives = ['all_reduce_perf', 'all_gather_perf']
        
        if algos is None:
            algos = ['RING', 'TREE']
        
        if protos is None:
            protos = ['SIMPLE']
        
        # Parse node range
        node_list = self._parse_nodes(nodes)
        
        configs = []
        for collective in collectives:
            for algo in algos:
                for n in node_list:
                    config = SweepConfig(
                        collective=collective,
                        nodes=n,
                        min_size=min_size,
                        max_size=max_size,
                        channels=channels,
                        algos=[algo],
                        protos=protos,
                    )
                    configs.append(config)
        
        return configs
    
    def _merge_ranges(
        self,
        hotspots: List[Hotspot],
    ) -> List[Tuple[int, int, str, int]]:
        """
        Merge overlapping/adjacent hotspot ranges.
        
        Returns:
            List of (start_bytes, end_bytes, algo, channels) tuples
        """
        if not hotspots:
            return []
        
        merged = []
        current_start = hotspots[0].start_bytes
        current_end = hotspots[0].end_bytes
        current_algo = hotspots[0].algo
        current_channels = hotspots[0].nchannels
        
        for hs in hotspots[1:]:
            # Check if this hotspot is adjacent/overlapping (within 2x size)
            if hs.start_bytes <= current_end * 2:
                # Extend current range
                current_end = max(current_end, hs.end_bytes)
                # Keep the most common config (use first for simplicity)
            else:
                # Save current range and start new
                merged.append((current_start, current_end, current_algo, current_channels))
                current_start = hs.start_bytes
                current_end = hs.end_bytes
                current_algo = hs.algo
                current_channels = hs.nchannels
        
        # Don't forget the last range
        merged.append((current_start, current_end, current_algo, current_channels))
        
        return merged
    
    def _calculate_channel_range(self, current_channels: int) -> str:
        """Calculate channel sweep range around current value."""
        if current_channels <= 0:
            current_channels = 32  # Default
        
        min_ch = max(
            self.min_channels,
            int(current_channels / self.channel_expansion)
        )
        max_ch = min(
            self.max_channels,
            int(current_channels * self.channel_expansion)
        )
        
        # Round to step size
        min_ch = (min_ch // self.channel_step) * self.channel_step
        max_ch = ((max_ch + self.channel_step - 1) // self.channel_step) * self.channel_step
        
        # Ensure minimum range
        if max_ch - min_ch < self.channel_step * 4:
            min_ch = max(self.min_channels, current_channels - self.channel_step * 4)
            max_ch = min(self.max_channels, current_channels + self.channel_step * 4)
        
        return f"{min_ch}:{max_ch}:{self.channel_step}"
    
    def _calculate_size_step(self, start_bytes: int, end_bytes: int) -> Optional[str]:
        """Calculate appropriate step size for the range."""
        span = end_bytes - start_bytes
        
        # Choose step to get reasonable number of data points (8-16)
        target_points = 12
        ideal_step = span / target_points
        
        # Find closest power-of-2 friendly step
        for step in self.SIZE_STEPS:
            step_bytes = self._parse_size(step)
            if step_bytes >= ideal_step:
                return step
        
        return self.SIZE_STEPS[-1]  # Use largest if range is huge
    
    def _get_alternative_algos(self, current_algo: str) -> List[str]:
        """Get alternative algorithms to try."""
        alternatives = {
            'RING': ['TREE'],
            'TREE': ['RING'],
            'Direct': ['RING', 'TREE'],
        }
        return alternatives.get(current_algo, ['RING', 'TREE'])
    
    def _parse_nodes(self, nodes_str: str) -> List[int]:
        """Parse node range string."""
        if '-' in nodes_str:
            start, end = nodes_str.split('-', 1)
            return list(range(int(start), int(end) + 1))
        return [int(nodes_str)]
    
    def _parse_size(self, size_str: str) -> int:
        """Parse size string to bytes."""
        size_str = size_str.upper().strip()
        
        multipliers = {
            'B': 1,
            'K': 1024,
            'KB': 1024,
            'M': 1024 * 1024,
            'MB': 1024 * 1024,
            'G': 1024 * 1024 * 1024,
            'GB': 1024 * 1024 * 1024,
        }
        
        for suffix, mult in sorted(multipliers.items(), key=lambda x: -len(x[0])):
            if size_str.endswith(suffix):
                return int(float(size_str[:-len(suffix)]) * mult)
        
        return int(size_str)
    
    @staticmethod
    def _format_bytes(size_bytes: int) -> str:
        """Convert bytes to human-readable format."""
        for unit in ['B', 'K', 'M', 'G']:
            if size_bytes < 1024:
                if size_bytes == int(size_bytes):
                    return f"{int(size_bytes)}{unit}"
                return f"{size_bytes:.0f}{unit}"
            size_bytes /= 1024
        return f"{size_bytes:.0f}T"
    
    def generate_sweep_script(
        self,
        configs: List[SweepConfig],
        output_path: Path,
        sweep_script: str = './rccl_sweep.py',
    ) -> None:
        """
        Generate a shell script to run all planned sweeps.
        
        Args:
            configs: List of sweep configurations
            output_path: Path for output script
            sweep_script: Path to rccl_sweep.py
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("# Auto-generated sweep script\n")
            f.write(f"# Generated sweeps: {len(configs)}\n")
            f.write("\n")
            f.write("set -e\n")
            f.write("\n")
            
            for i, config in enumerate(configs, 1):
                f.write(f"# Sweep {i}: {config.description}\n")
                f.write(f"{config.to_command(sweep_script)}\n")
                f.write("\n")
            
            f.write("echo 'All sweeps completed!'\n")
        
        # Make executable
        output_path.chmod(0o755)


def main():
    """Command-line interface for sweep planning."""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(
        description='Plan RCCL sweeps based on hotspot analysis'
    )
    parser.add_argument(
        '--hotspots-json',
        type=Path,
        help='Path to hotspots JSON file from hotspot_analyzer'
    )
    parser.add_argument(
        '--output-script',
        type=Path,
        default=None,
        help='Output path for generated sweep script'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List sweep commands to stdout'
    )
    
    args = parser.parse_args()
    
    planner = SweepPlanner()
    
    if args.hotspots_json:
        # Load hotspots from JSON
        with open(args.hotspots_json) as f:
            hotspot_data = json.load(f)
        
        hotspots = [
            Hotspot(**{k: v for k, v in hs.items() 
                      if k not in ['start_bytes_human', 'end_bytes_human']})
            for hs in hotspot_data
        ]
        
        configs = planner.plan_from_hotspots(hotspots)
    else:
        # Generate initial sweep plan
        configs = planner.plan_initial_sweep()
    
    if args.list:
        for config in configs:
            print(config.to_command())
    
    if args.output_script:
        planner.generate_sweep_script(configs, args.output_script)
        print(f"Generated sweep script: {args.output_script}")


if __name__ == '__main__':
    main()

