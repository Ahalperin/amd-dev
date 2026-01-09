#!/usr/bin/env python3
"""
Hotspot Analyzer Module

Wraps detect_hotspots functionality for pipeline integration.
Provides structured hotspot data and sweep suggestions.
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from detect_hotspots import (
    parse_metrics_csv,
    detect_hotspots as _detect_hotspots,
    HotSpot as _HotSpot,
    format_bytes,
)


@dataclass
class Hotspot:
    """Represents a detected performance hotspot with sweep suggestions."""
    collective: str
    num_nodes: int
    num_gpus: int
    start_bytes: int
    end_bytes: int
    expected_busbw: float
    actual_busbw_min: float
    actual_busbw_max: float
    drop_percent_max: float
    drop_percent_min: float
    algo: str
    proto: str
    nchannels: int
    data_points: int
    
    # Sweep suggestions (computed)
    suggested_channels: Optional[str] = None
    suggested_algos: Optional[List[str]] = None
    
    @classmethod
    def from_legacy(cls, hs: _HotSpot) -> 'Hotspot':
        """Convert from legacy HotSpot dataclass."""
        return cls(
            collective=hs.collective,
            num_nodes=hs.num_nodes,
            num_gpus=hs.num_gpus,
            start_bytes=hs.start_bytes,
            end_bytes=hs.end_bytes,
            expected_busbw=hs.expected_busbw,
            actual_busbw_min=hs.actual_busbw_min,
            actual_busbw_max=hs.actual_busbw_max,
            drop_percent_max=hs.drop_percent_max,
            drop_percent_min=hs.drop_percent_min,
            algo=hs.algo,
            proto=hs.proto,
            nchannels=hs.nchannels,
            data_points=hs.data_points,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d['start_bytes_human'] = format_bytes(self.start_bytes)
        d['end_bytes_human'] = format_bytes(self.end_bytes)
        return d
    
    @property
    def severity(self) -> str:
        """Classify hotspot severity based on drop percentage."""
        if self.drop_percent_max >= 30:
            return 'critical'
        elif self.drop_percent_max >= 20:
            return 'high'
        elif self.drop_percent_max >= 10:
            return 'medium'
        return 'low'


class HotspotAnalyzer:
    """
    Analyzes performance metrics to detect hotspots and generate sweep suggestions.
    """
    
    # Alternative algorithms to try for each current algorithm
    ALGO_ALTERNATIVES = {
        'RING': ['TREE', 'Direct'],
        'TREE': ['RING', 'Direct'],
        'Direct': ['RING', 'TREE'],
    }
    
    def __init__(
        self,
        threshold: float = 0.10,
        min_drop_gbps: float = 0.0,
    ):
        """
        Initialize the analyzer.
        
        Args:
            threshold: Drop percentage threshold (0.10 = 10%)
            min_drop_gbps: Minimum absolute busbw drop in GB/s to consider
        """
        self.threshold = threshold
        self.min_drop_gbps = min_drop_gbps
    
    def analyze(self, metrics_csv: Path) -> List[Hotspot]:
        """
        Analyze metrics and detect hotspots.
        
        Args:
            metrics_csv: Path to optimized_metrics.csv or merged_metrics.csv
            
        Returns:
            List of detected hotspots with sweep suggestions
        """
        # Parse metrics
        rows = parse_metrics_csv(metrics_csv)
        
        # Detect hotspots using legacy function
        legacy_hotspots = _detect_hotspots(rows, self.threshold, self.min_drop_gbps)
        
        # Convert and enhance with suggestions
        hotspots = []
        for legacy_hs in legacy_hotspots:
            hs = Hotspot.from_legacy(legacy_hs)
            self._add_sweep_suggestions(hs)
            hotspots.append(hs)
        
        return hotspots
    
    def _add_sweep_suggestions(self, hotspot: Hotspot) -> None:
        """Add sweep suggestions based on hotspot characteristics."""
        # Suggest channel sweep range
        current_channels = hotspot.nchannels or 32
        
        # Expand channel search around current value
        min_channels = max(4, current_channels // 2)
        max_channels = min(256, current_channels * 2)
        step = max(4, (max_channels - min_channels) // 8)
        
        hotspot.suggested_channels = f"{min_channels}:{max_channels}:{step}"
        
        # Suggest alternative algorithms
        hotspot.suggested_algos = self.ALGO_ALTERNATIVES.get(
            hotspot.algo, ['RING', 'TREE']
        )
    
    def to_json(self, hotspots: List[Hotspot]) -> str:
        """Convert hotspots to JSON string."""
        return json.dumps(
            [hs.to_dict() for hs in hotspots],
            indent=2,
        )
    
    def write_json(self, hotspots: List[Hotspot], output_path: Path) -> None:
        """Write hotspots to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(self.to_json(hotspots))
    
    def get_summary(self, hotspots: List[Hotspot]) -> Dict[str, Any]:
        """Get summary statistics of hotspots."""
        if not hotspots:
            return {
                'total': 0,
                'by_severity': {},
                'by_collective': {},
                'by_nodes': {},
            }
        
        summary = {
            'total': len(hotspots),
            'by_severity': {},
            'by_collective': {},
            'by_nodes': {},
            'max_drop_percent': max(hs.drop_percent_max for hs in hotspots),
            'avg_drop_percent': sum(hs.drop_percent_max for hs in hotspots) / len(hotspots),
        }
        
        for hs in hotspots:
            # By severity
            sev = hs.severity
            summary['by_severity'][sev] = summary['by_severity'].get(sev, 0) + 1
            
            # By collective
            coll = hs.collective
            summary['by_collective'][coll] = summary['by_collective'].get(coll, 0) + 1
            
            # By nodes
            nodes = hs.num_nodes
            summary['by_nodes'][nodes] = summary['by_nodes'].get(nodes, 0) + 1
        
        return summary
    
    def prioritize_hotspots(
        self,
        hotspots: List[Hotspot],
        max_hotspots: int = 10,
    ) -> List[Hotspot]:
        """
        Prioritize hotspots for targeted sweeps.
        
        Prioritizes by:
        1. Severity (critical first)
        2. Drop percentage
        3. Number of data points affected
        
        Args:
            hotspots: List of detected hotspots
            max_hotspots: Maximum number to return
            
        Returns:
            Prioritized list of hotspots for targeted sweeps
        """
        severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        
        sorted_hotspots = sorted(
            hotspots,
            key=lambda hs: (
                severity_order.get(hs.severity, 4),
                -hs.drop_percent_max,
                -hs.data_points,
            )
        )
        
        return sorted_hotspots[:max_hotspots]


def main():
    """Command-line interface for hotspot analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze RCCL metrics for performance hotspots'
    )
    parser.add_argument(
        'input_file',
        type=Path,
        help='Path to optimized_metrics.csv or merged_metrics.csv'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=None,
        help='Output JSON file path'
    )
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=0.10,
        help='Drop percentage threshold (default: 0.10 = 10%%)'
    )
    parser.add_argument(
        '--min-drop',
        type=float,
        default=0.0,
        help='Minimum absolute busbw drop in GB/s'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output JSON to stdout'
    )
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Print summary statistics'
    )
    
    args = parser.parse_args()
    
    # Analyze
    analyzer = HotspotAnalyzer(
        threshold=args.threshold,
        min_drop_gbps=args.min_drop,
    )
    hotspots = analyzer.analyze(args.input_file)
    
    # Output
    if args.json:
        print(analyzer.to_json(hotspots))
    elif args.summary:
        summary = analyzer.get_summary(hotspots)
        print(json.dumps(summary, indent=2))
    else:
        print(f"Detected {len(hotspots)} hotspots")
        for hs in hotspots:
            print(f"  [{hs.severity}] {hs.collective} @ {format_bytes(hs.start_bytes)}-{format_bytes(hs.end_bytes)}: "
                  f"-{hs.drop_percent_max:.1f}%")
    
    if args.output:
        analyzer.write_json(hotspots, args.output)
        print(f"Written to: {args.output}")


if __name__ == '__main__':
    main()

