#!/usr/bin/env python3
"""
Auto-Tune Pipeline Module

Orchestrates the full RCCL tuning workflow:
1. Run sweeps using rccl_sweep.py
2. Merge metrics from all runs
3. Optimize metrics to select best configs
4. Detect hotspots in optimized metrics
5. If hotspots found, plan and run targeted sweeps
6. Generate final tuner configuration
"""

import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import shutil

from .hotspot_analyzer import HotspotAnalyzer, Hotspot
from .sweep_planner import SweepPlanner, SweepConfig
from .config_generator import TunerConfigGenerator


@dataclass
class PipelineConfig:
    """Configuration for the auto-tune pipeline."""
    # Output directory
    output_dir: Path = Path('./sweep_results')
    
    # Initial sweep settings
    nodes: str = "1-2"
    channels: str = "32:256:32"
    collectives: List[str] = field(default_factory=lambda: ['all_reduce', 'all_gather'])
    algos: List[str] = field(default_factory=lambda: ['RING', 'TREE'])
    protos: List[str] = field(default_factory=lambda: ['SIMPLE'])
    min_size: str = "1M"
    max_size: str = "512M"
    
    # Hotspot detection settings
    hotspot_threshold: float = 0.10
    hotspot_min_drop_gbps: float = 0.0
    
    # Refinement settings
    max_iterations: int = 3
    max_hotspots_per_iteration: int = 10
    
    # Output paths
    tuner_output: Optional[Path] = None
    report_output: Optional[Path] = None
    
    # Tool paths (relative to script directory)
    sweep_script: str = "./rccl_sweep.py"
    merge_script: str = "./merge_metrics.py"
    optimize_script: str = "./optimize_metrics.py"
    filter_script: str = "./filter_metrics.py"
    servers_file: str = "./servers.txt"


@dataclass
class PipelineResult:
    """Result of a pipeline run."""
    success: bool
    iterations: int
    total_sweeps: int
    hotspots_detected: int
    hotspots_remaining: int
    tuner_config_path: Optional[Path]
    report_path: Optional[Path]
    duration_seconds: float
    error_message: Optional[str] = None


class AutoTunePipeline:
    """
    Orchestrates the full RCCL auto-tuning workflow.
    """
    
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        verbose: bool = True,
        dry_run: bool = False,
    ):
        """
        Initialize the pipeline.
        
        Args:
            config: Pipeline configuration
            verbose: Print progress information
            dry_run: If True, print commands without executing sweeps
        """
        self.config = config or PipelineConfig()
        self.verbose = verbose
        self.dry_run = dry_run
        
        # Ensure output directory exists
        self.config.output_dir = Path(self.config.output_dir)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.hotspot_analyzer = HotspotAnalyzer(
            threshold=self.config.hotspot_threshold,
            min_drop_gbps=self.config.hotspot_min_drop_gbps,
        )
        self.sweep_planner = SweepPlanner()
        self.config_generator = TunerConfigGenerator(merge_ranges=True)
        
        # Track state
        self.iteration = 0
        self.total_sweeps = 0
        self.hotspots_detected = 0
        
        # Track already-targeted hotspots to avoid repeating same sweeps
        # Key: (collective, num_nodes, start_bytes, end_bytes)
        self._targeted_hotspots: set = set()
        
        # Parse constraints for filtering
        self._node_range = self._parse_node_range(self.config.nodes)
        self._collective_set = self._normalize_collectives(self.config.collectives)
        self._algo_set = set(a.upper() for a in self.config.algos)
    
    def _parse_node_range(self, nodes_str: str) -> set:
        """Parse node range string to set of valid node counts."""
        if '-' in nodes_str:
            start, end = nodes_str.split('-', 1)
            return set(range(int(start), int(end) + 1))
        return {int(nodes_str)}
    
    def _normalize_collectives(self, collectives: List[str]) -> set:
        """Normalize collective names to match metrics format (with _perf suffix)."""
        normalized = set()
        for coll in collectives:
            if coll.endswith('_perf'):
                normalized.add(coll)
            else:
                normalized.add(f"{coll.replace('-', '_')}_perf")
        return normalized
    
    def _filter_hotspots_by_constraints(self, hotspots: List[Hotspot]) -> List[Hotspot]:
        """
        Filter hotspots to only those matching initial constraints.
        
        Filters by:
        - Node count within specified range
        - Collective in specified list
        """
        filtered = []
        for hs in hotspots:
            # Check node count
            if hs.num_nodes not in self._node_range:
                continue
            
            # Check collective (normalize for comparison)
            hs_collective = hs.collective
            if not hs_collective.endswith('_perf'):
                hs_collective = f"{hs_collective}_perf"
            if hs_collective not in self._collective_set:
                continue
            
            filtered.append(hs)
        
        return filtered
    
    def _get_hotspot_signature(self, hs: Hotspot) -> tuple:
        """
        Get a unique signature for a hotspot to track if it's been targeted.
        
        Uses (collective, num_nodes, start_bytes, end_bytes) as the key.
        """
        return (hs.collective, hs.num_nodes, hs.start_bytes, hs.end_bytes)
    
    def _filter_already_targeted(self, hotspots: List[Hotspot]) -> List[Hotspot]:
        """
        Filter out hotspots that have already been targeted in previous iterations.
        
        Returns:
            List of hotspots that haven't been targeted yet
        """
        new_hotspots = []
        for hs in hotspots:
            sig = self._get_hotspot_signature(hs)
            if sig not in self._targeted_hotspots:
                new_hotspots.append(hs)
        return new_hotspots
    
    def _mark_hotspots_targeted(self, hotspots: List[Hotspot]) -> None:
        """Mark hotspots as having been targeted."""
        for hs in hotspots:
            sig = self._get_hotspot_signature(hs)
            self._targeted_hotspots.add(sig)
    
    def run(self) -> PipelineResult:
        """
        Run the full auto-tuning pipeline.
        
        Returns:
            PipelineResult with status and outputs
        """
        start_time = time.time()
        
        try:
            self._log("=" * 60)
            self._log("  RCCL Auto-Tune Pipeline")
            self._log("=" * 60)
            self._log(f"Output directory: {self.config.output_dir}")
            self._log(f"Max iterations: {self.config.max_iterations}")
            self._log("")
            
            # Run initial sweep
            self._log("Phase 1: Running initial sweeps...")
            self._run_initial_sweeps()
            
            # Filter errors, merge, and optimize
            self._log("\nPhase 2: Filtering, merging, and optimizing metrics...")
            self._filter_error_entries()
            self._merge_metrics()
            optimized_path = self._optimize_metrics()
            
            # Hotspot refinement loop
            # In dry-run mode, only show one iteration since data doesn't change
            max_iter = 1 if self.dry_run else self.config.max_iterations
            remaining_hotspots = []
            
            for self.iteration in range(1, max_iter + 1):
                self._log(f"\nPhase 3.{self.iteration}: Hotspot detection (iteration {self.iteration})...")
                
                all_hotspots = self.hotspot_analyzer.analyze(optimized_path)
                
                # Filter hotspots to only those matching initial constraints
                hotspots = self._filter_hotspots_by_constraints(all_hotspots)
                
                # Filter out hotspots that have already been targeted (avoid repeating same sweeps)
                new_hotspots = self._filter_already_targeted(hotspots)
                already_targeted_count = len(hotspots) - len(new_hotspots)
                
                self.hotspots_detected += len(new_hotspots)
                
                if not new_hotspots:
                    if already_targeted_count > 0:
                        self._log(f"  All {already_targeted_count} hotspots have already been targeted - stopping refinement")
                        self._log(f"  (These hotspots may require manual tuning or different parameters)")
                    elif all_hotspots:
                        self._log("  No hotspots detected within constraints - optimization complete!")
                        self._log(f"  (Filtered out {len(all_hotspots)} hotspots outside node/collective constraints)")
                    else:
                        self._log("  No hotspots detected - optimization complete!")
                    remaining_hotspots = hotspots
                    break
                
                self._log(f"  Detected {len(new_hotspots)} new hotspots (within constraints)")
                if already_targeted_count > 0:
                    self._log(f"  (Skipping {already_targeted_count} already-targeted hotspots)")
                
                # Prioritize and limit hotspots
                priority_hotspots = self.hotspot_analyzer.prioritize_hotspots(
                    new_hotspots,
                    max_hotspots=self.config.max_hotspots_per_iteration,
                )
                
                # Mark these hotspots as targeted so we don't repeat them
                self._mark_hotspots_targeted(priority_hotspots)
                
                # Plan targeted sweeps - only include alternatives from our algo set
                targeted_configs = self.sweep_planner.plan_from_hotspots(
                    priority_hotspots,
                    include_alternatives=True,
                    allowed_algos=self._algo_set,
                )
                
                if not targeted_configs:
                    self._log("  No targeted sweeps planned")
                    remaining_hotspots = hotspots
                    break
                
                self._log(f"  Planned {len(targeted_configs)} targeted sweeps")
                
                # Run targeted sweeps
                self._run_targeted_sweeps(targeted_configs)
                
                # Re-filter, re-merge, and re-optimize
                self._filter_error_entries()
                self._merge_metrics()
                optimized_path = self._optimize_metrics()
                
                remaining_hotspots = hotspots
            
            # In dry-run, note the iteration limit
            if self.dry_run and self.config.max_iterations > 1:
                self._log(f"\n  [DRY RUN] Would continue for up to {self.config.max_iterations} iterations total")
            
            # Generate final outputs
            self._log("\nPhase 4: Generating tuner configuration...")
            tuner_path = self._generate_tuner_config(optimized_path)
            report_path = self._generate_report(optimized_path)
            
            duration = time.time() - start_time
            
            self._log("\n" + "=" * 60)
            self._log("  Pipeline Complete!")
            self._log("=" * 60)
            self._log(f"Iterations: {self.iteration}")
            self._log(f"Total sweeps: {self.total_sweeps}")
            self._log(f"Hotspots detected: {self.hotspots_detected}")
            self._log(f"Hotspots remaining: {len(remaining_hotspots)}")
            self._log(f"Duration: {duration:.1f}s")
            self._log(f"Tuner config: {tuner_path}")
            self._log(f"Report: {report_path}")
            
            return PipelineResult(
                success=True,
                iterations=self.iteration,
                total_sweeps=self.total_sweeps,
                hotspots_detected=self.hotspots_detected,
                hotspots_remaining=len(remaining_hotspots),
                tuner_config_path=tuner_path,
                report_path=report_path,
                duration_seconds=duration,
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self._log(f"\nError: {e}")
            return PipelineResult(
                success=False,
                iterations=self.iteration,
                total_sweeps=self.total_sweeps,
                hotspots_detected=self.hotspots_detected,
                hotspots_remaining=0,
                tuner_config_path=None,
                report_path=None,
                duration_seconds=duration,
                error_message=str(e),
            )
    
    def _run_initial_sweeps(self) -> None:
        """Run initial broad sweeps."""
        for collective in self.config.collectives:
            for algo in self.config.algos:
                cmd = [
                    sys.executable, self.config.sweep_script,
                    '--servers', self.config.servers_file,
                    '--nodes', self.config.nodes,
                    '--collective', collective,
                    '--channels', self.config.channels,
                    '--algo', algo,
                    '--proto', ','.join(self.config.protos),
                    '--min-size', self.config.min_size,
                    '--max-size', self.config.max_size,
                ]
                
                self._run_sweep_command(cmd)
    
    def _run_targeted_sweeps(self, configs: List[SweepConfig]) -> None:
        """Run targeted sweeps based on hotspot analysis."""
        for config in configs:
            cmd = [sys.executable, self.config.sweep_script]
            cmd.extend(['--servers', self.config.servers_file])
            cmd.extend(config.to_command_args())
            
            self._run_sweep_command(cmd)
    
    def _run_sweep_command(self, cmd: List[str]) -> None:
        """Execute a sweep command."""
        cmd_str = ' '.join(cmd)
        self.total_sweeps += 1
        
        if self.dry_run:
            self._log(f"  [DRY RUN] {cmd_str}")
            return
        
        self._log(f"  Running: {cmd_str}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.config.output_dir.parent,
                capture_output=True,
                text=True,
            )
            
            if result.returncode != 0:
                self._log(f"    Warning: Sweep returned {result.returncode}")
                if result.stderr:
                    self._log(f"    stderr: {result.stderr[:200]}")
                    
        except Exception as e:
            self._log(f"    Error running sweep: {e}")
    
    def _filter_error_entries(self) -> int:
        """
        Filter out entries with errors from all metrics.csv files.
        
        Runs filter_metrics.py --prune-err on each run_*/metrics.csv file
        to remove entries with non-zero errors_oop or errors_ip.
        
        Returns:
            Total number of entries pruned across all files
        """
        total_pruned = 0
        
        # Find all run_* directories
        run_dirs = sorted(self.config.output_dir.glob('run_*'))
        
        if not run_dirs:
            if self.dry_run:
                self._log(f"  [DRY RUN] Would filter errors from metrics.csv files")
            return 0
        
        if self.dry_run:
            self._log(f"  [DRY RUN] Would filter errors from {len(run_dirs)} run directories")
            return 0
        
        for run_dir in run_dirs:
            metrics_file = run_dir / 'metrics.csv'
            if not metrics_file.exists():
                continue
            
            cmd = [
                sys.executable,
                self.config.filter_script,
                str(metrics_file),
                '--prune-err',
            ]
            
            result = subprocess.run(
                cmd,
                cwd=self.config.output_dir.parent,
                capture_output=True,
                text=True,
            )
            
            if result.returncode == 0 and result.stdout:
                # Parse "Pruned N entries with errors" from output
                for line in result.stdout.split('\n'):
                    if 'Pruned' in line and 'entries' in line:
                        try:
                            parts = line.split()
                            pruned = int(parts[1])
                            total_pruned += pruned
                        except (IndexError, ValueError):
                            pass
        
        if total_pruned > 0:
            self._log(f"  Filtered {total_pruned} error entries from metrics files")
        
        return total_pruned
    
    def _merge_metrics(self) -> Path:
        """Merge all metrics files."""
        merged_path = self.config.output_dir / 'merged_metrics.csv'
        
        if self.dry_run:
            self._log(f"  [DRY RUN] Would merge metrics to {merged_path}")
            return merged_path
        
        cmd = [
            sys.executable,
            self.config.merge_script,
            '--base-path', str(self.config.output_dir),
            '-o', str(merged_path),
        ]
        
        self._log(f"  Merging to: {merged_path}")
        
        result = subprocess.run(
            cmd,
            cwd=self.config.output_dir.parent,
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Merge failed: {result.stderr}")
        
        return merged_path
    
    def _optimize_metrics(self) -> Path:
        """Optimize merged metrics."""
        merged_path = self.config.output_dir / 'merged_metrics.csv'
        optimized_path = self.config.output_dir / 'optimized_metrics.csv'
        
        if self.dry_run:
            self._log(f"  [DRY RUN] Would optimize to {optimized_path}")
            return optimized_path
        
        cmd = [
            sys.executable,
            self.config.optimize_script,
            str(merged_path),
            '-o', str(optimized_path),
        ]
        
        self._log(f"  Optimizing to: {optimized_path}")
        
        result = subprocess.run(
            cmd,
            cwd=self.config.output_dir.parent,
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Optimization failed: {result.stderr}")
        
        return optimized_path
    
    def _generate_tuner_config(self, optimized_path: Path) -> Path:
        """Generate tuner configuration file."""
        if self.config.tuner_output:
            output_path = Path(self.config.tuner_output)
        else:
            output_path = self.config.output_dir / 'generated_tuner.csv'
        
        if self.dry_run:
            self._log(f"  [DRY RUN] Would generate tuner config: {output_path}")
            return output_path
        
        self.config_generator.generate(
            optimized_path,
            output_path,
            include_algo_proto=True,  # Include actual algo/proto from metrics
        )
        
        self._log(f"  Generated: {output_path}")
        return output_path
    
    def _generate_report(self, optimized_path: Path) -> Path:
        """Generate CSV report of tuning results."""
        if self.config.report_output:
            output_path = Path(self.config.report_output)
        else:
            output_path = self.config.output_dir / 'tuning_report.csv'
        
        if self.dry_run:
            self._log(f"  [DRY RUN] Would generate report: {output_path}")
            return output_path
        
        # Read optimized metrics and generate report
        entries = self.config_generator.generate(
            optimized_path,
            self.config.output_dir / '_temp.conf',  # Temporary
            include_algo_proto=True,
        )
        
        self.config_generator.generate_csv_report(entries, output_path)
        
        # Clean up temp file
        temp_conf = self.config.output_dir / '_temp.conf'
        if temp_conf.exists():
            temp_conf.unlink()
        
        self._log(f"  Generated: {output_path}")
        return output_path
    
    def _log(self, message: str) -> None:
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            print(message)


def load_config_yaml(config_path: Path) -> PipelineConfig:
    """Load pipeline configuration from YAML file."""
    import yaml
    
    with open(config_path) as f:
        data = yaml.safe_load(f)
    
    config = PipelineConfig()
    
    # Map YAML keys to config attributes
    if 'sweep' in data:
        sweep = data['sweep']
        if 'nodes' in sweep:
            config.nodes = str(sweep['nodes'])
        if 'channels' in sweep:
            config.channels = str(sweep['channels'])
        if 'collectives' in sweep:
            config.collectives = sweep['collectives']
        if 'algos' in sweep:
            config.algos = sweep['algos']
        if 'protos' in sweep:
            config.protos = sweep['protos']
        if 'min_size' in sweep:
            config.min_size = str(sweep['min_size'])
        if 'max_size' in sweep:
            config.max_size = str(sweep['max_size'])
    
    if 'hotspot' in data:
        hotspot = data['hotspot']
        if 'threshold' in hotspot:
            config.hotspot_threshold = float(hotspot['threshold'])
        if 'min_drop_gbps' in hotspot:
            config.hotspot_min_drop_gbps = float(hotspot['min_drop_gbps'])
        if 'max_iterations' in hotspot:
            config.max_iterations = int(hotspot['max_iterations'])
    
    if 'output' in data:
        output = data['output']
        if 'dir' in output:
            config.output_dir = Path(output['dir'])
        if 'tuner_conf' in output:
            config.tuner_output = Path(output['tuner_conf'])
        if 'report_csv' in output:
            config.report_output = Path(output['report_csv'])
    
    return config


def main():
    """Command-line interface for the pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run the RCCL auto-tuning pipeline'
    )
    parser.add_argument(
        '--config',
        type=Path,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default='./sweep_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--nodes', '-n',
        default='1-2',
        help='Node range (e.g., "1-2" or "1-4")'
    )
    parser.add_argument(
        '--channels', '-c',
        default='32:256:32',
        help='Channel sweep range (e.g., "32:256:32")'
    )
    parser.add_argument(
        '--max-iterations',
        type=int,
        default=3,
        help='Maximum hotspot refinement iterations'
    )
    parser.add_argument(
        '--hotspot-threshold',
        type=float,
        default=0.10,
        help='Hotspot drop threshold (e.g., 0.10 for 10%%)'
    )
    parser.add_argument(
        '--dry-run', '-d',
        action='store_true',
        help='Show commands without executing'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Reduce output verbosity'
    )
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = load_config_yaml(args.config)
    else:
        config = PipelineConfig(
            output_dir=Path(args.output_dir),
            nodes=args.nodes,
            channels=args.channels,
            max_iterations=args.max_iterations,
            hotspot_threshold=args.hotspot_threshold,
        )
    
    # Run pipeline
    pipeline = AutoTunePipeline(
        config=config,
        verbose=not args.quiet,
        dry_run=args.dry_run,
    )
    
    result = pipeline.run()
    
    if not result.success:
        sys.exit(1)


if __name__ == '__main__':
    main()

