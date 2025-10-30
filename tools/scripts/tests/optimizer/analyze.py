#!/usr/bin/env python3
"""
Analysis and visualization tools for RCCL optimization results.
"""

import argparse
import sqlite3
import json
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

# Set plotting style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class ResultsAnalyzer:
    """Analyzes and visualizes RCCL optimization results."""
    
    def __init__(self, db_path: str):
        """Initialize analyzer.
        
        Args:
            db_path: Path to results database
        """
        self.db_path = Path(db_path)
        
        if not self.db_path.exists():
            print(f"Error: Database not found: {db_path}")
            sys.exit(1)
        
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        
        # Load data
        self.df = self._load_data()
        
    def _load_data(self) -> pd.DataFrame:
        """Load optimization results into DataFrame."""
        query = """
            SELECT 
                id, timestamp, parameters, 
                busbw_oop, busbw_ip, algbw_oop, algbw_ip,
                time_oop, time_ip, execution_time, status
            FROM optimization_runs
            WHERE status = 'success' AND busbw_oop IS NOT NULL
            ORDER BY timestamp
        """
        
        rows = []
        cursor = self.conn.cursor()
        cursor.execute(query)
        
        for row in cursor.fetchall():
            row_dict = dict(row)
            # Parse parameters JSON
            params = json.loads(row_dict['parameters'])
            row_dict.update(params)
            del row_dict['parameters']
            rows.append(row_dict)
        
        if not rows:
            print("Warning: No successful runs found in database")
            return pd.DataFrame()
        
        return pd.DataFrame(rows)
    
    def print_summary(self):
        """Print summary statistics."""
        if self.df.empty:
            print("No data to analyze")
            return
        
        print("\n" + "="*80)
        print("RCCL Optimization Results Summary")
        print("="*80 + "\n")
        
        # Basic statistics
        print(f"Total successful runs: {len(self.df)}")
        print(f"Date range: {self.df['timestamp'].min()} to {self.df['timestamp'].max()}")
        print()
        
        # Performance statistics
        metrics = ['busbw_oop', 'busbw_ip', 'algbw_oop', 'algbw_ip']
        stats_data = []
        
        for metric in metrics:
            if metric in self.df.columns:
                stats_data.append({
                    'Metric': metric,
                    'Mean': f"{self.df[metric].mean():.2f}",
                    'Std': f"{self.df[metric].std():.2f}",
                    'Min': f"{self.df[metric].min():.2f}",
                    'Max': f"{self.df[metric].max():.2f}",
                    'Improvement': f"{(self.df[metric].max() - self.df[metric].min()) / self.df[metric].min() * 100:.1f}%"
                })
        
        print("Performance Statistics (GB/s):")
        print(tabulate(stats_data, headers='keys', tablefmt='simple'))
        print()
        
        # Best configuration
        best_idx = self.df['busbw_oop'].idxmax()
        best_run = self.df.loc[best_idx]
        
        print("Best Configuration:")
        print(f"  Bus BW (out-of-place): {best_run['busbw_oop']:.2f} GB/s")
        print(f"  Bus BW (in-place):     {best_run['busbw_ip']:.2f} GB/s")
        print(f"  Run ID: {best_run['id']}")
        print()
        
        # Get parameter columns
        param_cols = [col for col in self.df.columns 
                     if col not in ['id', 'timestamp', 'busbw_oop', 'busbw_ip', 
                                   'algbw_oop', 'algbw_ip', 'time_oop', 'time_ip',
                                   'execution_time', 'status']]
        
        print("  Parameters:")
        for param in param_cols:
            print(f"    {param} = {best_run[param]}")
        print()
    
    def plot_convergence(self, output_file: str = None):
        """Plot optimization convergence over time.
        
        Args:
            output_file: Optional file to save plot
        """
        if self.df.empty:
            print("No data to plot")
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Bus bandwidth over iterations
        ax = axes[0]
        iterations = range(1, len(self.df) + 1)
        
        ax.plot(iterations, self.df['busbw_oop'], 'o-', label='Out-of-place', alpha=0.7)
        ax.plot(iterations, self.df['busbw_ip'], 's-', label='In-place', alpha=0.7)
        
        # Plot best so far
        best_so_far = self.df['busbw_oop'].cummax()
        ax.plot(iterations, best_so_far, 'r--', linewidth=2, label='Best so far', alpha=0.8)
        
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Bus Bandwidth (GB/s)', fontsize=12)
        ax.set_title('Optimization Convergence', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Execution time
        ax = axes[1]
        ax.plot(iterations, self.df['execution_time'], 'o-', color='green', alpha=0.7)
        ax.axhline(self.df['execution_time'].mean(), color='red', linestyle='--', 
                  label=f'Mean: {self.df["execution_time"].mean():.1f}s', alpha=0.7)
        
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Execution Time (s)', fontsize=12)
        ax.set_title('Test Execution Time', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved convergence plot to {output_file}")
        else:
            plt.show()
    
    def plot_parameter_importance(self, objective: str = 'busbw_oop', output_file: str = None):
        """Plot parameter importance using correlation analysis.
        
        Args:
            objective: Objective metric to analyze
            output_file: Optional file to save plot
        """
        if self.df.empty:
            print("No data to plot")
            return
        
        # Get parameter columns
        param_cols = [col for col in self.df.columns 
                     if col not in ['id', 'timestamp', 'busbw_oop', 'busbw_ip', 
                                   'algbw_oop', 'algbw_ip', 'time_oop', 'time_ip',
                                   'execution_time', 'status']]
        
        if not param_cols:
            print("No parameters to analyze")
            return
        
        # Calculate correlations
        correlations = {}
        for param in param_cols:
            # For categorical parameters, calculate mean objective for each value
            if self.df[param].dtype == 'object' or len(self.df[param].unique()) < 10:
                grouped = self.df.groupby(param)[objective].mean()
                # Use range as proxy for importance
                if len(grouped) > 1:
                    importance = (grouped.max() - grouped.min()) / self.df[objective].mean()
                    correlations[param] = importance
            else:
                # For continuous parameters, use correlation
                correlations[param] = abs(self.df[param].corr(self.df[objective]))
        
        if not correlations:
            print("Could not calculate parameter importance")
            return
        
        # Sort by importance
        sorted_params = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        params, importance = zip(*sorted_params)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, max(6, len(params) * 0.5)))
        
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(params)))
        bars = ax.barh(params, importance, color=colors)
        
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title(f'Parameter Importance for {objective}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (p, v) in enumerate(zip(params, importance)):
            ax.text(v, i, f' {v:.3f}', va='center', fontsize=10)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved parameter importance plot to {output_file}")
        else:
            plt.show()
    
    def plot_parameter_distribution(self, output_file: str = None):
        """Plot distribution of each parameter's effect on performance.
        
        Args:
            output_file: Optional file to save plot
        """
        if self.df.empty:
            print("No data to plot")
            return
        
        # Get parameter columns
        param_cols = [col for col in self.df.columns 
                     if col not in ['id', 'timestamp', 'busbw_oop', 'busbw_ip', 
                                   'algbw_oop', 'algbw_ip', 'time_oop', 'time_ip',
                                   'execution_time', 'status']]
        
        if not param_cols:
            print("No parameters to analyze")
            return
        
        # Create subplots
        n_params = len(param_cols)
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
        if n_params == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, param in enumerate(param_cols):
            ax = axes[i]
            
            # Group by parameter value and plot
            grouped = self.df.groupby(param)['busbw_oop'].agg(['mean', 'std', 'count'])
            
            x_pos = range(len(grouped))
            ax.bar(x_pos, grouped['mean'], yerr=grouped['std'], capsize=5, alpha=0.7)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(grouped.index, rotation=45, ha='right')
            ax.set_ylabel('Bus BW (GB/s)', fontsize=10)
            ax.set_title(param, fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add count labels
            for j, (idx, row) in enumerate(grouped.iterrows()):
                ax.text(j, row['mean'] + row['std'] + 5, f"n={int(row['count'])}", 
                       ha='center', fontsize=8, alpha=0.7)
        
        # Hide extra subplots
        for i in range(n_params, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Parameter Effect on Performance', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved parameter distribution plot to {output_file}")
        else:
            plt.show()
    
    def export_report(self, output_dir: str):
        """Export comprehensive analysis report.
        
        Args:
            output_dir: Directory to save report files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nGenerating analysis report in {output_path}/...\n")
        
        # Export CSV
        csv_file = output_path / "optimization_results.csv"
        self.df.to_csv(csv_file, index=False)
        print(f"✓ Exported data to {csv_file}")
        
        # Generate plots
        conv_file = output_path / "convergence_plot.png"
        self.plot_convergence(str(conv_file))
        
        imp_file = output_path / "parameter_importance.png"
        self.plot_parameter_importance(output_file=str(imp_file))
        
        dist_file = output_path / "parameter_distribution.png"
        self.plot_parameter_distribution(output_file=str(dist_file))
        
        # Generate text report
        report_file = output_path / "analysis_report.txt"
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("RCCL Optimization Analysis Report\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Database: {self.db_path}\n")
            f.write(f"Generated: {pd.Timestamp.now()}\n\n")
            
            # Summary stats
            f.write("Summary Statistics\n")
            f.write("-"*80 + "\n")
            f.write(f"Total runs: {len(self.df)}\n")
            f.write(f"Best bus BW (OOP): {self.df['busbw_oop'].max():.2f} GB/s\n")
            f.write(f"Worst bus BW (OOP): {self.df['busbw_oop'].min():.2f} GB/s\n")
            f.write(f"Mean bus BW (OOP): {self.df['busbw_oop'].mean():.2f} GB/s\n")
            f.write(f"Improvement: {(self.df['busbw_oop'].max() - self.df['busbw_oop'].min()) / self.df['busbw_oop'].min() * 100:.1f}%\n\n")
            
            # Best config
            best_idx = self.df['busbw_oop'].idxmax()
            best_run = self.df.loc[best_idx]
            
            f.write("Best Configuration\n")
            f.write("-"*80 + "\n")
            param_cols = [col for col in self.df.columns 
                         if col not in ['id', 'timestamp', 'busbw_oop', 'busbw_ip', 
                                       'algbw_oop', 'algbw_ip', 'time_oop', 'time_ip',
                                       'execution_time', 'status']]
            for param in param_cols:
                f.write(f"{param}={best_run[param]}\n")
        
        print(f"✓ Generated report: {report_file}")
        print(f"\n✓ Analysis complete! All files saved to {output_path}/")
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze RCCL optimization results',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'database',
        help='Path to results database'
    )
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Print summary statistics'
    )
    parser.add_argument(
        '--plot-convergence',
        action='store_true',
        help='Show convergence plot'
    )
    parser.add_argument(
        '--plot-importance',
        action='store_true',
        help='Show parameter importance plot'
    )
    parser.add_argument(
        '--plot-distribution',
        action='store_true',
        help='Show parameter distribution plot'
    )
    parser.add_argument(
        '--export-report',
        metavar='DIR',
        help='Export comprehensive report to directory'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Show all analysis (summary + all plots)'
    )
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = ResultsAnalyzer(args.database)
    
    try:
        # If no specific action, default to summary
        if not any([args.summary, args.plot_convergence, args.plot_importance,
                   args.plot_distribution, args.export_report, args.all]):
            args.summary = True
        
        if args.summary or args.all:
            analyzer.print_summary()
        
        if args.plot_convergence or args.all:
            analyzer.plot_convergence()
        
        if args.plot_importance or args.all:
            analyzer.plot_parameter_importance()
        
        if args.plot_distribution or args.all:
            analyzer.plot_parameter_distribution()
        
        if args.export_report:
            analyzer.export_report(args.export_report)
        
    finally:
        analyzer.close()


if __name__ == '__main__':
    main()


