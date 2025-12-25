#!/usr/bin/env python3
"""
Plot bus bandwidth (busbw_ip) vs message size for RCCL sweep results.

Usage:
    python plot_busbw.py <collective_type> <num_nodes> [--db <path>] [--output <path>]

Examples:
    python plot_busbw.py all_reduce_perf 1
    python plot_busbw.py all_reduce_perf 2 --output all_reduce_2node.png
    python plot_busbw.py reduce_scatter_perf 1 --db /path/to/sweep_results.db
"""

import argparse
import sqlite3
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
except ImportError:
    print("Error: matplotlib is required. Install with: pip install matplotlib")
    sys.exit(1)


def format_bytes(size_bytes):
    """Convert bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.0f}{unit}" if size_bytes == int(size_bytes) else f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}TB"


def query_data(db_path, collective, num_nodes):
    """Query the database for busbw_ip vs size_bytes data."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Find matching runs
    cursor.execute("""
        SELECT r.id, r.session_id, s.start_time, r.num_gpus, r.avg_busbw, r.max_busbw
        FROM sweep_runs r
        JOIN sweep_sessions s ON r.session_id = s.id
        WHERE r.collective = ? AND r.num_nodes = ?
        ORDER BY s.start_time, r.id
    """, (collective, num_nodes))
    
    runs = cursor.fetchall()
    
    if not runs:
        conn.close()
        return None, []
    
    # Get metrics for all matching runs
    all_data = []
    for run_id, session_id, start_time, num_gpus, avg_busbw, max_busbw in runs:
        cursor.execute("""
            SELECT size_bytes, busbw_ip, algo, proto, nchannels
            FROM sweep_metrics
            WHERE run_id = ?
            ORDER BY size_bytes
        """, (run_id,))
        
        metrics = cursor.fetchall()
        
        # Get min/max size for label
        if metrics:
            min_size = format_bytes(metrics[0][0])
            max_size = format_bytes(metrics[-1][0])
            label = f"Session {session_id} ({min_size}-{max_size})"
        else:
            label = f"Session {session_id}"
        
        all_data.append({
            'run_id': run_id,
            'session_id': session_id,
            'start_time': start_time,
            'num_gpus': num_gpus,
            'avg_busbw': avg_busbw,
            'max_busbw': max_busbw,
            'label': label,
            'metrics': metrics
        })
    
    conn.close()
    
    run_info = {
        'collective': collective,
        'num_nodes': num_nodes,
        'num_gpus': runs[0][3] if runs else 0
    }
    
    return run_info, all_data


def plot_busbw(run_info, all_data, output_path=None):
    """Create the bus bandwidth plot."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = plt.cm.tab10.colors
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h', '*']
    
    for idx, data in enumerate(all_data):
        metrics = data['metrics']
        if not metrics:
            continue
        
        sizes = [m[0] for m in metrics]
        busbw = [m[1] if m[1] is not None else 0 for m in metrics]
        
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        
        ax.plot(sizes, busbw, 
                marker=marker, 
                markersize=4,
                linewidth=1.5,
                color=color,
                label=data['label'],
                alpha=0.8)
    
    # Configure axes
    ax.set_xscale('log', base=2)
    ax.set_xlabel('Message Size (bytes)', fontsize=12)
    ax.set_ylabel('Bus Bandwidth - In-Place (GB/s)', fontsize=12)
    
    # Format x-axis with human-readable sizes
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format_bytes(x)))
    
    # Title
    collective_name = run_info['collective'].replace('_perf', '').replace('_', ' ').title()
    title = f"{collective_name} - {run_info['num_nodes']} Node(s), {run_info['num_gpus']} GPUs"
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Grid
    ax.grid(True, which='major', linestyle='-', alpha=0.3)
    ax.grid(True, which='minor', linestyle=':', alpha=0.2)
    
    # Legend
    if len(all_data) > 1:
        ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Plot bus bandwidth vs message size for RCCL sweep results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Collective types:
  all_reduce_perf, reduce_scatter_perf, all_gather_perf,
  alltoall_perf, broadcast_perf, reduce_perf

Examples:
  %(prog)s all_reduce_perf 1
  %(prog)s all_reduce_perf 2 --output all_reduce_2node.png
        """
    )
    
    parser.add_argument('collective', type=str,
                        help='Collective type (e.g., all_reduce_perf)')
    parser.add_argument('num_nodes', type=int,
                        help='Number of nodes (e.g., 1 or 2)')
    parser.add_argument('--db', type=str, default=None,
                        help='Path to sweep_results.db (default: ./sweep_results/sweep_results.db)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output file path (default: display interactively)')
    
    args = parser.parse_args()
    
    # Determine database path
    if args.db:
        db_path = Path(args.db)
    else:
        db_path = Path(__file__).parent / 'sweep_results' / 'sweep_results.db'
    
    if not db_path.exists():
        print(f"Error: Database not found: {db_path}")
        sys.exit(1)
    
    # Query data
    run_info, all_data = query_data(str(db_path), args.collective, args.num_nodes)
    
    if not run_info:
        print(f"Error: No data found for collective='{args.collective}', num_nodes={args.num_nodes}")
        
        # List available options
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT collective, num_nodes FROM sweep_runs ORDER BY collective, num_nodes")
        available = cursor.fetchall()
        conn.close()
        
        if available:
            print("\nAvailable combinations:")
            for coll, nodes in available:
                print(f"  {coll} {nodes}")
        sys.exit(1)
    
    # Create plot
    plot_busbw(run_info, all_data, args.output)


if __name__ == '__main__':
    main()

