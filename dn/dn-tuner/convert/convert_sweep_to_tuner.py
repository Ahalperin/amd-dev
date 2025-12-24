#!/usr/bin/env python3
"""
Convert sweep results CSV to RCCL tuner config format.

Source format (best_channels_analysis.csv):
    collective, num_nodes, message_size, message_size_human, best_num_channels, inplace_busbw

Target format (rccl_tuner.conf):
    collective_type, min_bytes, max_bytes, algorithm, protocol, channels, nNodes, nRanks, numPipeOps, regBuff
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

# Mapping from source collective names to target collective names
COLLECTIVE_MAP = {
    "all_gather_perf": "allgather",
    "all_reduce_perf": "allreduce",
    "broadcast_perf": "broadcast",
    "reduce_perf": "reduce",
    "reduce_scatter_perf": "reducescatter",
    "allgather_perf": "allgather",
    "allreduce_perf": "allreduce",
    "sendrecv_perf": "sendrecv",
    "scatter_perf": "scatter",
    "gather_perf": "gather",
    "alltoall_perf": "alltoall",
}

# Default values for fields not present in source
DEFAULT_ALGORITHM = -1
DEFAULT_PROTOCOL = -1
DEFAULT_NUM_PIPE_OPS = -1
DEFAULT_REG_BUFF = -1
RANKS_PER_NODE = 8


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert sweep results CSV to RCCL tuner config format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python convert_sweep_to_tuner.py input.csv output.conf
    python convert_sweep_to_tuner.py input.csv output.conf --ranks-per-node 8
    python convert_sweep_to_tuner.py input.csv  # outputs to stdout
        """,
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to source CSV file (best_channels_analysis.csv format)",
    )
    parser.add_argument(
        "output_file",
        type=Path,
        nargs="?",
        default=None,
        help="Path to output config file (default: stdout)",
    )
    parser.add_argument(
        "--ranks-per-node",
        type=int,
        default=RANKS_PER_NODE,
        help=f"Number of ranks per node (default: {RANKS_PER_NODE})",
    )
    parser.add_argument(
        "--header",
        action="store_true",
        help="Include header comment in output",
    )
    return parser.parse_args()


def map_collective_name(source_name: str) -> str:
    """Map source collective name to target format."""
    source_lower = source_name.lower().strip()
    if source_lower in COLLECTIVE_MAP:
        return COLLECTIVE_MAP[source_lower]
    # Try removing _perf suffix if not found
    if source_lower.endswith("_perf"):
        base_name = source_lower[:-5]
        return base_name.replace("_", "")
    return source_lower.replace("_", "")


def parse_row(row: dict, ranks_per_node: int) -> dict:
    """Parse a source row and extract relevant fields."""
    # Get values from source
    collective = row.get("collective", row.get("collective_type", ""))
    num_nodes = int(row.get("num_nodes", row.get("nNodes", 1)))
    message_size = int(row.get("message_size", row.get("bytes", 0)))
    channels = int(row.get("best_num_channels", row.get("channels", 0)))

    # Calculate derived values
    n_ranks = num_nodes * ranks_per_node

    return {
        "collective_type": map_collective_name(collective),
        "message_size": message_size,
        "algorithm": DEFAULT_ALGORITHM,
        "protocol": DEFAULT_PROTOCOL,
        "channels": channels,
        "nNodes": num_nodes,
        "nRanks": n_ranks,
        "numPipeOps": DEFAULT_NUM_PIPE_OPS,
        "regBuff": DEFAULT_REG_BUFF,
    }


def compute_byte_ranges(rows: list[dict]) -> list[dict]:
    """
    Compute min_bytes and max_bytes for each row.
    
    Rows are grouped by (collective_type, nNodes) and sorted by message_size.
    Within each group:
    - First row: min_bytes = 1
    - Subsequent rows: min_bytes = previous row's message_size + 1
    - max_bytes = current row's message_size
    """
    # Group rows by (collective_type, nNodes)
    groups = defaultdict(list)
    for row in rows:
        key = (row["collective_type"], row["nNodes"])
        groups[key].append(row)

    # Sort each group by message_size and compute byte ranges
    result = []
    for key in groups:
        group = groups[key]
        group.sort(key=lambda r: r["message_size"])

        for i, row in enumerate(group):
            if i == 0:
                min_bytes = 1
            else:
                min_bytes = group[i - 1]["message_size"] + 1

            row["min_bytes"] = min_bytes
            row["max_bytes"] = row["message_size"]
            result.append(row)

    return result


def convert_csv(input_path: Path, output_path: Path | None, ranks_per_node: int, include_header: bool):
    """Convert source CSV to target config format."""
    # Read input CSV
    with open(input_path, "r", newline="") as infile:
        reader = csv.DictReader(infile)
        rows = list(reader)

    if not rows:
        print("Warning: Input file is empty", file=sys.stderr)
        return

    # Parse rows
    parsed_rows = [parse_row(row, ranks_per_node) for row in rows]

    # Compute byte ranges (min_bytes, max_bytes) based on grouping
    converted_rows = compute_byte_ranges(parsed_rows)

    # Prepare output
    output_lines = []

    if include_header:
        output_lines.append("# Generated RCCL tuner config from sweep results")
        output_lines.append(f"# Source: {input_path}")
        output_lines.append(f"# Ranks per node: {ranks_per_node}")
        output_lines.append("# Format: collective_type,min_bytes,max_bytes,algorithm,protocol,channels,nNodes,nRanks,numPipeOps,regBuff")

    # Format each row
    for row in converted_rows:
        line = "{collective_type},{min_bytes},{max_bytes},{algorithm},{protocol},{channels},{nNodes},{nRanks},{numPipeOps},{regBuff}".format(**row)
        output_lines.append(line)

    # Write output
    output_content = "\n".join(output_lines) + "\n"

    if output_path:
        with open(output_path, "w") as outfile:
            outfile.write(output_content)
        print(f"Converted {len(converted_rows)} rows to {output_path}", file=sys.stderr)
    else:
        print(output_content, end="")


def main():
    args = parse_args()

    if not args.input_file.exists():
        print(f"Error: Input file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)

    convert_csv(args.input_file, args.output_file, args.ranks_per_node, args.header)


if __name__ == "__main__":
    main()

