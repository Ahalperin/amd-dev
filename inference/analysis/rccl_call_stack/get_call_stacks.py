#!/usr/bin/env python3
"""
Script to analyze Perfetto trace files and find call stacks where NCCL kernels were launched from.
"""

import argparse
import sys
from perfetto.trace_processor import TraceProcessor


def find_function_slices(tp, function_name):
    """Find all CPU slices matching the given function name."""
    function_name_escaped = function_name.replace("'", "''")
    compare_op = 'like' if '%' in function_name else '='
    query = f"""
    SELECT
      id,
      name,
      ts,
      dur,
      ts + dur as end_ts
    FROM slice
    WHERE
      name {compare_op} '{function_name_escaped}'
      AND cat != 'kernel'
    ORDER BY ts
    """
    return list(tp.query(query))


def select_slice_interactively(slices):
    """Interactively ask the user to select a slice from the list."""
    if len(slices) == 0:
        print("No slices found matching the function name.", file=sys.stderr)
        return None
    
    print(f"\nFound {len(slices)} matching slices:\n", file=sys.stderr)
    print(f"{'Index':<8} {'Start Time':<20} {'Duration':<20} {'End Time':<20}", file=sys.stderr)
    print("-" * 80, file=sys.stderr)
    
    for idx, slice_info in enumerate(slices):
        start_ts = slice_info.ts
        duration = slice_info.dur
        end_ts = slice_info.end_ts
        print(f"{idx:<8} {start_ts:<20} {duration:<20} {end_ts:<20}", file=sys.stderr)
    
    while True:
        try:
            choice = input(f"\nSelect a slice index (0-{len(slices)-1}): ").strip()
            index = int(choice)
            if 0 <= index < len(slices):
                return slices[index]
            else:
                print(f"Index must be between 0 and {len(slices)-1}.", file=sys.stderr)
        except ValueError:
            print("Please enter a valid integer.", file=sys.stderr)
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled by user.", file=sys.stderr)
            sys.exit(1)


def get_available_kernels(tp, start_ts, duration):
    """Query for all distinct kernel names that ran during the time window."""
    end_ts = start_ts + duration
    query = f"""
    SELECT DISTINCT
      name,
      COUNT(*) as count
    FROM slice
    WHERE
      cat = 'kernel'
      AND ts >= {start_ts}
      AND ts < {end_ts}
    GROUP BY name
    ORDER BY count DESC
    """
    return list(tp.query(query))


def build_query(kernel_name, start_ts, duration, max_depth):
    """Build the SQL query with the provided parameters."""
    end_ts = start_ts + duration
    # Escape single quotes in kernel name for SQL
    kernel_name_escaped = kernel_name.replace("'", "''")
    return f"""WITH recursive
  -- Start with the specific CPU slices you found
  target_cpu_slices AS (
    SELECT
      cpu_slice.id,
      cpu_slice.name,
      cpu_slice.ts,
      cpu_slice.parent_id
    FROM slice AS gpu_slice
    LEFT JOIN flow ON gpu_slice.id = flow.slice_in
    LEFT JOIN slice AS cpu_slice ON cpu_slice.id = flow.slice_out
    WHERE
      gpu_slice.cat = 'kernel'
      AND gpu_slice.name = '{kernel_name_escaped}'
      AND cpu_slice.ts > {start_ts}
      AND cpu_slice.ts < {end_ts}
  ),
  -- Recursively find all ancestors of those slices
  ancestors (slice_id, name, parent_id, depth, orig_id) AS (
    SELECT
      id,
      name,
      parent_id,
      0 AS depth,
      id AS orig_id
    FROM target_cpu_slices
    UNION ALL
    SELECT
      s.id,
      RTRIM(s.name, '0123456789'), -- Remove layer number so equivalent stacks are grouped together
      s.parent_id,
      a.depth + 1,
      orig_id
    FROM slice AS s
    JOIN ancestors AS a ON s.id = a.parent_id
    WHERE a.depth < {max_depth} -- Limit depth
  ),
  -- Calculate the concatenated string for each initial target slice
  -- Build the ordered stack strings
  stacked_results AS (
    SELECT orig_id, GROUP_CONCAT(name, ' <- ') as stack FROM ancestors GROUP BY orig_id ORDER BY depth
  )
-- Select the final results and group by the pre-computed stack string
SELECT
  stack, count(*) as count
FROM stacked_results
GROUP BY stack
ORDER BY count DESC"""


def main():
    parser = argparse.ArgumentParser(
        description='Analyze Perfetto trace files to find call stacks where NCCL kernels were launched from.'
    )
    parser.add_argument(
        'trace_file',
        type=str,
        help='Path to the Perfetto trace file (.perfetto-trace or .pb)'
    )
    parser.add_argument(
        '--function-name',
        type=str,
        default='/sgl-workspace/sglang/python/sglang/srt/managers/tp_worker.py(232): forward_batch_generation',
        help='Function name to search for in CPU slices (default: /sgl-workspace/sglang/python/sglang/srt/managers/tp_worker.py(232): forward_batch_generation)'
    )
    parser.add_argument(
        '--index',
        type=int,
        help='Index of the slice to use (if not provided, will interactively ask)'
    )
    parser.add_argument(
        '--start-ts',
        type=int,
        help='Start timestamp (overrides function-based selection)'
    )
    parser.add_argument(
        '--duration',
        type=int,
        help='Duration in timestamp units (overrides function-based selection)'
    )
    parser.add_argument(
        '--kernel-name',
        type=str,
        default='ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)',
        help='Name of the NCCL kernel to search for (default: ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>))'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output file path (default: print to stdout)'
    )
    parser.add_argument(
        '--stack-depth',
        type=int,
        default=12,
        help='Maximum depth for call stack traversal (default: 12)'
    )

    args = parser.parse_args()

    try:
        # Open the trace file
        print(f"Loading trace file: {args.trace_file}", file=sys.stderr)
        tp = TraceProcessor(trace=args.trace_file)

        # Determine start_ts and duration
        # If both start_ts and duration are provided, use them directly
        # Otherwise, find them from function slices
        if args.start_ts is not None and args.duration is not None:
            start_ts = args.start_ts
            duration = args.duration
            print(f"Using provided timestamps: start_ts={start_ts}, duration={duration}", file=sys.stderr)
        else:
            print(f"Searching for function: {args.function_name}", file=sys.stderr)
            slices = find_function_slices(tp, args.function_name)
            
            if len(slices) == 0:
                print(f"Error: No slices found matching function name: {args.function_name}", file=sys.stderr)
                sys.exit(1)
            
            # Select slice
            if args.index is not None:
                if args.index < 0 or args.index >= len(slices):
                    print(f"Error: Index {args.index} is out of range (0-{len(slices)-1})", file=sys.stderr)
                    sys.exit(1)
                selected_slice = slices[args.index]
            else:
                selected_slice = select_slice_interactively(slices)
                if selected_slice is None:
                    sys.exit(1)
            
            start_ts = selected_slice.ts
            duration = selected_slice.dur
            print(f"Using slice: start_ts={start_ts}, duration={duration}", file=sys.stderr)

        # Build the query
        query = build_query(args.kernel_name, start_ts, duration, args.stack_depth)

        # Execute the query
        print("Executing query...", file=sys.stderr)
        result = list(tp.query(query))

        # Process and output results
        output_file = open(args.output, 'w') if args.output else sys.stdout

        if len(result) == 0:
            print("No call stacks found matching the criteria.", file=output_file)
            print("\nQuerying for available kernel names in the time window...", file=output_file)
            available_kernels = get_available_kernels(tp, start_ts, duration)
            
            if len(available_kernels) == 0:
                print("No kernels found in the specified time window.", file=output_file)
            else:
                # Check if the target kernel is in the list
                target_kernel_found = any(row.name == args.kernel_name for row in available_kernels)
                
                if target_kernel_found:
                    print(f"\nThe kernel '{args.kernel_name}' was found in the trace.", file=output_file)
                    print("This kernel was probably launched as part of a graph and we don't have tracking data for it.", file=output_file)
                else:
                    print(f"\nCould not find the kernel '{args.kernel_name}' in the available kernels.", file=output_file)
                    possible_relevant_keywords = ['nccl', 'reduce', 'gather', 'broadcast', 'scatter']
                    filtered_kernels = [
                        row for row in available_kernels
                        if any(keyword in row.name.lower() for keyword in possible_relevant_keywords)
                    ]
                    
                    if len(filtered_kernels) == 0:
                        print("No relevant kernels found in the time window.", file=output_file)
                    else:
                        print(f"\nFound {len(filtered_kernels)} possibly relevant kernels:\n", file=output_file)
                        print(f"{'Count':<10} {'Kernel Name'}", file=output_file)
                        print("-" * 100, file=output_file)
                        for row in filtered_kernels:
                            count = row.count
                            name = row.name
                            print(f"{count:<10} {name}", file=output_file)
        else:
            print(f"\nFound {len(result)} unique call stacks:\n", file=output_file)
            for row in result:
                count = row.count
                stack = row.stack.replace(" <- ", " <-\n")
                print(f"{stack}\nAppeared {count} times\n", file=output_file)

        if args.output:
            output_file.close()
            print(f"\nResults written to: {args.output}", file=sys.stderr)

        tp.close()

    except Exception as e:
        print(f"Error processing trace file: {e}", file=sys.stderr)
        raise


if __name__ == '__main__':
    main()