#!/usr/bin/env python3
"""
Analyze channel 0 timing details
"""
import json

print("Loading NPKit JSON file...")
with open('/home/dn/amd-dev/rccl_mpi_profile/2025-11-06_00.37/npkit_json_rank_0/npkit_event_trace.json', 'r') as f:
    data = json.load(f)

# Match BEGIN and END events for channel 0
begin_stack = {}
channel_0_ops = []

for event in data['traceEvents']:
    pid = event.get('pid')
    tid = event.get('tid')
    key = (pid, tid)
    
    if event['ph'] == 'B' and 'name' in event:
        if key not in begin_stack:
            begin_stack[key] = []
        begin_stack[key].append(event)
    elif event['ph'] == 'E':
        if key in begin_stack and len(begin_stack[key]) > 0:
            begin_event = begin_stack[key].pop()
            duration_us = event['ts'] - begin_event['ts']
            
            if (begin_event['name'] == 'NPKIT_EVENT_ALL_REDUCE_RING_ENTRY' and 
                'args' in begin_event and 
                begin_event['args'].get('rank') == 0 and
                begin_event['args'].get('buf_idx') == 0):  # Channel 0
                
                channel_0_ops.append({
                    'seq': begin_event['args']['seq'],
                    'duration_us': duration_us,
                    'size_bytes': begin_event['args'].get('size_0', 0),
                    'begin_ts': begin_event['ts'],
                    'end_ts': event['ts']
                })

# Sort by sequence
channel_0_ops.sort(key=lambda x: x['seq'])

print("\n" + "="*80)
print("CHANNEL 0 DETAILED TIMING")
print("="*80)
print("\nChannel Info:")
print("  Ring Pattern: 8 -> 0 -> 1")
print("  Role: Receive from NETWORK (rank 8), Send via XGMI (to rank 1)")
print(f"  Total Operations: {len(channel_0_ops)}")

if channel_0_ops:
    durations = [op['duration_us'] for op in channel_0_ops]
    avg_duration = sum(durations) / len(durations)
    min_duration = min(durations)
    max_duration = max(durations)
    std_dev = (sum((d - avg_duration)**2 for d in durations) / len(durations))**0.5
    
    print(f"\nStatistics:")
    print(f"  Average duration: {avg_duration:.2f} μs")
    print(f"  Min duration:     {min_duration:.2f} μs")
    print(f"  Max duration:     {max_duration:.2f} μs")
    print(f"  Std deviation:    {std_dev:.2f} μs")
    
    # Data size
    sizes = [op['size_bytes'] for op in channel_0_ops if op['size_bytes'] > 0]
    if sizes:
        avg_size_mb = sum(sizes) / len(sizes) / 1024 / 1024
        print(f"  Average data size: {avg_size_mb:.2f} MB")
        
        # Bandwidth
        bw_gbps = (avg_size_mb) / (avg_duration / 1e6) / 1024
        print(f"  Average bandwidth: {bw_gbps:.2f} GB/s")
    
    print(f"\nAll Operations:")
    print(f"  {'Seq':>4} {'Duration (μs)':>14} {'Size (MB)':>12} {'Begin Timestamp':>20} {'End Timestamp':>20}")
    print("  " + "-"*76)
    
    for op in channel_0_ops:
        size_mb = op['size_bytes'] / 1024 / 1024 if op['size_bytes'] > 0 else 0
        print(f"  {op['seq']:>4} {op['duration_us']:>14.2f} {size_mb:>12.2f} {op['begin_ts']:>20.2f} {op['end_ts']:>20.2f}")
    
    # Find extremes
    fastest_op = min(channel_0_ops, key=lambda x: x['duration_us'])
    slowest_op = max(channel_0_ops, key=lambda x: x['duration_us'])
    
    print(f"\n  Fastest operation: seq {fastest_op['seq']}, {fastest_op['duration_us']:.2f} μs")
    print(f"  Slowest operation: seq {slowest_op['seq']}, {slowest_op['duration_us']:.2f} μs")
    print(f"  Speed ratio: {slowest_op['duration_us'] / fastest_op['duration_us']:.2f}x")

else:
    print("\nNo operations found for channel 0")

print("\n" + "="*80)


