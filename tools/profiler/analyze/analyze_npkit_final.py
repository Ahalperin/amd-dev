#!/usr/bin/env python3
"""
Analyze NPKit JSON for rank 0 all-reduce events
Properly matches BEGIN (ph='B') and END (ph='E') events by pid/tid
"""
import json
from collections import defaultdict

# Channel role mapping based on RCCL log (Ring patterns for rank 0)
channel_roles = {
    0: (8, 1, "recv_NET_send_XGMI"),      # 8 -> 0 -> 1
    1: (1, 2, "recv_XGMI_send_XGMI"),     # 1 -> 0 -> 2
    2: (5, 3, "recv_XGMI_send_XGMI"),     # 5 -> 0 -> 3
    3: (7, 4, "recv_XGMI_send_XGMI"),     # 7 -> 0 -> 4
    4: (3, 5, "recv_XGMI_send_XGMI"),     # 3 -> 0 -> 5
    5: (6, 8, "recv_XGMI_send_NET"),      # 6 -> 0 -> 8
    6: (4, 7, "recv_XGMI_send_XGMI"),     # 4 -> 0 -> 7
    7: (2, 6, "recv_XGMI_send_XGMI"),     # 2 -> 0 -> 6
    8: (1, 2, "recv_XGMI_send_XGMI"),
    9: (5, 3, "recv_XGMI_send_XGMI"),
    10: (7, 4, "recv_XGMI_send_XGMI"),
    11: (3, 5, "recv_XGMI_send_XGMI"),
    12: (6, 8, "recv_XGMI_send_NET"),
    13: (4, 7, "recv_XGMI_send_XGMI"),
    14: (2, 6, "recv_XGMI_send_XGMI"),
    15: (8, 1, "recv_NET_send_XGMI"),
    16: (5, 3, "recv_XGMI_send_XGMI"),
    17: (7, 4, "recv_XGMI_send_XGMI"),
    18: (3, 5, "recv_XGMI_send_XGMI"),
    19: (6, 8, "recv_XGMI_send_NET"),
    20: (4, 7, "recv_XGMI_send_XGMI"),
    21: (2, 6, "recv_XGMI_send_XGMI"),
    22: (8, 1, "recv_NET_send_XGMI"),
    23: (1, 2, "recv_XGMI_send_XGMI"),
    24: (7, 4, "recv_XGMI_send_XGMI"),
    25: (3, 5, "recv_XGMI_send_XGMI"),
    26: (6, 8, "recv_XGMI_send_NET"),
    27: (4, 7, "recv_XGMI_send_XGMI"),
    28: (2, 6, "recv_XGMI_send_XGMI"),
    29: (8, 1, "recv_NET_send_XGMI"),
    30: (1, 2, "recv_XGMI_send_XGMI"),
    31: (5, 3, "recv_XGMI_send_XGMI"),
    32: (3, 5, "recv_XGMI_send_XGMI"),
    33: (6, 8, "recv_XGMI_send_NET"),
    34: (4, 7, "recv_XGMI_send_XGMI"),
    35: (2, 6, "recv_XGMI_send_XGMI"),
    36: (8, 1, "recv_NET_send_XGMI"),
    37: (1, 2, "recv_XGMI_send_XGMI"),
    38: (5, 3, "recv_XGMI_send_XGMI"),
    39: (7, 4, "recv_XGMI_send_XGMI"),
    40: (6, 8, "recv_XGMI_send_NET"),
    41: (4, 7, "recv_XGMI_send_XGMI"),
    42: (2, 6, "recv_XGMI_send_XGMI"),
    43: (8, 1, "recv_NET_send_XGMI"),
    44: (1, 2, "recv_XGMI_send_XGMI"),
    45: (5, 3, "recv_XGMI_send_XGMI"),
    46: (7, 4, "recv_XGMI_send_XGMI"),
    47: (3, 5, "recv_XGMI_send_XGMI"),
    48: (4, 7, "recv_XGMI_send_XGMI"),
    49: (2, 6, "recv_XGMI_send_XGMI"),
    50: (8, 1, "recv_NET_send_XGMI"),
    51: (1, 2, "recv_XGMI_send_XGMI"),
    52: (5, 3, "recv_XGMI_send_XGMI"),
    53: (7, 4, "recv_XGMI_send_XGMI"),
    54: (3, 5, "recv_XGMI_send_XGMI"),
    55: (6, 8, "recv_XGMI_send_NET"),
    56: (2, 6, "recv_XGMI_send_XGMI"),
    57: (8, 1, "recv_NET_send_XGMI"),
    58: (1, 2, "recv_XGMI_send_XGMI"),
    59: (5, 3, "recv_XGMI_send_XGMI"),
    60: (7, 4, "recv_XGMI_send_XGMI"),
    61: (3, 5, "recv_XGMI_send_XGMI"),
    62: (6, 8, "recv_XGMI_send_NET"),
    63: (4, 7, "recv_XGMI_send_XGMI"),
}

print("Loading NPKit JSON file...")
with open('/home/dn/amd-dev/rccl_mpi_profile/2025-11-06_00.37/npkit_json_rank_0/npkit_event_trace.json', 'r') as f:
    data = json.load(f)

print(f"Total events in file: {len(data['traceEvents'])}")

# Match BEGIN and END events by (pid, tid)
# Stack to track begin events
begin_stack = {}  # (pid, tid) -> list of begin events
durations = []

for event in data['traceEvents']:
    pid = event.get('pid')
    tid = event.get('tid')
    key = (pid, tid)
    
    if event['ph'] == 'B' and 'name' in event:
        # Push begin event onto stack
        if key not in begin_stack:
            begin_stack[key] = []
        begin_stack[key].append(event)
    
    elif event['ph'] == 'E':
        # Pop matching begin event
        if key in begin_stack and len(begin_stack[key]) > 0:
            begin_event = begin_stack[key].pop()
            
            # Calculate duration
            duration_us = event['ts'] - begin_event['ts']
            
            # Store if it's an ALL_REDUCE_RING_ENTRY event from rank 0
            if (begin_event['name'] == 'NPKIT_EVENT_ALL_REDUCE_RING_ENTRY' and 
                'args' in begin_event and begin_event['args'].get('rank') == 0):
                
                durations.append({
                    'channel': begin_event['args']['buf_idx'],
                    'seq': begin_event['args']['seq'],
                    'duration_us': duration_us,
                    'size_bytes': begin_event['args'].get('size_0', 0),
                    'begin_ts': begin_event['ts'],
                    'end_ts': event['ts']
                })

print(f"Matched {len(durations)} all-reduce operations for rank 0")

# Organize by role
results_by_role = defaultdict(list)

for d in durations:
    role = channel_roles.get(d['channel'], (None, None, "unknown"))[2]
    results_by_role[role].append(d)

# Print analysis
print("\n" + "="*80)
print("ALL-REDUCE TIMING ANALYSIS BY RANK 0's ROLE IN CHANNEL")
print("="*80)

for role in sorted(results_by_role.keys()):
    events = results_by_role[role]
    if not events:
        continue
    
    durations_list = [e['duration_us'] for e in events]
    avg_duration = sum(durations_list) / len(durations_list)
    min_duration = min(durations_list)
    max_duration = max(durations_list)
    std_dev = (sum((d - avg_duration)**2 for d in durations_list) / len(durations_list))**0.5
    
    # Get average size
    sizes = [e['size_bytes'] for e in events if e['size_bytes'] > 0]
    avg_size_mb = sum(sizes) / len(sizes) / 1024 / 1024 if sizes else 0
    
    print(f"\n{'='*80}")
    print(f"Role: {role.upper().replace('_', ' ')}")
    print(f"{'='*80}")
    print(f"Total operations:  {len(events)}")
    print(f"Average data size: {avg_size_mb:.2f} MB")
    print(f"Average duration:  {avg_duration:.2f} μs")
    print(f"Min duration:      {min_duration:.2f} μs")
    print(f"Max duration:      {max_duration:.2f} μs")
    print(f"Std deviation:     {std_dev:.2f} μs")
    
    # Calculate effective bandwidth
    if avg_size_mb > 0 and avg_duration > 0:
        # Bandwidth = size / time
        # size in MB, time in microseconds
        # GB/s = (MB * 1024) / (μs / 1e6) / 1024 = MB / μs
        bw_gbps = (avg_size_mb) / (avg_duration / 1e6) / 1024
        print(f"Avg bandwidth:     {bw_gbps:.2f} GB/s")
    
    # Show channel breakdown
    channels = defaultdict(list)
    for e in events:
        channels[e['channel']].append(e['duration_us'])
    
    print(f"\nChannels used ({len(channels)} total):")
    for ch in sorted(channels.keys())[:5]:
        ch_durations = channels[ch]
        ch_avg = sum(ch_durations) / len(ch_durations)
        recv_from, send_to, _ = channel_roles.get(ch, (None, None, None))
        print(f"  Channel {ch:2d} (rank {recv_from} -> 0 -> {send_to}): "
              f"{len(ch_durations):3d} ops, avg {ch_avg:.2f} μs")
    
    if len(channels) > 5:
        print(f"  ... and {len(channels) - 5} more channels")

# Compare roles
print("\n" + "="*80)
print("ROLE COMPARISON SUMMARY")
print("="*80)
print(f"{'Role':<35} {'Ops':>6} {'Avg μs':>10} {'Min μs':>10} {'Max μs':>10} {'Std μs':>10} {'GB/s':>8}")
print("-"*90)

for role in ['recv_NET_send_XGMI', 'recv_XGMI_send_NET', 'recv_XGMI_send_XGMI']:
    if role not in results_by_role:
        continue
    
    events = results_by_role[role]
    durations_list = [e['duration_us'] for e in events]
    avg_duration = sum(durations_list) / len(durations_list)
    min_duration = min(durations_list)
    max_duration = max(durations_list)
    std_dev = (sum((d - avg_duration)**2 for d in durations_list) / len(durations_list))**0.5
    
    sizes = [e['size_bytes'] for e in events if e['size_bytes'] > 0]
    avg_size_mb = sum(sizes) / len(sizes) / 1024 / 1024 if sizes else 0
    bw_gbps = (avg_size_mb) / (avg_duration / 1e6) / 1024 if avg_duration > 0 and avg_size_mb > 0 else 0
    
    role_display = role.replace('_', ' ')
    print(f"{role_display:<35} {len(events):>6} {avg_duration:>10.2f} {min_duration:>10.2f} {max_duration:>10.2f} {std_dev:>10.2f} {bw_gbps:>8.2f}")

# Detailed analysis
print("\n" + "="*80)
print("DETAILED ANALYSIS - Sample Operations")
print("="*80)

for role in ['recv_NET_send_XGMI', 'recv_XGMI_send_NET', 'recv_XGMI_send_XGMI']:
    if role not in results_by_role:
        continue
    
    print(f"\n{role.upper().replace('_', ' ')}:")
    print(f"  {'#':<3} {'Ch':>3} {'Seq':>4} {'From':>4} {'To':>4} {'Duration μs':>12} {'Size MB':>10}")
    print("  " + "-"*52)
    
    events = results_by_role[role][:5]
    for i, e in enumerate(events, 1):
        recv_from, send_to, _ = channel_roles.get(e['channel'], (None, None, None))
        size_mb = e['size_bytes'] / 1024 / 1024 if e['size_bytes'] > 0 else 0
        print(f"  {i:<3} {e['channel']:>3} {e['seq']:>4} {recv_from:>4} {send_to:>4} {e['duration_us']:>12.2f} {size_mb:>10.2f}")

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)

# Calculate relative performance
roles_to_compare = ['recv_NET_send_XGMI', 'recv_XGMI_send_NET', 'recv_XGMI_send_XGMI']
role_avgs = {}

for role in roles_to_compare:
    if role in results_by_role:
        events = results_by_role[role]
        avg_duration = sum(e['duration_us'] for e in events) / len(events)
        role_avgs[role] = avg_duration

if len(role_avgs) >= 2:
    print("\nRelative Performance:")
    baseline = role_avgs.get('recv_XGMI_send_XGMI', 1)
    for role, avg in sorted(role_avgs.items()):
        relative = (avg / baseline) if baseline > 0 else 0
        print(f"  {role.replace('_', ' '):<30}: {relative:.2f}x vs XGMI-only")

print("\n" + "="*80)


