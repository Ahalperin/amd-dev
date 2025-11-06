#!/usr/bin/env python3
"""
Find the slowest and fastest channels for rank 0
"""
import json
from collections import defaultdict

# Channel role mapping based on RCCL log (Ring patterns for rank 0)
channel_roles = {
    0: (8, 1, "recv_NET_send_XGMI"),      1: (1, 2, "recv_XGMI_send_XGMI"),
    2: (5, 3, "recv_XGMI_send_XGMI"),     3: (7, 4, "recv_XGMI_send_XGMI"),
    4: (3, 5, "recv_XGMI_send_XGMI"),     5: (6, 8, "recv_XGMI_send_NET"),
    6: (4, 7, "recv_XGMI_send_XGMI"),     7: (2, 6, "recv_XGMI_send_XGMI"),
    8: (1, 2, "recv_XGMI_send_XGMI"),     9: (5, 3, "recv_XGMI_send_XGMI"),
    10: (7, 4, "recv_XGMI_send_XGMI"),    11: (3, 5, "recv_XGMI_send_XGMI"),
    12: (6, 8, "recv_XGMI_send_NET"),     13: (4, 7, "recv_XGMI_send_XGMI"),
    14: (2, 6, "recv_XGMI_send_XGMI"),    15: (8, 1, "recv_NET_send_XGMI"),
    16: (5, 3, "recv_XGMI_send_XGMI"),    17: (7, 4, "recv_XGMI_send_XGMI"),
    18: (3, 5, "recv_XGMI_send_XGMI"),    19: (6, 8, "recv_XGMI_send_NET"),
    20: (4, 7, "recv_XGMI_send_XGMI"),    21: (2, 6, "recv_XGMI_send_XGMI"),
    22: (8, 1, "recv_NET_send_XGMI"),     23: (1, 2, "recv_XGMI_send_XGMI"),
    24: (7, 4, "recv_XGMI_send_XGMI"),    25: (3, 5, "recv_XGMI_send_XGMI"),
    26: (6, 8, "recv_XGMI_send_NET"),     27: (4, 7, "recv_XGMI_send_XGMI"),
    28: (2, 6, "recv_XGMI_send_XGMI"),    29: (8, 1, "recv_NET_send_XGMI"),
    30: (1, 2, "recv_XGMI_send_XGMI"),    31: (5, 3, "recv_XGMI_send_XGMI"),
    32: (3, 5, "recv_XGMI_send_XGMI"),    33: (6, 8, "recv_XGMI_send_NET"),
    34: (4, 7, "recv_XGMI_send_XGMI"),    35: (2, 6, "recv_XGMI_send_XGMI"),
    36: (8, 1, "recv_NET_send_XGMI"),     37: (1, 2, "recv_XGMI_send_XGMI"),
    38: (5, 3, "recv_XGMI_send_XGMI"),    39: (7, 4, "recv_XGMI_send_XGMI"),
    40: (6, 8, "recv_XGMI_send_NET"),     41: (4, 7, "recv_XGMI_send_XGMI"),
    42: (2, 6, "recv_XGMI_send_XGMI"),    43: (8, 1, "recv_NET_send_XGMI"),
    44: (1, 2, "recv_XGMI_send_XGMI"),    45: (5, 3, "recv_XGMI_send_XGMI"),
    46: (7, 4, "recv_XGMI_send_XGMI"),    47: (3, 5, "recv_XGMI_send_XGMI"),
    48: (4, 7, "recv_XGMI_send_XGMI"),    49: (2, 6, "recv_XGMI_send_XGMI"),
    50: (8, 1, "recv_NET_send_XGMI"),     51: (1, 2, "recv_XGMI_send_XGMI"),
    52: (5, 3, "recv_XGMI_send_XGMI"),    53: (7, 4, "recv_XGMI_send_XGMI"),
    54: (3, 5, "recv_XGMI_send_XGMI"),    55: (6, 8, "recv_XGMI_send_NET"),
    56: (2, 6, "recv_XGMI_send_XGMI"),    57: (8, 1, "recv_NET_send_XGMI"),
    58: (1, 2, "recv_XGMI_send_XGMI"),    59: (5, 3, "recv_XGMI_send_XGMI"),
    60: (7, 4, "recv_XGMI_send_XGMI"),    61: (3, 5, "recv_XGMI_send_XGMI"),
    62: (6, 8, "recv_XGMI_send_NET"),     63: (4, 7, "recv_XGMI_send_XGMI"),
}

print("Loading NPKit JSON file...")
with open('/home/dn/amd-dev/rccl_mpi_profile/2025-11-06_00.37/npkit_json_rank_0/npkit_event_trace.json', 'r') as f:
    data = json.load(f)

# Match BEGIN and END events
begin_stack = {}
durations = []

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
                'args' in begin_event and begin_event['args'].get('rank') == 0):
                
                durations.append({
                    'channel': begin_event['args']['buf_idx'],
                    'seq': begin_event['args']['seq'],
                    'duration_us': duration_us,
                    'size_bytes': begin_event['args'].get('size_0', 0),
                })

print(f"Matched {len(durations)} all-reduce operations for rank 0")

# Calculate per-channel statistics
channel_stats = defaultdict(lambda: {'durations': [], 'count': 0})

for d in durations:
    ch = d['channel']
    channel_stats[ch]['durations'].append(d['duration_us'])
    channel_stats[ch]['count'] += 1

# Calculate averages
channel_averages = []
for ch, stats in channel_stats.items():
    avg = sum(stats['durations']) / len(stats['durations'])
    min_dur = min(stats['durations'])
    max_dur = max(stats['durations'])
    std_dev = (sum((d - avg)**2 for d in stats['durations']) / len(stats['durations']))**0.5
    
    recv_from, send_to, role = channel_roles.get(ch, (None, None, "unknown"))
    
    channel_averages.append({
        'channel': ch,
        'avg': avg,
        'min': min_dur,
        'max': max_dur,
        'std': std_dev,
        'count': stats['count'],
        'recv_from': recv_from,
        'send_to': send_to,
        'role': role
    })

# Sort by average
channel_averages.sort(key=lambda x: x['avg'])

print("\n" + "="*90)
print("TOP 10 FASTEST CHANNELS (Lowest Average Duration)")
print("="*90)
print(f"{'Rank':>4} {'Ch':>3} {'Recv':>5} {'Send':>5} {'Role':<25} {'Ops':>4} {'Avg μs':>10} {'Min μs':>10} {'Max μs':>10} {'Std μs':>10}")
print("-"*90)

for i, ch in enumerate(channel_averages[:10], 1):
    role_display = ch['role'].replace('recv_', '').replace('send_', '→').replace('_', ' ')
    print(f"{i:>4} {ch['channel']:>3} {ch['recv_from']:>5} {ch['send_to']:>5} {role_display:<25} "
          f"{ch['count']:>4} {ch['avg']:>10.2f} {ch['min']:>10.2f} {ch['max']:>10.2f} {ch['std']:>10.2f}")

print("\n" + "="*90)
print("TOP 10 SLOWEST CHANNELS (Highest Average Duration)")
print("="*90)
print(f"{'Rank':>4} {'Ch':>3} {'Recv':>5} {'Send':>5} {'Role':<25} {'Ops':>4} {'Avg μs':>10} {'Min μs':>10} {'Max μs':>10} {'Std μs':>10}")
print("-"*90)

for i, ch in enumerate(channel_averages[-10:][::-1], 1):
    role_display = ch['role'].replace('recv_', '').replace('send_', '→').replace('_', ' ')
    print(f"{i:>4} {ch['channel']:>3} {ch['recv_from']:>5} {ch['send_to']:>5} {role_display:<25} "
          f"{ch['count']:>4} {ch['avg']:>10.2f} {ch['min']:>10.2f} {ch['max']:>10.2f} {ch['std']:>10.2f}")

# Find absolute fastest and slowest operations
all_ops = []
for d in durations:
    recv_from, send_to, role = channel_roles.get(d['channel'], (None, None, "unknown"))
    all_ops.append({
        'channel': d['channel'],
        'seq': d['seq'],
        'duration': d['duration_us'],
        'recv_from': recv_from,
        'send_to': send_to,
        'role': role
    })

all_ops.sort(key=lambda x: x['duration'])

print("\n" + "="*90)
print("ABSOLUTE FASTEST OPERATIONS")
print("="*90)
print(f"{'Rank':>4} {'Ch':>3} {'Seq':>4} {'Recv':>5} {'Send':>5} {'Role':<25} {'Duration μs':>12}")
print("-"*75)

for i, op in enumerate(all_ops[:10], 1):
    role_display = op['role'].replace('recv_', '').replace('send_', '→').replace('_', ' ')
    print(f"{i:>4} {op['channel']:>3} {op['seq']:>4} {op['recv_from']:>5} {op['send_to']:>5} "
          f"{role_display:<25} {op['duration']:>12.2f}")

print("\n" + "="*90)
print("ABSOLUTE SLOWEST OPERATIONS")
print("="*90)
print(f"{'Rank':>4} {'Ch':>3} {'Seq':>4} {'Recv':>5} {'Send':>5} {'Role':<25} {'Duration μs':>12}")
print("-"*75)

for i, op in enumerate(all_ops[-10:][::-1], 1):
    role_display = op['role'].replace('recv_', '').replace('send_', '→').replace('_', ' ')
    print(f"{i:>4} {op['channel']:>3} {op['seq']:>4} {op['recv_from']:>5} {op['send_to']:>5} "
          f"{role_display:<25} {op['duration']:>12.2f}")

# Summary statistics
fastest_ch = channel_averages[0]
slowest_ch = channel_averages[-1]
fastest_op = all_ops[0]
slowest_op = all_ops[-1]

print("\n" + "="*90)
print("SUMMARY")
print("="*90)
print(f"\nFASTEST CHANNEL (by average):")
print(f"  Channel {fastest_ch['channel']}: {fastest_ch['recv_from']} -> 0 -> {fastest_ch['send_to']} ({fastest_ch['role']})")
print(f"  Average: {fastest_ch['avg']:.2f} μs over {fastest_ch['count']} operations")

print(f"\nSLOWEST CHANNEL (by average):")
print(f"  Channel {slowest_ch['channel']}: {slowest_ch['recv_from']} -> 0 -> {slowest_ch['send_to']} ({slowest_ch['role']})")
print(f"  Average: {slowest_ch['avg']:.2f} μs over {slowest_ch['count']} operations")

print(f"\nFASTEST SINGLE OPERATION:")
print(f"  Channel {fastest_op['channel']}, seq {fastest_op['seq']}: {fastest_op['recv_from']} -> 0 -> {fastest_op['send_to']} ({fastest_op['role']})")
print(f"  Duration: {fastest_op['duration']:.2f} μs")

print(f"\nSLOWEST SINGLE OPERATION:")
print(f"  Channel {slowest_op['channel']}, seq {slowest_op['seq']}: {slowest_op['recv_from']} -> 0 -> {slowest_op['send_to']} ({slowest_op['role']})")
print(f"  Duration: {slowest_op['duration']:.2f} μs")

ratio = slowest_ch['avg'] / fastest_ch['avg'] if fastest_ch['avg'] > 0 else 0
print(f"\nPerformance spread: {ratio:.2f}x (slowest channel is {ratio:.2f}x slower than fastest)")

print("\n" + "="*90)


