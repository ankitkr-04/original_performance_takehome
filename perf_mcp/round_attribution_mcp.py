#!/usr/bin/env python3
"""
Tier 1 MCP: Round/Group Attribution
=====================================
Tag cycles by round, group id, leftover group, drain to show where time is spent.
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional
from collections import defaultdict
import re

sys.path.insert(0, str(Path(__file__).parent.parent))

from perf_mcp.trace_mcp import parse_trace_json


@dataclass
class RoundStats:
    """Statistics for a single round"""
    round_num: int
    start_cycle: int
    end_cycle: int
    duration: int
    load_ops: int
    valu_ops: int
    alu_ops: int
    store_ops: int


def attribute_cycles_to_rounds(trace_path: str, rounds: int = 16, 
                                n_vectors: int = 32, num_parallel: int = 6,
                                verbose: bool = True) -> List[RoundStats]:
    """
    Attribute cycles to specific rounds and groups.
    
    This requires some heuristics based on the kernel structure:
    - Initialization phase (loads, constants, broadcasts)
    - Round 0-2: No gathers (uses preloaded nodes)
    - Round 3+: Gathers + hash computation
    - Tail: Final stores
    
    Args:
        trace_path: Path to trace.json
        rounds: Number of rounds executed
        n_vectors: Total number of vectors (batch_size / VLEN)
        num_parallel: Number of vectors processed in parallel
        verbose: Print detailed output
    
    Returns:
        List of RoundStats
    """
    events = parse_trace_json(trace_path)
    
    # Count operations per cycle
    cycle_ops = defaultdict(lambda: {'load': 0, 'valu': 0, 'alu': 0, 'store': 0})
    max_cycle = 0
    
    for event in events:
        if event.get('ph') == 'X' and 'ts' in event and event.get('dur', 0) > 0:
            ts = event['ts']
            max_cycle = max(max_cycle, ts)
            
            # Determine engine type from thread name
            for e in events:
                if e.get('ph') == 'M' and e.get('name') == 'thread_name' and e.get('tid') == event.get('tid'):
                    engine = e['args']['name'].split('-')[0]
                    if engine in ['load', 'valu', 'alu', 'store']:
                        cycle_ops[ts][engine] += 1
                    break
    
    # Heuristic round detection
    round_stats = []
    
    # Phase 1: Initialization (heavy loads, broadcasts)
    init_end = 0
    for cycle in range(max_cycle + 1):
        ops = cycle_ops[cycle]
        # Init phase has lots of loads and broadcasts, minimal compute
        if ops['valu'] > 4:  # Hash computation starts
            init_end = cycle - 1
            break
    
    print(f"\n{'='*70}")
    print(f"ROUND ATTRIBUTION: {trace_path}")
    print(f"{'='*70}")
    print(f"\nInitialization phase: cycles 0-{init_end} ({init_end + 1} cycles)")
    
    # Phase 2: Compute rounds
    # Rounds 0-2: No loads (uses preloaded), pure valu
    # Rounds 3+: Interleaved loads + valu
    
    n_groups = n_vectors // num_parallel
    leftover = n_vectors % num_parallel
    
    current_cycle = init_end + 1
    
    for r in range(rounds):
        round_start = current_cycle
        round_loads = 0
        round_valus = 0
        round_alus = 0
        round_stores = 0
        
        if r <= 2:
            # Rounds 0-2: Pure valu compute (no gathers)
            # Estimate: ~18-20 valu cycles per group * n_groups
            estimated_cycles_per_group = 20
            estimated_round_cycles = estimated_cycles_per_group * (n_groups + (1 if leftover > 0 else 0))
            
            for _ in range(estimated_round_cycles):
                if current_cycle > max_cycle:
                    break
                ops = cycle_ops[current_cycle]
                round_loads += ops['load']
                round_valus += ops['valu']
                round_alus += ops['alu']
                round_stores += ops['store']
                current_cycle += 1
                
                # Stop if we hit loads (means next round started)
                if ops['load'] > 0 and r < 2:
                    break
        else:
            # Rounds 3+: Interleaved gather + compute
            # Each group needs ~24 cycles (gather) + overlap with hash
            # Estimate: ~30 cycles per group
            estimated_cycles_per_group = 30
            estimated_round_cycles = estimated_cycles_per_group * (n_groups + (1 if leftover > 0 else 0))
            
            for _ in range(estimated_round_cycles):
                if current_cycle > max_cycle:
                    break
                ops = cycle_ops[current_cycle]
                round_loads += ops['load']
                round_valus += ops['valu']
                round_alus += ops['alu']
                round_stores += ops['store']
                current_cycle += 1
        
        round_end = current_cycle - 1
        duration = round_end - round_start + 1
        
        round_stats.append(RoundStats(
            round_num=r,
            start_cycle=round_start,
            end_cycle=round_end,
            duration=duration,
            load_ops=round_loads,
            valu_ops=round_valus,
            alu_ops=round_alus,
            store_ops=round_stores
        ))
    
    # Phase 3: Tail (final stores)
    tail_start = current_cycle
    tail_loads = 0
    tail_valus = 0
    tail_alus = 0
    tail_stores = 0
    
    for cycle in range(tail_start, max_cycle + 1):
        ops = cycle_ops[cycle]
        tail_loads += ops['load']
        tail_valus += ops['valu']
        tail_alus += ops['alu']
        tail_stores += ops['store']
    
    tail_duration = max_cycle - tail_start + 1 if tail_start <= max_cycle else 0
    
    if verbose:
        print(f"\n{'='*70}")
        print("ROUND-BY-ROUND BREAKDOWN:")
        print(f"{'='*70}")
        print(f"{'Round':>6} | {'Start':>6} | {'End':>6} | {'Duration':>8} | {'Load':>6} | {'Valu':>6} | {'ALU':>6} | {'Store':>6}")
        print(f"{'-'*70}")
        
        for rs in round_stats:
            print(f"{rs.round_num:6d} | {rs.start_cycle:6d} | {rs.end_cycle:6d} | "
                  f"{rs.duration:8d} | {rs.load_ops:6d} | {rs.valu_ops:6d} | "
                  f"{rs.alu_ops:6d} | {rs.store_ops:6d}")
        
        print(f"{'-'*70}")
        print(f"{'TAIL':>6} | {tail_start:6d} | {max_cycle:6d} | {tail_duration:8d} | "
              f"{tail_loads:6d} | {tail_valus:6d} | {tail_alus:6d} | {tail_stores:6d}")
        
        # Summary statistics
        print(f"\n{'='*70}")
        print("SUMMARY:")
        print(f"{'='*70}")
        
        early_rounds = [rs for rs in round_stats if rs.round_num <= 2]
        late_rounds = [rs for rs in round_stats if rs.round_num > 2]
        
        if early_rounds:
            avg_early = sum(rs.duration for rs in early_rounds) / len(early_rounds)
            print(f"Rounds 0-2 (no gather): avg {avg_early:.1f} cycles/round")
        
        if late_rounds:
            avg_late = sum(rs.duration for rs in late_rounds) / len(late_rounds)
            print(f"Rounds 3+ (with gather): avg {avg_late:.1f} cycles/round")
        
        print(f"Tail/drain: {tail_duration} cycles")
        
        # Identify slowest rounds
        slowest = sorted(round_stats, key=lambda x: x.duration, reverse=True)[:5]
        print(f"\nSlowest rounds:")
        for rs in slowest:
            print(f"  Round {rs.round_num}: {rs.duration} cycles")
    
    return round_stats


def export_round_csv(round_stats: List[RoundStats], output_path: str, tail_info: Dict):
    """Export round statistics to CSV"""
    with open(output_path, 'w') as f:
        f.write("round,start_cycle,end_cycle,duration,load_ops,valu_ops,alu_ops,store_ops\n")
        for rs in round_stats:
            f.write(f"{rs.round_num},{rs.start_cycle},{rs.end_cycle},{rs.duration},"
                   f"{rs.load_ops},{rs.valu_ops},{rs.alu_ops},{rs.store_ops}\n")
        # Add tail
        f.write(f"tail,{tail_info['start']},{tail_info['end']},{tail_info['duration']},"
               f"{tail_info['loads']},{tail_info['valus']},{tail_info['alus']},{tail_info['stores']}\n")
    print(f"✓ Round stats exported to: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Round Attribution MCP - Analyze cycle attribution by round")
    parser.add_argument('trace_path', help='Path to trace.json')
    parser.add_argument('--rounds', type=int, default=16, help='Number of rounds (default: 16)')
    parser.add_argument('--vectors', type=int, default=32, help='Number of vectors (default: 32)')
    parser.add_argument('--parallel', type=int, default=6, help='Parallel vectors (default: 6)')
    parser.add_argument('--csv', help='Export to CSV file')
    
    args = parser.parse_args()
    
    round_stats = attribute_cycles_to_rounds(
        args.trace_path,
        rounds=args.rounds,
        n_vectors=args.vectors,
        num_parallel=args.parallel,
        verbose=True
    )
    
    if args.csv:
        # Need to re-parse for tail info
        events = parse_trace_json(args.trace_path)
        max_cycle = max(e['ts'] for e in events if 'ts' in e)
        tail_start = round_stats[-1].end_cycle + 1 if round_stats else 0
        
        export_round_csv(round_stats, args.csv, {
            'start': tail_start,
            'end': max_cycle,
            'duration': max_cycle - tail_start + 1,
            'loads': 0, 'valus': 0, 'alus': 0, 'stores': 0
        })


if __name__ == "__main__":
    main()
