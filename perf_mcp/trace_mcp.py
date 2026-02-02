#!/usr/bin/env python3
"""
Tier 1 MCP: Trace Analyzer
==========================
Parse trace.json and compute per-cycle utilization, bubble cycles, idle streaks.
Can also run kernel with trace automatically.
"""

import json
import gzip
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import defaultdict

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from problem import SLOT_LIMITS, Machine, build_mem_image, Tree, Input, N_CORES
from perf_takehome import KernelBuilder


@dataclass
class CycleStats:
    """Statistics for a single cycle"""
    cycle: int
    load_usage: int
    valu_usage: int
    alu_usage: int
    store_usage: int
    flow_usage: int
    total_slots: int
    is_bubble: bool


@dataclass
class TraceStats:
    """Overall trace statistics"""
    total_cycles: int
    per_cycle: List[CycleStats]
    load_utilization: float  # Average % of load slots used
    valu_utilization: float  # Average % of valu slots used
    alu_utilization: float   # Average % of alu slots used
    store_utilization: float # Average % of store slots used
    bubble_cycles: int
    longest_idle_streak: int
    avg_slots_per_instr: float
    tail_drain_cycles: int


def parse_trace_json(trace_path: str) -> List[Dict]:
    """Parse trace.json (handles both .json and .json.gz)"""
    if trace_path.endswith('.gz'):
        with gzip.open(trace_path, 'rt') as f:
            content = f.read()
    else:
        with open(trace_path, 'r') as f:
            content = f.read()
    
    # Handle malformed JSON from trace generation
    content = content.strip()
    
    # Remove all variations of trailing commas before ]
    import re
    content = re.sub(r',(\s*)\]', r'\1]', content)
    
    # Ensure it ends with ]
    if not content.endswith(']'):
        if content.endswith(','):
            content = content[:-1] + ']'
        else:
            content += ']'
    
    events = json.loads(content)
    return events


def analyze_trace(trace_path: str, verbose: bool = False) -> TraceStats:
    """
    Analyze trace.json and compute utilization metrics.
    
    Args:
        trace_path: Path to trace.json or trace.json.gz
        verbose: Print detailed per-cycle stats
    
    Returns:
        TraceStats with all computed metrics
    """
    events = parse_trace_json(trace_path)
    
    # Group events by cycle (ts field) and engine
    cycle_events = defaultdict(lambda: defaultdict(list))
    max_cycle = 0
    
    for event in events:
        if event.get('ph') == 'X' and 'ts' in event:  # Duration event
            ts = event['ts']
            dur = event.get('dur', 0)
            if dur > 0:  # Skip zero-duration init events
                tid_name = None
                # Find thread name
                for e in events:
                    if e.get('ph') == 'M' and e.get('name') == 'thread_name':
                        if e.get('tid') == event.get('tid'):
                            tid_name = e['args']['name']
                            break
                
                if tid_name:
                    # Parse engine name (e.g., "load-0" -> "load")
                    engine = tid_name.split('-')[0]
                    if engine in SLOT_LIMITS:
                        cycle_events[ts][engine].append(event)
                        max_cycle = max(max_cycle, ts)
    
    # Compute per-cycle statistics
    per_cycle_stats = []
    total_load = 0
    total_valu = 0
    total_alu = 0
    total_store = 0
    bubble_count = 0
    
    current_idle_streak = 0
    longest_idle_streak = 0
    
    last_gather_cycle = 0
    first_store_cycle = max_cycle + 1
    
    for cycle in range(max_cycle + 1):
        engines = cycle_events[cycle]
        
        load_usage = len(engines.get('load', []))
        valu_usage = len(engines.get('valu', []))
        alu_usage = len(engines.get('alu', []))
        store_usage = len(engines.get('store', []))
        flow_usage = len(engines.get('flow', []))
        
        total_slots = load_usage + valu_usage + alu_usage + store_usage + flow_usage
        is_bubble = total_slots == 0
        
        stats = CycleStats(
            cycle=cycle,
            load_usage=load_usage,
            valu_usage=valu_usage,
            alu_usage=alu_usage,
            store_usage=store_usage,
            flow_usage=flow_usage,
            total_slots=total_slots,
            is_bubble=is_bubble
        )
        per_cycle_stats.append(stats)
        
        total_load += load_usage
        total_valu += valu_usage
        total_alu += alu_usage
        total_store += store_usage
        
        if is_bubble:
            bubble_count += 1
            current_idle_streak += 1
        else:
            longest_idle_streak = max(longest_idle_streak, current_idle_streak)
            current_idle_streak = 0
        
        # Track last gather (load) and first store
        if load_usage > 0:
            last_gather_cycle = cycle
        if store_usage > 0 and first_store_cycle > max_cycle:
            first_store_cycle = cycle
    
    total_cycles = max_cycle + 1
    
    # Compute averages
    load_util = (total_load / (total_cycles * SLOT_LIMITS['load'])) * 100 if total_cycles > 0 else 0
    valu_util = (total_valu / (total_cycles * SLOT_LIMITS['valu'])) * 100 if total_cycles > 0 else 0
    alu_util = (total_alu / (total_cycles * SLOT_LIMITS['alu'])) * 100 if total_cycles > 0 else 0
    store_util = (total_store / (total_cycles * SLOT_LIMITS['store'])) * 100 if total_cycles > 0 else 0
    
    total_instr_slots = total_load + total_valu + total_alu + total_store
    avg_slots_per_instr = total_instr_slots / total_cycles if total_cycles > 0 else 0
    
    # Tail drain: cycles after last gather
    tail_drain = total_cycles - last_gather_cycle - 1 if last_gather_cycle > 0 else 0
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"TRACE ANALYSIS: {trace_path}")
        print(f"{'='*70}")
        print(f"\nTotal Cycles: {total_cycles}")
        print(f"\nEngine Utilization:")
        print(f"  Load:  {load_util:6.2f}% (avg {total_load/total_cycles:.2f}/{SLOT_LIMITS['load']} slots/cycle)")
        print(f"  Valu:  {valu_util:6.2f}% (avg {total_valu/total_cycles:.2f}/{SLOT_LIMITS['valu']} slots/cycle)")
        print(f"  ALU:   {alu_util:6.2f}% (avg {total_alu/total_cycles:.2f}/{SLOT_LIMITS['alu']} slots/cycle)")
        print(f"  Store: {store_util:6.2f}% (avg {total_store/total_cycles:.2f}/{SLOT_LIMITS['store']} slots/cycle)")
        print(f"\nBubble Cycles: {bubble_count}")
        print(f"Longest Idle Streak: {longest_idle_streak}")
        print(f"Avg Slots per Instruction: {avg_slots_per_instr:.2f}")
        print(f"Tail Drain Cycles: {tail_drain}")
        print(f"\nLast Gather Cycle: {last_gather_cycle}")
        print(f"First Store Cycle: {first_store_cycle}")
        
        # Show first 20 and last 20 cycles
        print(f"\n{'='*70}")
        print("FIRST 20 CYCLES:")
        print(f"{'='*70}")
        print(f"{'Cycle':>5} | {'Load':>4} | {'Valu':>4} | {'ALU':>4} | {'Store':>5} | {'Total':>5} | Bubble")
        print(f"{'-'*70}")
        for stats in per_cycle_stats[:20]:
            print(f"{stats.cycle:5d} | {stats.load_usage:4d} | {stats.valu_usage:4d} | "
                  f"{stats.alu_usage:4d} | {stats.store_usage:5d} | {stats.total_slots:5d} | "
                  f"{'YES' if stats.is_bubble else ''}")
        
        if total_cycles > 40:
            print(f"\n{'='*70}")
            print("LAST 20 CYCLES:")
            print(f"{'='*70}")
            print(f"{'Cycle':>5} | {'Load':>4} | {'Valu':>4} | {'ALU':>4} | {'Store':>5} | {'Total':>5} | Bubble")
            print(f"{'-'*70}")
            for stats in per_cycle_stats[-20:]:
                print(f"{stats.cycle:5d} | {stats.load_usage:4d} | {stats.valu_usage:4d} | "
                      f"{stats.alu_usage:4d} | {stats.store_usage:5d} | {stats.total_slots:5d} | "
                      f"{'YES' if stats.is_bubble else ''}")
    
    return TraceStats(
        total_cycles=total_cycles,
        per_cycle=per_cycle_stats,
        load_utilization=load_util,
        valu_utilization=valu_util,
        alu_utilization=alu_util,
        store_utilization=store_util,
        bubble_cycles=bubble_count,
        longest_idle_streak=longest_idle_streak,
        avg_slots_per_instr=avg_slots_per_instr,
        tail_drain_cycles=tail_drain
    )


def run_and_trace(forest_height: int = 10, rounds: int = 16, batch_size: int = 256, 
                  output_path: str = "trace.json", seed: int = 123) -> int:
    """
    Run kernel with trace enabled and return cycle count.
    
    Args:
        forest_height: Height of the binary tree
        rounds: Number of rounds to execute
        batch_size: Number of parallel inputs
        output_path: Where to save trace.json
        seed: Random seed
    
    Returns:
        Cycle count
    """
    import random
    random.seed(seed)
    
    print(f"Running kernel: forest_height={forest_height}, rounds={rounds}, batch_size={batch_size}")
    
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)
    
    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        trace=True
    )
    
    from problem import reference_kernel2
    for i, ref_mem in enumerate(reference_kernel2(mem, {})):
        machine.run()
        inp_values_p = ref_mem[6]
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
    
    print(f"✓ Kernel completed successfully")
    print(f"  Cycles: {machine.cycle}")
    print(f"  Trace saved to: {output_path}")
    
    return machine.cycle


def export_csv(stats: TraceStats, output_path: str):
    """Export trace statistics to CSV"""
    with open(output_path, 'w') as f:
        f.write("cycle,load,valu,alu,store,total,is_bubble\n")
        for s in stats.per_cycle:
            f.write(f"{s.cycle},{s.load_usage},{s.valu_usage},{s.alu_usage},"
                   f"{s.store_usage},{s.total_slots},{int(s.is_bubble)}\n")
    print(f"✓ CSV exported to: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Trace MCP - Run and analyze kernel traces")
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze existing trace.json')
    analyze_parser.add_argument('trace_path', help='Path to trace.json or trace.json.gz')
    analyze_parser.add_argument('--csv', help='Export to CSV file')
    analyze_parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed output')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run kernel and generate trace')
    run_parser.add_argument('--rounds', type=int, default=16, help='Number of rounds (default: 16)')
    run_parser.add_argument('--batch', type=int, default=256, help='Batch size (default: 256)')
    run_parser.add_argument('--height', type=int, default=10, help='Forest height (default: 10)')
    run_parser.add_argument('--output', '-o', default='trace.json', help='Output trace file')
    run_parser.add_argument('--seed', type=int, default=123, help='Random seed')
    run_parser.add_argument('--analyze', action='store_true', help='Analyze after running')
    run_parser.add_argument('--csv', help='Export analysis to CSV')
    
    args = parser.parse_args()
    
    if args.command == 'analyze':
        stats = analyze_trace(args.trace_path, verbose=args.verbose)
        if args.csv:
            export_csv(stats, args.csv)
    
    elif args.command == 'run':
        cycles = run_and_trace(
            forest_height=args.height,
            rounds=args.rounds,
            batch_size=args.batch,
            output_path=args.output,
            seed=args.seed
        )
        
        if args.analyze:
            print(f"\n{'='*70}")
            print("AUTO-ANALYZING GENERATED TRACE")
            stats = analyze_trace(args.output, verbose=True)
            if args.csv:
                export_csv(stats, args.csv)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
