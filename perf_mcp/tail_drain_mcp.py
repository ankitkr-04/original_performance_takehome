#!/usr/bin/env python3
"""
Tier 2 MCP: Tail/Drain Analyzer
=================================
Measure cycles after last gather, cycles before store, pure-store cycles.
"""

import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from trace_mcp import parse_trace_json
from collections import defaultdict


def analyze_tail_drain(trace_path: str, verbose: bool = True) -> Dict:
    """
    Analyze tail and drain phases of kernel execution.
    
    Args:
        trace_path: Path to trace.json
        verbose: Print detailed analysis
    
    Returns:
        Dictionary with tail/drain metrics
    """
    events = parse_trace_json(trace_path)
    
    # Track operations per cycle
    cycle_ops = defaultdict(lambda: {'load': 0, 'valu': 0, 'alu': 0, 'store': 0})
    max_cycle = 0
    
    for event in events:
        if event.get('ph') == 'X' and 'ts' in event and event.get('dur', 0) > 0:
            ts = event['ts']
            max_cycle = max(max_cycle, ts)
            
            # Determine engine
            for e in events:
                if e.get('ph') == 'M' and e.get('name') == 'thread_name' and e.get('tid') == event.get('tid'):
                    engine = e['args']['name'].split('-')[0]
                    if engine in ['load', 'valu', 'alu', 'store']:
                        cycle_ops[ts][engine] += 1
                    break
    
    # Find phases
    last_load_cycle = -1
    first_store_cycle = max_cycle + 1
    last_valu_cycle = -1
    
    pure_store_cycles = 0
    store_with_compute = 0
    
    for cycle in range(max_cycle + 1):
        ops = cycle_ops[cycle]
        
        if ops['load'] > 0:
            last_load_cycle = cycle
        
        if ops['valu'] > 0:
            last_valu_cycle = cycle
        
        if ops['store'] > 0:
            if first_store_cycle > max_cycle:
                first_store_cycle = cycle
            
            # Pure store: only store operations, no compute
            if ops['valu'] == 0 and ops['alu'] == 0 and ops['load'] == 0:
                pure_store_cycles += 1
            else:
                store_with_compute += 1
    
    # Calculate metrics
    tail_after_load = max_cycle - last_load_cycle if last_load_cycle >= 0 else 0
    tail_after_compute = max_cycle - last_valu_cycle if last_valu_cycle >= 0 else 0
    drain_before_store = first_store_cycle if first_store_cycle <= max_cycle else 0
    store_phase_length = max_cycle - first_store_cycle + 1 if first_store_cycle <= max_cycle else 0
    
    # Analyze store phase composition
    store_efficiency = (store_with_compute / (pure_store_cycles + store_with_compute) * 100) if (pure_store_cycles + store_with_compute) > 0 else 0
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"TAIL/DRAIN ANALYSIS: {trace_path}")
        print(f"{'='*70}")
        
        print(f"\n┌{'─'*68}┐")
        print(f"│ {'EXECUTION PHASES':^66} │")
        print(f"├{'─'*68}┤")
        print(f"│ {'Total cycles:':<40} {max_cycle + 1:>26} │")
        print(f"│ {'Last load (gather) at cycle:':<40} {last_load_cycle:>26} │")
        print(f"│ {'Last compute (valu) at cycle:':<40} {last_valu_cycle:>26} │")
        print(f"│ {'First store at cycle:':<40} {first_store_cycle if first_store_cycle <= max_cycle else 'N/A':>26} │")
        print(f"└{'─'*68}┘")
        
        print(f"\n┌{'─'*68}┐")
        print(f"│ {'TAIL METRICS':^66} │")
        print(f"├{'─'*68}┤")
        print(f"│ {'Cycles after last gather:':<40} {tail_after_load:>26} │")
        print(f"│ {'Cycles after last compute:':<40} {tail_after_compute:>26} │")
        print(f"│ {'Percent of total:':<40} {(tail_after_load/(max_cycle+1)*100):>24.1f}% │")
        print(f"└{'─'*68}┘")
        
        print(f"\n┌{'─'*68}┐")
        print(f"│ {'STORE PHASE ANALYSIS':^66} │")
        print(f"├{'─'*68}┤")
        print(f"│ {'Store phase starts at:':<40} {first_store_cycle if first_store_cycle <= max_cycle else 'N/A':>26} │")
        print(f"│ {'Store phase length:':<40} {store_phase_length:>26} │")
        print(f"│ {'Pure store cycles (no overlap):':<40} {pure_store_cycles:>26} │")
        print(f"│ {'Store with compute cycles:':<40} {store_with_compute:>26} │")
        print(f"│ {'Store overlap efficiency:':<40} {store_efficiency:>24.1f}% │")
        print(f"└{'─'*68}┘")
        
        # Show tail cycles detail
        print(f"\n┌{'─'*68}┐")
        print(f"│ {'TAIL CYCLES BREAKDOWN':^66} │")
        print(f"├{'─'*68}┤")
        print(f"│ {'Cycle':<8} │ {'Load':>6} │ {'Valu':>6} │ {'ALU':>6} │ {'Store':>6} │ {'Phase':<18} │")
        print(f"├{'─'*68}┤")
        
        tail_start = max(0, max_cycle - 19)  # Show last 20 cycles
        for cycle in range(tail_start, max_cycle + 1):
            ops = cycle_ops[cycle]
            phase = ""
            if cycle > last_valu_cycle:
                phase = "Post-compute"
            elif cycle > last_load_cycle:
                phase = "Post-gather"
            elif cycle >= first_store_cycle:
                phase = "Store phase"
            
            print(f"│ {cycle:<8} │ {ops['load']:>6} │ {ops['valu']:>6} │ "
                  f"{ops['alu']:>6} │ {ops['store']:>6} │ {phase:<18} │")
        print(f"└{'─'*68}┘")
        
        # Recommendations
        print(f"\n┌{'─'*68}┐")
        print(f"│ {'RECOMMENDATIONS':^66} │")
        print(f"├{'─'*68}┤")
        
        tail_ratio = tail_after_load / (max_cycle + 1) if max_cycle >= 0 else 0
        
        if tail_ratio > 0.05:
            print(f"│ ⚠ Long tail after gather ({tail_after_load} cycles, {tail_ratio*100:.1f}%)          │")
            print(f"│   Consider moving stores earlier or adding more computation       │")
        
        if pure_store_cycles > store_phase_length * 0.5:
            print(f"│ ⚠ Many pure store cycles ({pure_store_cycles}/{store_phase_length})                       │")
            print(f"│   Overlap stores with final computation if possible               │")
        
        if store_efficiency < 50:
            print(f"│ ⚠ Low store overlap ({store_efficiency:.1f}%)                                  │")
            print(f"│   Try to interleave stores with trailing operations               │")
        
        if tail_after_load < 10:
            print(f"│ ✓ Excellent tail optimization ({tail_after_load} cycles)                      │")
        
        print(f"└{'─'*68}┘")
    
    return {
        'total_cycles': max_cycle + 1,
        'last_load_cycle': last_load_cycle,
        'last_valu_cycle': last_valu_cycle,
        'first_store_cycle': first_store_cycle,
        'tail_after_load': tail_after_load,
        'tail_after_compute': tail_after_compute,
        'pure_store_cycles': pure_store_cycles,
        'store_with_compute': store_with_compute,
        'store_efficiency': store_efficiency
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Tail/Drain MCP - Analyze tail and drain phases")
    parser.add_argument('trace_path', help='Path to trace.json')
    
    args = parser.parse_args()
    
    analyze_tail_drain(args.trace_path, verbose=True)


if __name__ == "__main__":
    main()
