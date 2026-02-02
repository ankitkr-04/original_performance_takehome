#!/usr/bin/env python3
"""
Tier 2 MCP: Slot Packing Analyzer
===================================
Scan instruction stream for avg slots used per engine, worst bundles, single-engine runs.
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from problem import SLOT_LIMITS


@dataclass
class BundleStats:
    """Statistics for an instruction bundle"""
    instr_num: int
    load_slots: int
    valu_slots: int
    alu_slots: int
    store_slots: int
    flow_slots: int
    total_slots: int
    efficiency: float  # % of available slots used


def analyze_slot_packing(instrs: List[Dict], verbose: bool = True) -> Dict:
    """
    Analyze instruction slot packing efficiency.
    
    Args:
        instrs: List of instruction dictionaries
        verbose: Print detailed analysis
    
    Returns:
        Dictionary with packing statistics
    """
    bundle_stats = []
    
    total_available = (SLOT_LIMITS['alu'] + SLOT_LIMITS['valu'] + 
                       SLOT_LIMITS['load'] + SLOT_LIMITS['store'] + SLOT_LIMITS['flow'])
    
    # Analyze each instruction bundle
    for i, instr in enumerate(instrs):
        load_used = len(instr.get('load', []))
        valu_used = len(instr.get('valu', []))
        alu_used = len(instr.get('alu', []))
        store_used = len(instr.get('store', []))
        flow_used = len(instr.get('flow', []))
        
        total_used = load_used + valu_used + alu_used + store_used + flow_used
        efficiency = (total_used / total_available) * 100 if total_available > 0 else 0
        
        bundle_stats.append(BundleStats(
            instr_num=i,
            load_slots=load_used,
            valu_slots=valu_used,
            alu_slots=alu_used,
            store_slots=store_used,
            flow_slots=flow_used,
            total_slots=total_used,
            efficiency=efficiency
        ))
    
    # Compute statistics
    avg_load = sum(b.load_slots for b in bundle_stats) / len(bundle_stats) if bundle_stats else 0
    avg_valu = sum(b.valu_slots for b in bundle_stats) / len(bundle_stats) if bundle_stats else 0
    avg_alu = sum(b.alu_slots for b in bundle_stats) / len(bundle_stats) if bundle_stats else 0
    avg_store = sum(b.store_slots for b in bundle_stats) / len(bundle_stats) if bundle_stats else 0
    avg_efficiency = sum(b.efficiency for b in bundle_stats) / len(bundle_stats) if bundle_stats else 0
    
    # Find worst bundles (lowest efficiency)
    worst_bundles = sorted(bundle_stats, key=lambda x: x.efficiency)[:int(len(bundle_stats) * 0.05) or 1]
    
    # Find longest runs of single-engine dominance
    load_only_runs = []
    valu_only_runs = []
    current_run = {'engine': None, 'start': 0, 'length': 0}
    
    for i, b in enumerate(bundle_stats):
        engines_used = []
        if b.load_slots > 0:
            engines_used.append('load')
        if b.valu_slots > 0:
            engines_used.append('valu')
        if b.alu_slots > 0:
            engines_used.append('alu')
        if b.store_slots > 0:
            engines_used.append('store')
        
        dominant_engine = None
        if len(engines_used) == 1:
            dominant_engine = engines_used[0]
        elif b.valu_slots > 0 and b.valu_slots > b.load_slots and b.valu_slots > b.alu_slots:
            if b.load_slots <= 2:  # Allow minimal loads
                dominant_engine = 'valu'
        elif b.load_slots > 0 and b.load_slots == 2 and b.valu_slots == 0:
            dominant_engine = 'load'
        
        if dominant_engine == current_run['engine']:
            current_run['length'] += 1
        else:
            if current_run['length'] > 0:
                if current_run['engine'] == 'load':
                    load_only_runs.append((current_run['start'], current_run['length']))
                elif current_run['engine'] == 'valu':
                    valu_only_runs.append((current_run['start'], current_run['length']))
            current_run = {'engine': dominant_engine, 'start': i, 'length': 1}
    
    if verbose:
        print(f"\n{'='*70}")
        print("SLOT PACKING ANALYSIS")
        print(f"{'='*70}")
        
        print(f"\nTotal Instructions: {len(bundle_stats)}")
        print(f"Max Available Slots: {total_available} per instruction")
        
        print(f"\n┌{'─'*68}┐")
        print(f"│ {'AVERAGE SLOT USAGE':^66} │")
        print(f"├{'─'*68}┤")
        print(f"│ {'Engine':<20} │ {'Avg Used':>12} │ {'Max':>12} │ {'Util %':>12} │")
        print(f"├{'─'*68}┤")
        print(f"│ {'Load':<20} │ {avg_load:>12.2f} │ {SLOT_LIMITS['load']:>12} │ {(avg_load/SLOT_LIMITS['load']*100):>11.1f}% │")
        print(f"│ {'Valu':<20} │ {avg_valu:>12.2f} │ {SLOT_LIMITS['valu']:>12} │ {(avg_valu/SLOT_LIMITS['valu']*100):>11.1f}% │")
        print(f"│ {'ALU':<20} │ {avg_alu:>12.2f} │ {SLOT_LIMITS['alu']:>12} │ {(avg_alu/SLOT_LIMITS['alu']*100):>11.1f}% │")
        print(f"│ {'Store':<20} │ {avg_store:>12.2f} │ {SLOT_LIMITS['store']:>12} │ {(avg_store/SLOT_LIMITS['store']*100):>11.1f}% │")
        print(f"├{'─'*68}┤")
        print(f"│ {'Overall Efficiency':<20} │ {avg_efficiency:>12.1f}% │ {'':<12} │ {'':<12} │")
        print(f"└{'─'*68}┘")
        
        print(f"\n┌{'─'*68}┐")
        print(f"│ {'WORST 5% BUNDLES (Lowest Efficiency)':^66} │")
        print(f"├{'─'*68}┤")
        print(f"│ {'Instr#':>6} │ {'Load':>4} │ {'Valu':>4} │ {'ALU':>4} │ {'Store':>5} │ {'Total':>5} │ {'Eff %':>7} │")
        print(f"├{'─'*68}┤")
        
        for b in worst_bundles[:20]:
            print(f"│ {b.instr_num:>6} │ {b.load_slots:>4} │ {b.valu_slots:>4} │ "
                  f"{b.alu_slots:>4} │ {b.store_slots:>5} │ {b.total_slots:>5} │ {b.efficiency:>6.1f}% │")
        print(f"└{'─'*68}┘")
        
        print(f"\n┌{'─'*68}┐")
        print(f"│ {'SINGLE-ENGINE RUNS':^66} │")
        print(f"├{'─'*68}┤")
        
        longest_load = max(load_only_runs, key=lambda x: x[1]) if load_only_runs else (0, 0)
        longest_valu = max(valu_only_runs, key=lambda x: x[1]) if valu_only_runs else (0, 0)
        
        print(f"│ {'Longest load-only run:':<40} {longest_load[1]:>6} instrs{' '*10}│")
        print(f"│ {'  (starting at instr':<40} {longest_load[0]:>6}){' '*16}│")
        print(f"│ {'Longest valu-only run:':<40} {longest_valu[1]:>6} instrs{' '*10}│")
        print(f"│ {'  (starting at instr':<40} {longest_valu[0]:>6}){' '*16}│")
        print(f"│ {'':<66} │")
        print(f"│ {'Total load-only runs:':<40} {len(load_only_runs):>6}{' '*20}│")
        print(f"│ {'Total valu-only runs:':<40} {len(valu_only_runs):>6}{' '*20}│")
        print(f"└{'─'*68}┘")
        
        # Recommendations
        print(f"\n┌{'─'*68}┐")
        print(f"│ {'RECOMMENDATIONS':^66} │")
        print(f"├{'─'*68}┤")
        
        if avg_efficiency < 40:
            print(f"│ ⚠ Low overall efficiency ({avg_efficiency:.1f}%)                           │")
            print(f"│   Consider better instruction packing                              │")
        
        if longest_load[1] > 10:
            print(f"│ ⚠ Long load-only run ({longest_load[1]} instrs)                              │")
            print(f"│   Interleave compute operations with loads                         │")
        
        if longest_valu[1] > 20:
            print(f"│ ⚠ Long valu-only run ({longest_valu[1]} instrs)                              │")
            print(f"│   Consider software pipelining to overlap with loads              │")
        
        if avg_load < 1.0 and SLOT_LIMITS['load'] == 2:
            print(f"│ ⚠ Load slots underutilized (avg {avg_load:.2f}/{SLOT_LIMITS['load']})                   │")
            print(f"│   Can gather more data in parallel                                 │")
        
        print(f"└{'─'*68}┘")
    
    return {
        'avg_load': avg_load,
        'avg_valu': avg_valu,
        'avg_alu': avg_alu,
        'avg_store': avg_store,
        'avg_efficiency': avg_efficiency,
        'worst_bundles': worst_bundles,
        'longest_load_run': longest_load,
        'longest_valu_run': longest_valu
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Slot Packing MCP - Analyze instruction packing efficiency")
    parser.add_argument('--kernel', help='Load kernel from perf_takehome.py')
    parser.add_argument('--rounds', type=int, default=16)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--height', type=int, default=10)
    
    args = parser.parse_args()
    
    # Load kernel
    from perf_takehome import KernelBuilder
    from problem import Tree
    
    kb = KernelBuilder()
    
    import random
    random.seed(123)
    forest = Tree.generate(args.height)
    
    kb.build_kernel(forest.height, len(forest.values), args.batch, args.rounds)
    
    analyze_slot_packing(kb.instrs, verbose=True)


if __name__ == "__main__":
    main()
