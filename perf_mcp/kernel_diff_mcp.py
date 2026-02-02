#!/usr/bin/env python3
"""
Tier 1 MCP: Kernel Diff
========================
Compare two kernel versions showing cycle deltas, round attribution, utilization differences.
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional
import difflib

sys.path.insert(0, str(Path(__file__).parent.parent))

from perf_mcp.trace_mcp import analyze_trace, run_and_trace
import json


@dataclass
class DiffStats:
    """Comparison statistics between two kernels"""
    version_a_cycles: int
    version_b_cycles: int
    delta_cycles: int
    delta_percent: float
    
    # Utilization deltas
    load_util_delta: float
    valu_util_delta: float
    alu_util_delta: float
    store_util_delta: float
    
    # Other metrics
    bubble_delta: int
    tail_drain_delta: int
    avg_slots_delta: float


def compare_kernels(trace_a: str, trace_b: str, label_a: str = "Version A", 
                   label_b: str = "Version B", verbose: bool = True) -> DiffStats:
    """
    Compare two kernel traces and show differences.
    
    Args:
        trace_a: Path to first trace
        trace_b: Path to second trace
        label_a: Label for first version
        label_b: Label for second version
        verbose: Print detailed comparison
    
    Returns:
        DiffStats with comparison metrics
    """
    print(f"Loading {label_a}: {trace_a}")
    stats_a = analyze_trace(trace_a, verbose=False)
    
    print(f"Loading {label_b}: {trace_b}")
    stats_b = analyze_trace(trace_b, verbose=False)
    
    delta_cycles = stats_b.total_cycles - stats_a.total_cycles
    delta_percent = (delta_cycles / stats_a.total_cycles * 100) if stats_a.total_cycles > 0 else 0
    
    diff = DiffStats(
        version_a_cycles=stats_a.total_cycles,
        version_b_cycles=stats_b.total_cycles,
        delta_cycles=delta_cycles,
        delta_percent=delta_percent,
        load_util_delta=stats_b.load_utilization - stats_a.load_utilization,
        valu_util_delta=stats_b.valu_utilization - stats_a.valu_utilization,
        alu_util_delta=stats_b.alu_utilization - stats_a.alu_utilization,
        store_util_delta=stats_b.store_utilization - stats_a.store_utilization,
        bubble_delta=stats_b.bubble_cycles - stats_a.bubble_cycles,
        tail_drain_delta=stats_b.tail_drain_cycles - stats_a.tail_drain_cycles,
        avg_slots_delta=stats_b.avg_slots_per_instr - stats_a.avg_slots_per_instr
    )
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"KERNEL COMPARISON: {label_a} vs {label_b}")
        print(f"{'='*70}")
        
        # Cycle comparison
        print(f"\n┌{'─'*68}┐")
        print(f"│ {'CYCLE COUNT':^66} │")
        print(f"├{'─'*68}┤")
        print(f"│ {label_a:<30}: {stats_a.total_cycles:>8} cycles{' '*20}│")
        print(f"│ {label_b:<30}: {stats_b.total_cycles:>8} cycles{' '*20}│")
        print(f"├{'─'*68}┤")
        
        delta_sign = "+" if delta_cycles > 0 else ""
        delta_color = "SLOWER" if delta_cycles > 0 else "FASTER"
        print(f"│ {'Delta:':<30} {delta_sign}{delta_cycles:>8} cycles ({delta_sign}{delta_percent:+.2f}%){' '*8}│")
        print(f"│ {'':<30} {'>>> ' + delta_color if delta_cycles != 0 else 'SAME':<36}│")
        print(f"└{'─'*68}┘")
        
        # Utilization comparison
        print(f"\n┌{'─'*68}┐")
        print(f"│ {'ENGINE UTILIZATION DELTA':^66} │")
        print(f"├{'─'*68}┤")
        print(f"│ {'Engine':<12} │ {label_a[:12]:^12} │ {label_b[:12]:^12} │ {'Delta':^12} │")
        print(f"├{'─'*68}┤")
        
        def format_delta(delta):
            sign = "+" if delta > 0 else ""
            return f"{sign}{delta:.2f}%"
        
        print(f"│ {'Load':<12} │ {stats_a.load_utilization:>11.2f}% │ {stats_b.load_utilization:>11.2f}% │ {format_delta(diff.load_util_delta):>12} │")
        print(f"│ {'Valu':<12} │ {stats_a.valu_utilization:>11.2f}% │ {stats_b.valu_utilization:>11.2f}% │ {format_delta(diff.valu_util_delta):>12} │")
        print(f"│ {'ALU':<12} │ {stats_a.alu_utilization:>11.2f}% │ {stats_b.alu_utilization:>11.2f}% │ {format_delta(diff.alu_util_delta):>12} │")
        print(f"│ {'Store':<12} │ {stats_a.store_utilization:>11.2f}% │ {stats_b.store_utilization:>11.2f}% │ {format_delta(diff.store_util_delta):>12} │")
        print(f"└{'─'*68}┘")
        
        # Other metrics
        print(f"\n┌{'─'*68}┐")
        print(f"│ {'OTHER METRICS':^66} │")
        print(f"├{'─'*68}┤")
        print(f"│ {'Metric':<20} │ {label_a[:12]:^12} │ {label_b[:12]:^12} │ {'Delta':^12} │")
        print(f"├{'─'*68}┤")
        print(f"│ {'Bubble Cycles':<20} │ {stats_a.bubble_cycles:>12} │ {stats_b.bubble_cycles:>12} │ {diff.bubble_delta:>+12} │")
        print(f"│ {'Tail Drain':<20} │ {stats_a.tail_drain_cycles:>12} │ {stats_b.tail_drain_cycles:>12} │ {diff.tail_drain_delta:>+12} │")
        print(f"│ {'Avg Slots/Instr':<20} │ {stats_a.avg_slots_per_instr:>12.2f} │ {stats_b.avg_slots_per_instr:>12.2f} │ {diff.avg_slots_delta:>+12.2f} │")
        print(f"└{'─'*68}┘")
        
        # Bottleneck analysis
        print(f"\n┌{'─'*68}┐")
        print(f"│ {'BOTTLENECK ANALYSIS':^66} │")
        print(f"├{'─'*68}┤")
        
        bottlenecks_a = []
        bottlenecks_b = []
        
        if stats_a.load_utilization < 50:
            bottlenecks_a.append(f"Load underutilized ({stats_a.load_utilization:.1f}%)")
        if stats_a.valu_utilization < 50:
            bottlenecks_a.append(f"Valu underutilized ({stats_a.valu_utilization:.1f}%)")
        if stats_a.tail_drain_cycles > stats_a.total_cycles * 0.05:
            bottlenecks_a.append(f"Long tail drain ({stats_a.tail_drain_cycles} cycles)")
        
        if stats_b.load_utilization < 50:
            bottlenecks_b.append(f"Load underutilized ({stats_b.load_utilization:.1f}%)")
        if stats_b.valu_utilization < 50:
            bottlenecks_b.append(f"Valu underutilized ({stats_b.valu_utilization:.1f}%)")
        if stats_b.tail_drain_cycles > stats_b.total_cycles * 0.05:
            bottlenecks_b.append(f"Long tail drain ({stats_b.tail_drain_cycles} cycles)")
        
        print(f"│ {label_a + ':':<66} │")
        if bottlenecks_a:
            for b in bottlenecks_a:
                print(f"│   - {b:<62} │")
        else:
            print(f"│   {'No obvious bottlenecks':<64} │")
        
        print(f"│ {'':<66} │")
        print(f"│ {label_b + ':':<66} │")
        if bottlenecks_b:
            for b in bottlenecks_b:
                print(f"│   - {b:<62} │")
        else:
            print(f"│   {'No obvious bottlenecks':<64} │")
        
        print(f"└{'─'*68}┘")
    
    return diff


def compare_code(file_a: str, file_b: str, label_a: str = "Version A", 
                 label_b: str = "Version B"):
    """
    Show unified diff between two kernel implementations.
    """
    with open(file_a, 'r') as f:
        lines_a = f.readlines()
    
    with open(file_b, 'r') as f:
        lines_b = f.readlines()
    
    diff = difflib.unified_diff(
        lines_a, lines_b,
        fromfile=label_a,
        tofile=label_b,
        lineterm=''
    )
    
    print(f"\n{'='*70}")
    print(f"CODE DIFF: {label_a} vs {label_b}")
    print(f"{'='*70}\n")
    
    for line in diff:
        if line.startswith('+'):
            print(f"\033[92m{line}\033[0m")  # Green
        elif line.startswith('-'):
            print(f"\033[91m{line}\033[0m")  # Red
        elif line.startswith('@@'):
            print(f"\033[94m{line}\033[0m")  # Blue
        else:
            print(line)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Kernel Diff MCP - Compare kernel versions")
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Diff command
    diff_parser = subparsers.add_parser('diff', help='Compare two traces')
    diff_parser.add_argument('trace_a', help='First trace file')
    diff_parser.add_argument('trace_b', help='Second trace file')
    diff_parser.add_argument('--label-a', default='Version A', help='Label for first version')
    diff_parser.add_argument('--label-b', default='Version B', help='Label for second version')
    diff_parser.add_argument('--json', help='Export comparison to JSON')
    
    # Code diff command
    code_parser = subparsers.add_parser('code-diff', help='Show code differences')
    code_parser.add_argument('file_a', help='First Python file')
    code_parser.add_argument('file_b', help='Second Python file')
    code_parser.add_argument('--label-a', default='Version A', help='Label for first version')
    code_parser.add_argument('--label-b', default='Version B', help='Label for second version')
    
    args = parser.parse_args()
    
    if args.command == 'diff':
        diff = compare_kernels(args.trace_a, args.trace_b, args.label_a, args.label_b)
        
        if args.json:
            import json
            with open(args.json, 'w') as f:
                json.dump({
                    'version_a': {
                        'label': args.label_a,
                        'cycles': diff.version_a_cycles
                    },
                    'version_b': {
                        'label': args.label_b,
                        'cycles': diff.version_b_cycles
                    },
                    'delta': {
                        'cycles': diff.delta_cycles,
                        'percent': diff.delta_percent
                    },
                    'utilization_delta': {
                        'load': diff.load_util_delta,
                        'valu': diff.valu_util_delta,
                        'alu': diff.alu_util_delta,
                        'store': diff.store_util_delta
                    }
                }, f, indent=2)
            print(f"\n✓ Comparison exported to: {args.json}")
    
    elif args.command == 'code-diff':
        compare_code(args.file_a, args.file_b, args.label_a, args.label_b)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
