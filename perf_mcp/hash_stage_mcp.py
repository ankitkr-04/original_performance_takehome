#!/usr/bin/env python3
"""
Tier 3 MCP: Hash Stage Cost Analyzer
======================================
Recognize hash patterns, estimate cycles/stage, non-collapsed ops.
"""

import sys
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from problem import HASH_STAGES


def analyze_hash_stages(instrs: List[Dict], verbose: bool = True) -> Dict:
    """
    Analyze hash computation patterns and costs.
    
    Args:
        instrs: List of instruction dictionaries
        verbose: Print detailed analysis
    
    Returns:
        Dictionary with hash analysis
    """
    # Identify hash operations by pattern matching
    hash_ops = []
    
    # Hash patterns to look for:
    # 1. XOR operations (hash ^= value)
    # 2. multiply_add (optimized hash stages)
    # 3. Sequences of op1, op3, op2 (non-collapsed stages)
    
    xor_ops = []
    multiply_add_ops = []
    hash_stage_sequences = []
    
    for i, instr in enumerate(instrs):
        valu_ops = instr.get('valu', [])
        
        for op in valu_ops:
            if len(op) >= 2:
                if op[0] == '^':
                    xor_ops.append((i, op))
                elif op[0] == 'multiply_add':
                    multiply_add_ops.append((i, op))
        
        # Look for hash stage pattern: multiple ops in sequence
        if len(valu_ops) >= 2:
            ops_in_instr = [op[0] for op in valu_ops if len(op) > 0]
            # Check if it matches hash pattern
            has_add = '+' in ops_in_instr
            has_xor = '^' in ops_in_instr
            has_shift = '<<' in ops_in_instr or '>>' in ops_in_instr
            
            if (has_add or has_xor) and has_shift:
                hash_stage_sequences.append((i, valu_ops))
    
    # Estimate hash costs
    # From HASH_STAGES: 6 stages
    num_stages = len(HASH_STAGES)
    
    # Count collapsed vs non-collapsed implementations
    collapsed_stages = len(multiply_add_ops)
    non_collapsed_stages = len(hash_stage_sequences)
    
    # Estimate cycles per hash stage
    # Collapsed: 1 cycle (multiply_add)
    # Non-collapsed: 3 cycles (op1, op3, op2 in separate cycles)
    
    collapsed_cost = collapsed_stages * 1
    non_collapsed_cost = non_collapsed_stages * 3
    
    # Estimate number of hash computations
    # Each hash needs: 1 XOR + 6 stages
    estimated_hashes = len(xor_ops) // 2  # Rough estimate
    
    if verbose:
        print(f"\n{'='*70}")
        print("HASH STAGE COST ANALYSIS")
        print(f"{'='*70}")
        
        print(f"\n┌{'─'*68}┐")
        print(f"│ {'HASH STAGES REFERENCE':^66} │")
        print(f"├{'─'*68}┤")
        
        for i, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            stage_desc = f"Stage {i}: {op1}(val, {val1:08X}) {op2} {op3}(val, {val3})"
            print(f"│ {stage_desc:<64} │")
        print(f"└{'─'*68}┘")
        
        print(f"\n┌{'─'*68}┐")
        print(f"│ {'HASH OPERATION COUNTS':^66} │")
        print(f"├{'─'*68}┤")
        print(f"│ {'XOR operations (value ^= node):':<40} {len(xor_ops):>26} │")
        print(f"│ {'multiply_add ops (collapsed):':<40} {collapsed_stages:>26} │")
        print(f"│ {'Hash stage sequences (non-collapsed):':<40} {non_collapsed_stages:>26} │")
        print(f"│ {'Estimated hash computations:':<40} {estimated_hashes:>26} │")
        print(f"└{'─'*68}┘")
        
        print(f"\n┌{'─'*68}┐")
        print(f"│ {'CYCLE COST ESTIMATION':^66} │")
        print(f"├{'─'*68}┤")
        print(f"│ {'Collapsed stages cost:':<40} {collapsed_cost:>20} cycles │")
        print(f"│ {'Non-collapsed stages cost:':<40} {non_collapsed_cost:>20} cycles │")
        print(f"│ {'Total hash compute cost:':<40} {collapsed_cost + non_collapsed_cost:>20} cycles │")
        print(f"└{'─'*68}┘")
        
        # Analyze stage-by-stage patterns
        print(f"\n┌{'─'*68}┐")
        print(f"│ {'OPTIMIZATION OPPORTUNITIES':^66} │")
        print(f"├{'─'*68}┤")
        
        # Check for algebraic optimizations
        collapsed_ratio = collapsed_stages / (collapsed_stages + non_collapsed_stages) if (collapsed_stages + non_collapsed_stages) > 0 else 0
        
        print(f"│ {'Collapsed stage ratio:':<40} {collapsed_ratio*100:>24.1f}% │")
        
        if collapsed_ratio < 0.5:
            print(f"│ {'':<66} │")
            print(f"│ ⚠ Many non-collapsed stages detected                              │")
            print(f"│   Consider using multiply_add for stages with pattern:            │")
            print(f"│   val = (val * C1 + C2) where C1 = 1 + (1 << N)                   │")
        
        # Check for specific collapsible stages
        collapsible = []
        for i, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            if op1 == '+' and op2 == '+' and op3 == '<<':
                aux_val = 1 + (1 << val3)
                collapsible.append(i)
        
        if collapsible:
            print(f"│ {'':<66} │")
            print(f"│ {'Collapsible stages:':<40} {len(collapsible)}/{len(HASH_STAGES)}{' '*20}│")
            print(f"│   Stages {str(collapsible):<40}{' '*15}│")
        
        print(f"└{'─'*68}┘")
        
        # Recommendations
        print(f"\n┌{'─'*68}┐")
        print(f"│ {'RECOMMENDATIONS':^66} │")
        print(f"├{'─'*68}┤")
        
        if collapsed_ratio < 1.0 and len(collapsible) > 0:
            print(f"│ 💡 Collapse hash stages using multiply_add                        │")
            print(f"│    Potential savings: ~{non_collapsed_cost - collapsed_stages:<5} cycles{' '*29}│")
        
        if collapsed_ratio > 0.8:
            print(f"│ ✓ Excellent hash stage optimization ({collapsed_ratio*100:.0f}% collapsed)        │")
        
        # Check if hash ops are interleaved with loads
        hash_cycles = set(i for i, _ in xor_ops) | set(i for i, _ in multiply_add_ops)
        load_cycles = set(i for i, instr in enumerate(instrs) if instr.get('load'))
        
        overlapped = len(hash_cycles & load_cycles)
        overlap_ratio = overlapped / len(hash_cycles) if hash_cycles else 0
        
        print(f"│ {'':<66} │")
        print(f"│ {'Hash/Load overlap ratio:':<40} {overlap_ratio*100:>24.1f}% │")
        
        if overlap_ratio < 0.3:
            print(f"│ 💡 Interleave hash computation with load operations               │")
            print(f"│    Software pipeline to hide gather latency                       │")
        elif overlap_ratio > 0.7:
            print(f"│ ✓ Good overlap between hash and gather operations                 │")
        
        print(f"└{'─'*68}┘")
    
    return {
        'xor_ops': len(xor_ops),
        'collapsed_stages': collapsed_stages,
        'non_collapsed_stages': non_collapsed_stages,
        'collapsed_cost': collapsed_cost,
        'non_collapsed_cost': non_collapsed_cost,
        'total_cost': collapsed_cost + non_collapsed_cost,
        'estimated_hashes': estimated_hashes
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Hash Stage Cost MCP - Analyze hash computation efficiency")
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
    
    analyze_hash_stages(kb.instrs, verbose=True)


if __name__ == "__main__":
    main()
