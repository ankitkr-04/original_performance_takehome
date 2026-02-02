"""
Instruction diff - compare instruction streams between two kernels side by side.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from problem import Tree, VLEN


def format_instr(instr):
    """Format a single instruction compactly."""
    parts = []
    if 'load' in instr:
        ops = instr['load']
        op_types = set(op[0] for op in ops)
        parts.append(f"L:{len(ops)}({','.join(sorted(op_types))})")
    if 'valu' in instr:
        ops = instr['valu']
        op_types = set(op[0] for op in ops)
        parts.append(f"V:{len(ops)}({','.join(sorted(op_types))})")
    if 'alu' in instr:
        ops = instr['alu']
        parts.append(f"A:{len(ops)}")
    if 'store' in instr:
        ops = instr['store']
        parts.append(f"S:{len(ops)}")
    if 'flow' in instr:
        parts.append("FLOW")
    return " | ".join(parts) if parts else "(empty)"


def diff_instructions(
    kernel_a: str = "perf_takehome",
    kernel_b: str = "takehome_diff",
    start: int = 0,
    count: int = 50,
    tree_height: int = 10,
    rounds: int = 16,
    batch_size: int = 256,
):
    """
    Show instruction streams from two kernels side by side.
    
    Useful for finding where implementations diverge.
    """
    # Import kernels
    if kernel_a == "perf_takehome":
        from perf_takehome import KernelBuilder as KB_A
    else:
        exec(f"from {kernel_a} import KernelBuilder as KB_A")
        KB_A = locals()['KB_A']
    
    if kernel_b == "takehome_diff":
        from takehome_diff import KernelBuilder as KB_B
    else:
        exec(f"from {kernel_b} import KernelBuilder as KB_B")
        KB_B = locals()['KB_B']
    
    random.seed(123)
    forest = Tree.generate(tree_height)
    
    kb_a = KB_A()
    kb_a.build_kernel(forest.height, len(forest.values), batch_size, rounds)
    
    kb_b = KB_B()
    kb_b.build_kernel(forest.height, len(forest.values), batch_size, rounds)
    
    instrs_a = kb_a.instrs
    instrs_b = kb_b.instrs
    
    # Build comparison
    lines = []
    lines.append(f"{'Idx':>4} | {kernel_a:^35} | {kernel_b:^35} | Match")
    lines.append("-" * 90)
    
    end = min(start + count, max(len(instrs_a), len(instrs_b)))
    
    matches = 0
    mismatches = 0
    
    for i in range(start, end):
        a_str = format_instr(instrs_a[i]) if i < len(instrs_a) else "(end)"
        b_str = format_instr(instrs_b[i]) if i < len(instrs_b) else "(end)"
        
        match = "✓" if a_str == b_str else "✗"
        if a_str == b_str:
            matches += 1
        else:
            mismatches += 1
        
        lines.append(f"{i:4d} | {a_str:^35} | {b_str:^35} | {match}")
    
    lines.append("-" * 90)
    lines.append(f"Match rate: {matches}/{matches+mismatches} ({100*matches/(matches+mismatches):.1f}%)")
    lines.append(f"Total instructions: {kernel_a}={len(instrs_a)}, {kernel_b}={len(instrs_b)}")
    
    return {
        "comparison": "\n".join(lines),
        "matches": matches,
        "mismatches": mismatches,
        "len_a": len(instrs_a),
        "len_b": len(instrs_b),
    }


def find_divergence(
    kernel_a: str = "perf_takehome",
    kernel_b: str = "takehome_diff",
    tree_height: int = 10,
    rounds: int = 16,
    batch_size: int = 256,
):
    """
    Find where two kernels first diverge in their instruction streams.
    """
    if kernel_a == "perf_takehome":
        from perf_takehome import KernelBuilder as KB_A
    else:
        from takehome_diff import KernelBuilder as KB_A
    
    if kernel_b == "takehome_diff":
        from takehome_diff import KernelBuilder as KB_B
    else:
        from perf_takehome import KernelBuilder as KB_B
    
    random.seed(123)
    forest = Tree.generate(tree_height)
    
    kb_a = KB_A()
    kb_a.build_kernel(forest.height, len(forest.values), batch_size, rounds)
    
    kb_b = KB_B()
    kb_b.build_kernel(forest.height, len(forest.values), batch_size, rounds)
    
    instrs_a = kb_a.instrs
    instrs_b = kb_b.instrs
    
    # Find first divergence
    first_diff = None
    for i in range(min(len(instrs_a), len(instrs_b))):
        if format_instr(instrs_a[i]) != format_instr(instrs_b[i]):
            first_diff = i
            break
    
    if first_diff is None and len(instrs_a) != len(instrs_b):
        first_diff = min(len(instrs_a), len(instrs_b))
    
    # Find regions of difference
    diff_regions = []
    in_diff = False
    region_start = None
    
    for i in range(max(len(instrs_a), len(instrs_b))):
        a_str = format_instr(instrs_a[i]) if i < len(instrs_a) else "(end)"
        b_str = format_instr(instrs_b[i]) if i < len(instrs_b) else "(end)"
        
        if a_str != b_str:
            if not in_diff:
                in_diff = True
                region_start = i
        else:
            if in_diff:
                in_diff = False
                diff_regions.append((region_start, i - 1))
    
    if in_diff:
        diff_regions.append((region_start, max(len(instrs_a), len(instrs_b)) - 1))
    
    return {
        "first_divergence": first_diff,
        "diff_regions": diff_regions[:10],  # First 10 regions
        "total_diff_regions": len(diff_regions),
        "len_a": len(instrs_a),
        "len_b": len(instrs_b),
    }


if __name__ == "__main__":
    print("=== Finding Divergence ===")
    div = find_divergence()
    print(f"First divergence at instruction: {div['first_divergence']}")
    print(f"Diff regions: {div['diff_regions'][:5]}")
    print(f"Total diff regions: {div['total_diff_regions']}")
    
    if div['first_divergence']:
        print(f"\n=== Instructions around divergence (idx {div['first_divergence']}) ===")
        diff = diff_instructions(start=max(0, div['first_divergence'] - 5), count=20)
        print(diff['comparison'])
