#!/usr/bin/env python3
"""
Kernel Comparison MCP - Deep comparison between two kernel implementations
=========================================================================
Compares cycle counts, instruction breakdown, pipelining effectiveness,
and identifies exactly what changed between versions.
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from problem import SLOT_LIMITS, VLEN, Machine, Tree, Input, build_mem_image, reference_kernel2


@dataclass
class KernelMetrics:
    """Metrics for a single kernel"""
    name: str
    total_instrs: int
    total_cycles: int
    
    # Instruction type counts
    load_only_instrs: int
    valu_only_instrs: int
    combined_instrs: int
    store_instrs: int
    
    # Utilization
    load_utilization: float
    valu_utilization: float
    
    # Pipelining
    interleave_ratio: float  # combined / total
    
    # Scratch usage
    scratch_used: int


@dataclass
class KernelComparison:
    """Comparison between two kernels"""
    kernel_a: KernelMetrics
    kernel_b: KernelMetrics
    
    # Deltas
    cycle_delta: int
    cycle_delta_pct: float
    instr_delta: int
    
    # Specific differences
    combined_instr_delta: int
    load_only_delta: int
    valu_only_delta: int
    
    # Correctness
    both_correct: bool
    correctness_note: str
    
    # Analysis
    winner: str
    analysis: List[str]
    recommendations: List[str]


def analyze_kernel(name: str, kb: Any) -> KernelMetrics:
    """Analyze a single kernel's instruction stream."""
    instrs = kb.instrs
    
    load_only = 0
    valu_only = 0
    combined = 0
    store_only = 0
    
    total_loads = 0
    total_valus = 0
    
    for instr in instrs:
        has_load = 'load' in instr and len(instr['load']) > 0
        has_valu = 'valu' in instr and len(instr['valu']) > 0
        has_store = 'store' in instr and len(instr['store']) > 0
        
        if has_load:
            total_loads += len(instr['load'])
        if has_valu:
            total_valus += len(instr['valu'])
        
        if has_load and has_valu:
            combined += 1
        elif has_load and not has_valu:
            load_only += 1
        elif has_valu and not has_load:
            valu_only += 1
        elif has_store:
            store_only += 1
    
    load_util = (total_loads / (len(instrs) * SLOT_LIMITS['load']) * 100) if instrs else 0
    valu_util = (total_valus / (len(instrs) * SLOT_LIMITS['valu']) * 100) if instrs else 0
    interleave = combined / len(instrs) if instrs else 0
    
    scratch_used = kb.scratch_ptr if hasattr(kb, 'scratch_ptr') else 0
    
    return KernelMetrics(
        name=name,
        total_instrs=len(instrs),
        total_cycles=len(instrs),  # 1 cycle per instruction
        load_only_instrs=load_only,
        valu_only_instrs=valu_only,
        combined_instrs=combined,
        store_instrs=store_only,
        load_utilization=load_util,
        valu_utilization=valu_util,
        interleave_ratio=interleave,
        scratch_used=scratch_used
    )


def verify_correctness(kb: Any, tree_height: int, rounds: int, batch_size: int) -> Tuple[bool, str]:
    """Verify kernel produces correct results."""
    try:
        import random
        random.seed(42)
        
        forest = Tree.generate(tree_height)
        inp = Input.generate(forest, batch_size, rounds)
        mem = build_mem_image(forest, inp)
        
        machine = Machine(
            mem,
            kb.instrs,
            kb.debug_info(),
            trace=False
        )
        
        value_trace = {}
        for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
            machine.run()
            inp_values_p = ref_mem[6]
            if machine.mem[inp_values_p:inp_values_p + len(inp.values)] != ref_mem[inp_values_p:inp_values_p + len(inp.values)]:
                return False, f"Mismatch at round {i}"
        
        return True, "All rounds correct"
    except Exception as e:
        return False, f"Error: {str(e)}"


def compare_kernels(
    kernel_a_module: str,
    kernel_b_module: str,
    tree_height: int = 10,
    rounds: int = 16,
    batch_size: int = 256
) -> KernelComparison:
    """
    Compare two kernel implementations.
    
    Args:
        kernel_a_module: Module name for first kernel (e.g., 'perf_takehome')
        kernel_b_module: Module name for second kernel (e.g., 'takehome_diff')
    """
    import importlib
    
    # Import and build kernel A
    mod_a = importlib.import_module(kernel_a_module)
    kb_a = mod_a.KernelBuilder()
    # CORRECT: Tree has 2^(height+1) - 1 nodes
    n_nodes = 2 ** (tree_height + 1) - 1
    kb_a.build_kernel(tree_height, n_nodes, batch_size, rounds)
    
    # Import and build kernel B
    mod_b = importlib.import_module(kernel_b_module)
    kb_b = mod_b.KernelBuilder()
    kb_b.build_kernel(tree_height, n_nodes, batch_size, rounds)
    
    # Analyze both
    metrics_a = analyze_kernel(kernel_a_module, kb_a)
    metrics_b = analyze_kernel(kernel_b_module, kb_b)
    
    # Check correctness
    correct_a, note_a = verify_correctness(kb_a, tree_height, rounds, batch_size)
    correct_b, note_b = verify_correctness(kb_b, tree_height, rounds, batch_size)
    
    both_correct = correct_a and correct_b
    correctness_note = f"A: {note_a}, B: {note_b}"
    
    # Calculate deltas
    cycle_delta = metrics_b.total_cycles - metrics_a.total_cycles
    cycle_delta_pct = (cycle_delta / metrics_a.total_cycles * 100) if metrics_a.total_cycles else 0
    instr_delta = metrics_b.total_instrs - metrics_a.total_instrs
    
    combined_delta = metrics_b.combined_instrs - metrics_a.combined_instrs
    load_only_delta = metrics_b.load_only_instrs - metrics_a.load_only_instrs
    valu_only_delta = metrics_b.valu_only_instrs - metrics_a.valu_only_instrs
    
    # Determine winner
    if not correct_b and correct_a:
        winner = kernel_a_module
    elif not correct_a and correct_b:
        winner = kernel_b_module
    elif metrics_b.total_cycles < metrics_a.total_cycles:
        winner = kernel_b_module
    else:
        winner = kernel_a_module
    
    # Generate analysis
    analysis = []
    
    if cycle_delta < 0:
        analysis.append(f"✓ {kernel_b_module} is {-cycle_delta} cycles faster ({-cycle_delta_pct:.1f}% improvement)")
    elif cycle_delta > 0:
        analysis.append(f"✗ {kernel_b_module} is {cycle_delta} cycles slower ({cycle_delta_pct:.1f}% regression)")
    else:
        analysis.append("= Same cycle count")
    
    if combined_delta > 0:
        analysis.append(f"✓ Better interleaving: +{combined_delta} combined load+valu instructions")
    elif combined_delta < 0:
        analysis.append(f"✗ Worse interleaving: {combined_delta} fewer combined instructions")
    
    if load_only_delta < 0:
        analysis.append(f"✓ Fewer load-only instructions: {load_only_delta}")
    elif load_only_delta > 0:
        analysis.append(f"✗ More load-only instructions: +{load_only_delta}")
    
    # Generate recommendations
    recommendations = []
    
    if combined_delta < 0:
        recommendations.append(
            "The new kernel has worse instruction interleaving. "
            "Check emit_gather_with_hash to ensure loads and valus are combined."
        )
    
    if not correct_b:
        recommendations.append(
            f"⚠ {kernel_b_module} produces incorrect results! Debug before optimizing."
        )
    
    if cycle_delta > 0 and combined_delta < 0:
        recommendations.append(
            f"The cycle regression ({cycle_delta}) is likely due to worse interleaving ({combined_delta} fewer combined instrs). "
            "Each un-interleaved instruction costs 1 extra cycle."
        )
    
    return KernelComparison(
        kernel_a=metrics_a,
        kernel_b=metrics_b,
        cycle_delta=cycle_delta,
        cycle_delta_pct=cycle_delta_pct,
        instr_delta=instr_delta,
        combined_instr_delta=combined_delta,
        load_only_delta=load_only_delta,
        valu_only_delta=valu_only_delta,
        both_correct=both_correct,
        correctness_note=correctness_note,
        winner=winner,
        analysis=analysis,
        recommendations=recommendations
    )


def format_kernel_comparison(comp: KernelComparison) -> str:
    """Format comparison as readable report."""
    lines = [
        "=" * 70,
        "KERNEL COMPARISON",
        "=" * 70,
        "",
        f"Comparing: {comp.kernel_a.name} vs {comp.kernel_b.name}",
        "",
        "┌" + "─" * 68 + "┐",
        "│                         METRICS                                   │",
        "├" + "─" * 68 + "┤",
        f"│ Metric              │ {comp.kernel_a.name:>20} │ {comp.kernel_b.name:>20} │",
        "├" + "─" * 68 + "┤",
        f"│ Total Cycles        │ {comp.kernel_a.total_cycles:>20} │ {comp.kernel_b.total_cycles:>20} │",
        f"│ Combined Instrs     │ {comp.kernel_a.combined_instrs:>20} │ {comp.kernel_b.combined_instrs:>20} │",
        f"│ Load-only Instrs    │ {comp.kernel_a.load_only_instrs:>20} │ {comp.kernel_b.load_only_instrs:>20} │",
        f"│ VALU-only Instrs    │ {comp.kernel_a.valu_only_instrs:>20} │ {comp.kernel_b.valu_only_instrs:>20} │",
        f"│ Interleave Ratio    │ {comp.kernel_a.interleave_ratio:>19.1%} │ {comp.kernel_b.interleave_ratio:>19.1%} │",
        f"│ Load Utilization    │ {comp.kernel_a.load_utilization:>19.1f}% │ {comp.kernel_b.load_utilization:>19.1f}% │",
        f"│ VALU Utilization    │ {comp.kernel_a.valu_utilization:>19.1f}% │ {comp.kernel_b.valu_utilization:>19.1f}% │",
        f"│ Scratch Used        │ {comp.kernel_a.scratch_used:>20} │ {comp.kernel_b.scratch_used:>20} │",
        "└" + "─" * 68 + "┘",
        "",
        "┌" + "─" * 68 + "┐",
        "│                         DELTAS                                    │",
        "├" + "─" * 68 + "┤",
        f"│ Cycle Delta:           {comp.cycle_delta:+6d} ({comp.cycle_delta_pct:+.1f}%)                       │",
        f"│ Combined Instr Delta:  {comp.combined_instr_delta:+6d}                                    │",
        f"│ Load-only Delta:       {comp.load_only_delta:+6d}                                    │",
        f"│ VALU-only Delta:       {comp.valu_only_delta:+6d}                                    │",
        "└" + "─" * 68 + "┘",
        "",
        f"Correctness: {comp.correctness_note}",
        f"Winner: {comp.winner}",
        ""
    ]
    
    if comp.analysis:
        lines.append("ANALYSIS:")
        for item in comp.analysis:
            lines.append(f"  {item}")
        lines.append("")
    
    if comp.recommendations:
        lines.append("RECOMMENDATIONS:")
        for i, rec in enumerate(comp.recommendations, 1):
            lines.append(f"  {i}. {rec}")
    
    return "\n".join(lines)


if __name__ == "__main__":
    comparison = compare_kernels('perf_takehome', 'takehome_diff')
    print(format_kernel_comparison(comparison))
